"""
═══════════════════════════════════════════════════════════════════════════════
TRANSLATOR v3 - NLLB avec nettoyage texte OCR + GPU FIX
═══════════════════════════════════════════════════════════════════════════════

FIX MAJEUR: Nettoyage du texte OCR AVANT traduction.
EasyOCR produit souvent des artefacts :
  - Mots collés : "IWANTED" → "I WANTED"
  - Ponctuation collée : "STRENGTHI" → "STRENGTH!"  
  - Underscores : "HIM_" → "HIM."

GPU FIX: Force explicitement le modèle sur CUDA avec vérification

Config: model_name dans settings.py
  - "facebook/nllb-200-distilled-600M"  (rapide, ~1.5GB VRAM)
  - "facebook/nllb-200-distilled-1.3B"  (meilleur, ~3GB VRAM)
"""

import re
import json
import numpy as np
import torch
from typing import List, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from utils import CacheManager, ImageUtils

try:
    import torch
except ImportError:
    raise RuntimeError("PyTorch requis: pip install torch")


# ═════════════════════════════════════════════════════════════════════════════
# NETTOYAGE TEXTE OCR
# ═════════════════════════════════════════════════════════════════════════════
def should_skip_translation(self, text: str) -> bool:
    if not text:
        return True
    
    # Skip si coréen (pas de lettres latines)
    if not any(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' for c in text):
        return True
    
    # ... reste du code
def clean_ocr_text(text: str) -> str:
    """
    Nettoie le texte brut OCR avant traduction.
    Corrige les artefacts typiques sur du texte manga.
    """
    if not text:
        return text
    
    # Retirer underscores (OCR lit _ au lieu de . ou espace)
    text = re.sub(r'_+$', '.', text)
    text = re.sub(r'_+', ' ', text)
    
    # Fix mots collés avec I majuscule en début
    # "IWANTED" → "I WANTED", "ICOULD" → "I COULD"
    text = re.sub(r'\bI([A-Z]{2,})', r'I \1', text)
    
    # Fix I parasite en fin de mot majuscule
    # "STRENGTHI" → "STRENGTH!", "SURVIVEI" → "SURVIVE!"  
    text = re.sub(r'([A-Z]{3,})I\b', r'\1!', text)
    
    # Fix ; → , (OCR confond souvent)
    text = text.replace(';', ',')
    
    # Fix : en fin de phrase → .
    text = re.sub(r':\s*$', '.', text)

    # Fix fréquent PP-OCR: "1." reconnu à la place de "I."
    # ex: "YOU'RE TELLING ME THAT 1. THE BEST..." -> "... I. THE BEST..."
    text = re.sub(r'\b1\.(?=\s+[A-Z])', 'I.', text)

    # OCR ponctuation: "I. THE" est souvent "I, THE"
    text = re.sub(r'\bI\.(?=\s+THE\b)', 'I,', text)

    # Un "1" isolé entre mots majuscules est souvent un "I"
    text = re.sub(r'(?<=[A-Z])\s+1\s+(?=[A-Z])', ' I ', text)

    # Corrections OCR ciblées (webtoon)
    text = re.sub(r'\bDALIGHTER\b', 'DAUGHTER', text, flags=re.IGNORECASE)
    text = re.sub(r'\bDAIGTHER\b', 'DAUGHTER', text, flags=re.IGNORECASE)
    text = re.sub(r'\bWHERE\s+WE\s+ARE\?', 'WHERE ARE WE?', text, flags=re.IGNORECASE)
    
    # Nettoyer espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


class TranslationGroup:
    def __init__(self, detections: list):
        self.detections = detections
        self.combined_text: Optional[str] = None
        self.translation: Optional[str] = None
    
    def get_center(self) -> Tuple[float, float]:
        if not self.detections:
            return (0, 0)
        centers = [((d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2) for d in self.detections]
        return (
            sum(c[0] for c in centers) / len(centers),
            sum(c[1] for c in centers) / len(centers)
        )


class NLLBTranslator:
    """Traducteur NLLB/LLM local avec nettoyage OCR et GPU fix"""

    SINGLE_WORD_NON_NAME_STOPLIST = {
        "perhaps", "later", "hello", "look", "wait", "please", "help",
        "where", "what", "when", "why", "how", "there", "here",
        "yes", "no", "stop", "start", "wake", "time", "moment"
    }

    MULTI_WORD_NON_NAME_STOPLIST = {
        "oh", "there", "here", "please", "help", "wait", "look",
        "yes", "no", "what", "when", "where", "why", "how",
        "stop", "start", "go", "come", "now", "again"
    }

    LANGUAGE_STOPWORDS = {
        "en": {"the", "and", "you", "are", "what", "there", "here", "oh", "this", "that"},
        "fr": {"le", "la", "les", "et", "vous", "que", "est", "pas", "une", "des"},
        "es": {"el", "la", "los", "las", "y", "que", "una", "por", "para", "está"},
        "de": {"der", "die", "das", "und", "ist", "nicht", "ein", "eine", "mit", "ich"},
        "it": {"il", "lo", "la", "gli", "le", "e", "che", "non", "una", "con"},
        "pt": {"o", "a", "os", "as", "e", "que", "não", "uma", "com", "para"},
    }
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.cfg = config.translation
        self.backend = getattr(self.cfg, 'backend', 'nllb')
        self.tokenizer = None
        self.model = None
        self.cache = None
        self.name_memory_file = config.TRANSLATION_CACHE_DIR / "name_memory_v1.json"
        self.name_memory = self._load_name_memory()
        
        if self.cfg.enable_cache:
            cache_file = config.TRANSLATION_CACHE_DIR / self.cfg.cache_file
            self.cache = CacheManager(cache_file, max_size_mb=config.performance.cache_max_size_mb)
        
        self._load_model()

    def _load_name_memory(self) -> dict:
        try:
            if self.name_memory_file.exists():
                with open(self.name_memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
        except Exception:
            pass
        return {}

    def _save_name_memory(self):
        try:
            self.name_memory_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.name_memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.name_memory, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @staticmethod
    def _normalize_text_key(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.strip().upper())
        return normalized

    @staticmethod
    def _post_process_french(text: str) -> str:
        if not text:
            return text
        text = re.sub(r"\bJe y\b", "J'y", text)
        text = re.sub(r"\bje y\b", "j'y", text)
        text = re.sub(r"\bde déchets\b", "d'ordure", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _is_mostly_uppercase(text: str) -> bool:
        letters = [ch for ch in text if ch.isalpha()]
        if len(letters) < 6:
            return False
        upper_count = sum(1 for ch in letters if ch.isupper())
        return (upper_count / max(1, len(letters))) >= 0.75

    @staticmethod
    def _normalize_case_for_translation(text: str) -> str:
        """Normalise les textes OCR en MAJUSCULES pour améliorer la traduction."""
        if not text:
            return text

        if not NLLBTranslator._is_mostly_uppercase(text):
            return text

        normalized = text.lower()
        normalized = re.sub(r"\bi\b", "I", normalized)

        # Majuscule sur le premier caractère alphabétique
        chars = list(normalized)
        for idx, ch in enumerate(chars):
            if ch.isalpha():
                chars[idx] = ch.upper()
                break
        normalized = ''.join(chars)
        return normalized

    @staticmethod
    def _name_key(text: str) -> Optional[str]:
        if not text:
            return None
        cleaned = re.sub(r"[^A-Za-z'\-\s]", " ", text.upper())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return None
        words = cleaned.split()
        if 2 <= len(words) <= 4 and all(len(w) >= 2 for w in words):
            if all(re.fullmatch(r"[A-Z][A-Z'\-]*", w) for w in words):
                return " ".join(words)
        return None

    @classmethod
    def _looks_like_uppercase_name_sequence(cls, text: str) -> bool:
        tokens = re.findall(r"[A-Z][A-Z'\-]+", text.upper())
        if not (2 <= len(tokens) <= 4):
            return False

        # Si le groupe contient des mots usuels de dialogue, ce n'est pas un nom.
        if any(tok.lower() in cls.MULTI_WORD_NON_NAME_STOPLIST for tok in tokens):
            return False

        # Évite de classer des segments trop courts comme des noms.
        # Ex: "OH THERE" -> rejeté, "GHISLAIN PERDIUM" -> accepté.
        if any(len(tok) < 3 for tok in tokens):
            return False

        return True

    def _detect_source_language(self, text: str) -> str:
        return self._detect_source_language_with_confidence(text)[0]

    def _detect_source_language_with_confidence(self, text: str) -> Tuple[str, float]:
        fallback = self.cfg.fallback_source_lang if self.cfg.fallback_source_lang in self.cfg.lang_codes else self.cfg.source_lang

        if not self.cfg.auto_detect_source_lang:
            return self.cfg.source_lang, 1.0

        if not text:
            return fallback, 0.4

        # Scripts non-latins (detection robuste)
        if re.search(r"[\uAC00-\uD7AF]", text):
            return "ko", 0.99
        if re.search(r"[\u3040-\u30FF]", text):
            return "ja", 0.99
        if re.search(r"[\u4E00-\u9FFF]", text):
            return "zh", 0.99
        if re.search(r"[\u0400-\u04FF]", text):
            return "ru", 0.99

        # Latin: heuristique légère par stopwords
        words = re.findall(r"[A-Za-zÀ-ÿ']+", text.lower())
        if not words:
            return fallback, 0.4

        scores = {}
        for lang, stopwords in self.LANGUAGE_STOPWORDS.items():
            scores[lang] = sum(1 for w in words if w in stopwords)

        best_lang = max(scores, key=scores.get)
        sorted_scores = sorted(scores.values(), reverse=True)
        best_score = sorted_scores[0] if sorted_scores else 0
        second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0
        if best_score >= 1:
            margin = max(0, best_score - second_score)
            confidence = min(0.95, 0.60 + 0.15 * best_score + 0.08 * margin)
            return best_lang, float(confidence)

        # Si latin sans signal fort, on garde l'anglais (cas OCR webtoon le plus fréquent)
        return ("en" if "en" in self.cfg.lang_codes else fallback), 0.5

    def detect_source_language(self, text: str) -> str:
        """API publique pour debug/inspection de la langue source détectée."""
        return self._detect_source_language_with_confidence(text)[0]

    def detect_source_language_with_confidence(self, text: str) -> Tuple[str, float]:
        """Retourne (langue, confiance) pour le texte OCR."""
        return self._detect_source_language_with_confidence(text)

    @staticmethod
    def _is_single_proper_name(text: str) -> bool:
        stripped = text.strip()
        if not re.fullmatch(r"[A-Z][a-z]{2,20}[.!?]?", stripped):
            return False
        token = re.sub(r"[.!?]$", "", stripped).lower()
        if token in NLLBTranslator.SINGLE_WORD_NON_NAME_STOPLIST:
            return False
        return True
    
    def _load_model(self):
        if self.backend == 'local_llm':
            self._load_local_llm_model()
            return

        try:
            from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
            
            model_name = self.cfg.model_name
            print(f"⏳ Chargement NLLB: {model_name}...")
            
            self.tokenizer = NllbTokenizer.from_pretrained(
                model_name,
                cache_dir=str(config.TRANSLATION_CACHE_DIR),
                trust_remote_code=True
            )
            
            # ✅ FIX GPU: Forcer dtype correctement
            dtype = torch.float16 if self.device == 'cuda' and self.cfg.use_fp16 else torch.float32
            quantization_config = None

            if self.device == 'cuda' and self.cfg.use_bitsandbytes:
                try:
                    from transformers import BitsAndBytesConfig
                    if self.cfg.bnb_4bit:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=dtype,
                            bnb_4bit_quant_type='nf4',
                            bnb_4bit_use_double_quant=True,
                        )
                        print("   Quantization: bitsandbytes 4-bit (nf4)")
                    elif self.cfg.bnb_8bit:
                        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                        print("   Quantization: bitsandbytes 8-bit")
                    else:
                        print("   Quantization: bitsandbytes activé mais aucun mode sélectionné")
                except Exception as quant_error:
                    print(f"⚠️  BitsAndBytes indisponible ({quant_error}) -> fallback FP16/FP32")
                    quantization_config = None
            
            print(f"   Dtype: {dtype}")
            print(f"   Device: {self.device}")

            model_kwargs = {
                'cache_dir': str(config.TRANSLATION_CACHE_DIR),
                'trust_remote_code': True,
                'use_safetensors': True,
                'low_cpu_mem_usage': True,
            }

            if quantization_config is not None:
                model_kwargs['quantization_config'] = quantization_config
                model_kwargs['device_map'] = 'auto'
            else:
                model_kwargs['dtype'] = dtype

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                **model_kwargs,
            )
            
            # ✅ FIX GPU: Vérifier et mettre sur GPU
            print(f"   Model device before .to(): {next(self.model.parameters()).device}")
            
            if self.device == 'cuda' and quantization_config is None:
                if not torch.cuda.is_available():
                    print("⚠️  CUDA not available! Falling back to CPU (SLOW)")
                    self.device = 'cpu'
                else:
                    print(f"   CUDA available: {torch.cuda.is_available()}")
                    print(f"   CUDA device: {torch.cuda.current_device()}")
                    print(f"   Moving model to GPU...")
                    
                    # Mettre le modèle sur GPU
                    self.model = self.model.to('cuda')
                    
                    # Vérifier que c'est bien sur GPU
                    device_after = next(self.model.parameters()).device
                    print(f"   Model device after .to(cuda): {device_after}")
                    
                    # Logs VRAM
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / (1024**3)
                        reserved = torch.cuda.memory_reserved() / (1024**3)
                        print(f"   VRAM allocated: {allocated:.2f} GB")
                        print(f"   VRAM reserved: {reserved:.2f} GB")
            elif self.device == 'cuda' and quantization_config is not None:
                print("   Device map géré automatiquement par bitsandbytes")
            
            self.model.eval()
            print(f"✅ NLLB chargé ! ({model_name})")
            print(f"✅ Model running on: {next(self.model.parameters()).device}\n")
            
        except Exception as e:
            raise RuntimeError(f"Erreur chargement NLLB: {e}")

    def _load_local_llm_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

            model_name = self.cfg.llm_model_name
            print(f"⏳ Chargement LLM local: {model_name}...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(config.TRANSLATION_CACHE_DIR),
                trust_remote_code=True,
            )

            dtype = torch.float16 if self.device == 'cuda' and self.cfg.use_fp16 else torch.float32
            quantization_config = None

            require_cuda = bool(getattr(self.cfg, 'llm_require_cuda', True))
            if self.device == 'cuda' and not torch.cuda.is_available():
                message = "CUDA demandé pour le LLM mais torch ne voit pas de GPU (build CPU ou drivers absents)."
                if require_cuda:
                    raise RuntimeError(message)
                print(f"⚠️  {message} Fallback CPU activé.")
                self.device = 'cpu'

            if self.device == 'cuda' and self.cfg.use_bitsandbytes:
                try:
                    if self.cfg.bnb_4bit:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=dtype,
                            bnb_4bit_quant_type='nf4',
                            bnb_4bit_use_double_quant=True,
                        )
                        print("   Quantization LLM: bitsandbytes 4-bit (nf4)")
                    elif self.cfg.bnb_8bit:
                        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                        print("   Quantization LLM: bitsandbytes 8-bit")
                except Exception as quant_error:
                    print(f"⚠️  BitsAndBytes indisponible ({quant_error}) -> fallback FP16/FP32")
                    quantization_config = None

            model_kwargs = {
                'cache_dir': str(config.TRANSLATION_CACHE_DIR),
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,
            }

            if quantization_config is not None:
                model_kwargs['quantization_config'] = quantization_config
                model_kwargs['device_map'] = 'auto'
            else:
                model_kwargs['dtype'] = dtype

            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

            if self.device == 'cuda' and quantization_config is None and torch.cuda.is_available():
                self.model = self.model.to('cuda')

            self.model.eval()
            print(f"✅ LLM local chargé ! ({model_name})")
            print(f"✅ Model running on: {next(self.model.parameters()).device}\n")

        except Exception as e:
            raise RuntimeError(f"Erreur chargement LLM local: {e}")

    def _build_llm_prompt(self, text: str, source_lang_code: str) -> str:
        source_lang = source_lang_code or self.cfg.source_lang
        target_lang = self.cfg.target_lang
        template = getattr(self.cfg, 'llm_prompt_template', None) or (
            "Translate from {source_lang} to {target_lang}. Output only the translation.\n"
            "TEXT:\n{text}\nTRANSLATION:"
        )
        return template.format(source_lang=source_lang, target_lang=target_lang, text=text)

    @staticmethod
    def _extract_llm_translation(raw_output: str, prompt: str) -> str:
        if not raw_output:
            return ""
        content = raw_output[len(prompt):] if raw_output.startswith(prompt) else raw_output
        content = content.strip()

        if "TRANSLATION:" in content:
            content = content.split("TRANSLATION:", 1)[1].strip()

        marker_match = re.split(r"\n\s*(Human|User|Assistant|System)\s*:\s*", content, maxsplit=1)
        if marker_match:
            content = marker_match[0].strip()

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return ""

        first_line = lines[0]
        if first_line.startswith(('"', "'")) and first_line.endswith(('"', "'")) and len(first_line) >= 2:
            first_line = first_line[1:-1].strip()

        return first_line

    def _translate_with_local_llm(self, source_text: str, source_lang_code: str) -> str:
        prompt = self._build_llm_prompt(source_text, source_lang_code)

        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {
                    'role': 'system',
                    'content': 'You are a translation engine. Return only the translated text on a single line.',
                },
                {'role': 'user', 'content': prompt},
            ]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors='pt',
            )
            if isinstance(inputs, torch.Tensor):
                input_ids = inputs
                attention_mask = torch.ones_like(input_ids)
                inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        else:
            inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=self.cfg.max_length)

        if self.device == 'cuda' and torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        generate_kwargs = {
            'max_new_tokens': int(getattr(self.cfg, 'llm_max_new_tokens', 220)),
            'repetition_penalty': float(getattr(self.cfg, 'llm_repetition_penalty', 1.05)),
            'pad_token_id': self.tokenizer.eos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }

        temperature = float(getattr(self.cfg, 'llm_temperature', 0.0))
        top_p = float(getattr(self.cfg, 'llm_top_p', 1.0))
        if temperature > 0:
            generate_kwargs['do_sample'] = True
            generate_kwargs['temperature'] = temperature
            generate_kwargs['top_p'] = top_p
        else:
            generate_kwargs['do_sample'] = False

        with torch.no_grad():
            generated = self.model.generate(**inputs, **generate_kwargs)

        input_token_count = int(inputs['input_ids'].shape[-1])
        new_tokens = generated[0][input_token_count:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        translation = self._extract_llm_translation(raw, prompt)
        return translation or source_text
    
    def group_detections_by_context(self, detections: list) -> List[TranslationGroup]:
        if not self.cfg.enable_context_grouping or not detections:
            return [TranslationGroup([d]) for d in detections]
        
        sorted_dets = sorted(detections, key=lambda d: d.y1)
        groups = []
        current_group = [sorted_dets[0]]
        
        for det in sorted_dets[1:]:
            last_det = current_group[-1]
            distance = ImageUtils.distance_between_boxes(last_det.bbox, det.bbox)
            
            if (distance < self.cfg.context_distance_threshold and 
                len(current_group) < self.cfg.max_group_size):
                current_group.append(det)
            else:
                groups.append(TranslationGroup(current_group))
                current_group = [det]
        
        if current_group:
            groups.append(TranslationGroup(current_group))
        
        return groups
    
    def should_skip_translation(self, text: str) -> bool:
        if not text:
            return True
        text = text.strip()
        if self.cfg.skip_numeric_only:
            if all(c.isdigit() or c.isspace() or c in '.,;:' for c in text):
                return True
        if self.cfg.skip_single_char and len(text) == 1:
            return True
        if self.cfg.skip_if_no_letters:
            if not any(c.isalpha() for c in text):
                return True
        return False
    
    def translate(self, text: str) -> str:
        if not text or self.should_skip_translation(text):
            return text
        
        # ★ NETTOYAGE OCR ★
        source_text = clean_ocr_text(text.strip())

        # Translation Memory des noms
        name_key = self._name_key(source_text)
        if name_key and name_key in self.name_memory:
            return self.name_memory[name_key]

        # Glossaire forcé
        forced_map = getattr(self.cfg, 'forced_translations', {}) or {}
        forced_key = self._normalize_text_key(source_text)
        forced_translation = forced_map.get(forced_key)
        if forced_translation:
            return forced_translation

        # Noms propres simples (ex: "Miso.") : conserver tel quel
        if self._is_single_proper_name(source_text):
            return source_text

        translation_input = self._normalize_case_for_translation(source_text)
        source_lang_code = self._detect_source_language(source_text)

        if source_lang_code == self.cfg.target_lang:
            return source_text

        # Heuristique: noms propres/entités courtes en MAJUSCULES -> conserver
        # ex: "GHISLAIN PERDIUM."
        source_words = re.findall(r"[A-Z][A-Z'\-]+", source_text.upper())
        word_count = len(source_text.split())
        if source_words and len(source_words) <= 3 and 2 <= word_count <= 4:
            alpha_ratio = sum(c.isalpha() for c in source_text) / max(1, len(source_text))
            if (
                alpha_ratio > 0.65
                and source_text.upper() == source_text
                and self._looks_like_uppercase_name_sequence(source_text)
            ):
                if name_key:
                    self.name_memory[name_key] = source_text
                    self._save_name_memory()
                return source_text
        
        # Cache (sur texte nettoyé)
        if self.cache:
            cached = self.cache.get(source_text, source_lang_code, self.cfg.target_lang)
            if cached:
                # Ne pas garder un cache "identique à la source" pour une vraie traduction.
                # Permet de corriger les anciennes mauvaises sorties (ex: "OH THERE" -> "OH THERE").
                if (
                    source_lang_code != self.cfg.target_lang
                    and cached.strip().lower() == source_text.strip().lower()
                    and any(c.isalpha() for c in source_text)
                ):
                    cached = None
                else:
                    return cached
        
        try:
            if self.backend == 'local_llm':
                translation = self._translate_with_local_llm(translation_input, source_lang_code)
            else:
                src_lang = self.cfg.lang_codes.get(source_lang_code, 'eng_Latn')
                tgt_lang = self.cfg.lang_codes.get(self.cfg.target_lang, 'fra_Latn')

                self.tokenizer.src_lang = src_lang
                inputs = self.tokenizer(
                    translation_input, return_tensors="pt", padding=True,
                    truncation=True, max_length=self.cfg.max_length
                )

                # ✅ FIX GPU: Mettre inputs sur le même device que le modèle
                if self.device == 'cuda' and torch.cuda.is_available():
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}

                forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

                with torch.no_grad():
                    generated = self.model.generate(
                        **inputs,
                        max_length=self.cfg.max_length,
                        num_beams=self.cfg.num_beams,
                        early_stopping=self.cfg.early_stopping,
                        forced_bos_token_id=forced_bos_token_id
                    )

                translation = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

            # Règle de reformulation (out_text rhétorique)
            # "YOU'RE TELLING ME THAT I, THE ..." -> "Vous me dites que moi, ..."
            m = re.match(r"^YOU'?RE\s+TELLING\s+ME\s+THAT\s+I[,\.]\s+(.+)$", source_text.upper())
            if m and self.cfg.target_lang == 'fr':
                tail = m.group(1).strip()
                tail_translation = self.translate(tail)
                tail_translation = tail_translation[0].lower() + tail_translation[1:] if tail_translation else tail_translation
                if tail_translation:
                    punct = '.' if source_text.strip().endswith('.') else ''
                    translation = f"Vous me dites que moi, {tail_translation.rstrip('.!?')}{punct}"

            # Garde-fou: éviter une traduction anormalement courte
            # (souvent une hallucination/résumé sur phrases OCR longues)
            src_len = len(source_text)
            tr_len = len(translation)
            if src_len >= 24 and tr_len < max(12, int(src_len * 0.40)):
                return source_text
            
            # ✅ NOUVEAU: Post-traitement pour corriger traductions bizarres
            # "Miso." → "C'est le Miso." est faux, remettre en "Miso."
            if source_text.lower().strip() == "miso." and translation.lower().startswith("c'est le"):
                translation = "Miso."

            if self.cfg.target_lang == 'fr':
                translation = self._post_process_french(translation)
            
            if self.cache:
                self.cache.set(source_text, translation, source_lang_code, self.cfg.target_lang)

            if name_key:
                self.name_memory[name_key] = translation
                self._save_name_memory()
            
            return translation
            
        except Exception as e:
            print(f"⚠️ Erreur traduction: {e}")
            return text
    
    def translate_group(self, group: TranslationGroup) -> str:
        texts = [d.text_original for d in group.detections if d.text_original]
        if not texts:
            return ""
        
        if len(texts) == 1:
            translation = self.translate(texts[0])
            for det in group.detections:
                if det.text_original:
                    det.text_translated = translation
            group.translation = translation
            return translation
        
        combined = self.cfg.group_separator.join(texts)
        group.combined_text = combined
        translation = self.translate(combined)
        group.translation = translation
        
        translated_parts = translation.split(self.cfg.group_separator)
        if len(translated_parts) != len(texts):
            translated_parts = [self.translate(t) for t in texts]
        
        dets_with_text = [d for d in group.detections if d.text_original]
        for det, trans in zip(dets_with_text, translated_parts):
            det.text_translated = trans.strip()
        
        return translation
    
    
    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        ✅ SIMPLIFIÉ: Traduction INDIVIDUELLE robuste
        
        La traduction par batch crée trop de problèmes :
        - Séparation échoue souvent
        - Force des traductions bizarres (ex: "Miso" → "C'est le Miso")
        - Plus lent que traduction individuelle
        
        On traduit simplement bulle par bulle.
        Le cache réutilise les traductions = pas de perte de contexte.
        
        Args:
            texts: Liste des textes à traduire
            
        Returns:
            Liste des traductions
        """
        if not texts:
            return []
        
        results = []
        for i, text in enumerate(texts):
            trans = self.translate(text)
            results.append(trans)
            print(f"      [{i+1}/{len(texts)}] \"{text[:40]}...\" → \"{trans[:40]}...\"")
        
        return results
    
    def get_cache_stats(self) -> dict:
        if self.cache:
            return self.cache.get_stats()
        return {}
    
    def __del__(self):
        if self.cache:
            self.cache._save()
        self._save_name_memory()
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None