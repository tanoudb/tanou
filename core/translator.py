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
    """Traducteur NLLB avec nettoyage OCR et GPU fix"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.cfg = config.translation
        self.tokenizer = None
        self.model = None
        self.cache = None
        
        if self.cfg.enable_cache:
            cache_file = config.TRANSLATION_CACHE_DIR / self.cfg.cache_file
            self.cache = CacheManager(cache_file, max_size_mb=config.performance.cache_max_size_mb)
        
        self._load_model()
    
    def _load_model(self):
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
            
            print(f"   Dtype: {dtype}")
            print(f"   Device: {self.device}")
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=str(config.TRANSLATION_CACHE_DIR),
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )
            
            # ✅ FIX GPU: Vérifier et mettre sur GPU
            print(f"   Model device before .to(): {next(self.model.parameters()).device}")
            
            if self.device == 'cuda':
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
            
            self.model.eval()
            print(f"✅ NLLB chargé ! ({model_name})")
            print(f"✅ Model running on: {next(self.model.parameters()).device}\n")
            
        except Exception as e:
            raise RuntimeError(f"Erreur chargement NLLB: {e}")
    
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

        # Heuristique: noms propres/entités courtes en MAJUSCULES -> conserver
        # ex: "GHISLAIN PERDIUM."
        source_words = re.findall(r"[A-Z][A-Z'\-]+", source_text.upper())
        word_count = len(source_text.split())
        if source_words and len(source_words) <= 3 and 2 <= word_count <= 4:
            alpha_ratio = sum(c.isalpha() for c in source_text) / max(1, len(source_text))
            if alpha_ratio > 0.65 and source_text.upper() == source_text:
                return source_text
        
        # Cache (sur texte nettoyé)
        if self.cache:
            cached = self.cache.get(source_text, self.cfg.source_lang, self.cfg.target_lang)
            if cached:
                return cached
        
        try:
            src_lang = self.cfg.lang_codes.get(self.cfg.source_lang, 'eng_Latn')
            tgt_lang = self.cfg.lang_codes.get(self.cfg.target_lang, 'fra_Latn')
            
            self.tokenizer.src_lang = src_lang
            inputs = self.tokenizer(
                source_text, return_tensors="pt", padding=True,
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
            
            if self.cache:
                self.cache.set(source_text, translation, self.cfg.source_lang, self.cfg.target_lang)
            
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
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None