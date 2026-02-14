"""
Moteur OCR avec support PP-OCRv5
Intégré dans l'architecture manhwa trad v2
"""

import cv2
import numpy as np
import traceback as tb
import re
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import sys

# Imports du projet
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config
from utils import ImageUtils, TextFilter

# Imports des backends
from .backends import OCRBackend, PPOCRv5Backend, EasyOCRBackend


# ═════════════════════════════════════════════════════════════════════════════
# PRIORITÉ DES BACKENDS
# ═════════════════════════════════════════════════════════════════════════════

BACKEND_PRIORITY = [
    ('ppocr-v5', PPOCRv5Backend),
    ('easyocr',      EasyOCRBackend),
]


# ═════════════════════════════════════════════════════════════════════════════
# MOTEUR OCR PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

class OCREngine:
    """
    Moteur OCR principal avec sélection automatique du backend
    
    Backends supportés (par ordre de priorité):
    1. PP-OCRv5 (GPU recommandé)
    2. EasyOCR (fallback robuste)
    
    Utilise config.ocr et utils du projet existant
    """
    
    def __init__(self, device: str = 'cuda', paddle_env_path: Optional[str] = None):
        """
        Args:
            device: 'cuda' ou 'cpu'
            paddle_env_path: Chemin vers l'env Paddle/NVIDIA (optionnel)
        """
        self.device = device
        self.cfg = config.ocr
        self.paddle_env_path = paddle_env_path or getattr(config, 'paddle_env_path', None)
        self.backend: Optional[OCRBackend] = None
        
        # Utilise le TextFilter du projet
        self.text_filter = TextFilter(
            watermark_patterns=config.filters.watermark_patterns,
            sfx_patterns=config.filters.sfx_patterns
        )
        
        # Chargement du backend
        self._load_best_backend()
    
    def predict_full_image(self, image_path: Path) -> List[Dict]:
        """
        Analyse l'image entière. 
        Pour l'instant, on laisse le pipeline faire le fallback par crop 
        car c'est plus précis sur les longs webtoons.
        """
        # IMPORTANT : Retourner une liste vide [] et non None pour éviter le crash
        return []
    
    def _load_best_backend(self):
        """Charge le meilleur backend disponible"""
        # Forçage si config.ocr.backend est défini
        forced = getattr(self.cfg, 'backend', None)
        if forced:
            for name, backend_class in BACKEND_PRIORITY:
                if name == forced:
                    try:
                        backend = backend_class()
                        self._initialize_backend(backend)
                        self.backend = backend
                        print(f"✅ Backend forcé '{forced}' chargé sur {self.device}")
                        return
                    except Exception as e:
                        print(f"⚠️  Backend forcé '{forced}' échoué: {e}")
                        tb.print_exc()
                        print("   → Passage au fallback automatique...")
                        break
        
        # Auto-sélection par ordre de priorité
        for name, backend_class in BACKEND_PRIORITY:
            try:
                backend = backend_class()
                self._initialize_backend(backend)
                self.backend = backend
                print(f"✅ Backend OCR: {backend.name} sur {self.device}")
                return
            except ImportError:
                print(f"   ℹ️  {name} non installé, essai suivant...")
            except Exception as e:
                print(f"   ⚠️  {name} erreur: {e}")
                tb.print_exc()
                print(f"   → Essai suivant...")
        
        raise RuntimeError(
            "Aucun backend OCR disponible !\n"
            "Installation:\n"
            "  PP-OCRv5: pip install paddleocr paddlepaddle-gpu\n"
            "  EasyOCR:      pip install easyocr"
        )
    
    def _initialize_backend(self, backend: OCRBackend):
        """
        Initialise un backend avec les paramètres appropriés
        
        Args:
            backend: Instance du backend à initialiser
        """
        if isinstance(backend, PPOCRv5Backend):
            # PP-OCRv5 peut nécessiter le chemin de l'env pour gestion DLL
            backend.load(self.device, paddle_env_path=self.paddle_env_path)
        
        elif isinstance(backend, EasyOCRBackend):
            # EasyOCR : langues et cache depuis config
            lang_map = {'en': 'en', 'ko': 'ko', 'ja': 'ja',
                       'zh': 'ch_sim', 'fr': 'fr', 'es': 'es', 'de': 'de'}
            ocr_lang = lang_map.get(config.translation.source_lang, 'en')
            languages = list(set([ocr_lang, 'en']))
            
            backend.load(
                self.device,
                languages=languages,
                model_storage_dir=str(config.OCR_CACHE_DIR)
            )
        
        else:
            # Backend générique
            backend.load(self.device)
    
    # ── Prétraitement ─────────────────────────────────────────────────────────
    
    def preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Prétraite l'image avant OCR
        
        ✅ NOUVEAU: Upscale proportionnel intelligent pour micro-texte
        Retourne aussi le coefficient d'upscale pour tracking
        
        Returns:
            (img_preprocessed, upscale_factor)
        """
        h, w = img.shape[:2]
        upscale_factor = 1.0
        
        # ─────────────────────────────────────────────────────────────────────
        # ÉTAPE 1: Upscale proportionnel si texte très petit
        # ─────────────────────────────────────────────────────────────────────
        
        if h < 80:  # Crop très petit
            upscale_factor = 150 / h  # Atteindre ~150px de haut
            new_w = int(w * upscale_factor)
            new_h = 150
            
            print(f"   [UPSCALE PROP] {h}x{w}px → {new_h}x{new_w}px (x{upscale_factor:.2f})")
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]
        
        elif h < 100:  # Crop petit
            upscale_factor = 120 / h
            new_w = int(w * upscale_factor)
            new_h = 120
            
            print(f"   [UPSCALE PROP] {h}x{w}px → {new_h}x{new_w}px (x{upscale_factor:.2f})")
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]
        
        # ─────────────────────────────────────────────────────────────────────
        # ÉTAPE 2: Upscale minimum si encore trop petit
        # ─────────────────────────────────────────────────────────────────────
        
        if h < 64:
            min_upscale = 64 / h
            upscale_factor *= min_upscale
            img = cv2.resize(img, (max(1, int(w * 64 / h)), 64),
                           interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]
        
        # ─────────────────────────────────────────────────────────────────────
        # ÉTAPE 3: Resize intelligent si configuré
        # ─────────────────────────────────────────────────────────────────────
        
        if self.cfg.auto_resize:
            img = ImageUtils.smart_resize(
                img,
                min_height=self.cfg.min_text_height,
                max_factor=self.cfg.max_resize_factor,
                interpolation=self.cfg.resize_interpolation
            )
        
        return img, upscale_factor
    
    # ── Post-traitement ───────────────────────────────────────────────────────
    
    def post_process_text(self, text: str) -> str:
        """
        Nettoie le texte après OCR (utilise utils.TextFilter du projet)
        
        Args:
            text: Texte brut
            
        Returns:
            Texte nettoyé
        """
        if not text:
            return ""
        
        # Nettoyage via TextFilter du projet
        text = self.text_filter.clean_text(text)

        # Normalisation artefacts OCR fréquents
        text = re.sub(r'\b1\.(?=\s+[A-Z])', 'I.', text)
        text = re.sub(r'\bI\.(?=\s+THE\b)', 'I,', text)
        text = re.sub(r'(?<=[A-Z])\s+1\s+(?=[A-Z])', ' I ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Suppression des caractères isolés si configuré
        if self.cfg.remove_isolated_chars:
            words = text.split()
            words = [w for w in words if len(w) > 1 or w.isalnum()]
            text = ' '.join(words)
        
        return text
    
    # ── Validation ────────────────────────────────────────────────────────────
    
    def is_valid_text(self, text: str, confidence: float) -> Tuple[bool, Optional[str]]:
        """
        Valide un texte extrait (utilise config.ocr et utils.TextFilter)
        
        Args:
            text: Texte à valider
            confidence: Score de confiance
            
        Returns:
            (is_valid, skip_reason)
        """
        # Vérification confiance
        if confidence < self.cfg.min_confidence:
            return False, "low_confidence"
        
        # Vérification longueur
        if len(text.strip()) < self.cfg.min_text_length:
            return False, "too_short"
        
        # Filtre watermark/SFX via TextFilter du projet
        should_skip, reason = self.text_filter.should_skip(
            text,
            min_length=self.cfg.min_text_length,
            max_numeric_ratio=self.cfg.max_numeric_ratio
        )
        if should_skip:
            return False, reason
        
        # Filtre numérique uniquement
        if self.cfg.filter_numeric_only:
            if self.text_filter.is_numeric_only(text, self.cfg.max_numeric_ratio):
                return False, "numeric_only"
        
        # Filtre caractères spéciaux uniquement
        if self.cfg.filter_special_chars_only:
            if self.text_filter.is_special_chars_only(text):
                return False, "special_chars_only"
        
        return True, None
    
    # ── Extraction principale ─────────────────────────────────────────────────
    
    def extract_text(self, img: np.ndarray) -> Tuple[
            Optional[str], float, bool, Optional[str], List[Dict], float]:
        """
        Pipeline complet d'extraction de texte
        
        Args:
            img: Image BGR (numpy array)
            
        Returns:
            Tuple contenant:
            - text: Texte extrait (None si invalide)
            - confidence: Score de confiance moyen
            - is_valid: Booléen de validité
            - skip_reason: Raison du rejet (si is_valid=False)
            - text_regions: Liste des régions détectées
            - upscale_factor: Coefficient d'upscale appliqué
        """
        # Prétraitement avec coefficient
        img_proc, upscale_factor = self.preprocess_image(img)
        
        # OCR - Paddle ou fallback
        text, confidence, text_regions = self.backend.read_text(img_proc)
        
        # Post-traitement
        text = self.post_process_text(text)
        
        # Validation
        is_valid, skip_reason = self.is_valid_text(text, confidence)
        
        if not is_valid:
            return None, confidence, False, skip_reason, [], upscale_factor
        
        return text, confidence, True, None, text_regions, upscale_factor
    
    # ── Accesseurs ────────────────────────────────────────────────────────────
    
    def get_backend_name(self) -> str:
        """Retourne le nom du backend actuellement chargé"""
        return self.backend.name if self.backend else "none"
    
    # ── Nettoyage ─────────────────────────────────────────────────────────────
    
    def __del__(self):
        """Décharge proprement le backend"""
        if self.backend:
            self.backend.unload()