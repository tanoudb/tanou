"""
Backend EasyOCR
Fallback robuste avec prétraitements optimisés pour le texte manga
"""

import re
import cv2
import logging
import traceback as tb
import numpy as np
from typing import Tuple, List, Dict

from .base import OCRBackend


class EasyOCRBackend(OCRBackend):
    """
    EasyOCR avec prétraitements améliorés v5 :
    - Upscale minimum 64px
    - Binarisation Otsu + inversion auto
    - Normalisation uppercase (texte manga = quasi toujours caps)
    - Corrections chiffre/lettre : 0↔O, 1↔I, 5↔S en contexte alphabétique
    """
    
    def __init__(self):
        self.reader = None
    
    @property
    def name(self) -> str:
        return "EasyOCR"
    
    def load(self, device: str, languages: List[str] = None, 
             model_storage_dir: str = None) -> None:
        """
        Charge EasyOCR
        
        Args:
            device: 'cuda' ou 'cpu'
            languages: Liste des codes langues (ex: ['ko', 'en'])
            model_storage_dir: Répertoire de cache des modèles
        """
        try:
            import easyocr
        except ImportError:
            raise ImportError(
                "EasyOCR non installé.\n"
                "Installation: pip install easyocr"
            )
        
        # Langues par défaut
        if languages is None:
            languages = ['en']
        
        # Dédupliquer et assurer 'en' présent
        languages = list(set(languages + ['en']))
        
        use_gpu = (device == 'cuda')
        
        print(f"⏳ Chargement EasyOCR (langues: {languages}, GPU: {use_gpu})...")
        
        kwargs = {
            'lang_list': languages,
            'gpu': use_gpu,
            'download_enabled': True,
        }
        
        if model_storage_dir:
            kwargs['model_storage_directory'] = str(model_storage_dir)
        
        if use_gpu:
            kwargs['cudnn_benchmark'] = True
        
        self.reader = easyocr.Reader(**kwargs)
        print("✅ EasyOCR chargé !")
    
    # ── Prétraitement image ───────────────────────────────────────────────────
    
    @staticmethod
    def _preprocess(img: np.ndarray) -> np.ndarray:
        """
        Prétraite l'image pour améliorer la reconnaissance
        
        Args:
            img: Image BGR
            
        Returns:
            Image prétraitée BGR
        """
        h, w = img.shape[:2]
        
        # Upscale minimum 64px
        if h < 64:
            new_w = max(1, int(w * 64 / h))
            img = cv2.resize(img, (new_w, 64), interpolation=cv2.INTER_CUBIC)
        
        # Binarisation Otsu
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Inversion si fond sombre
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)
        
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    # ── Normalisation texte manga ─────────────────────────────────────────────
    
    @staticmethod
    def _fix_manga_text(text: str) -> str:
        """
        Normalise le texte manga :
        - Force uppercase (texte manga = quasi toujours caps)
        - Corrige les confusions chiffres/lettres courantes
        
        Args:
            text: Texte brut d'EasyOCR
            
        Returns:
            Texte normalisé
        """
        if not text:
            return text
        
        text = text.upper()
        
        # Corrections chiffre→lettre en contexte alphabétique
        # Entre deux lettres
        text = re.sub(r'(?<=[A-Z])0(?=[A-Z])', 'O', text)
        text = re.sub(r'(?<=[A-Z])1(?=[A-Z])', 'I', text)
        text = re.sub(r'(?<=[A-Z])5(?=[A-Z])', 'S', text)
        
        # En début de mot (ex: "5TART" → "START")
        text = re.sub(r'\b0([A-Z])', r'O\1', text)
        text = re.sub(r'\b1([A-Z])', r'I\1', text)
        text = re.sub(r'\b5([A-Z])', r'S\1', text)
        
        # En fin de mot (ex: "CLOS0" → "CLOSE")
        text = re.sub(r'([A-Z])0\b', r'\1O', text)
        text = re.sub(r'([A-Z])5\b', r'\1S', text)
        
        return text.strip()
    
    def read_text(self, img: np.ndarray) -> Tuple[str, float, List[Dict]]:
        """
        Extrait le texte avec EasyOCR
        
        Args:
            img: Image BGR (numpy array)
            
        Returns:
            (texte_combiné, confiance_moyenne, régions_détaillées)
        """
        if self.reader is None:
            return "", 0.0, []
        
        try:
            # Prétraitement
            img_proc = self._preprocess(img)
            
            # OCR avec paramètres optimisés
            results = self.reader.readtext(
                img_proc,
                detail=1,
                paragraph=False,
                min_size=10,
                text_threshold=0.5,
                low_text=0.3,
                width_ths=0.7,
                height_ths=0.7,
            )
            
            if not results:
                return "", 0.0, []
            
            texts = []
            confidences = []
            text_regions = []
            
            for bbox_pts, text, conf in results:
                # Normalisation manga
                text = self._fix_manga_text(str(text).strip())
                conf = float(conf)
                
                # Filtre basique de qualité
                if text and conf >= 0.3:
                    texts.append(text)
                    confidences.append(conf)
                    text_regions.append({
                        'bbox': bbox_pts,
                        'text': text,
                        'conf': conf
                    })
            
            if not texts:
                return "", 0.0, []
            
            combined_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences)
            
            return combined_text, avg_confidence, text_regions
            
        except Exception as e:
            print(f"⚠️  EasyOCR read_text error: {e}")
            tb.print_exc()
            return "", 0.0, []
    
    def unload(self) -> None:
        """Décharge le lecteur et libère la mémoire"""
        if self.reader is not None:
            del self.reader
            self.reader = None
