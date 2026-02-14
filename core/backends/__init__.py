"""
Package des backends OCR
Expose les backends disponibles et l'interface abstraite
"""

from .base import OCRBackend
from .ppocr_v5 import PPOCRv5Backend
from .easyocr import EasyOCRBackend

__all__ = [
    'OCRBackend',
    'PPOCRv5Backend',
    'EasyOCRBackend',
]
