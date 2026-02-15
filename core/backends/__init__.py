"""
Package des backends OCR
Expose les backends disponibles et l'interface abstraite
"""

from .base import OCRBackend
from .paddleocr_vl_v15 import PaddleOCRVLV15Backend
from .ppocr_v5 import PPOCRv5Backend
from .easyocr import EasyOCRBackend

__all__ = [
    'OCRBackend',
    'PaddleOCRVLV15Backend',
    'PPOCRv5Backend',
    'EasyOCRBackend',
]
