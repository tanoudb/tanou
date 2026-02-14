from .detector import YOLODetector, Detection
from .ocr import OCREngine
from .renderer import TextRenderer
from .translator import NLLBTranslator 

# Alias pour la compatibilité si nécessaire
Translator = NLLBTranslator 

__all__ = [
    'YOLODetector', 
    'Detection',
    'OCREngine', 
    'TextRenderer',
    'Translator', 
    'NLLBTranslator',
]