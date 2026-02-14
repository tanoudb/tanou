"""
═══════════════════════════════════════════════════════════════════════════════
WEBTOON TRANSLATOR V5 PREMIUM - CONFIGURATION CENTRALISÉE
═══════════════════════════════════════════════════════════════════════════════
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS & DIRECTORIES
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent.parent.absolute()
ASSETS_DIR = BASE_DIR / "assets"
MODEL_DIR = ASSETS_DIR / "models"
CACHE_DIR = ASSETS_DIR / "cache"
_DEFAULT_PADDLE_ENV = BASE_DIR / ".venv311"
if not _DEFAULT_PADDLE_ENV.exists():
    _DEFAULT_PADDLE_ENV = BASE_DIR / ".venv"
PADDLE_ENV_PATH = Path(os.environ.get("PADDLE_ENV_PATH", str(_DEFAULT_PADDLE_ENV)))
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"

YOLO_MODEL_PATH = MODEL_DIR / "manhwa_v3.pt"  # Nouveau modèle 4 classes
OCR_CACHE_DIR = CACHE_DIR / "ocr_weights"
TRANSLATION_CACHE_DIR = CACHE_DIR / "translation_models"

# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE & PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PerformanceConfig:
    device: str = "cuda"
    use_fp16: bool = True
    aggressive_cleanup: bool = True
    max_batch_size: int = 8
    prefetch_images: bool = False
    num_workers: int = 4
    enable_cache: bool = True
    cache_max_size_mb: int = 500


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTION - YOLO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectionConfig:
    """Configuration détection YOLO"""
    
    # SLICING ADAPTATIF
    enable_adaptive_slicing: bool = True
    base_window_height: int = 2048
    overlap_ratio: float = 0.30
    auto_calibrate_window: bool = False
    min_window_height: int = 1024
    max_window_height: int = 4096
    
    # MULTI-SCALE
    enable_multi_scale: bool = True
    detection_scales: List[float] = field(default_factory=lambda: [1.0, 0.75, 0.5])
    
    multi_scale_fusion: str = "weighted"
    scale_weights: Dict[float, float] = field(default_factory=lambda: {
        1.0: 1.0,
        0.75: 0.8,
        0.5: 0.6
    })
    
    # ── CONFIDENCE - YOLO v3 (4 classes) ──
    # names: ['System', 'bulle', 'out_text', 'sfx']
    confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'System': 0.55,
        'bulle': 0.70,       # Seuil bas pour ne rater aucune bulle
        'out_text': 0.30,    # Texte hors bulle (baissé pour capturer plus)
        'sfx': 0.50          # Effets sonores
    })
    
    # NMS - YOLO v3
    nms_iou_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'System': 0.5,
        'bulle': 0.4,
        'out_text': 0.6,
        'sfx': 0.5
    })
    
    multi_scale_nms_iou: float = 0.6
    inter_class_iou_threshold: float = 0.7
    
    # ── FILTRAGE BORDURES - DÉSACTIVÉ ──
    # Le filtrage des bordures supprimait des détections valides
    # dans les zones d'overlap du sliding window. La NMS suffit
    # pour éliminer les doublons.
    filter_border_detections: bool = False   # CHANGÉ: True → False
    border_margin_px: int = 50
    
    min_box_area: int = 300       # Baissé de 400 → 300
    max_box_area: int = 800000    # Augmenté de 500000 → 800000
    min_box_ratio: float = 0.1
    max_box_ratio: float = 10.0
    
    # ── CLASSES À TRADUIRE - YOLO v3 ──
    # bulle = bulles de dialogue (à traduire)
    # out_text = texte hors bulle (à traduire)
    # sfx = effets sonores (optionnel, souvent onomatopées)
    # System = UI/titre (ne pas traduire)
    translatable_classes: List[str] = field(default_factory=lambda: [
        'bulle', 'out_text'
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# OCR - EASYOCR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OCRConfig:
    """Configuration OCR - Priorité PP-OCRv5"""
    backend = 'ppocr-v5'
    use_fp16: bool = True
    # Ajoute la ligne ci-dessous :
    paddle_env_path: Path = PADDLE_ENV_PATH
    
    # PREPROCESSING
    enable_preprocessing: bool = True
    auto_resize: bool = True
    min_text_height: int = 20
    max_resize_factor: int = 3
    resize_interpolation: str = "lanczos"
    enable_contrast_enhancement: bool = False
    enable_denoising: bool = False
    convert_to_rgb: bool = True
    
    # ── VALIDATION - SEUILS ASSOUPLIS ──
    min_confidence: float = 0.1
    min_text_length: int = 2
    
    enable_spell_check: bool = False
    remove_isolated_chars: bool = True
    
    filter_numeric_only: bool = True
    filter_special_chars_only: bool = True
    max_numeric_ratio: float = 0.8


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSLATION - NLLB
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TranslationConfig:
    # CHANGÉ : 600M -> 1.3B pour une meilleure précision
    model_name: str = "facebook/nllb-200-distilled-1.3B"
    

    use_fp16: bool = True  # CRITIQUE pour tes 8 Go de RAM
    
    source_lang: str = "en"
    target_lang: str = "fr"
    
    lang_codes: Dict[str, str] = field(default_factory=lambda: {
        'en': 'eng_Latn',
        'fr': 'fra_Latn',
        'ko': 'kor_Hang',
        'ja': 'jpn_Jpan',
        'es': 'spa_Latn',
        'de': 'deu_Latn',
        'it': 'ita_Latn',
        'pt': 'por_Latn',
        'ru': 'rus_Cyrl',
        'zh': 'zho_Hans'
    })
    
    max_length: int = 512
    num_beams: int = 5
    early_stopping: bool = True
    
    enable_context_grouping: bool = True
    context_distance_threshold: int = 300
    max_group_size: int = 5
    group_separator: str = " | "
    
    enable_cache: bool = True
    cache_file: str = "translation_cache_v5.json"
    
    skip_numeric_only: bool = True
    skip_single_char: bool = True
    skip_if_no_letters: bool = True
    preserve_ellipsis: bool = True
    preserve_emphasis: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# RENDERING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RenderingConfig:
    inpainting_method: str = "simple_fill"
    margin_px: int = 5
    inpaint_radius: int = 7
    inpaint_method: str = "telea"
    background_detection: str = "median_border"
    background_sample_ratio: float = 0.1
    
    font_paths: List[str] = field(default_factory=lambda: [
        str(BASE_DIR / "assets" / "fonts" / "Expressives (cris, creepy, peur etc.)" / "creepy"/"BloodyMurder BB.ttf"),
        "C:/Windows/Fonts/cour.ttf",             # Courier
        "C:/Windows/Fonts/times.ttf",            # Times New Roman
        
        ])
    
    enable_dynamic_sizing: bool = True
    target_fill_ratio: float = 0.80
    min_font_size: int = 12
    max_font_size: int = 80
    font_size_step: int = 2
    max_iterations: int = 15
    
    horizontal_align: str = "center"
    vertical_align: str = "center"
    line_spacing_ratio: float = 0.20
    word_wrap_ratio: float = 0.90
    padding_horizontal: int = 8
    padding_vertical: int = 6
    
    auto_text_color: bool = True
    luminosity_threshold: int = 128
    default_text_color: Tuple[int, int, int] = (0, 0, 0)
    default_outline_color: Tuple[int, int, int] = (255, 255, 255)
    enable_outline: bool = True
    outline_width: int = 2
    outline_method: str = "stroke"
    enable_shadow: bool = False
    shadow_offset: Tuple[int, int] = (2, 2)
    shadow_blur: int = 3


# ═══════════════════════════════════════════════════════════════════════════════
# FILTERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FilterConfig:
    enable_watermark_filter: bool = True
    
    watermark_patterns: List[str] = field(default_factory=lambda: [
        r'asurascans?\.com',
        r'reaper-?scans?\.com',
        r'flamescans?\.org',
        r'\.com',
        r'\.net',
        r'\.org',
        r'www\.',
        r'http',
        r'chapter\s*\d+',
        r'page\s*\d+',
        r'read\s+on',
        r'scan\s+by'
    ])
    
    enable_sfx_filter: bool = True
    
    sfx_patterns: List[str] = field(default_factory=lambda: [
        r'^(.)\1{2,}$',
        r'^[!?.*#@\-_]{2,}$',
    ])
    
    # ── FILTRAGE BORDS - ASSOUPLI ──
    # Sur les images longues (14400px), le titre en haut et les crédits
    # en bas sont filtrés, mais 5% de 14400px = 720px c'est beaucoup.
    # On réduit pour ne filtrer que le strict bord.
    filter_top_edge: bool = True
    top_edge_threshold: float = 0.02    # Réduit de 0.05 → 0.02
    filter_bottom_edge: bool = True
    bottom_edge_threshold: float = 0.98  # Réduit de 0.95 → 0.98


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LoggingConfig:
    level: str = "INFO"
    console_format: str = "%(levelname)s | %(message)s"
    file_format: str = "%(asctime)s | %(levelname)s | %(module)s | %(message)s"
    enable_file_logging: bool = True
    log_file: str = "webtoon_v5.log"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 3
    log_statistics: bool = True
    log_timings: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION GLOBALE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    YOLO_MODEL_PATH: Path = YOLO_MODEL_PATH 
    INPUT_DIR: Path = INPUT_DIR
    OUTPUT_DIR: Path = OUTPUT_DIR
    LOGS_DIR: Path = LOGS_DIR
    OCR_CACHE_DIR: Path = OCR_CACHE_DIR
    TRANSLATION_CACHE_DIR: Path = TRANSLATION_CACHE_DIR
    
    
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        for directory in [INPUT_DIR, OUTPUT_DIR, LOGS_DIR, MODEL_DIR, CACHE_DIR, 
                         OCR_CACHE_DIR, TRANSLATION_CACHE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Config':
        return cls(**data)


config = Config()