"""
═══════════════════════════════════════════════════════════════════════════════
WEBTOON TRANSLATOR V5 PREMIUM
═══════════════════════════════════════════════════════════════════════════════

Usage:
    python main.py                  # Traite input/ → output/
    python main.py --debug          # Mode debug (sauvegarde détections)
    python main.py --input custom/  # Dossier custom
    python main.py --image test.png # Image unique
"""

import argparse
import sys
from pathlib import Path
from core.ocr import OCREngine
from core.dll_manager import setup_nvidia_environment
from config import config, INPUT_DIR, OUTPUT_DIR, LOGS_DIR
from utils import init_logger, MemoryManager
from pipeline import TranslationPipeline
import os
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'

def print_banner():
    """Affiche le banner de démarrage"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║               🚀 WEBTOON TRANSLATOR V5🚀                                 ║
║                                                                           ║
║               Architecture Modulaire                                      ║
║                    Traduction Manhwa Universelle                          ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Point d'entrée principal"""
    
    parser = argparse.ArgumentParser(
        description='Webtoon Translator V5 Premium',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=INPUT_DIR,
        help=f'Dossier input (défaut: {INPUT_DIR})'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=OUTPUT_DIR,
        help=f'Dossier output (défaut: {OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--image',
        type=Path,
        help='Traiter une image unique'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Niveau de logging (défaut: INFO)'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Désactiver le cache de traduction'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Mode debug: sauvegarde image annotée des détections + crops dans output/debug/'
    )
    
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Afficher la configuration et quitter'
    )
    
    args = parser.parse_args()
    
    # Banner
    print_banner()
    
    # Show config
    if args.show_config:
        print(f"\nDevice: {MemoryManager.get_device()}")
        print(f"FP16: {config.performance.use_fp16}")
        print(f"\nDetection:")
        print(f"  Slicing: {config.detection.enable_adaptive_slicing}")
        print(f"  Window: {config.detection.base_window_height}px")
        print(f"  Scales: {config.detection.detection_scales}")
        print(f"\nOCR: {config.ocr.backend}")
        print(f"  Source lang: {config.translation.source_lang}")
        print(f"\nTranslation:")
        print(f"  {config.translation.source_lang.upper()} → {config.translation.target_lang.upper()}")
        print(f"  Cache: {config.translation.enable_cache}")
        print(f"\nRendering:")
        print(f"  Inpainting: {config.rendering.inpainting_method}")
        print(f"  Font size: {config.rendering.min_font_size}-{config.rendering.max_font_size}")
        return
    
    # Appliquer arguments
    if args.no_cache:
        config.translation.enable_cache = False
    
    # Initialiser logger
    log_file = LOGS_DIR / "webtoon_v5.log"
    logger = init_logger(log_file=log_file, level=args.log_level)
    
    logger.header("WEBTOON TRANSLATOR V5 PREMIUM")
    
    # Vérifications
    if args.image:
        if not args.image.exists():
            logger.error(f"Image introuvable: {args.image}")
            sys.exit(1)
        input_path = args.image
        mode = "single"
    else:
        if not args.input.exists():
            logger.error(f"Dossier input introuvable: {args.input}")
            sys.exit(1)
        input_path = args.input
        mode = "batch"
    
    # Vérifier modèle YOLO
    if not config.YOLO_MODEL_PATH.exists():
        logger.error(f"Modèle YOLO introuvable: {config.YOLO_MODEL_PATH}")
        logger.info("Placez le modèle manhwa_v2.pt dans assets/models/")
        sys.exit(1)
    
    # Stats système
    logger.section("SYSTÈME")
    logger.stat("Device", MemoryManager.get_device())
    
    ram = MemoryManager.get_ram_usage()
    logger.stat("RAM", f"{ram['available_gb']:.1f} GB disponible")
    
    vram = MemoryManager.get_vram_usage()
    if vram:
        logger.stat("VRAM", f"{vram['allocated_gb']:.2f} GB allouée")
    
    # Configuration
    logger.section("CONFIGURATION")
    logger.stat("Mode", mode)
    logger.stat("Debug", "OUI" if args.debug else "non")
    logger.stat("Input", str(input_path))
    logger.stat("Output", str(args.output))
    logger.stat("Slicing", "Adaptatif" if config.detection.enable_adaptive_slicing else "Fixe")
    logger.stat("Multi-scale", str(config.detection.detection_scales))
    logger.stat("OCR", str(config.ocr.backend))
    logger.stat("Translation", f"{config.translation.source_lang.upper()} → {config.translation.target_lang.upper()}")
    logger.stat("Cache", "Activé" if config.translation.enable_cache else "Désactivé")
    
    # Créer pipeline avec mode debug
    logger.info("")
    pipeline = TranslationPipeline(logger, debug=args.debug)
    
    # Traiter
    logger.start_timer()
    
    try:
        if mode == "single":
            stats = pipeline.process_image(input_path, args.output)
            if not stats.get('success', True):
                logger.error("Échec du traitement")
                sys.exit(1)
        else:
            stats = pipeline.process_directory(input_path, args.output)
            if stats.get('failed', 0) > 0:
                logger.warning(f"{stats['failed']} images ont échoué")
    
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Interruption utilisateur")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        logger.end_timer()
    
    logger.header("✅ TERMINÉ")
    
    if args.debug:
        logger.info(f"🐛 Fichiers debug dans: {args.output / 'debug'}")


if __name__ == "__main__":
    main()