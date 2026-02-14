"""
═══════════════════════════════════════════════════════════════════════════════
PIPELINE - Orchestration complète de la traduction
═══════════════════════════════════════════════════════════════════════════════

Mode --debug : sauvegarde image annotée + crops OCR dans output/debug/
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import time
import json

from config import config
from utils import MemoryManager, model_context, memory_profiler, WebtoonLogger
from core import YOLODetector, OCREngine, NLLBTranslator, TextRenderer, Detection
from core.dll_manager import setup_nvidia_environment

# Couleurs par classe pour le debug (v3 + legacy v2)
DEBUG_COLORS = {
    'bulle':        (0, 255, 0),     # Vert
    'out_text':     (0, 165, 255),   # Orange
    'sfx':          (255, 0, 255),   # Magenta
    'System':       (128, 128, 128), # Gris
    'Bubble':       (0, 255, 0),
    'Box':          (255, 0, 0),
    'Outer_Text':   (0, 165, 255),
    'Small_Text':   (0, 255, 255),
    'Continuation': (255, 0, 255),
}


class TranslationPipeline:
    """Pipeline complet de traduction manhwa"""
    
    def __init__(self, logger: WebtoonLogger, debug: bool = False):
        self.logger = logger
        self.device = MemoryManager.get_device()
        self.debug = debug
        
        self.logger.info(f"🖥️  Device: {self.device}")
        
        # --- NOUVEAU : Configuration DLL (Une seule fois) ---
        if hasattr(config.ocr, 'paddle_env_path'):
            self.logger.info(f"⚙️  Configuration de l'environnement NVIDIA...")
            setup_nvidia_environment(config.ocr.paddle_env_path, verbose=True)
        # ---------------------------------------------------

        if self.debug:
            self.logger.info(f"🐛 Mode DEBUG activé")
            
        MemoryManager.log_memory_status(self.logger)
        
        try:
            # Modifié pour passer le paddle_env_path explicitement
            self.ocr_engine = OCREngine(
                device=self.device, 
                paddle_env_path=config.ocr.paddle_env_path
            )
            self.logger.info(f"   🔤 OCR initialisé: {self.ocr_engine.get_backend_name()}")
        except Exception as e:
            self.logger.error(f"Échec initialisation OCR: {e}")
            self.ocr_engine = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # DEBUG : Visualisation détections
    # ─────────────────────────────────────────────────────────────────────────
    
    def save_debug_detections(self, img: np.ndarray, all_detections: List[Detection],
                              translatable_detections: List[Detection],
                              output_dir: Path, image_name: str):
        """
        Image debug LISIBLE :
        - Chaque détection numérotée avec couleur par classe
        - Épaisseur forte pour les traduisibles, fine pour les autres
        - Légende en haut avec code couleur
        - Score de confiance affiché
        """
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # ── Image annotée ──
        img_debug = img.copy()
        h_img, w_img = img_debug.shape[:2]
        
        # Police plus grande 
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.7, min(w_img / 700, 1.5))  # Adapte à la largeur
        
        for idx, det in enumerate(all_detections):
            color = DEBUG_COLORS.get(det.class_name, (255, 255, 255))
            is_translatable = det.class_name in config.detection.translatable_classes
            thickness = 4 if is_translatable else 2
            
            # Rectangle
            cv2.rectangle(img_debug, (det.x1, det.y1), (det.x2, det.y2), color, thickness)
            
            # Label : #numero classe score%
            label = f"#{idx} {det.class_name} {det.score:.0%}"
            label_size = cv2.getTextSize(label, font, font_scale, 2)[0]
            
            # Fond du label (plus grand, lisible)
            pad = 6
            cv2.rectangle(img_debug,
                          (det.x1, det.y1 - label_size[1] - 2 * pad),
                          (det.x1 + label_size[0] + 2 * pad, det.y1),
                          color, -1)
            
            cv2.putText(img_debug, label,
                        (det.x1 + pad, det.y1 - pad),
                        font, font_scale,
                        (255, 255, 255), 2, cv2.LINE_AA)
        
        # ── Légende en haut ──
        legend_h = 50
        legend = np.zeros((legend_h, w_img, 3), dtype=np.uint8)
        x_offset = 10
        
        # Compter par classe
        class_counts = {}
        for det in all_detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
        
        for cls_name, count in class_counts.items():
            color = DEBUG_COLORS.get(cls_name, (255, 255, 255))
            is_tr = cls_name in config.detection.translatable_classes
            marker = "✓" if is_tr else "✗"
            txt = f"{cls_name}:{count} [{marker}]"
            
            # Pastille couleur
            cv2.rectangle(legend, (x_offset, 10), (x_offset + 25, 40), color, -1)
            x_offset += 30
            
            cv2.putText(legend, txt, (x_offset, 32),
                        font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            txt_w = cv2.getTextSize(txt, font, 0.6, 1)[0][0]
            x_offset += txt_w + 20
        
        # Coller légende au-dessus de l'image
        img_debug = np.vstack([legend, img_debug])
        
        debug_path = debug_dir / f"{image_name}_detections.png"
        cv2.imwrite(str(debug_path), img_debug)
        self.logger.info(f"   🐛 Debug détections: {debug_path}")
        
        # ── Crops individuels ──
        crops_dir = debug_dir / f"{image_name}_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        
        for i, det in enumerate(translatable_detections):
            crop = img[det.y1:det.y2, det.x1:det.x2]
            if crop.size > 0:
                crop_path = crops_dir / f"crop_{i:02d}_{det.class_name}_{det.score:.2f}.png"
                cv2.imwrite(str(crop_path), crop)
        
        self.logger.info(f"   🐛 {len(translatable_detections)} crops dans: {crops_dir}")
    
    def save_debug_ocr(self, output_dir: Path, image_name: str,
                       detections: List[Detection]):
        """Sauvegarde un résumé OCR en texte"""
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        ocr_path = debug_dir / f"{image_name}_ocr_results.txt"
        
        with open(ocr_path, 'w', encoding='utf-8') as f:
            f.write(f"OCR Results for {image_name}\n")
            f.write("=" * 60 + "\n\n")
            
            for i, det in enumerate(detections):
                f.write(f"[{i+1}] {det.class_name} (score={det.score:.2f})\n")
                f.write(f"    bbox: [{det.x1}, {det.y1}, {det.x2}, {det.y2}]\n")
                f.write(f"    size: {det.x2-det.x1}x{det.y2-det.y1}px\n")
                f.write(f"    OCR text: \"{det.text_original or '(none)'}\" \n")
                f.write(f"    OCR conf: {det.ocr_confidence:.2f}\n")
                f.write(f"    Traduit: \"{det.text_translated or '(none)'}\" \n")
                f.write("\n")
        
        self.logger.info(f"   🐛 Résultats OCR: {ocr_path}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # TRAITEMENT IMAGE UNIQUE
    # ─────────────────────────────────────────────────────────────────────────
    
    def process_image(self, image_path: Path, output_dir: Path) -> Dict:
        """Pipeline complet pour une image"""
        self.logger.header(f"📸 {image_path.name}")
        
        start_time = time.time()
        image_stem = image_path.stem
        
        # Charger image
        img = cv2.imread(str(image_path))
        if img is None:
            self.logger.error(f"Impossible de charger {image_path}")
            return {'success': False, 'error': 'load_failed'}
        
        h, w = img.shape[:2]
        self.logger.info(f"📏 {w}x{h}px\n")
        
        stats = {
            'image': image_path.name,
            'width': w,
            'height': h,
            'detections': 0,
            'translatable': 0,
            'translated': 0,
            'skipped': 0,
            'skip_reasons': {},
            'time_seconds': 0
        }
        
        # ─────────────────────────────────────────────────────────────────
        # PHASE 1 : DETECTION
        # ─────────────────────────────────────────────────────────────────
        
        self.logger.phase("Detection", 1, 4)
        
        detections = []
        
        # ── Barres noires haut/bas pour mieux détecter les bulles au bord ──
        # YOLO a du mal avec les objets coupés par le bord de l'image.
        # On ajoute du padding noir (10% de la hauteur) puis on ajuste les coords.
        pad_h = int(h * 0.03)
        black_bar_top = np.zeros((pad_h, w, 3), dtype=np.uint8)
        black_bar_bot = np.zeros((pad_h, w, 3), dtype=np.uint8)
        img_padded = np.vstack([black_bar_top, img, black_bar_bot])
        
        self.logger.info(f"   🔲 Barres noires: +{pad_h}px haut/bas ({w}x{img_padded.shape[0]}px)")
        
        with model_context(lambda: YOLODetector(config.YOLO_MODEL_PATH, self.device)) as detector:
            detections = detector.detect(img_padded, logger=self.logger)
            
            # Ajuster les coordonnées Y en retirant le padding
            for det in detections:
                new_y1 = max(0, int(det.bbox[1]) - pad_h)
                new_y2 = min(h, int(det.bbox[3]) - pad_h)
                det.bbox = [det.bbox[0], new_y1, det.bbox[2], new_y2]
            
            # Filtrer les détections complètement dans les barres noires
            detections = [d for d in detections if d.y2 > 0 and d.y1 < h]
            
            translatable_detections = detector.get_translatable_detections(detections)
        
        stats['detections'] = len(detections)
        stats['translatable'] = len(translatable_detections)
        
        # ── DEBUG : sauvegarder visualisation ──
        if self.debug:
            self.save_debug_detections(img, detections, translatable_detections,
                                       output_dir, image_stem)
        
        self.logger.info(f"\n✅ {len(translatable_detections)} zones à traduire")
        
        if not translatable_detections:
            self.logger.warning("Aucune zone traduisible détectée")
            stats['time_seconds'] = time.time() - start_time
            return stats
        
        # ─────────────────────────────────────────────────────────────────
        # PHASE 2 : OCR
        # ─────────────────────────────────────────────────────────────────
        
        self.logger.phase("OCR", 2, 4)

        if not self.ocr_engine:
            self.logger.error("OCR engine non initialisé — saut de l'OCR")
        else:
            self.logger.info(f"   🔤 Backend OCR: {self.ocr_engine.get_backend_name()}")

            # OCR global (si implémenté par le backend) puis fallback par crop
            vl_regions = []
            try:
                vl_regions = self.ocr_engine.predict_full_image(image_path)
            except Exception as e:
                self.logger.warning(f"   ⚠️  OCR global predict failed: {e}")
                vl_regions = []

            if not vl_regions:
                self.logger.info("   ℹ️  Aucun résultat OCR global — fallback par crop individuel")

            # Mapper les régions VL aux détections YOLO (par intersection)
            def _iou_area(boxA, boxB):
                if not boxA or not boxB:
                    return 0
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])
                interW = max(0, xB - xA)
                interH = max(0, yB - yA)
                return interW * interH

            for i, det in enumerate(translatable_detections):
                self.logger.info(f"   [{i+1}/{len(translatable_detections)}] {det.class_name} ({det.x2-det.x1}x{det.y2-det.y1}px)")

                # Trouver la meilleure région OCR qui intersecte la détection
                det_box = [det.x1, det.y1, det.x2, det.y2]
                det_area = max(1, (det_box[2]-det_box[0]) * (det_box[3]-det_box[1]))
                best = None
                best_score = 0.0
                for r in vl_regions:
                    rbox = r.get('bbox')
                    if not rbox:
                        continue
                    inter = _iou_area(det_box, rbox)
                    # préférence sur proportion d'intersection par rapport à la détection
                    score = inter / det_area
                    if score > best_score:
                        best_score = score
                        best = r

                if best and best_score > 0.05:
                    text = self.ocr_engine.post_process_text(best.get('text', ''))
                    conf = 0.95
                    is_valid, skip_reason = self.ocr_engine.is_valid_text(text, conf)
                    det.ocr_confidence = conf
                    
                    # 🐛 DEBUG : voir pourquoi c'est rejeté
                    self.logger.info(f"      DEBUG OCR global: text='{text}', conf={conf}, valid={is_valid}, reason={skip_reason}")
                    
                    if not is_valid:
                        self.logger.info(f"      ⚠️  Ignoré: {skip_reason} (conf={conf:.2f})")
                        stats['skipped'] += 1
                        stats['skip_reasons'][skip_reason] = stats['skip_reasons'].get(skip_reason, 0) + 1
                        continue

                    det.text_original = text
                    # Normalize region bbox into polygon points relative to the detection
                    try:
                        raw_bbox = best.get('bbox')
                        poly = None
                        if raw_bbox and isinstance(raw_bbox, list):
                            # Case: bbox is [x1,y1,x2,y2]
                            if len(raw_bbox) == 4 and all(isinstance(v, (int, float)) for v in raw_bbox):
                                bx1, by1, bx2, by2 = map(int, raw_bbox)
                                poly = [[bx1 - det.x1, by1 - det.y1], [bx2 - det.x1, by1 - det.y1],
                                        [bx2 - det.x1, by2 - det.y1], [bx1 - det.x1, by2 - det.y1]]
                            else:
                                # Assume list of points in absolute coords
                                pts = []
                                for p in raw_bbox:
                                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                                        pts.append([int(p[0]) - det.x1, int(p[1]) - det.y1])
                                if pts:
                                    poly = pts

                        region_rel = {'bbox': poly if poly is not None else [], 'text': best.get('text', ''), 'conf': best.get('conf', 0.0)}
                        det.text_regions = [region_rel] if poly else []
                    except Exception:
                        det.text_regions = []

                    self.logger.info(f"      ✓ \"{text}\" ({conf:.0%}) [{self.ocr_engine.get_backend_name()}]")
                    continue

                # Si aucune région OCR globale valide, fallback: OCR sur crop
                crop = img[det.y1:det.y2, det.x1:det.x2]
                if crop.size == 0:
                    self.logger.info(f"      ⚠️  Crop vide, ignoré")
                    stats['skipped'] += 1
                    continue

                text, confidence, is_valid, skip_reason, text_regions, upscale_factor = self.ocr_engine.extract_text(crop)
                det.ocr_upscale_factor = upscale_factor  # ✅ Sauvegarder le coefficient
                det.ocr_confidence = confidence
                
                # 🐛 DEBUG : voir pourquoi fallback est rejeté
                self.logger.info(f"      DEBUG FALLBACK: text='{text}', conf={confidence}, valid={is_valid}, reason={skip_reason}")
                
                if not is_valid:
                    self.logger.info(f"      ⚠️  Ignoré: {skip_reason} (conf={confidence:.2f})")
                    stats['skipped'] += 1
                    stats['skip_reasons'][skip_reason] = stats['skip_reasons'].get(skip_reason, 0) + 1
                    continue

                det.text_original = text
                det.text_regions = text_regions
                self.logger.info(f"      ✓ \"{text}\" ({confidence:.0%}) [{len(text_regions)} régions]")
        
        # Filtrer détections sans texte
        valid_detections = [d for d in translatable_detections if d.text_original]
        
        self.logger.info(f"\n✅ {len(valid_detections)} textes extraits")
        
        if not valid_detections:
            self.logger.warning("Aucun texte valide extrait")
            # Debug : sauvegarder résultats OCR même si aucun valide
            if self.debug:
                self.save_debug_ocr(output_dir, image_stem, translatable_detections)
            stats['time_seconds'] = time.time() - start_time
            return stats
        
        # ✅ NOUVEAU: Décharger OCR pour libérer ~2GB de VRAM avant traduction
        self.logger.info("\n🧹 Déchargement OCR pour libérer RAM/VRAM...")
        self.ocr_engine = None  # Libérer le modèle OCR
        MemoryManager.cleanup_aggressive()  # Nettoyer agressivement
        
        vram = MemoryManager.get_vram_usage()
        if vram:
            self.logger.info(f"   💾 VRAM après: {vram['allocated_gb']:.2f} GB")
        
        # ─────────────────────────────────────────────────────────────────
        # PHASE 3 : TRADUCTION
        # ─────────────────────────────────────────────────────────────────
        
        self.logger.phase("Translation", 3, 4)
        
        # Traductions pour tous les textes
        texts_to_translate = [d.text_original for d in valid_detections]
        
        if texts_to_translate:
            with model_context(lambda: NLLBTranslator(self.device)) as translator:
                self.logger.info(f"\n   🌍 Traduction individuelle ({len(texts_to_translate)} bulles)")
                
                # Traduire simplement bulle par bulle
                translations = translator.translate_batch(texts_to_translate)
                
                # Assigner les traductions
                for det, trans in zip(valid_detections, translations):
                    det.text_translated = trans
                
                cache_stats = translator.get_cache_stats()
                if cache_stats:
                    self.logger.info(f"\n   💾 Cache: {cache_stats['entries']} entrées, hit rate={cache_stats['hit_rate']}")
        
        stats['translated'] = len(valid_detections)
        
        self.logger.info(f"\n✅ {len(valid_detections)} traductions")
        
        # ── DEBUG : sauvegarder résultats OCR + traduction ──
        if self.debug:
            self.save_debug_ocr(output_dir, image_stem, translatable_detections)
        
        # ─────────────────────────────────────────────────────────────────
        # PHASE 4 : RENDERING
        # ─────────────────────────────────────────────────────────────────
        
        self.logger.phase("Rendering", 4, 4)
        
        img_translated = img.copy()
        renderer = TextRenderer()
        
        for i, det in enumerate(valid_detections):
            if not det.text_translated:
                continue
            
            self.logger.info(f"   [{i+1}/{len(valid_detections)}] \"{det.text_translated}\"")
            
            img_translated = renderer.render_text(
                img_translated,
                det.text_translated,
                det.x1, det.y1, det.x2, det.y2,
                text_regions=getattr(det, 'text_regions', None)
            )
        
        # Sauvegarder
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_image_path = output_dir / f"{image_stem}_translated.png"
        cv2.imwrite(str(output_image_path), img_translated)
        
        self.logger.info(f"\n💾 {output_image_path.name}")
        
        # Métadonnées
        metadata_path = output_dir / f"{image_stem}_metadata.json"
        metadata = {
            'source': str(image_path),
            'output': str(output_image_path),
            'dimensions': {'width': w, 'height': h},
            'stats': stats,
            'detections': [
                {
                    'class': d.class_name,
                    'bbox': d.bbox,
                    'original': d.text_original,
                    'translated': d.text_translated,
                    'confidence': d.ocr_confidence
                }
                for d in valid_detections if d.text_translated
            ]
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)  # ensure_ascii=False !
        
        stats['time_seconds'] = time.time() - start_time
        self.logger.info(f"⏱️  {stats['time_seconds']:.1f}s")
        
        return stats
    
    # ─────────────────────────────────────────────────────────────────────────
    # TRAITEMENT BATCH
    # ─────────────────────────────────────────────────────────────────────────
    
    def process_directory(self, input_dir: Path, output_dir: Path) -> Dict:
        """Traite toutes les images d'un dossier"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        image_files = [
            f for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        if not image_files:
            self.logger.error(f"Aucune image dans {input_dir}")
            return {'success': False, 'error': 'no_images'}
        
        self.logger.header(f"🚀 TRADUCTION BATCH - {len(image_files)} IMAGES")
        self.logger.stat("Input", str(input_dir))
        self.logger.stat("Output", str(output_dir))
        
        global_stats = {
            'total_images': len(image_files),
            'processed': 0,
            'failed': 0,
            'total_detections': 0,
            'total_translated': 0,
            'total_skipped': 0,
            'total_time_seconds': 0,
            'results': []
        }
        
        start_time = time.time()
        
        for i, img_path in enumerate(image_files):
            self.logger.info(f"\n{'═'*80}")
            self.logger.info(f"IMAGE {i+1}/{len(image_files)}")
            self.logger.info(f"{'═'*80}")
            
            try:
                stats = self.process_image(img_path, output_dir)
                
                if stats.get('success', True):
                    global_stats['processed'] += 1
                    global_stats['total_detections'] += stats.get('detections', 0)
                    global_stats['total_translated'] += stats.get('translated', 0)
                    global_stats['total_skipped'] += stats.get('skipped', 0)
                else:
                    global_stats['failed'] += 1
                
                global_stats['results'].append(stats)
                
            except Exception as e:
                self.logger.error(f"Erreur: {e}")
                global_stats['failed'] += 1
                global_stats['results'].append({
                    'image': img_path.name,
                    'success': False,
                    'error': str(e)
                })
            
            if config.performance.aggressive_cleanup:
                MemoryManager.cleanup_medium()
        
        global_stats['total_time_seconds'] = time.time() - start_time
        
        self.logger.header("🎉 TRAITEMENT TERMINÉ")
        self.logger.summary({
            'Images traitées': f"{global_stats['processed']}/{global_stats['total_images']}",
            'Détections': global_stats['total_detections'],
            'Traductions': global_stats['total_translated'],
            'Ignorées': global_stats['total_skipped'],
            'Échecs': global_stats['failed'],
            'Temps total': f"{global_stats['total_time_seconds']:.1f}s",
            'Temps moyen': f"{global_stats['total_time_seconds'] / max(1, global_stats['processed']):.1f}s/image"
        })
        
        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(global_stats, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"\n📊 {summary_path}")
        
        return global_stats