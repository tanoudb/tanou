"""
═══════════════════════════════════════════════════════════════════════════════
PIPELINE - Orchestration complète de la traduction
═══════════════════════════════════════════════════════════════════════════════

Mode --debug : sauvegarde image annotée + crops OCR dans output/debug/
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
import json

from config import config
from utils import MemoryManager, model_context, memory_profiler, WebtoonLogger
from core import YOLODetector, OCREngine, NLLBTranslator, TextRenderer, Detection, SmartSegmenter
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

        self.segmenter = SmartSegmenter(logger=self.logger)

    def _sort_detections_reading_order(self, detections: List[Detection]) -> List[Detection]:
        """Tri lecture naturelle: haut→bas, puis gauche→droite par ligne."""
        if not detections:
            return detections

        sorted_by_y = sorted(detections, key=lambda d: ((d.y1 + d.y2) / 2, d.x1))
        median_h = sorted([(d.y2 - d.y1) for d in sorted_by_y])[len(sorted_by_y) // 2]
        row_threshold = max(30, int(median_h * 0.55))

        rows: List[List[Detection]] = []
        current_row: List[Detection] = []
        current_row_y: Optional[float] = None

        for det in sorted_by_y:
            cy = (det.y1 + det.y2) / 2
            if current_row_y is None or abs(cy - current_row_y) <= row_threshold:
                current_row.append(det)
                current_row_y = cy if current_row_y is None else (current_row_y + cy) / 2
            else:
                rows.append(sorted(current_row, key=lambda d: d.x1))
                current_row = [det]
                current_row_y = cy

        if current_row:
            rows.append(sorted(current_row, key=lambda d: d.x1))

        ordered: List[Detection] = []
        for row in rows:
            ordered.extend(row)
        return ordered

    def _extract_text_with_retry(self, img: np.ndarray, det: Detection):
        """OCR principal + retry ciblé si confiance faible."""
        crop = img[det.y1:det.y2, det.x1:det.x2]
        if crop.size == 0:
            return None, 0.0, False, "empty_crop", [], 1.0, "none"

        text, confidence, is_valid, skip_reason, text_regions, upscale_factor = self.ocr_engine.extract_text(crop)
        if is_valid and confidence >= 0.45:
            return text, confidence, is_valid, skip_reason, text_regions, upscale_factor, "base"

        # Retry 1: crop élargi (récupère ponctuation/bords de lettres)
        h_img, w_img = img.shape[:2]
        margin = 8
        x1 = max(0, det.x1 - margin)
        y1 = max(0, det.y1 - margin)
        x2 = min(w_img, det.x2 + margin)
        y2 = min(h_img, det.y2 + margin)
        crop_expand = img[y1:y2, x1:x2]

        if crop_expand.size > 0:
            t2, c2, v2, r2, reg2, u2 = self.ocr_engine.extract_text(crop_expand)
            if v2 and c2 >= max(0.35, confidence):
                return t2, c2, v2, r2, reg2, u2, "expanded"

        # Retry 2: contraste CLAHE + sharpen
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        enhanced = cv2.filter2D(enhanced, -1, sharpen_kernel)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        t3, c3, v3, r3, reg3, u3 = self.ocr_engine.extract_text(enhanced_bgr)
        if v3 and c3 >= max(0.35, confidence):
            return t3, c3, v3, r3, reg3, u3, "clahe_sharpen"

        return text, confidence, is_valid, skip_reason, text_regions, upscale_factor, "base"

    @staticmethod
    def _compute_global_confidence(det_score: float, ocr_conf: float, lang_conf: float) -> float:
        """Score global [0..1] basé sur détection + OCR + langue."""
        det_score = min(1.0, max(0.0, float(det_score)))
        ocr_conf = min(1.0, max(0.0, float(ocr_conf)))
        lang_conf = min(1.0, max(0.0, float(lang_conf)))
        return round(0.35 * det_score + 0.45 * ocr_conf + 0.20 * lang_conf, 3)

    @staticmethod
    def _extract_ocr_lines_from_regions(text_regions: List[Dict]) -> List[str]:
        if not text_regions:
            return []

        ranked = []
        for region in text_regions:
            text = (region.get('text') or '').strip()
            poly = region.get('bbox')
            if not text or not poly or len(poly) < 3:
                continue
            try:
                ys = [float(p[1]) for p in poly]
                y_center = sum(ys) / max(1, len(ys))
            except Exception:
                y_center = 0.0
            ranked.append((y_center, text))

        ranked.sort(key=lambda x: x[0])
        lines: List[str] = []
        for _, txt in ranked:
            if txt not in lines:
                lines.append(txt)
        return lines
    
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

    @staticmethod
    def _wrap_debug_text(text: str, max_chars: int = 52) -> List[str]:
        words = (text or "").split()
        if not words:
            return [""]

        lines: List[str] = []
        current: List[str] = []
        current_len = 0
        for word in words:
            add_len = len(word) + (1 if current else 0)
            if current and (current_len + add_len) > max_chars:
                lines.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += add_len

        if current:
            lines.append(" ".join(current))
        return lines

    @staticmethod
    def _estimate_debug_text_color(img: np.ndarray, det: Detection) -> Tuple[int, int, int]:
        fallback = DEBUG_COLORS.get(getattr(det, 'class_name', ''), (255, 255, 255))
        text_regions = getattr(det, 'text_regions', None) or []

        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or not text_regions:
            return fallback

        h, w = crop.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for region in text_regions:
            pts = region.get('bbox') if isinstance(region, dict) else None
            if not pts:
                continue
            arr = np.array(pts, dtype=np.int32)
            if arr.ndim != 2 or arr.shape[0] < 3:
                continue
            arr[:, 0] = np.clip(arr[:, 0], 0, max(0, w - 1))
            arr[:, 1] = np.clip(arr[:, 1], 0, max(0, h - 1))
            cv2.fillPoly(mask, [arr], 255)

        if np.sum(mask) == 0:
            return fallback

        pixels = crop[mask > 0]
        if pixels.size == 0:
            return fallback

        bgr = np.median(pixels.reshape(-1, 3), axis=0)
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    def save_debug_double_page_ocr(self, img: np.ndarray, output_dir: Path, image_name: str,
                                   detections: List[Detection]):
        """
        Génère une vue debug double-page orientée crops (lisibilité maximale):
        - gauche: crop OCR zoomé, surligné avec régions OCR
        - droite: feuille blanche, retranscription OCR encadrée en couleur
        - indicateurs couleur + confiance visibles sur chaque ligne
        """
        if img is None or img.size == 0:
            return

        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        ordered = [d for d in self._sort_detections_reading_order(detections) if getattr(d, 'text_original', None)]
        if not ordered:
            return

        margin = 16
        gap = 16
        header_h = 56
        row_h = 232
        left_w = 920
        right_w = 920
        total_w = margin + left_w + gap + right_w + margin
        total_h = header_h + margin + len(ordered) * (row_h + gap)

        page = np.full((total_h, total_w, 3), 245, dtype=np.uint8)

        cv2.putText(page, "OCR Debug (Crop-based Double Page)", (margin, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(page, "Left: OCR crop highlighted | Right: OCR transcript + color/confidence",
                    (margin, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)

        y_cursor = header_h + margin
        for idx, det in enumerate(ordered, start=1):
            color = self._estimate_debug_text_color(img, det)
            class_name = str(getattr(det, 'class_name', 'text'))

            lx1, ly1 = margin, y_cursor
            lx2, ly2 = lx1 + left_w, y_cursor + row_h
            rx1, ry1 = lx2 + gap, y_cursor
            rx2, ry2 = rx1 + right_w, y_cursor + row_h

            # Cartes de fond colorées (léger tint)
            tint = np.array(color, dtype=np.float32)
            left_tint = np.clip((0.10 * tint + 0.90 * np.array([255, 255, 255])), 0, 255).astype(np.uint8)
            right_tint = np.clip((0.06 * tint + 0.94 * np.array([255, 255, 255])), 0, 255).astype(np.uint8)

            page[ly1:ly2, lx1:lx2] = left_tint
            page[ry1:ry2, rx1:rx2] = right_tint
            cv2.rectangle(page, (lx1, ly1), (lx2, ly2), color, 3)
            cv2.rectangle(page, (rx1, ry1), (rx2, ry2), color, 3)

            # Crop OCR
            crop = img[det.y1:det.y2, det.x1:det.x2].copy()
            if crop.size > 0:
                for region in getattr(det, 'text_regions', None) or []:
                    raw = region.get('bbox') if isinstance(region, dict) else None
                    if not raw:
                        continue
                    pts = np.array(raw, dtype=np.int32)
                    if pts.ndim != 2 or pts.shape[0] < 3:
                        continue
                    pts[:, 0] = np.clip(pts[:, 0], 0, max(0, crop.shape[1] - 1))
                    pts[:, 1] = np.clip(pts[:, 1], 0, max(0, crop.shape[0] - 1))
                    cv2.polylines(crop, [pts], True, color, 2, cv2.LINE_AA)

                inner_pad = 14
                slot_w = max(10, left_w - 2 * inner_pad)
                slot_h = max(10, row_h - 2 * inner_pad - 28)
                ch, cw = crop.shape[:2]
                scale = min(slot_w / max(1, cw), slot_h / max(1, ch))
                if scale <= 0:
                    scale = 1.0
                nw = max(1, int(cw * scale))
                nh = max(1, int(ch * scale))
                crop_rs = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA)

                px = lx1 + inner_pad + (slot_w - nw) // 2
                py = ly1 + 30 + inner_pad + (slot_h - nh) // 2
                page[py:py + nh, px:px + nw] = crop_rs

            cv2.putText(page, f"#{idx}  {class_name}  conf={det.ocr_confidence:.0%}",
                        (lx1 + 12, ly1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # Panneau transcript (droite)
            text = (det.text_original or "").strip() or "(none)"
            lines = self._wrap_debug_text(text, max_chars=66)

            cv2.putText(page, "OCR transcript", (rx1 + 12, ry1 + 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (25, 25, 25), 1, cv2.LINE_AA)

            # carré couleur + score (en haut à droite de la carte)
            sw = 18
            cv2.rectangle(page, (rx2 - 170, ry1 + 8), (rx2 - 170 + sw, ry1 + 8 + sw), color, -1)
            cv2.rectangle(page, (rx2 - 170, ry1 + 8), (rx2 - 170 + sw, ry1 + 8 + sw), (0, 0, 0), 1)
            cv2.putText(page, f"{det.ocr_confidence:.0%}", (rx2 - 145, ry1 + 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (20, 20, 20), 1, cv2.LINE_AA)

            text_y = ry1 + 52
            for line in lines[:6]:
                cv2.putText(page, line, (rx1 + 14, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.58, (15, 15, 15), 1, cv2.LINE_AA)
                text_y += 28

            y_cursor += row_h + gap

        double_page = page
        out_path = debug_dir / f"{image_name}_ocr_double_page.png"
        cv2.imwrite(str(out_path), double_page)
        self.logger.info(f"   🐛 Debug double-page OCR: {out_path}")

    def save_debug_mask_bundle(
        self,
        img: np.ndarray,
        output_dir: Path,
        image_name: str,
        index: int,
        det: Detection,
        mask_regions: Optional[List[Dict]],
    ) -> None:
        """Sauvegarde crop original + masque segmenté pour debug fin."""
        debug_dir = output_dir / "debug" / f"{image_name}_pipeline"
        debug_dir.mkdir(parents=True, exist_ok=True)

        crop = img[det.y1:det.y2, det.x1:det.x2]
        if crop.size == 0:
            return

        cv2.imwrite(str(debug_dir / f"{index:02d}_crop.png"), crop)

        h, w = crop.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for region in mask_regions or []:
            pts = region.get('bbox') if isinstance(region, dict) else None
            if not pts:
                continue
            arr = np.array(pts, dtype=np.int32)
            if arr.ndim != 2 or arr.shape[0] < 3:
                continue
            arr[:, 0] = np.clip(arr[:, 0], 0, max(0, w - 1))
            arr[:, 1] = np.clip(arr[:, 1], 0, max(0, h - 1))
            cv2.fillPoly(mask, [arr], 255)

        cv2.imwrite(str(debug_dir / f"{index:02d}_mask.png"), mask)

    def save_debug_render_bundle(
        self,
        output_dir: Path,
        image_name: str,
        index: int,
        before_crop: np.ndarray,
        after_crop: np.ndarray,
        det: Detection,
    ) -> None:
        debug_dir = output_dir / "debug" / f"{image_name}_pipeline"
        debug_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(debug_dir / f"{index:02d}_render_before.png"), before_crop)
        cv2.imwrite(str(debug_dir / f"{index:02d}_render_after.png"), after_crop)

        text_path = debug_dir / f"{index:02d}_texts.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"class: {det.class_name}\n")
            f.write(f"bbox: {det.bbox}\n")
            f.write(f"ocr: {det.text_original or ''}\n")
            f.write(f"translation: {det.text_translated or ''}\n")
            f.write(f"style: {getattr(det, 'text_style', 'dialogue')}\n")
            f.write(f"mask_regions: {len(getattr(det, 'mask_regions', []) or [])}\n")
    
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
        
        # ── Padding noir optionnel haut/bas pour les bords ──
        # Désactivé par défaut (WEBTOON_USE_BLACK_PADDING=false)
        use_black_padding = getattr(config.detection, 'use_black_padding', False)
        pad_h = int(h * max(0.0, float(getattr(config.detection, 'black_padding_ratio', 0.03)))) if use_black_padding else 0

        if pad_h > 0:
            black_bar_top = np.zeros((pad_h, w, 3), dtype=np.uint8)
            black_bar_bot = np.zeros((pad_h, w, 3), dtype=np.uint8)
            img_padded = np.vstack([black_bar_top, img, black_bar_bot])
            self.logger.info(f"   🔲 Barres noires actives: +{pad_h}px haut/bas ({w}x{img_padded.shape[0]}px)")
        else:
            img_padded = img
            self.logger.info("   🔲 Barres noires: désactivées")
        
        with model_context(lambda: YOLODetector(config.YOLO_MODEL_PATH, self.device)) as detector:
            detections = detector.detect(img_padded, logger=self.logger)
            
            # Ajuster les coordonnées Y en retirant le padding (si activé)
            if pad_h > 0:
                for det in detections:
                    new_y1 = max(0, int(det.bbox[1]) - pad_h)
                    new_y2 = min(h, int(det.bbox[3]) - pad_h)
                    det.bbox = [det.bbox[0], new_y1, det.bbox[2], new_y2]
            
            # Filtrer les détections hors image
            detections = [d for d in detections if d.y2 > 0 and d.y1 < h]
            
            translatable_detections = detector.get_translatable_detections(detections)

        # Garde-fou: ne jamais traduire les SFX, même si la config change.
        pre_filter_count = len(translatable_detections)
        translatable_detections = [
            d for d in translatable_detections
            if str(getattr(d, 'class_name', '')).lower() != 'sfx'
        ]
        if len(translatable_detections) != pre_filter_count:
            self.logger.info(
                f"   🔇 SFX exclus de la traduction: {pre_filter_count - len(translatable_detections)}"
            )

        translatable_detections = self._sort_detections_reading_order(translatable_detections)
        
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

            # Récupération des zones potentiellement manquées par YOLO via OCR global
            if vl_regions:
                recovered = 0

                def _iou_ratio(box_a, box_b):
                    xA = max(box_a[0], box_b[0])
                    yA = max(box_a[1], box_b[1])
                    xB = min(box_a[2], box_b[2])
                    yB = min(box_a[3], box_b[3])
                    inter = max(0, xB - xA) * max(0, yB - yA)
                    area = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
                    return inter / area

                existing_boxes = [[d.x1, d.y1, d.x2, d.y2] for d in translatable_detections]
                for region in vl_regions:
                    rbox = region.get('bbox')
                    rtext = (region.get('text') or '').strip()
                    rconf = float(region.get('conf', 0.0) or 0.0)
                    if not rbox or len(rbox) != 4 or not rtext:
                        continue

                    rw = max(1, int(rbox[2] - rbox[0]))
                    rh = max(1, int(rbox[3] - rbox[1]))
                    area = rw * rh

                    words = [w for w in rtext.replace("\n", " ").split(" ") if w.strip()]
                    looks_fragment = len(words) < 4 or len(rtext) < 20

                    # skip fragments en MAJUSCULES très courts (souvent morceaux de bulle déjà couverte)
                    letters = [c for c in rtext if c.isalpha()]
                    upper_ratio = (sum(1 for c in letters if c.isupper()) / max(1, len(letters))) if letters else 0.0
                    if upper_ratio > 0.9 and len(words) <= 4:
                        looks_fragment = True

                    if rconf < 0.85 or area < 12000 or looks_fragment:
                        continue

                    overlap = max((_iou_ratio(rbox, b) for b in existing_boxes), default=0.0)
                    if overlap > 0.20:
                        continue

                    margin = 16
                    x1 = max(0, int(rbox[0]) - margin)
                    y1 = max(0, int(rbox[1]) - margin)
                    x2 = min(w, int(rbox[2]) + margin)
                    y2 = min(h, int(rbox[3]) + margin)
                    if x2 - x1 < 20 or y2 - y1 < 20:
                        continue

                    pseudo = Detection(class_name='out_text', bbox=[x1, y1, x2, y2], score=max(0.35, min(0.80, rconf)))
                    translatable_detections.append(pseudo)
                    existing_boxes.append([x1, y1, x2, y2])
                    recovered += 1

                    if recovered >= 3:
                        break

                if recovered > 0:
                    translatable_detections = self._sort_detections_reading_order(translatable_detections)
                    stats['translatable'] = len(translatable_detections)
                    self.logger.info(f"   🩹 Zones récupérées via OCR global: +{recovered}")

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

                    det.mask_regions = self.segmenter.segment_detection(img, det, det.text_regions)
                    if self.debug:
                        self.save_debug_mask_bundle(img, output_dir, image_stem, i + 1, det, det.mask_regions)
                    continue

                # Si aucune région OCR globale valide, fallback: OCR sur crop
                crop = img[det.y1:det.y2, det.x1:det.x2]
                if crop.size == 0:
                    self.logger.info(f"      ⚠️  Crop vide, ignoré")
                    stats['skipped'] += 1
                    continue

                text, confidence, is_valid, skip_reason, text_regions, upscale_factor, ocr_strategy = self._extract_text_with_retry(img, det)
                det.ocr_upscale_factor = upscale_factor  # ✅ Sauvegarder le coefficient
                det.ocr_confidence = confidence
                
                # 🐛 DEBUG : voir pourquoi fallback est rejeté
                self.logger.info(
                    f"      DEBUG FALLBACK: text='{text}', conf={confidence}, valid={is_valid}, reason={skip_reason}, strategy={ocr_strategy}"
                )
                
                if not is_valid:
                    self.logger.info(f"      ⚠️  Ignoré: {skip_reason} (conf={confidence:.2f})")
                    stats['skipped'] += 1
                    stats['skip_reasons'][skip_reason] = stats['skip_reasons'].get(skip_reason, 0) + 1
                    continue

                det.text_original = text
                det.text_regions = text_regions
                det.ocr_lines = self._extract_ocr_lines_from_regions(text_regions)

                det.mask_regions = self.segmenter.segment_detection(img, det, text_regions)

                if self.debug:
                    self.save_debug_mask_bundle(img, output_dir, image_stem, i + 1, det, det.mask_regions)

                self.logger.info(f"      ✓ \"{text}\" ({confidence:.0%}) [{len(text_regions)} régions, {ocr_strategy}]")
        
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
                if self.debug:
                    self.logger.info("\n   🔎 Langue source détectée (par bulle)")
                    for idx, det in enumerate(valid_detections, start=1):
                        src_text = det.text_original or ""
                        detected_lang, lang_conf = translator.detect_source_language_with_confidence(src_text)
                        det.source_lang_detected = detected_lang
                        det.source_lang_confidence = lang_conf
                        det.global_confidence = self._compute_global_confidence(det.score, det.ocr_confidence, lang_conf)
                        preview = src_text.replace("\n", " ").strip()
                        if len(preview) > 80:
                            preview = preview[:77] + "..."
                        self.logger.info(
                            f"      [{idx:02d}] lang={detected_lang} ({lang_conf:.0%}) | global={det.global_confidence:.0%} | \"{preview}\""
                        )

                # Toujours alimenter les champs de confiance, même hors debug.
                for det in valid_detections:
                    src_text = det.text_original or ""
                    detected_lang, lang_conf = translator.detect_source_language_with_confidence(src_text)
                    det.source_lang_detected = detected_lang
                    det.source_lang_confidence = lang_conf
                    det.global_confidence = self._compute_global_confidence(det.score, det.ocr_confidence, lang_conf)

                system_detections = [d for d in valid_detections if str(getattr(d, 'class_name', '')).lower() == 'system']
                regular_detections = [d for d in valid_detections if str(getattr(d, 'class_name', '')).lower() != 'system']

                if config.translation.enable_context_grouping and regular_detections:
                    groups = translator.group_detections_by_context(regular_detections)
                    self.logger.info(
                        f"\n   🌍 Traduction contextuelle ({len(regular_detections)} bulles → {len(groups)} groupes)"
                    )
                    for group_idx, group in enumerate(groups, start=1):
                        translator.translate_group(group)
                        self.logger.info(
                            f"      Groupe {group_idx}/{len(groups)}: {len(group.detections)} bulle(s)"
                        )
                else:
                    self.logger.info(f"\n   🌍 Traduction individuelle ({len(regular_detections)} bulles)")

                    # Traduire simplement bulle par bulle
                    translations = translator.translate_batch([d.text_original for d in regular_detections])

                    # Assigner les traductions
                    for det, trans in zip(regular_detections, translations):
                        det.text_translated = trans

                # Traduction spécifique des cartes System: conserver structure titre + description
                for det in system_detections:
                    lines = [ln.strip() for ln in getattr(det, 'ocr_lines', []) if ln and ln.strip()]
                    if len(lines) >= 2:
                        raw_title = lines[0]
                        raw_body = " ".join(lines[1:])

                        title_for_translation = raw_title
                        if raw_title.isupper() and len(raw_title.split()) <= 5:
                            title_for_translation = raw_title.title()

                        body_for_translation = raw_body
                        if raw_body.isupper():
                            body_for_translation = raw_body.lower().capitalize()

                        title_tr = translator.translate(title_for_translation).strip()
                        body_tr = translator.translate(body_for_translation).strip()
                        if title_tr and body_tr:
                            det.text_translated = f"{title_tr}\n{body_tr}"
                        else:
                            det.text_translated = translator.translate(det.text_original or "")
                    else:
                        det.text_translated = translator.translate(det.text_original or "")
                
                cache_stats = translator.get_cache_stats()
                if cache_stats:
                    self.logger.info(f"\n   💾 Cache: {cache_stats['entries']} entrées, hit rate={cache_stats['hit_rate']}")
        
        stats['translated'] = len(valid_detections)
        
        self.logger.info(f"\n✅ {len(valid_detections)} traductions")
        
        # ── DEBUG : sauvegarder résultats OCR + traduction ──
        if self.debug:
            self.save_debug_ocr(output_dir, image_stem, translatable_detections)
            self.save_debug_double_page_ocr(img, output_dir, image_stem, valid_detections)
        
        # ─────────────────────────────────────────────────────────────────
        # PHASE 4 : RENDERING
        # ─────────────────────────────────────────────────────────────────
        
        self.logger.phase("Rendering", 4, 4)
        
        img_translated = img.copy()
        renderer = TextRenderer()
        
        for i, det in enumerate(valid_detections):
            if not det.text_translated:
                continue

            det.text_style = renderer.infer_text_style(
                det.text_translated,
                det.x2 - det.x1,
                det.y2 - det.y1,
                class_name=det.class_name,
            )
            det.text_color_rgb = renderer.extract_original_text_color(
                img,
                det.x1,
                det.y1,
                det.x2,
                det.y2,
                getattr(det, 'mask_regions', None) or getattr(det, 'text_regions', None),
            )
            det.font_hint = renderer.detect_font_hint(
                img,
                det.x1,
                det.y1,
                det.x2,
                det.y2,
                getattr(det, 'mask_regions', None) or getattr(det, 'text_regions', None),
            )
            
            self.logger.info(f"   [{i+1}/{len(valid_detections)}] \"{det.text_translated}\"")

            before_crop = None
            if self.debug:
                before_crop = img_translated[det.y1:det.y2, det.x1:det.x2].copy()
            
            img_translated = renderer.render_text(
                img_translated,
                det.text_translated,
                det.x1, det.y1, det.x2, det.y2,
                text_regions=getattr(det, 'text_regions', None),
                mask_regions=getattr(det, 'mask_regions', None),
                text_color_rgb=getattr(det, 'text_color_rgb', None),
                text_style=getattr(det, 'text_style', 'dialogue'),
                font_hint=getattr(det, 'font_hint', 'regular'),
                class_name=getattr(det, 'class_name', '')
            )

            if self.debug and before_crop is not None:
                after_crop = img_translated[det.y1:det.y2, det.x1:det.x2].copy()
                self.save_debug_render_bundle(output_dir, image_stem, i + 1, before_crop, after_crop, det)
        
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
                    'mask_regions_count': len(getattr(d, 'mask_regions', []) or []),
                    'text_style': getattr(d, 'text_style', 'dialogue'),
                    'text_color_rgb': getattr(d, 'text_color_rgb', None),
                    'font_hint': getattr(d, 'font_hint', 'regular'),
                    'confidence': d.ocr_confidence,
                    'detection_confidence': d.score,
                    'source_lang_detected': getattr(d, 'source_lang_detected', config.translation.source_lang),
                    'source_lang_confidence': getattr(d, 'source_lang_confidence', 0.5),
                    'global_confidence': getattr(d, 'global_confidence', self._compute_global_confidence(d.score, d.ocr_confidence, 0.5))
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