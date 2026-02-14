"""
═══════════════════════════════════════════════════════════════════════════════
RENDERER v6 - LaMa sur crop local + masque OCR + SKIP pour petites bulles
═══════════════════════════════════════════════════════════════════════════════

FIX PERF : LaMa travaille sur un CROP local autour de la bulle (avec marge),
pas sur l'image entière de 690x10000. ~50x plus rapide.

FIX QUALITÉ : Skip inpainting si pas de text_regions OCR (évite d'effacer 
des bulles vides ou des zones mal détectées).

✅ NOUVEAU: Skip inpainting pour bulles < 150px de haut
LaMa crée des artefacts sur micro-texte, on laisse juste le rendu texte.

Pipeline :
1. Extraire un crop local (bbox + marge de 30px)
2. Créer masque OCR en coords locales
3. LaMa inpaint sur le crop (sauf si trop petit)
4. Remettre le crop dans l'image
5. Dessiner le texte traduit
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from utils import ImageUtils

# ── Charger LaMa ──
try:
    from simple_lama_inpainting import SimpleLama
    LAMA_AVAILABLE = True
except ImportError:
    LAMA_AVAILABLE = False
    print("⚠️  simple-lama-inpainting non installé → pip install simple-lama-inpainting")


class TextRenderer:
    """Rendu texte avec LaMa inpainting local"""
    
    SHRINK_RATIO = 0.10
    CROP_MARGIN = 30  # Marge autour de la bbox pour le crop LaMa
    INPAINT_MIN_HEIGHT = 100  # ✅ RÉDUIT de 150px → 100px pour plus d'inpainting
    
    def __init__(self):
        self.cfg = config.rendering
        self.fonts = self._load_fonts()
        self.lama = None
        
        if LAMA_AVAILABLE:
            try:
                print("⏳ Chargement LaMa inpainting...")
                self.lama = SimpleLama()
                print("✅ LaMa chargé !")
            except Exception as e:
                print(f"⚠️  Erreur LaMa: {e}. Fallback cv2.inpaint.")
    
    def _load_fonts(self) -> List[str]:
        fonts = []
        for font_path in self.cfg.font_paths:
            if Path(font_path).exists():
                try:
                    ImageFont.truetype(font_path, 24)
                    fonts.append(font_path)
                except:
                    continue
        return fonts
    
    def get_font(self, size: int) -> Optional[ImageFont.FreeTypeFont]:
        for font_path in self.fonts:
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
        try:
            return ImageFont.load_default()
        except:
            return None
    
    def _get_inner_zone(self, x1: int, y1: int, x2: int, y2: int, 
                         img_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        h_img, w_img = img_shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))
        
        box_w, box_h = x2 - x1, y2 - y1
        sx = max(5, int(box_w * self.SHRINK_RATIO))
        sy = max(5, int(box_h * self.SHRINK_RATIO))
        
        return (x1 + sx, y1 + sy, x2 - sx, y2 - sy)
    
    # ─────────────────────────────────────────────────────────────────────────
    # INPAINTING LOCAL (LaMa sur crop, pas image entière)
    # ─────────────────────────────────────────────────────────────────────────
    
    def inpaint_region(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                        text_regions: Optional[List[Dict]] = None) -> np.ndarray:
        """
        Inpainting sur un crop local autour de la bulle.
        
        IMPORTANT : si pas de text_regions OCR, on skip l'inpainting.
        Ça évite d'effacer des bulles vides ou des faux positifs.
        
        ✅ NOUVEAU: Skip inpainting si bulle trop petite (< 150px haut)
        LaMa crée des artefacts sur micro-texte
        """
        h_img, w_img = img.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))
        
        if x2 - x1 < 10 or y2 - y1 < 10:
            return img
        
        # ✅ NOUVEAU: Skip inpainting pour bulles trop petites
        bubble_height = y2 - y1
        if bubble_height < self.INPAINT_MIN_HEIGHT:
            print(f"   [SKIP INPAINT] Bulle trop petite ({bubble_height}px < {self.INPAINT_MIN_HEIGHT}px)")
            return img  # Pas d'inpainting, juste du rendu texte
        
        # Pas de régions OCR → pas d'inpainting (on sait pas quoi effacer)
        if not text_regions or len(text_regions) == 0:
            return img
        
        # ── Crop local avec marge ──
        m = self.CROP_MARGIN
        crop_x1 = max(0, x1 - m)
        crop_y1 = max(0, y1 - m)
        crop_x2 = min(w_img, x2 + m)
        crop_y2 = min(h_img, y2 + m)
        
        crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        crop_h, crop_w = crop.shape[:2]
        
        # ── Masque OCR en coords locales (relatif au crop) ──
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        
        # Offset: les text_regions sont en coords du crop YOLO (relatif à x1,y1)
        # On doit les mettre en coords du crop local (relatif à crop_x1, crop_y1)
        for region in text_regions:
            bbox_points = region['bbox']
            local_points = []
            for pt in bbox_points:
                # pt est en coords du crop OCR (relatif au coin de la détection x1,y1)
                lx = int(pt[0]) + (x1 - crop_x1)
                ly = int(pt[1]) + (y1 - crop_y1)
                lx = max(0, min(lx, crop_w - 1))
                ly = max(0, min(ly, crop_h - 1))
                local_points.append([lx, ly])
            
            pts = np.array(local_points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        
        # Dilater pour couvrir anti-alias
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        if np.sum(mask) == 0:
            return img
        
        # ── Inpaint le crop ──
        if self.lama is not None:
            crop_inpainted = self._inpaint_lama(crop, mask)
        else:
            crop_inpainted = cv2.inpaint(crop, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        
        # ── Remettre le crop dans l'image ──
        img[crop_y1:crop_y2, crop_x1:crop_x2] = crop_inpainted
        
        return img
    
    def _inpaint_lama(self, crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """LaMa sur un crop local (rapide !)"""
        h_orig, w_orig = crop.shape[:2]
        
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask).convert('L')
        
        result_pil = self.lama(crop_pil, mask_pil)
        result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        
        # Sécurité taille
        if result.shape[:2] != (h_orig, w_orig):
            result = cv2.resize(result, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # COULEURS
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_text_colors(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        box_w, box_h = x2 - x1, y2 - y1
        cx1, cy1 = x1 + box_w // 4, y1 + box_h // 4
        cx2, cy2 = x2 - box_w // 4, y2 - box_h // 4
        
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            crop = img[y1:y2, x1:x2]
        
        bg_color = tuple(int(x) for x in np.median(crop.reshape(-1, 3), axis=0))
        luminosity = sum(bg_color) / 3
        
        if luminosity > self.cfg.luminosity_threshold:
            tc, oc = (0, 0, 0), (255, 255, 255)
        else:
            tc, oc = (255, 255, 255), (0, 0, 0)
        
        return (tc[2], tc[1], tc[0]), (oc[2], oc[1], oc[0])
    
    # ─────────────────────────────────────────────────────────────────────────
    # SIZING
    # ─────────────────────────────────────────────────────────────────────────
    
    def wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        words = text.split()
        lines, current = [], []
        for word in words:
            test = ' '.join(current + [word])
            try:
                w = font.getbbox(test)[2] - font.getbbox(test)[0]
            except:
                w = len(test) * (font.size // 2)
            if w <= max_width:
                current.append(word)
            else:
                if current:
                    lines.append(' '.join(current))
                current = [word]
        if current:
            lines.append(' '.join(current))
        return lines if lines else [""]
    
    def calculate_optimal_font_size(self, text: str, bbox_width: int, bbox_height: int) -> int:
        if not self.cfg.enable_dynamic_sizing:
            return max(self.cfg.min_font_size, min(bbox_height // 3, self.cfg.max_font_size))
        nb_chars = len(text)
        if nb_chars == 0:
            return self.cfg.min_font_size
        r = 1.0 - 2 * self.SHRINK_RATIO
        area = int(bbox_width * r) * int(bbox_height * r) * self.cfg.target_fill_ratio
        fs = int(math.sqrt(area / (nb_chars * 0.6)))
        return max(self.cfg.min_font_size, min(fs, self.cfg.max_font_size))
    
    def refine_font_size(self, text: str, font_size: int, bbox_width: int, bbox_height: int) -> int:
        r = 1.0 - 2 * self.SHRINK_RATIO
        uw = int(bbox_width * r) - 2 * self.cfg.padding_horizontal
        uh = int(bbox_height * r) - 2 * self.cfg.padding_vertical
        if uw <= 0 or uh <= 0:
            return font_size
        for _ in range(self.cfg.max_iterations):
            font = self.get_font(font_size)
            if not font:
                break
            lines = self.wrap_text(text, font, int(uw * self.cfg.word_wrap_ratio))
            try:
                lh = font.getbbox("Tg")[3] - font.getbbox("Tg")[1]
            except:
                lh = font_size
            sp = int(lh * self.cfg.line_spacing_ratio)
            th = len(lines) * lh + (len(lines) - 1) * sp
            fr = th / uh if uh > 0 else 1
            if abs(fr - self.cfg.target_fill_ratio) < 0.1:
                break
            font_size += self.cfg.font_size_step if fr < self.cfg.target_fill_ratio else -self.cfg.font_size_step
            font_size = max(self.cfg.min_font_size, min(font_size, self.cfg.max_font_size))
        return font_size

    def _fit_font_hard(self, text: str, font_size: int, inner_w: int, inner_h: int) -> Tuple[Optional[ImageFont.FreeTypeFont], int, List[str], int, int]:
        """Ajuste strictement la taille pour éviter tout débordement."""
        fs = max(self.cfg.min_font_size, min(font_size, self.cfg.max_font_size))

        while fs >= self.cfg.min_font_size:
            font = self.get_font(fs)
            if not font:
                return None, fs, [], 0, 0

            wrap_w = max(10, int(inner_w * self.cfg.word_wrap_ratio))
            lines = self.wrap_text(text, font, wrap_w)

            try:
                line_h = font.getbbox("Tg")[3] - font.getbbox("Tg")[1]
            except Exception:
                line_h = fs

            spacing = int(line_h * self.cfg.line_spacing_ratio)
            total_h = len(lines) * line_h + max(0, len(lines) - 1) * spacing

            line_widths = []
            for line in lines:
                try:
                    line_widths.append(font.getbbox(line)[2] - font.getbbox(line)[0])
                except Exception:
                    line_widths.append(len(line) * max(1, fs // 2))

            max_line_w = max(line_widths) if line_widths else 0

            if total_h <= inner_h and max_line_w <= inner_w:
                return font, fs, lines, line_h, spacing

            fs -= self.cfg.font_size_step

        font = self.get_font(self.cfg.min_font_size)
        if not font:
            return None, self.cfg.min_font_size, [], 0, 0
        lines = self.wrap_text(text, font, max(10, int(inner_w * self.cfg.word_wrap_ratio)))
        try:
            line_h = font.getbbox("Tg")[3] - font.getbbox("Tg")[1]
        except Exception:
            line_h = self.cfg.min_font_size
        spacing = int(line_h * self.cfg.line_spacing_ratio)
        return font, self.cfg.min_font_size, lines, line_h, spacing
    
    # ─────────────────────────────────────────────────────────────────────────
    # RENDU
    # ─────────────────────────────────────────────────────────────────────────
    
    def render_text(self, img: np.ndarray, text: str, 
                     x1: int, y1: int, x2: int, y2: int,
                     text_regions: Optional[List[Dict]] = None) -> np.ndarray:
        img = self.inpaint_region(img, x1, y1, x2, y2, text_regions=text_regions)
        img = self.insert_text(img, text, x1, y1, x2, y2)
        return img
    
    def insert_text(self, img: np.ndarray, text: str, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        if not text:
            return img
        
        ix1, iy1, ix2, iy2 = self._get_inner_zone(x1, y1, x2, y2, img.shape)
        tw, th = ix2 - ix1, iy2 - iy1
        if tw <= 0 or th <= 0:
            return img
        
        text_color, outline_color = self.get_text_colors(img, x1, y1, x2, y2)
        
        bw, bh = x2 - x1, y2 - y1
        fs = self.calculate_optimal_font_size(text, bw, bh)
        if self.cfg.enable_dynamic_sizing:
            fs = self.refine_font_size(text, fs, bw, bh)

        inner_w = max(10, tw - 2 * self.cfg.padding_horizontal)
        inner_h = max(10, th - 2 * self.cfg.padding_vertical)
        font, fs, lines, lh, sp = self._fit_font_hard(text, fs, inner_w, inner_h)
        if not font:
            return img
        
        img_pil = ImageUtils.cv2_to_pil(img)
        draw = ImageDraw.Draw(img_pil)
        
        total_h = len(lines) * lh + (len(lines) - 1) * sp
        
        if self.cfg.vertical_align == 'center':
            ys = iy1 + (th - total_h) // 2
        elif self.cfg.vertical_align == 'top':
            ys = iy1 + self.cfg.padding_vertical
        else:
            ys = iy2 - total_h - self.cfg.padding_vertical
        
        for i, line in enumerate(lines):
            try:
                lw = font.getbbox(line)[2] - font.getbbox(line)[0]
            except:
                lw = len(line) * (fs // 2)
            
            if self.cfg.horizontal_align == 'center':
                xp = ix1 + (tw - lw) // 2
            elif self.cfg.horizontal_align == 'left':
                xp = ix1 + self.cfg.padding_horizontal
            else:
                xp = ix2 - lw - self.cfg.padding_horizontal
            
            yp = ys + i * (lh + sp)

            # Clamp final pour ne jamais dessiner hors zone
            xp = max(ix1 + self.cfg.padding_horizontal, min(xp, ix2 - lw - self.cfg.padding_horizontal))
            yp = max(iy1 + self.cfg.padding_vertical, min(yp, iy2 - lh - self.cfg.padding_vertical))
            
            if self.cfg.enable_outline:
                for ox in range(-self.cfg.outline_width, self.cfg.outline_width + 1):
                    for oy in range(-self.cfg.outline_width, self.cfg.outline_width + 1):
                        if ox != 0 or oy != 0:
                            draw.text((xp + ox, yp + oy), line, font=font, fill=outline_color)
            
            draw.text((xp, yp), line, font=font, fill=text_color)
        
        return ImageUtils.pil_to_cv2(img_pil)