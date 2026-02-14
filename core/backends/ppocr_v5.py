"""
Backend PP-OCRv5
OCR texte standard avec gestion DLL NVIDIA
"""

import os
import sys
import logging
import traceback as tb
import numpy as np
from typing import Tuple, List, Dict, Optional
from pathlib import Path

from .base import OCRBackend


class PPOCRv5Backend(OCRBackend):
    """Backend PP-OCRv5 (PaddleOCR standard)."""

    def __init__(self):
        self.pipeline = None
        self._dll_configured = False

    @property
    def name(self) -> str:
        return "PP-OCRv5"

    @staticmethod
    def _silence_paddle():
        for logger_name in ["ppocr", "paddle", "paddleocr", "ppdet", "ppcls",
                            "PaddleOCR", "paddlex", "ppocrlite"]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)

    def _ensure_dll_setup(self, paddle_env_path: Optional[str] = None):
        if self._dll_configured:
            return

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from dll_manager import setup_nvidia_environment

        if paddle_env_path is None:
            venv_path = Path(sys.executable).parent.parent
            if (venv_path / "Lib" / "site-packages" / "nvidia").exists():
                paddle_env_path = str(venv_path)
            else:
                raise RuntimeError("Impossible de détecter l'environnement Paddle/NVIDIA.")

        setup_nvidia_environment(paddle_env_path, verbose=True)
        self._dll_configured = True

    def load(self, device: str, paddle_env_path: Optional[str] = None) -> None:
        self._silence_paddle()
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

        use_gpu = (device == 'cuda')
        if use_gpu:
            try:
                self._ensure_dll_setup(paddle_env_path)
            except Exception as e:
                print(f"⚠️  Configuration DLL échouée: {e}")

        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise ImportError(f"PaddleOCR non installé: {e}")

        print(f"⏳ Chargement PP-OCRv5 (GPU={use_gpu})...")

        try:
            device_str = 'gpu:0' if use_gpu else 'cpu'
            kwargs = {
                'lang': 'en',
                'ocr_version': 'PP-OCRv5',
                'use_textline_orientation': True,
                'device': device_str,
            }

            try:
                self.pipeline = PaddleOCR(**kwargs)
            except Exception:
                kwargs.pop('device', None)
                self.pipeline = PaddleOCR(**kwargs)

            print("✅ PP-OCRv5 chargé")
        except Exception as e:
            raise RuntimeError(f"Échec chargement PP-OCRv5: {e}")

    def read_text(self, img: np.ndarray) -> Tuple[str, float, List[Dict]]:
        if self.pipeline is None:
            return "", 0.0, []

        try:
            results = self.pipeline.ocr(img)
            if not results:
                return "", 0.0, []

            first_result = results[0]
            texts = []
            confidences = []
            regions = []

            # PaddleOCR 3.x: OCRResult (dict-like) avec rec_texts/rec_scores/dt_polys
            if hasattr(first_result, 'get'):
                rec_texts = first_result.get('rec_texts', []) or []
                rec_scores = first_result.get('rec_scores', []) or []
                dt_polys = first_result.get('dt_polys', []) or []

                for idx, text_value in enumerate(rec_texts):
                    text = str(text_value).strip()
                    if not text:
                        continue

                    conf = float(rec_scores[idx]) if idx < len(rec_scores) else 0.95

                    bbox_points = []
                    if idx < len(dt_polys):
                        poly = dt_polys[idx]
                        try:
                            bbox_points = poly.tolist() if hasattr(poly, 'tolist') else poly
                        except Exception:
                            bbox_points = poly

                    texts.append(text)
                    confidences.append(conf)
                    regions.append({'bbox': bbox_points, 'text': text, 'conf': conf})

            # PaddleOCR 2.x legacy: liste [[bbox, (text, conf)], ...]
            else:
                for line_result in first_result:
                    if not line_result or len(line_result) < 2:
                        continue

                    bbox_points = line_result[0]
                    text = str(line_result[1][0]).strip()
                    conf = float(line_result[1][1])

                    if not text:
                        continue

                    texts.append(text)
                    confidences.append(conf)
                    regions.append({'bbox': bbox_points, 'text': text, 'conf': conf})

            if not texts:
                return "", 0.0, []

            full_text = " ".join(texts)
            avg_conf = sum(confidences) / len(confidences)
            return full_text, avg_conf, regions

        except Exception as e:
            print(f"⚠️  Erreur PP-OCRv5: {e}")
            tb.print_exc()
            return "", 0.0, []

    def predict_full_image(self, image_path: Path) -> List[Dict]:
        return []

    def unload(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
