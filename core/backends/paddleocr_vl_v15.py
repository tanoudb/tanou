"""
Backend PaddleOCR-VL v1.5
"""

import os
import sys
import logging
import traceback as tb
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from pathlib import Path

from .base import OCRBackend


class PaddleOCRVLV15Backend(OCRBackend):
    def __init__(self):
        self.pipeline = None
        self._dll_configured = False

    @property
    def name(self) -> str:
        return "PaddleOCR-VL-v1.5"

    @staticmethod
    def _silence_paddle():
        for logger_name in ["ppocr", "paddle", "paddleocr", "ppdet", "ppcls", "PaddleOCR", "paddlex", "ppocrlite"]:
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
            from paddleocr import PaddleOCRVL
        except ImportError as e:
            raise ImportError(f"PaddleOCRVL non installé: {e}")

        print(f"⏳ Chargement PaddleOCR-VL v1.5 (GPU={use_gpu})...")

        try:
            kwargs = {
                'pipeline_version': 'v1.5',
                'use_layout_detection': True,
                'use_ocr_for_image_block': True,
                'merge_layout_blocks': False,
                'use_doc_orientation_classify': False,
                'use_doc_unwarping': False,
            }

            if use_gpu:
                kwargs['device'] = 'gpu:0'
            else:
                kwargs['device'] = 'cpu'

            try:
                self.pipeline = PaddleOCRVL(**kwargs)
            except Exception:
                kwargs.pop('device', None)
                self.pipeline = PaddleOCRVL(**kwargs)

            print("✅ PaddleOCR-VL v1.5 chargé")
        except Exception as e:
            raise RuntimeError(f"Échec chargement PaddleOCR-VL v1.5: {e}")

    @staticmethod
    def _normalize_bbox(raw_bbox: Any) -> List[List[int]]:
        if raw_bbox is None:
            return []

        try:
            arr = np.array(raw_bbox, dtype=np.float32)
        except Exception:
            return []

        if arr.ndim == 1 and arr.shape[0] >= 4:
            x1, y1, x2, y2 = [int(v) for v in arr[:4]]
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        if arr.ndim == 2 and arr.shape[1] >= 2:
            return [[int(p[0]), int(p[1])] for p in arr]

        return []

    def _collect_entries(self, node: Any, entries: List[Dict]) -> None:
        if node is None:
            return

        if isinstance(node, dict):
            text = None
            for key in ('text', 'rec_text', 'transcription', 'content', 'block_content'):
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    text = value.strip()
                    break

            conf = None
            for key in ('score', 'confidence', 'rec_score', 'prob'):
                value = node.get(key)
                if isinstance(value, (int, float)):
                    conf = float(value)
                    break

            bbox_raw = None
            for key in ('bbox', 'poly', 'polygon', 'dt_poly'):
                if key in node:
                    bbox_raw = node.get(key)
                    break

            if text:
                entries.append({
                    'text': text,
                    'conf': conf if conf is not None else 0.80,
                    'bbox': self._normalize_bbox(bbox_raw),
                })

            for value in node.values():
                self._collect_entries(value, entries)
            return

        if isinstance(node, (list, tuple)):
            for item in node:
                self._collect_entries(item, entries)

    def read_text(self, img: np.ndarray) -> Tuple[str, float, List[Dict]]:
        if self.pipeline is None:
            return "", 0.0, []

        try:
            raw_results = self.pipeline.predict(img)
            results = list(raw_results) if not isinstance(raw_results, list) else raw_results

            entries: List[Dict] = []
            for result in results:
                candidate = result
                if hasattr(result, 'to_dict'):
                    try:
                        candidate = result.to_dict()
                    except Exception:
                        candidate = result
                self._collect_entries(candidate, entries)

            if not entries:
                return "", 0.0, []

            texts = []
            confidences = []
            regions = []

            for item in entries:
                text = str(item.get('text', '')).strip()
                if not text:
                    continue
                conf = float(item.get('conf', 0.80))
                bbox = item.get('bbox') or []

                texts.append(text)
                confidences.append(conf)
                regions.append({'bbox': bbox, 'text': text, 'conf': conf})

            if not texts:
                return "", 0.0, []

            combined = " ".join(texts)
            avg_conf = sum(confidences) / max(1, len(confidences))
            return combined, avg_conf, regions

        except Exception as e:
            print(f"⚠️  Erreur PaddleOCR-VL v1.5: {e}")
            tb.print_exc()
            return "", 0.0, []

    def unload(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
