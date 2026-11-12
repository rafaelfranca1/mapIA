"""Renderização de imagens de depuração (Antes | Máscara dourada | Depois)."""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .geometry import BBox
from .models import ProcessOutcome

_BLUE = (0, 120, 255)
_RED = (255, 0, 0)
_GOLD = (255, 200, 0)


class DebugRenderer:
    """Azul = recorte de borda branca; vermelho = recorte final."""

    def __init__(self, height: int = 512) -> None:
        self.height = height

    def render(self, original: np.ndarray, outcome: ProcessOutcome) -> np.ndarray:
        H = self.height
        h, w = original.shape[:2]
        s = H / h
        left = cv2.resize(original, (max(1, round(w * s)), H), interpolation=cv2.INTER_AREA)
        crop = outcome.crop
        if crop.outer_bbox is not None and crop.outer_bbox != crop.bbox:
            self._rect(left, crop.outer_bbox, s, _BLUE, 1)
        if crop.bbox is not None:
            self._rect(left, crop.bbox, s, _RED, 2)

        panels: List[Tuple[str, np.ndarray]] = [(f"Antes {w}x{h}", left)]
        if crop.debug_mask is not None and crop.outer_bbox is not None:
            panels.append(("Mascara dourada", self._mask_panel(left.shape, crop, s)))
        if outcome.image is None:
            right = np.full((H, H, 3), 200, dtype=np.uint8)
            cv2.putText(right, crop.status.name, (20, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        else:
            right = cv2.resize(outcome.image, (H, H), interpolation=cv2.INTER_NEAREST)
        panels.append(("Depois", right))

        parts = []
        for label, img in panels:
            img = img.copy()
            cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _BLUE, 2)
            parts += [img, np.full((H, 6, 3), 128, dtype=np.uint8)]
        return np.hstack(parts[:-1])

    @staticmethod
    def _rect(img: np.ndarray, bbox: BBox, s: float, color: Tuple[int, int, int], t: int) -> None:
        x0, y0, x1, y1 = (round(v * s) for v in bbox)
        cv2.rectangle(img, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), color, t)

    def _mask_panel(self, shape, crop, s: float) -> np.ndarray:
        canvas = np.zeros(shape, dtype=np.uint8)
        x0, y0, x1, y1 = crop.outer_bbox
        mw, mh = max(1, round((x1 - x0) * s)), max(1, round((y1 - y0) * s))
        m = cv2.resize(crop.debug_mask, (mw, mh), interpolation=cv2.INTER_NEAREST)
        ox, oy = round(x0 * s), round(y0 * s)
        ph, pw = min(mh, shape[0] - oy), min(mw, shape[1] - ox)
        if ph > 0 and pw > 0:
            view = canvas[oy:oy + ph, ox:ox + pw]
            view[m[:ph, :pw] > 0] = _GOLD
        if crop.bbox is not None:
            self._rect(canvas, crop.bbox, s, _RED, 2)
        return canvas
