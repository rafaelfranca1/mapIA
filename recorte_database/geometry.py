"""Tipos e utilitários geométricos."""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np

BBox = Tuple[int, int, int, int]  # (x0, y0, x1, y1) com x1/y1 exclusivos
RGB = Tuple[int, int, int]
Margin = Union[int, float]


def resolve_margin(margin: Margin, reference: int) -> int:
    """Converte uma margem em pixels. ``float`` < 1 = fração de ``reference``;
    qualquer outro valor = pixels."""
    if isinstance(margin, float) and margin < 1:
        return int(round(margin * reference))
    return int(margin)


def crop_array(rgb: np.ndarray, bbox: BBox) -> np.ndarray:
    """Recorta ``rgb`` (H, W, 3) pela bounding box."""
    x0, y0, x1, y1 = bbox
    return rgb[y0:y1, x0:x1]
