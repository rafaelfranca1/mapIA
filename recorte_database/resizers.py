"""Estratégias de padronização de dimensões (Strategy)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from .geometry import RGB


class Resizer(ABC):
    """Contrato: RGB uint8 (H, W, 3) -> RGB uint8 de tamanho fixo."""

    @abstractmethod
    def resize(self, rgb: np.ndarray) -> np.ndarray:
        """Retorna a imagem padronizada."""


class LetterboxResizer(Resizer):
    """Redimensiona mantendo a proporção (maior lado = ``size``) e centraliza
    em um canvas ``size x size`` com ``color``. Sem distorção."""

    def __init__(self, size: int, color: RGB = (0, 0, 0)) -> None:
        self.size = size
        self.color = color

    def resize(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        scale = self.size / max(h, w)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=interp)

        canvas = np.empty((self.size, self.size, 3), dtype=np.uint8)
        canvas[:] = self.color
        top, left = (self.size - new_h) // 2, (self.size - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized
        return canvas
