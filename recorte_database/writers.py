"""Gravadores de imagem (Strategy)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image


class ImageWriter(ABC):
    """Contrato: grava RGB uint8 em disco."""

    extension: str = ""

    def write(self, rgb: np.ndarray, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._save(Image.fromarray(rgb, mode="RGB"), path)

    @abstractmethod
    def _save(self, img: Image.Image, path: Path) -> None:
        """Serializa a imagem no formato concreto."""


class PngWriter(ImageWriter):
    extension = ".png"

    def __init__(self, compress_level: int = 3) -> None:
        self.compress_level = compress_level

    def _save(self, img: Image.Image, path: Path) -> None:
        img.save(path, format="PNG", compress_level=self.compress_level)


class JpegWriter(ImageWriter):
    extension = ".jpg"

    def __init__(self, quality: int = 95) -> None:
        self.quality = quality

    def _save(self, img: Image.Image, path: Path) -> None:
        img.save(path, format="JPEG", quality=self.quality, subsampling=0, optimize=True)
