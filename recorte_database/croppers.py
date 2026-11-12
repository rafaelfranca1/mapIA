"""Estratégias de recorte (Strategy). Cada ``Cropper`` recebe RGB uint8 e
devolve um ``CropResult`` com a bounding box a manter."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .geometry import Margin, resolve_margin
from .models import CropResult, CropStatus


class Cropper(ABC):
    """Contrato de recorte."""

    name: str = "cropper"

    @abstractmethod
    def crop_box(self, rgb: np.ndarray) -> CropResult:
        """Calcula a região a manter. ``bbox=None`` significa 'não salvar'."""


class NullCropper(Cropper):
    """Não recorta (mantém a imagem inteira)."""

    name = "none"

    def crop_box(self, rgb: np.ndarray) -> CropResult:
        h, w = rgb.shape[:2]
        return CropResult(CropStatus.OK, (0, 0, w, h), (0, 0, w, h), self.name)


class WhiteBorderCropper(Cropper):
    """Remove bordas brancas/quase brancas.

    Cinza < ``threshold`` = conteúdo; abertura morfológica remove pixels
    sujos isolados; bounding box + ``margin`` (px ou fração).
    """

    name = "white-border"

    def __init__(self, threshold: int = 245, margin: Margin = 5,
                 morph_kernel: int = 3, min_content_pixels: int = 20) -> None:
        self.threshold = threshold
        self.margin = margin
        self.morph_kernel = morph_kernel
        self.min_content_pixels = min_content_pixels

    def crop_box(self, rgb: np.ndarray) -> CropResult:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = (gray < self.threshold).astype(np.uint8)
        if self.morph_kernel > 1:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_kernel,) * 2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        if cv2.countNonZero(mask) < max(1, self.min_content_pixels):
            return CropResult(CropStatus.EMPTY, method=self.name, message="imagem vazia/branca")

        ys = np.flatnonzero(mask.any(axis=1))
        xs = np.flatnonzero(mask.any(axis=0))
        x0, x1 = int(xs[0]), int(xs[-1]) + 1
        y0, y1 = int(ys[0]), int(ys[-1]) + 1
        pad = resolve_margin(self.margin, max(x1 - x0, y1 - y0))
        h, w = gray.shape
        bbox = (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad))
        return CropResult(CropStatus.OK, bbox, bbox, self.name)


class GoldFrameCropper(Cropper):
    """Recorta **da moldura dourada para dentro**.

    1. ``base`` (borda branca) isola a área útil externa.
    2. Máscara de "dourado" em HSV (matiz/saturação/valor configuráveis).
    3. Para cada lado, perfil de fração dourada por linha/coluna (usando só a
       parte central do lado, para não contar a moldura perpendicular). A
       banda da moldura é o trecho contínuo, colado à borda, acima de
       ``band_ratio``; a borda interna da banda é o novo limite.
    4. Detecção feita em versão reduzida (``detect_max_side``): mais rápida e
       independente da resolução.

    Se a moldura não for detectada em ``min_sides`` lados (ou o recorte for
    implausível), aplica ``fallback``: ``keep`` (usa recorte de borda branca,
    com aviso) ou ``skip`` (rejeita a imagem).
    """

    name = "gold-frame"
    _MAX_START_FRACTION = 0.05   # a moldura deve começar a <=5% da borda externa
    _CENTER = (0.2, 0.8)         # trecho central do lado usado no perfil

    def __init__(self, base: Cropper, hue_range: Tuple[int, int] = (15, 35),
                 min_saturation: int = 90, min_value: int = 100,
                 band_ratio: float = 0.5, search_fraction: float = 0.25,
                 inner_margin: Margin = 0.002, min_sides: int = 4,
                 fallback: str = "keep", detect_max_side: int = 1500,
                 min_inner_ratio: float = 0.3) -> None:
        if fallback not in ("keep", "skip"):
            raise ValueError("fallback deve ser 'keep' ou 'skip'")
        self.base = base
        self.hue_range = hue_range
        self.min_saturation = min_saturation
        self.min_value = min_value
        self.band_ratio = band_ratio
        self.search_fraction = search_fraction
        self.inner_margin = inner_margin
        self.min_sides = min_sides
        self.fallback = fallback
        self.detect_max_side = detect_max_side
        self.min_inner_ratio = min_inner_ratio

    # ------------------------------------------------------------------ API
    def crop_box(self, rgb: np.ndarray) -> CropResult:
        outer = self.base.crop_box(rgb)
        if outer.status is not CropStatus.OK or outer.bbox is None:
            return outer  # vazia: nada a fazer
        x0, y0, x1, y1 = outer.bbox
        region = rgb[y0:y1, x0:x1]
        h, w = region.shape[:2]

        scale = min(1.0, self.detect_max_side / max(h, w))
        small = region if scale == 1.0 else cv2.resize(
            region, (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA)
        mask = self._gold_mask(small)
        found = {k: v for k, v in self._side_extents(mask).items() if v is not None}

        if len(found) < self.min_sides:
            return self._fallback(outer, mask, f"moldura dourada detectada em {len(found)}/4 lados")

        inv = 1.0 / scale
        pad = resolve_margin(self.inner_margin, min(w, h))
        off = {s: (round(found[s] * inv) + pad if s in found else 0)
               for s in ("left", "top", "right", "bottom")}
        bbox = (x0 + off["left"], y0 + off["top"], x1 - off["right"], y1 - off["bottom"])

        if (bbox[2] - bbox[0] < self.min_inner_ratio * w
                or bbox[3] - bbox[1] < self.min_inner_ratio * h):
            return self._fallback(outer, mask, "recorte interno implausível (muito pequeno)")
        return CropResult(CropStatus.OK, bbox, outer.bbox, self.name, debug_mask=mask)

    # ------------------------------------------------------------ internals
    def _gold_mask(self, rgb_small: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2HSV)
        lo = np.array([self.hue_range[0], self.min_saturation, self.min_value], np.uint8)
        hi = np.array([self.hue_range[1], 255, 255], np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def _side_extents(self, mask: np.ndarray) -> Dict[str, Optional[int]]:
        """Espessura (px, na escala reduzida) da moldura em cada lado."""
        binary = mask > 0
        h, w = binary.shape
        c0, c1 = self._CENTER
        cols = slice(int(w * c0), max(int(w * c0) + 1, int(w * c1)))
        rows = slice(int(h * c0), max(int(h * c0) + 1, int(h * c1)))
        row_prof = binary[:, cols].mean(axis=1)
        col_prof = binary[rows, :].mean(axis=0)
        return {
            "top": self._band_extent(row_prof, h),
            "bottom": self._band_extent(row_prof[::-1], h),
            "left": self._band_extent(col_prof, w),
            "right": self._band_extent(col_prof[::-1], w),
        }

    def _band_extent(self, profile: np.ndarray, side: int) -> Optional[int]:
        """Fim (exclusivo) da banda dourada colada à borda, ou None."""
        window = max(8, int(side * self.search_fraction))
        max_start = max(3, int(side * self._MAX_START_FRACTION))
        max_gap = max(2, int(side * 0.004))
        min_len = max(2, int(side * 0.003))

        idx = np.flatnonzero(profile[:window] >= self.band_ratio)
        if idx.size == 0 or idx[0] > max_start:
            return None
        last = int(idx[0])
        for i in idx[1:]:
            if i - last - 1 > max_gap:
                break
            last = int(i)
        extent = last + 1
        if extent - int(idx[0]) < min_len:
            return None
        if extent >= window - max_gap:  # não terminou na janela: cor pegou o miolo
            return None
        return extent

    def _fallback(self, outer: CropResult, mask: np.ndarray, reason: str) -> CropResult:
        if self.fallback == "skip":
            return CropResult(CropStatus.REJECTED, None, outer.bbox, self.name,
                              reason + " (descartada)", mask)
        return CropResult(CropStatus.FALLBACK, outer.bbox, outer.bbox,
                          "white-border (fallback)", reason + " (usado recorte de borda branca)", mask)
