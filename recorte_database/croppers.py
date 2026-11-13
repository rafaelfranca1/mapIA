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
       Tenta saturação mais estrita (max(min_sat, 90)) e, se falhar, a
       configurada — molduras vivas vs. desbotadas.
    3. Para cada lado, perfil de fração dourada por linha/coluna (usando só a
       parte central do lado, para não contar a moldura perpendicular). A
       banda da moldura é o trecho contínuo, colado à borda, acima de
       ``band_ratio``; a borda interna da banda é o novo limite.
    4. Detecção feita em versão reduzida (``detect_max_side``): mais rápida e
       independente da resolução.

    Se a moldura não for detectada em ``min_sides`` lados (ou o recorte for
    implausível), tenta ``secondary`` (ex.: ``InkFrameCropper``) e, se ainda
    falhar, aplica ``fallback``: ``keep`` (borda branca) ou ``skip``.
    """

    name = "gold-frame"
    _MAX_START_FRACTION = 0.15   # moldura pode estar inseta se a borda escura não for cortada
    _CENTER = (0.2, 0.8)         # trecho central do lado usado no perfil

    def __init__(self, base: Cropper, hue_range: Tuple[int, int] = (15, 35),
                 min_saturation: int = 70, min_value: int = 100,
                 band_ratio: float = 0.4, search_fraction: float = 0.25,
                 inner_margin: Margin = 0.002, min_sides: int = 3,
                 fallback: str = "keep", detect_max_side: int = 1500,
                 min_inner_ratio: float = 0.3,
                 secondary: Optional[Cropper] = None) -> None:
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
        self.secondary = secondary  # ex.: InkFrameCropper quando não há dourado

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

        # Saturação adaptativa: estrita primeiro (evita miolo amarelado),
        # depois a configurada (molduras desbotadas).
        sats: list[int] = []
        for sat in (max(self.min_saturation, 90), self.min_saturation):
            if sat not in sats:
                sats.append(sat)

        best_mask: Optional[np.ndarray] = None
        best_found: Dict[str, int] = {}
        for sat in sats:
            mask = self._gold_mask(small, min_saturation=sat)
            extents = self._side_extents(mask)
            found = {k: v for k, v in extents.items() if v is not None}
            if len(found) > len(best_found):
                best_mask, best_found = mask, found
            if len(found) >= self.min_sides:
                best_mask, best_found = mask, found
                break

        mask = best_mask if best_mask is not None else self._gold_mask(small)
        found = best_found

        if len(found) < self.min_sides:
            return self._fallback(
                outer, mask, f"moldura dourada detectada em {len(found)}/4 lados", rgb)

        inv = 1.0 / scale
        pad = resolve_margin(self.inner_margin, min(w, h))
        off = {s: (round(found[s] * inv) + pad if s in found else 0)
               for s in ("left", "top", "right", "bottom")}
        bbox = (x0 + off["left"], y0 + off["top"], x1 - off["right"], y1 - off["bottom"])

        if (bbox[2] - bbox[0] < self.min_inner_ratio * w
                or bbox[3] - bbox[1] < self.min_inner_ratio * h):
            return self._fallback(outer, mask, "recorte interno implausível (muito pequeno)", rgb)
        return CropResult(CropStatus.OK, bbox, outer.bbox, self.name, debug_mask=mask)

    # ------------------------------------------------------------ internals
    def _gold_mask(self, rgb_small: np.ndarray,
                   min_saturation: Optional[int] = None) -> np.ndarray:
        sat = self.min_saturation if min_saturation is None else min_saturation
        hsv = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2HSV)
        lo = np.array([self.hue_range[0], sat, self.min_value], np.uint8)
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

    def _fallback(self, outer: CropResult, mask: np.ndarray, reason: str,
                  rgb: Optional[np.ndarray] = None) -> CropResult:
        if self.secondary is not None and rgb is not None:
            alt = self.secondary.crop_box(rgb)
            if alt.status is CropStatus.OK and alt.bbox is not None:
                return CropResult(
                    CropStatus.OK, alt.bbox, outer.bbox, alt.method,
                    f"{reason}; usado {alt.method}",
                    alt.debug_mask if alt.debug_mask is not None else mask,
                )
        if self.fallback == "skip":
            return CropResult(CropStatus.REJECTED, None, outer.bbox, self.name,
                              reason + " (descartada)", mask)
        return CropResult(CropStatus.FALLBACK, outer.bbox, outer.bbox,
                          "white-border (fallback)", reason + " (usado recorte de borda branca)", mask)


class InkFrameCropper(Cropper):
    """Recorta pela **moldura de tinta** (retângulo preto/vermelho).

    Usa Canny + contornos aproximados a polígonos para achar o maior
    retângulo inseto na página (folha de atlas com orla desenhada). Não
    depende de cor dourada — cobre o caso em que ``GoldFrameCropper`` falha.
    """

    name = "ink-frame"

    def __init__(self, base: Optional[Cropper] = None,
                 detect_max_side: int = 1500,
                 canny_low: int = 40, canny_high: int = 120,
                 min_area_ratio: float = 0.22, max_area_ratio: float = 0.82,
                 inner_margin: Margin = 0.002,
                 min_inner_ratio: float = 0.3) -> None:
        self.base = base
        self.detect_max_side = detect_max_side
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.inner_margin = inner_margin
        self.min_inner_ratio = min_inner_ratio

    def crop_box(self, rgb: np.ndarray) -> CropResult:
        if self.base is not None:
            outer = self.base.crop_box(rgb)
            if outer.status is not CropStatus.OK or outer.bbox is None:
                return outer
            ox0, oy0, ox1, oy1 = outer.bbox
        else:
            h0, w0 = rgb.shape[:2]
            ox0, oy0, ox1, oy1 = 0, 0, w0, h0
            outer = CropResult(CropStatus.OK, (ox0, oy0, ox1, oy1), (ox0, oy0, ox1, oy1), "full")

        region = rgb[oy0:oy1, ox0:ox1]
        rh, rw = region.shape[:2]
        scale = min(1.0, self.detect_max_side / max(rh, rw))
        small = region if scale == 1.0 else cv2.resize(
            region, (max(1, round(rw * scale)), max(1, round(rh * scale))),
            interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        edges, box = self._find_rect(small)
        if box is None:
            return CropResult(
                CropStatus.FALLBACK, outer.bbox, outer.bbox, self.name,
                "moldura de tinta não detectada", edges)

        x0, y0, x1, y1 = box
        inv = 1.0 / scale
        pad = resolve_margin(self.inner_margin, min(rw, rh))
        # pad para dentro da moldura (mesmo critério do gold-frame)
        bx0 = min(rw - 1, max(0, int(round(x0 * inv)) + pad))
        by0 = min(rh - 1, max(0, int(round(y0 * inv)) + pad))
        bx1 = max(bx0 + 1, min(rw, int(round(x1 * inv)) - pad))
        by1 = max(by0 + 1, min(rh, int(round(y1 * inv)) - pad))
        bbox = (ox0 + bx0, oy0 + by0, ox0 + bx1, oy0 + by1)
        if (bbox[2] - bbox[0] < self.min_inner_ratio * rw
                or bbox[3] - bbox[1] < self.min_inner_ratio * rh):
            return CropResult(
                CropStatus.FALLBACK, outer.bbox, outer.bbox, self.name,
                "recorte de tinta implausível (muito pequeno)", edges)
        return CropResult(CropStatus.OK, bbox, outer.bbox, self.name, debug_mask=edges)

    def _find_rect(self, small: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        sh, sw = small.shape[:2]
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, k, iterations=1)
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, sw // 50), 1))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(9, sh // 50)))
        hv = cv2.bitwise_or(
            cv2.morphologyEx(edges, cv2.MORPH_OPEN, hk),
            cv2.morphologyEx(edges, cv2.MORPH_OPEN, vk),
        )
        comb = cv2.bitwise_or(edges, hv)
        area_img = float(sh * sw)
        best: Optional[Tuple[float, Tuple[int, int, int, int]]] = None
        min_peri = 2 * (sh + sw) * 0.15
        for mode in (cv2.RETR_LIST, cv2.RETR_EXTERNAL):
            contours, _ = cv2.findContours(comb, mode, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                peri = cv2.arcLength(cnt, True)
                if peri < min_peri:
                    continue
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                x, y, bw, bh = cv2.boundingRect(approx if len(approx) >= 4 else cnt)
                ratio = (bw * bh) / area_img
                if ratio < self.min_area_ratio or ratio > self.max_area_ratio:
                    continue
                ar = bw / max(bh, 1)
                if ar < 0.45 or ar > 2.2:
                    continue
                margin_ok = (
                    (x > sw * 0.02) + (y > sh * 0.02)
                    + (x + bw < sw * 0.98) + (y + bh < sh * 0.98)
                )
                if margin_ok < 2:
                    continue
                nvert = len(approx)
                rectness = (
                    cv2.contourArea(approx) / max(bw * bh, 1) if nvert >= 4 else 0.5
                )
                score = (
                    ratio
                    * (1.2 if 4 <= nvert <= 8 else 0.8)
                    * (0.6 + 0.4 * rectness)
                    * (1 + 0.05 * margin_ok)
                )
                cand = (x, y, x + bw, y + bh)
                if best is None or score > best[0]:
                    best = (score, cand)
        return comb, (None if best is None else best[1])
