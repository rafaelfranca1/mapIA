"""Conversões de profundidade de bits e espaço de cor para RGB uint8."""
from __future__ import annotations

import numpy as np
from PIL import Image


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Converte qualquer dtype numérico para ``uint8``.

    uint8: inalterado; uint16: escala linear (65535 -> 255); float em [0, 1]:
    x255; demais: normalização min-max.
    """
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr // 257).astype(np.uint8)
    a = np.nan_to_num(arr.astype(np.float64))
    lo, hi = float(a.min()), float(a.max())
    if np.issubdtype(arr.dtype, np.floating) and lo >= 0.0 and hi <= 1.0:
        return np.rint(a * 255.0).astype(np.uint8)
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.rint((a - lo) / (hi - lo) * 255.0).astype(np.uint8)


def composite_on_white(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Compõe ``rgb`` (uint8) sobre fundo branco usando ``alpha`` (uint8)."""
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    out = rgb.astype(np.float32) * a + 255.0 * (1.0 - a)
    return np.rint(out).astype(np.uint8)


def cmyk_to_rgb(cmyk: np.ndarray) -> np.ndarray:
    """CMYK uint8 -> RGB uint8 (conversão simples, sem perfil ICC)."""
    c = cmyk.astype(np.float32) / 255.0
    rgb = 255.0 * (1.0 - c[..., :3]) * (1.0 - c[..., 3:4])
    return np.rint(rgb).astype(np.uint8)


def pil_to_rgb(img: Image.Image) -> np.ndarray:
    """``PIL.Image`` de qualquer modo -> RGB uint8 (alfa composto sobre branco)."""
    mode = img.mode
    if mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
        mode = img.mode
    if mode in ("RGBA", "LA", "PA", "RGBa", "La"):
        rgba = np.asarray(img.convert("RGBA"))
        return composite_on_white(rgba[..., :3], rgba[..., 3])
    if mode.startswith("I;16") or mode in ("I", "F"):
        gray = to_uint8(np.asarray(img))
        return np.repeat(gray[..., None], 3, axis=2)
    return np.asarray(img.convert("RGB")).copy()  # 1, L, RGB, CMYK, YCbCr...


def array_to_rgb(arr: np.ndarray, is_cmyk: bool, planar_separate: bool) -> np.ndarray:
    """Arrays do ``tifffile`` (gray/RGB/RGBA/CMYK, 8/16-bit, planar) -> RGB uint8."""
    arr = np.squeeze(arr)
    if arr.ndim == 3 and (planar_separate or (arr.shape[0] <= 4 < arr.shape[2])):
        arr = np.moveaxis(arr, 0, -1)  # (C, H, W) -> (H, W, C)
    if arr.ndim == 2:
        return np.repeat(to_uint8(arr)[..., None], 3, axis=2)
    if arr.ndim != 3:
        raise ValueError(f"Shape de TIFF não suportado: {arr.shape}")
    arr = to_uint8(arr)
    channels = arr.shape[2]
    if channels == 1:
        return np.repeat(arr, 3, axis=2)
    if channels == 2:  # gray + alpha
        return np.repeat(composite_on_white(arr[..., :1], arr[..., 1]), 3, axis=2)
    if channels == 3:
        return arr
    if channels == 4:
        return cmyk_to_rgb(arr) if is_cmyk else composite_on_white(arr[..., :3], arr[..., 3])
    return arr[..., :3]
