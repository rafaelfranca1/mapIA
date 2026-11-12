"""Configuração central do pipeline. Todos os valores são *padrões* e podem
ser sobrescritos via CLI (``python main.py --help``)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #
INPUT_DIR: Path = Path("data/raw")
OUTPUT_DIR: Path = Path("data/processed")
DEBUG_DIR: Path = Path("data/debug")
LOG_FILE: Path = Path("processing.log")

# --------------------------------------------------------------------------- #
# Saída
# --------------------------------------------------------------------------- #
TARGET_SIZE: int = 512
"""Lado do quadrado final. 224 é o padrão da ResNet, mas perde muito detalhe em
mapas; 512 permite treinar em 384/512 ou reduzir para 224 em tempo de treino."""

OUTPUT_FORMAT: str = "jpg"  # "jpg" (leve) ou "png" (sem perdas)
JPEG_QUALITY: int = 95
PNG_COMPRESS_LEVEL: int = 3

PADDING_COLOR: str = "imagenet"
"""'black', 'white', 'imagenet' ou 'R,G,B'."""
PADDING_PRESETS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "imagenet": (124, 116, 104),
}

# --------------------------------------------------------------------------- #
# Recorte
# --------------------------------------------------------------------------- #
CROP_MODES = ("gold-frame", "white", "none")
CROP_MODE: str = "gold-frame"
"""gold-frame: recorta da moldura dourada para dentro; white: só remove borda
branca; none: não recorta."""

# Borda branca (também é o 1º passo do modo gold-frame)
WHITE_THRESHOLD: int = 245       # cinza < threshold = conteúdo
MORPH_KERNEL: int = 3            # abertura morfológica (1 desliga)
MIN_CONTENT_PIXELS: int = 20
CROP_MARGIN: Union[int, float] = 5  # int = px; float < 1 = fração

# Moldura dourada (HSV do OpenCV: H 0-179, S/V 0-255)
FRAME_HUE: Tuple[int, int] = (15, 35)
FRAME_MIN_SAT: int = 90          # suba se o pergaminho do miolo for detectado como moldura
FRAME_MIN_VAL: int = 100
FRAME_BAND_RATIO: float = 0.5    # fração dourada da linha/coluna p/ contar como moldura
FRAME_SEARCH_FRACTION: float = 0.25  # espessura máx. da moldura (fração do lado)
FRAME_INNER_MARGIN: Union[int, float] = 0.002  # respiro após a borda interna (0.002 = 0,2%)
FRAME_MIN_SIDES: int = 4         # lados em que a moldura deve ser encontrada
FRAME_FALLBACK: str = "keep"     # sem moldura: 'keep' (borda branca + aviso) ou 'skip'
DETECT_MAX_SIDE: int = 1500      # a detecção roda em versão reduzida (rápido)

# --------------------------------------------------------------------------- #
# Leitura / paralelismo
# --------------------------------------------------------------------------- #
PDF_DPI: int = 300
PDF_MAX_SIDE: int = 8000
"""Lado máx. (px) ao renderizar PDF; evita 'Overly large image' em plantas/mapas."""

WORKERS: int = max(1, (os.cpu_count() or 2) - 1)
DEBUG_SEED: int = 42


@dataclass(frozen=True)
class Settings:
    """Parâmetros efetivos (imutável e serializável para os workers)."""

    input_dir: Path = INPUT_DIR
    output_dir: Path = OUTPUT_DIR
    debug_dir: Path = DEBUG_DIR
    log_file: Path = LOG_FILE
    target_size: int = TARGET_SIZE
    output_format: str = OUTPUT_FORMAT
    jpeg_quality: int = JPEG_QUALITY
    png_compress_level: int = PNG_COMPRESS_LEVEL
    padding_color: Tuple[int, int, int] = PADDING_PRESETS[PADDING_COLOR]
    crop_mode: str = CROP_MODE
    white_threshold: int = WHITE_THRESHOLD
    morph_kernel: int = MORPH_KERNEL
    min_content_pixels: int = MIN_CONTENT_PIXELS
    crop_margin: Union[int, float] = CROP_MARGIN
    frame_hue: Tuple[int, int] = FRAME_HUE
    frame_min_sat: int = FRAME_MIN_SAT
    frame_min_val: int = FRAME_MIN_VAL
    frame_band_ratio: float = FRAME_BAND_RATIO
    frame_search_fraction: float = FRAME_SEARCH_FRACTION
    frame_inner_margin: Union[int, float] = FRAME_INNER_MARGIN
    frame_min_sides: int = FRAME_MIN_SIDES
    frame_fallback: str = FRAME_FALLBACK
    detect_max_side: int = DETECT_MAX_SIDE
    pdf_dpi: int = PDF_DPI
    pdf_max_side: int = PDF_MAX_SIDE
    workers: int = WORKERS


def parse_padding_color(value: str) -> Tuple[int, int, int]:
    """``'black' | 'white' | 'imagenet' | 'R,G,B'`` -> tupla RGB."""
    key = value.strip().lower()
    if key in PADDING_PRESETS:
        return PADDING_PRESETS[key]
    try:
        parts = tuple(int(p) for p in key.split(","))
    except ValueError as exc:
        raise ValueError(f"Cor de padding inválida: {value!r}") from exc
    if len(parts) != 3 or not all(0 <= p <= 255 for p in parts):
        raise ValueError(f"Cor de padding deve ser 'R,G,B' em 0-255: {value!r}")
    return parts  # type: ignore[return-value]


def parse_hue_range(value: str) -> Tuple[int, int]:
    """``'15,35'`` -> (15, 35)."""
    try:
        lo, hi = (int(p) for p in value.split(","))
    except ValueError as exc:
        raise ValueError(f"Faixa de matiz deve ser 'min,max' (0-179): {value!r}") from exc
    if not 0 <= lo <= hi <= 179:
        raise ValueError(f"Faixa de matiz inválida: {value!r}")
    return lo, hi
