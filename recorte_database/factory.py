"""Fábricas: traduzem ``Settings`` em objetos concretos (registries polimórficos).

Para adicionar um novo método de recorte/formato, implemente a ABC e registre
um construtor em ``CROPPER_BUILDERS`` / ``WRITER_BUILDERS``.
"""
from __future__ import annotations

from typing import Callable, Dict

from .config import Settings
from .croppers import Cropper, GoldFrameCropper, NullCropper, WhiteBorderCropper
from .processing import FileProcessor, ImageProcessor
from .readers import PdfReader, RasterReader, ReaderRegistry, TiffReader
from .resizers import LetterboxResizer, Resizer
from .writers import ImageWriter, JpegWriter, PngWriter


def _white(s: Settings) -> Cropper:
    return WhiteBorderCropper(s.white_threshold, s.crop_margin, s.morph_kernel, s.min_content_pixels)


def _gold(s: Settings) -> Cropper:
    return GoldFrameCropper(
        base=_white(s), hue_range=s.frame_hue, min_saturation=s.frame_min_sat,
        min_value=s.frame_min_val, band_ratio=s.frame_band_ratio,
        search_fraction=s.frame_search_fraction, inner_margin=s.frame_inner_margin,
        min_sides=s.frame_min_sides, fallback=s.frame_fallback,
        detect_max_side=s.detect_max_side)


CROPPER_BUILDERS: Dict[str, Callable[[Settings], Cropper]] = {
    "gold-frame": _gold,
    "white": _white,
    "none": lambda s: NullCropper(),
}

WRITER_BUILDERS: Dict[str, Callable[[Settings], ImageWriter]] = {
    "png": lambda s: PngWriter(s.png_compress_level),
    "jpg": lambda s: JpegWriter(s.jpeg_quality),
}


def build_readers(s: Settings) -> ReaderRegistry:
    return ReaderRegistry([PdfReader(s.pdf_dpi, s.pdf_max_side), TiffReader(), RasterReader()])


def build_cropper(s: Settings) -> Cropper:
    return CROPPER_BUILDERS[s.crop_mode](s)


def build_resizer(s: Settings) -> Resizer:
    return LetterboxResizer(s.target_size, s.padding_color)


def build_writer(s: Settings) -> ImageWriter:
    return WRITER_BUILDERS[s.output_format](s)


def build_file_processor(s: Settings) -> FileProcessor:
    return FileProcessor(build_readers(s),
                         ImageProcessor(build_cropper(s), build_resizer(s)),
                         build_writer(s), s.output_dir)
