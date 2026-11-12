"""Leitores de imagem (Strategy): um por família de formato."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, FrozenSet, Iterable, Iterator, Optional

import numpy as np
from PIL import Image, ImageOps

try:  # PyMuPDF >= 1.24 expõe "pymupdf"; versões antigas, "fitz"
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore[no-redef]

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None  # type: ignore[assignment]

from .conversions import array_to_rgb, pil_to_rgb
from .models import Page

log = logging.getLogger("dataset_prep")

# Scans grandes são legítimos neste fluxo; desliga o guard de "decompression bomb".
Image.MAX_IMAGE_PIXELS = None


class ImageReader(ABC):
    """Contrato: dado um arquivo, gera ``Page`` (RGB uint8) uma a uma."""

    extensions: ClassVar[FrozenSet[str]] = frozenset()

    @abstractmethod
    def read(self, path: Path) -> Iterator[Page]:
        """Gera as páginas do arquivo. Levanta exceção se o arquivo for inválido."""


class PdfReader(ImageReader):
    """PDF via PyMuPDF (sem Poppler). Limita o lado maior da renderização:
    páginas grandes (plantas/mapas) estouram o limite do MuPDF a 300 DPI."""

    extensions = frozenset({".pdf"})

    def __init__(self, dpi: int = 300, max_side: int = 8000) -> None:
        self.dpi = dpi
        self.max_side = max_side

    def read(self, path: Path) -> Iterator[Page]:
        doc = fitz.open(path)
        try:
            if doc.needs_pass:
                raise ValueError("PDF protegido por senha")
            n = len(doc)
            if n == 0:
                raise ValueError("PDF sem páginas")
            for i in range(n):
                yield Page(i, n, self._render(doc[i]))
        finally:
            doc.close()

    def _render(self, page) -> np.ndarray:
        longest_pt = max(page.rect.width, page.rect.height)
        scale = self.dpi / 72.0
        if longest_pt * scale > self.max_side:
            scale = self.max_side / longest_pt
            log.info("Página %.0fpt: DPI efetivo reduzido para %.0f (lado máx. %dpx)",
                     longest_pt, scale * 72, self.max_side)
        last_exc: Optional[Exception] = None
        for _ in range(4):  # em falha (FzErrorLimit/memória) reduz o DPI pela metade
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale),
                                      colorspace=fitz.csRGB, alpha=False)
                arr = np.frombuffer(pix.samples, dtype=np.uint8)
                return arr.reshape(pix.height, pix.width, pix.n).copy()
            except Exception as exc:
                last_exc = exc
                scale /= 2
                log.warning("Falha ao renderizar (%s); tentando com DPI %.0f", exc, scale * 72)
        raise RuntimeError(f"Não foi possível renderizar a página: {last_exc}")


class TiffReader(ImageReader):
    """TIFF multipágina. Pillow primeiro (CMYK/paleta/alfa); ``tifffile`` como
    fallback (RGB 16-bit, float32, planar)."""

    extensions = frozenset({".tif", ".tiff"})

    def read(self, path: Path) -> Iterator[Page]:
        try:
            img = Image.open(path)
            n = getattr(img, "n_frames", 1)
            first = pil_to_rgb(img)
        except Exception as exc:
            log.debug("Pillow falhou em %s (%s); usando tifffile", path, exc)
            yield from self._read_tifffile(path)
            return
        with img:
            yield Page(0, n, first)
            for i in range(1, n):
                img.seek(i)
                yield Page(i, n, pil_to_rgb(img))

    @staticmethod
    def _read_tifffile(path: Path) -> Iterator[Page]:
        if tifffile is None:
            raise RuntimeError("tifffile não instalado")
        with tifffile.TiffFile(path) as tif:
            n = len(tif.pages)
            for i, page in enumerate(tif.pages):
                photometric = getattr(page, "photometric", None)
                planar = getattr(page, "planarconfig", None)
                rgb = array_to_rgb(
                    page.asarray(),
                    is_cmyk=photometric == tifffile.PHOTOMETRIC.SEPARATED,
                    planar_separate=planar == tifffile.PLANARCONFIG.SEPARATE,
                )
                yield Page(i, n, rgb)


class RasterReader(ImageReader):
    """JPEG/PNG: respeita orientação EXIF; alfa composto sobre branco."""

    extensions = frozenset({".jpg", ".jpeg", ".png"})

    def read(self, path: Path) -> Iterator[Page]:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            yield Page(0, 1, pil_to_rgb(img))


class ReaderRegistry:
    """Escolhe o ``ImageReader`` pela extensão (despacho polimórfico)."""

    def __init__(self, readers: Iterable[ImageReader]) -> None:
        self._by_ext = {ext: r for r in readers for ext in r.extensions}

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset(self._by_ext)

    def reader_for(self, path: Path) -> ImageReader:
        try:
            return self._by_ext[path.suffix.lower()]
        except KeyError:
            raise ValueError(f"Extensão não suportada: {path.suffix}") from None
