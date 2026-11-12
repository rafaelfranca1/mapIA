"""Composição das estratégias: ``ImageProcessor`` (imagem) e ``FileProcessor`` (arquivo)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .croppers import Cropper
from .debug import DebugRenderer
from .geometry import crop_array
from .models import CropStatus, FileResult, Page, ProcessOutcome, Task
from .readers import ReaderRegistry
from .resizers import Resizer
from .writers import ImageWriter, PngWriter


class ImageProcessor:
    """Recorte + padronização. Não sabe *como* recortar/redimensionar: delega
    às estratégias injetadas (``Cropper`` e ``Resizer``)."""

    def __init__(self, cropper: Cropper, resizer: Resizer) -> None:
        self.cropper = cropper
        self.resizer = resizer

    def process(self, rgb: np.ndarray) -> ProcessOutcome:
        crop = self.cropper.crop_box(rgb)
        if crop.bbox is None:
            return ProcessOutcome(crop, None)
        return ProcessOutcome(crop, self.resizer.resize(crop_array(rgb, crop.bbox)))


class FileProcessor:
    """Processa um arquivo inteiro (todas as páginas). Nunca levanta exceção:
    falhas são registradas em ``FileResult.errors``."""

    def __init__(self, readers: ReaderRegistry, processor: ImageProcessor,
                 writer: ImageWriter, output_dir: Path,
                 debug_renderer: DebugRenderer | None = None) -> None:
        self.readers = readers
        self.processor = processor
        self.writer = writer
        self.output_dir = output_dir
        self.debug_renderer = debug_renderer or DebugRenderer()
        self._debug_writer = PngWriter(compress_level=1)

    def process(self, task: Task) -> FileResult:
        parts = task.rel_parent.parts
        res = FileResult(str(task.src), task.src.suffix.lower(), parts[0] if parts else "<raiz>")
        try:
            reader = self.readers.reader_for(task.src)
            for page in reader.read(task.src):
                try:
                    self._handle_page(task, page, res)
                except Exception as exc:  # erro em UMA página não derruba o arquivo
                    res.errors.append(f"página {page.index + 1}: {type(exc).__name__}: {exc}")
        except Exception as exc:  # corrompido / protegido / formato inválido
            res.errors.append(f"{type(exc).__name__}: {exc}")
        return res

    def _handle_page(self, task: Task, page: Page, res: FileResult) -> None:
        outcome = self.processor.process(page.rgb)
        crop = outcome.crop

        if task.debug_path is not None and page.index == 0:
            self._debug_writer.write(self.debug_renderer.render(page.rgb, outcome), task.debug_path)
            res.debug_saved = True
        if crop.message:
            res.notes.append(f"página {page.index + 1}: {crop.message}")

        if outcome.image is None:
            if crop.status is CropStatus.EMPTY:
                res.blank_pages += 1
            else:
                res.rejected_pages += 1
            return
        if crop.status is CropStatus.FALLBACK:
            res.fallback_pages += 1

        if task.write_output:
            tag = f"_page_{page.index + 1}" if page.total > 1 else ""
            out = self.output_dir / task.rel_parent / f"{task.out_stem}{tag}{self.writer.extension}"
            self.writer.write(outcome.image, out)
        res.generated += 1
