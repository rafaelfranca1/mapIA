"""Modelos de dados compartilhados entre as camadas."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, NamedTuple, Optional

import numpy as np

from .geometry import BBox


class Page(NamedTuple):
    """Uma página/frame decodificada (RGB uint8)."""
    index: int
    total: int
    rgb: np.ndarray


class CropStatus(Enum):
    OK = "ok"                # recorte feito pelo método principal
    FALLBACK = "fallback"    # método principal falhou; usado recorte alternativo
    EMPTY = "empty"          # imagem vazia/branca
    REJECTED = "rejected"    # descartada por política (ex.: sem moldura + skip)


@dataclass
class CropResult:
    """Resultado de um ``Cropper``."""
    status: CropStatus
    bbox: Optional[BBox] = None          # None => nada a salvar
    outer_bbox: Optional[BBox] = None    # recorte de borda branca (p/ debug)
    method: str = ""
    message: str = ""
    debug_mask: Optional[np.ndarray] = None  # máscara dourada (p/ debug), relativa a outer_bbox


@dataclass
class ProcessOutcome:
    crop: CropResult
    image: Optional[np.ndarray]  # imagem final (size x size) ou None


@dataclass(frozen=True)
class Task:
    src: Path
    rel_parent: Path
    out_stem: str
    write_output: bool
    debug_path: Optional[Path]


@dataclass
class FileResult:
    src: str
    ext: str
    cls: str
    generated: int = 0
    blank_pages: int = 0
    rejected_pages: int = 0
    fallback_pages: int = 0
    debug_saved: bool = False
    notes: List[str] = field(default_factory=list)   # avisos (WARNING)
    errors: List[str] = field(default_factory=list)  # erros (ERROR)
