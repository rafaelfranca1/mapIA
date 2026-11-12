"""CLI de preparação de dataset de imagens para ResNet (ImageFolder).

Calibrar:  python main.py -i data/raw --dry-run --debug-samples 30
Executar:  python main.py -i data/raw -o data/processed --workers 8
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Union

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name

from . import config
from .config import Settings, parse_hue_range, parse_padding_color
from .pipeline import DatasetPipeline


def _margin_type(v: str) -> Union[int, float]:
    f = float(v)
    return f if ("." in v and f < 1) else int(f)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Limpa, recorta e padroniza imagens para ResNet.")
    g = p.add_argument_group("entrada/saída")
    g.add_argument("-i", "--input", type=Path, default=config.INPUT_DIR)
    g.add_argument("-o", "--output", type=Path, default=config.OUTPUT_DIR)
    g.add_argument("--size", type=int, default=config.TARGET_SIZE, help="Lado final (224, 384, 512...)")
    g.add_argument("--format", choices=["png", "jpg"], default=config.OUTPUT_FORMAT)
    g.add_argument("--quality", type=int, default=config.JPEG_QUALITY, help="Qualidade JPEG")
    g.add_argument("--padding-color", default=config.PADDING_COLOR, help="black | white | imagenet | R,G,B")
    g.add_argument("--dpi", type=int, default=config.PDF_DPI, help="DPI de renderização dos PDFs")
    g.add_argument("--pdf-max-side", type=int, default=config.PDF_MAX_SIDE,
                   help="Lado máx. (px) ao renderizar PDFs")

    c = p.add_argument_group("recorte")
    c.add_argument("--crop-mode", choices=config.CROP_MODES, default=config.CROP_MODE)
    c.add_argument("--threshold", type=int, default=config.WHITE_THRESHOLD,
                   help="Borda branca: cinza < threshold = conteúdo")
    c.add_argument("--margin", type=_margin_type, default=config.CROP_MARGIN,
                   help="Respiro da borda branca: inteiro = px; decimal <1 = fração")
    c.add_argument("--morph-kernel", type=int, default=config.MORPH_KERNEL)
    c.add_argument("--frame-hue", default="%d,%d" % config.FRAME_HUE, help="Faixa de matiz 'min,max' (0-179)")
    c.add_argument("--frame-min-sat", type=int, default=config.FRAME_MIN_SAT)
    c.add_argument("--frame-min-val", type=int, default=config.FRAME_MIN_VAL)
    c.add_argument("--frame-band-ratio", type=float, default=config.FRAME_BAND_RATIO)
    c.add_argument("--frame-search", type=float, default=config.FRAME_SEARCH_FRACTION,
                   help="Espessura máx. da moldura (fração do lado)")
    c.add_argument("--frame-inner-margin", type=_margin_type, default=config.FRAME_INNER_MARGIN,
                   help="Respiro para dentro da moldura (px ou fração)")
    c.add_argument("--frame-min-sides", type=int, choices=[1, 2, 3, 4], default=config.FRAME_MIN_SIDES)
    c.add_argument("--frame-fallback", choices=["keep", "skip"], default=config.FRAME_FALLBACK,
                   help="Sem moldura detectada: keep = recorte de borda branca; skip = descarta")

    x = p.add_argument_group("execução")
    x.add_argument("--workers", type=int, default=config.WORKERS, help="Processos (1 = sequencial)")
    x.add_argument("--dry-run", action="store_true", help="Processa sem gravar o dataset")
    x.add_argument("--debug-samples", type=int, default=0, metavar="N",
                   help="Salva N comparações Antes | Máscara | Depois")
    x.add_argument("--debug-dir", type=Path, default=config.DEBUG_DIR)
    x.add_argument("--seed", type=int, default=config.DEBUG_SEED)
    x.add_argument("--log-file", type=Path, default=config.LOG_FILE)
    x.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    a = parse_args(argv)
    settings = replace(
        Settings(), input_dir=a.input, output_dir=a.output, debug_dir=a.debug_dir,
        log_file=a.log_file, target_size=a.size, output_format=a.format,
        jpeg_quality=a.quality, padding_color=parse_padding_color(a.padding_color),
        pdf_dpi=a.dpi, pdf_max_side=a.pdf_max_side, crop_mode=a.crop_mode,
        white_threshold=a.threshold, crop_margin=a.margin, morph_kernel=a.morph_kernel,
        frame_hue=parse_hue_range(a.frame_hue), frame_min_sat=a.frame_min_sat,
        frame_min_val=a.frame_min_val, frame_band_ratio=a.frame_band_ratio,
        frame_search_fraction=a.frame_search, frame_inner_margin=a.frame_inner_margin,
        frame_min_sides=a.frame_min_sides, frame_fallback=a.frame_fallback,
        workers=max(1, a.workers),
    )
    return DatasetPipeline(settings, dry_run=a.dry_run, debug_samples=a.debug_samples,
                           seed=a.seed, verbose=a.verbose).run()


if __name__ == "__main__":
    sys.exit(main())
