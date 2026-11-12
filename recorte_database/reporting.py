"""Logging e relatório final."""
from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path
from typing import List

from tqdm import tqdm

from .models import FileResult

log = logging.getLogger("dataset_prep")


class TqdmLoggingHandler(logging.StreamHandler):
    """Imprime logs sem quebrar a barra de progresso."""

    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record), file=sys.stderr)


def setup_logging(log_file: Path, verbose: bool) -> None:
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG if verbose else logging.INFO)
    fh.setFormatter(fmt)
    ch = TqdmLoggingHandler()
    ch.setLevel(logging.INFO if verbose else logging.ERROR)
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)


class Report:
    """Acumula ``FileResult`` e gera o relatório."""

    def __init__(self) -> None:
        self.results: List[FileResult] = []

    def add(self, res: FileResult) -> None:
        self.results.append(res)
        for e in res.errors:
            log.error("%s -> %s", res.src, e)
        for n in res.notes:
            log.warning("%s -> %s", res.src, n)
        log.info("%s -> %d imagem(ns) gerada(s)", res.src, res.generated)

    def render(self, elapsed: float, dry_run: bool, log_file: Path, debug_dir: Path) -> str:
        rs = self.results
        read = Counter(r.ext for r in rs)
        problems = [r for r in rs if r.errors]
        by_class: Counter = Counter()
        for r in rs:
            by_class[r.cls] += r.generated
        debug_n = sum(r.debug_saved for r in rs)

        L = ["", "=" * 64,
             "RELATÓRIO FINAL" + ("  [DRY-RUN: nada foi gravado em OUTPUT]" if dry_run else ""),
             "=" * 64, f"Arquivos lidos: {len(rs)}"]
        L += [f"  {ext:<6} {n}" for ext, n in sorted(read.items())]
        L += [f"Imagens geradas:                     {sum(r.generated for r in rs)}",
              f"  (dessas, sem moldura -> fallback): {sum(r.fallback_pages for r in rs)}",
              f"Páginas vazias descartadas:          {sum(r.blank_pages for r in rs)}",
              f"Páginas rejeitadas (sem moldura):    {sum(r.rejected_pages for r in rs)}",
              f"Arquivos com erro:                   {len(problems)}"]
        for r in problems[:30]:
            L.append(f"  - {r.src}: {' | '.join(r.errors)}")
        if len(problems) > 30:
            L.append(f"  ... +{len(problems) - 30} (veja {log_file})")
        if by_class:
            L.append("Imagens por classe:")
            L += [f"  {c}: {n}" for c, n in sorted(by_class.items())]
        if debug_n:
            L.append(f"Amostras de debug: {debug_n} em {debug_dir}")
        L.append(f"Tempo total: {elapsed:.1f}s")
        text = "\n".join(L)
        log.info(text)
        return text
