"""Orquestração: descoberta de arquivos, paralelismo e relatório."""
from __future__ import annotations

import logging
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Set

import cv2
from tqdm import tqdm

from .config import Settings
from .factory import build_file_processor, build_readers
from .models import FileResult, Task
from .processing import FileProcessor
from .reporting import Report, setup_logging

log = logging.getLogger("dataset_prep")

# Estado por processo: cada worker constrói seu FileProcessor uma única vez.
_WORKER: Optional[FileProcessor] = None


def _init_worker(settings: Settings) -> None:
    global _WORKER
    cv2.setNumThreads(1)  # o paralelismo é por arquivo; evita oversubscription
    _WORKER = build_file_processor(settings)


def _run_in_worker(task: Task) -> FileResult:
    assert _WORKER is not None
    return _WORKER.process(task)


class DatasetPipeline:
    """Executa o fluxo completo sobre ``settings.input_dir``."""

    def __init__(self, settings: Settings, *, dry_run: bool = False,
                 debug_samples: int = 0, seed: int = 42, verbose: bool = False) -> None:
        self.s = settings
        self.dry_run = dry_run
        self.debug_samples = debug_samples if debug_samples > 0 else (20 if dry_run else 0)
        self.seed = seed
        self.verbose = verbose

    # ----------------------------------------------------------- descoberta
    def discover_files(self) -> List[Path]:
        """Arquivos suportados (recursivo), ignorando ocultos e saída/debug."""
        exts = build_readers(self.s).supported_extensions
        skip = [p.resolve() for p in (self.s.output_dir, self.s.debug_dir)]
        return [p for p in sorted(self.s.input_dir.rglob("*"))
                if p.is_file() and not p.name.startswith(".")
                and p.suffix.lower() in exts
                and not any(p.resolve().is_relative_to(d) for d in skip)]

    def build_tasks(self, files: List[Path]) -> List[Task]:
        """Cria tarefas; resolve colisões (a.jpg + a.png) com sufixo da extensão."""
        debug_set: Set[Path] = set(
            random.Random(self.seed).sample(files, min(self.debug_samples, len(files))))
        rels = [f.relative_to(self.s.input_dir) for f in files]
        counts = Counter((r.parent, r.stem.lower()) for r in rels)
        tasks = []
        for f, rel in zip(files, rels):
            stem = (rel.stem if counts[(rel.parent, rel.stem.lower())] == 1
                    else f"{rel.stem}_{rel.suffix.lstrip('.').lower()}")
            dbg = (self.s.debug_dir / f"{'__'.join((*rel.parent.parts, stem))}.png"
                   if f in debug_set else None)
            tasks.append(Task(f, rel.parent, stem, not self.dry_run, dbg))
        return tasks

    # ------------------------------------------------------------ execução
    def run(self) -> int:
        s = self.s
        if not s.input_dir.is_dir():
            print(f"Diretório de entrada não existe: {s.input_dir}", file=sys.stderr)
            return 2
        setup_logging(s.log_file, self.verbose)
        t0 = time.perf_counter()
        files = self.discover_files()
        if not files:
            print("Nenhum arquivo suportado encontrado.", file=sys.stderr)
            return 1
        tasks = self.build_tasks(files)
        log.info("Iniciando: %d arquivos | %s", len(tasks), s)

        report = Report()
        if s.workers == 1:
            fp = build_file_processor(s)
            for t in tqdm(tasks, desc="Processando", unit="arq"):
                report.add(fp.process(t))
        else:
            with ProcessPoolExecutor(max_workers=s.workers, initializer=_init_worker,
                                     initargs=(s,)) as ex:
                futs = {ex.submit(_run_in_worker, t): t for t in tasks}
                for fut in tqdm(as_completed(futs), total=len(futs), desc="Processando", unit="arq"):
                    t = futs[fut]
                    try:
                        res = fut.result()
                    except Exception as exc:  # ex.: worker morto (BrokenProcessPool)
                        res = FileResult(str(t.src), t.src.suffix.lower(), "?",
                                         errors=[f"Falha no worker: {type(exc).__name__}: {exc}"])
                    report.add(res)

        print(report.render(time.perf_counter() - t0, self.dry_run, s.log_file, s.debug_dir))
        return 0
