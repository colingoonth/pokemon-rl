"""Lightweight CSV logger for training runs.

Console prints are fine for smoke runs but useless for multi-hour ELSA
runs. CSV is the lowest-friction durable log: every iteration writes
a row, post-hoc plotting just reads the file. W&B integration can sit
on top of this later as a sink.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any


class CSVLogger:
    """Append-mode CSV logger that auto-writes header on first row."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None
        self._file = None
        self._writer: csv.DictWriter | None = None
        self._t0 = time.time()

    def log(self, row: dict[str, Any]) -> None:
        row = {"wall_time_s": round(time.time() - self._t0, 3), **row}
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._file = self.path.open("w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        assert self._writer is not None and self._fieldnames is not None
        # Drop unexpected keys to keep the file stable
        clean = {k: row.get(k, "") for k in self._fieldnames}
        self._writer.writerow(clean)
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
