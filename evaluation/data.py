"""Where the experimental data lives, and what to do when it is absent.

This repository ships **without** experimental data: the frozen feature matrices,
the raw evidence tables and the result tables are project deliverables and are
not published here. The code that consumes them is complete and testable, but
the verification scripts can only run where the data has been exported.

Convention: a script that cannot run because its inputs are missing exits with
:data:`DATA_ABSENT_EXIT` (2) and prints what to do, rather than raising a
traceback or - worse - reporting success on zero inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "DATA_ABSENT_EXIT",
    "DEFAULT_BUNDLE",
    "DEFAULT_RAW",
    "missing_paths",
    "report_absent",
    "require_data",
]

#: Exit status meaning "inputs absent", distinct from 1 ("check failed").
DATA_ABSENT_EXIT = 2

DEFAULT_BUNDLE = "artifacts/evidence_bundle.npz"
DEFAULT_RAW = "artifacts/raw"

_HOWTO = (
    "The experimental data is not part of this repository.\n"
    "To run this check, export it from the research host first:\n"
    "    python tools/export_raw_evidence.py     # raw evidence tables\n"
    "    python tools/export_bundle.py           # feature matrix + reference scores\n"
    "and place the outputs under artifacts/."
)


def missing_paths(paths: Iterable[str | Path]) -> list[str]:
    """Return the subset of ``paths`` that does not exist."""
    return [str(p) for p in paths if not Path(p).exists()]


def report_absent(missing: Sequence[str], purpose: str) -> int:
    """Print a plain explanation and return :data:`DATA_ABSENT_EXIT`."""
    print("=" * 88)
    print(f"Cannot {purpose}: required inputs are missing")
    print("=" * 88)
    for path in missing:
        print(f"  missing: {path}")
    print()
    print(_HOWTO)
    return DATA_ABSENT_EXIT


def require_data(paths: Iterable[str | Path], purpose: str) -> int | None:
    """``None`` when everything is present, otherwise the exit status to use."""
    missing = missing_paths(paths)
    if missing:
        return report_absent(missing, purpose)
    return None
