#!/usr/bin/env python
"""Single entry point for QACD — matches the `run.py` referenced by the
technical integration table.

    python run.py demo                                  # offline end-to-end demo
    python run.py score --question "..." --answer "..." # score one answer
    python run.py fit --data dev.jsonl --out scorer.json --k 3
    python run.py serve --scorer scorer.json --port 8080

Every subcommand is implemented in :mod:`qacd.cli`; this file only exists so the
platform team has one obvious thing to run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qacd.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
