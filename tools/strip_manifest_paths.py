#!/usr/bin/env python
"""Take the machine's directory layout back out of an evidence manifest.

The manifests this repository ships were written by export tools that recorded
the *absolute* path of every source file. That is provenance, and provenance is
worth keeping -- but the directory part of it names a specific research server
and a specific home directory, and the manifests travel: they sit in the
submission package and their ``sources`` block is copied verbatim into
``results/manifest.json`` by ``run_frozen_evaluation.py``. ``NOTICE.md`` already
promises that absolute paths are omitted, so the files contradicted the notice.

What actually makes an input verifiable is its **hash**, not its location. This
replaces ``"path": "/somewhere/private/x.csv"`` with ``"file": "x.csv"`` and
changes nothing else, so every SHA256, byte count and row count survives and a
consumer can still check the bundle they were given.

Idempotent: running it twice is the same as running it once.

    python tools/strip_manifest_paths.py artifacts/bundle_manifest.json
    python tools/strip_manifest_paths.py --check artifacts/        # report only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

__all__ = ["is_absolute", "relativise", "strip_file"]


def is_absolute(value: str) -> bool:
    """Posix absolute, Windows absolute, or a UNC path."""
    if not isinstance(value, str) or not value:
        return False
    return value.startswith("/") or value.startswith("\\\\") or (
        len(value) > 2 and value[1] == ":" and value[2] in "\\/"
    )


def relativise(node: Any) -> Tuple[Any, int]:
    """Replace every absolute ``path`` with a ``file`` holding its basename.

    Returns the new tree and how many entries were rewritten, so a caller can
    report "nothing to do" rather than silently rewriting nothing.
    """
    changed = 0

    def walk(value: Any) -> Any:
        nonlocal changed
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for key, item in value.items():
                if key == "path" and is_absolute(item):
                    out["file"] = Path(item.replace("\\", "/")).name
                    changed += 1
                else:
                    out[key] = walk(item)
            return out
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value

    return walk(node), changed


def strip_file(path: Path, *, check: bool = False) -> int:
    """Rewrite one manifest. Returns the number of entries that were absolute."""
    raw = path.read_text(encoding="utf-8")
    original = json.loads(raw)
    cleaned, changed = relativise(original)
    if not changed:
        print(f"  {path}: already clean")
        return 0
    if check:
        print(f"  {path}: {changed} absolute path(s) still present")
        return changed
    path.write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")
    print(f"  {path}: {changed} absolute path(s) -> basenames")
    return changed


def collect(targets: List[str]) -> List[Path]:
    found: List[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_dir():
            found.extend(sorted(p for p in path.rglob("*manifest*.json")))
        elif path.is_file():
            found.append(path)
        else:
            raise SystemExit(f"not found: {target}")
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("targets", nargs="+", help="manifest files, or directories to search")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report absolute paths without rewriting; exits 1 if any remain",
    )
    args = parser.parse_args(argv)

    total = sum(strip_file(path, check=args.check) for path in collect(args.targets))
    if args.check and total:
        print(f"\n{total} absolute path(s) remain")
        return 1
    print(f"\n{total} rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
