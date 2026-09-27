#!/usr/bin/env python
"""Export the raw evidence tables so the ported feature engineering can be
verified offline against the frozen feature matrix.

Runs on the machine that holds the research repository, which is where the raw
evidence lives. It cannot run anywhere else: it imports the research protocol
module by path, and that module is not distributed with this repository.

This script is kept here as the provenance record for how the verification bundle
was assembled -- which inputs, and their SHA256 -- not as something a consumer of
this repository can execute. If you have been given a data bundle, you do not need
it; see REPRODUCING.md.

Copies the *inputs* of the feature stage, not the derived features, so that
``scripts/verify_feature_port.py`` exercises this repository's own ported code
rather than reading a pre-computed result.

Output, into --out:
    dev_features.csv.gz      claim-level belief/consistency features
    dev_claims.csv.gz        claim table (metadata; target columns dropped)
    dev_direct.csv.gz        raw direct-verifier readings
    dev_mechanical.csv.gz    mechanical OCR features
    test_*.csv.gz            same for the frozen test split
    raw_manifest.json        SHA256 of every source file and of each export
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path()  # set from --research-root
EXPERIMENTS = Path()  # set from --research-root
PROTOCOL = Path()  # set from --research-root
OUT_DIR = Path()  # set from --out

#: Columns that must never leave the research host inside this export.
TARGET_COLUMNS = ("correct", "hallucination_label", "label", "gold_letter", "answers")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol():
    import importlib.util

    spec = importlib.util.spec_from_file_location("qacd_frozen_protocol", PROTOCOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["qacd_frozen_protocol"] = module
    spec.loader.exec_module(module)
    return module


def drop_targets(src: Path, dst: Path) -> tuple[int, list[str]]:
    """Copy a CSV to gzip, dropping any target column."""
    import pandas as pd

    frame = pd.read_csv(src, keep_default_na=False)
    dropped = [c for c in frame.columns if c in TARGET_COLUMNS]
    if dropped:
        frame = frame.drop(columns=dropped)
    with gzip.open(dst, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
    return len(frame), dropped


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--research-root",
        required=True,
        help="root of the research repository; it must contain "
             "hallucination_calibration/experiments/ with the cached evidence and "
             "run_conformal_selective_prediction_strict.py",
    )
    parser.add_argument("--out", required=True, help="destination directory")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    global PROJECT, EXPERIMENTS, PROTOCOL, OUT_DIR
    args = _parse_args(argv)
    PROJECT = Path(args.research_root).expanduser().resolve()
    EXPERIMENTS = PROJECT / "hallucination_calibration" / "experiments"
    PROTOCOL = EXPERIMENTS / "run_conformal_selective_prediction_strict.py"
    OUT_DIR = Path(args.out).expanduser().resolve()
    if not PROTOCOL.is_file():
        raise SystemExit(
            f"research protocol not found: {PROTOCOL}\n"
            "  --research-root must point at a checkout of the research repository.\n"
            "  This script is author-side tooling; see REPRODUCING.md."
        )
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    P = load_protocol()

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "inputs of the feature stage, for offline verification of the ported code",
        "protocol_script": {"path": str(PROTOCOL), "sha256": sha256_file(PROTOCOL)},
        "exports": {},
        "sources": {},
    }

    for split, config in (("dev", P.DEV), ("test", P.TEST)):
        for key in ("features", "claims", "direct", "mechanical"):
            src = Path(config[key])
            dst = OUT_DIR / f"{split}_{key}.csv.gz"
            rows, dropped = drop_targets(src, dst)
            manifest["sources"][f"{split}_{key}"] = {
                "path": str(src),
                "sha256": sha256_file(src),
                "rows": rows,
                "target_columns_dropped": dropped,
            }
            manifest["exports"][dst.name] = {
                "bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
            }
            print(f"  {split:>4} {key:<10} {rows:>5} rows  targets dropped: {dropped or 'none'}")

    (OUT_DIR / "raw_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum(v["bytes"] for v in manifest["exports"].values())
    print()
    print(f"exported {len(manifest['exports'])} tables, {total / 1e6:.2f} MB -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
