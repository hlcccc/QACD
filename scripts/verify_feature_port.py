#!/usr/bin/env python
"""Verify the ported feature engineering element-wise against the frozen matrix.

The evidence bundle records the claim-level feature matrix that the frozen
calibrator was fitted on. It was produced by the research pipeline. This script
rebuilds that matrix from the **raw evidence tables** using only this
repository's ported code (:mod:`frozen`), then compares the two element by
element.

A single mis-transcribed derivation shows up immediately as a non-zero
difference, which is what makes the port trustworthy.

    python scripts/verify_feature_port.py

Exit status is 0 only when every dev and test cell matches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frozen.features import (  # noqa: E402
    add_direction_aligned_features,
    apply_imputer,
    fit_imputer,
    load_direct_features,
    select_feature_names,
)

#: Element-wise agreement tolerance. The derivations are pure arithmetic on
#: float64 columns, so anything above this is a real transcription difference.
TOLERANCE = 1e-12


def build_frame(raw_dir: Path, split: str) -> pd.DataFrame:
    """Assemble one split's claim frame from the raw tables, using frozen code."""
    return load_direct_features(
        raw_dir / f"{split}_features.csv.gz",
        raw_dir / f"{split}_claims.csv.gz",
        raw_dir / f"{split}_direct.csv.gz",
    ).merge(
        pd.read_csv(raw_dir / f"{split}_mechanical.csv.gz", keep_default_na=False).drop_duplicates("unit_id"),
        on="unit_id",
        how="left",
        suffixes=("", "_mechanical"),
    )


def image_of_claim(frame: pd.DataFrame, claim_table: Path) -> np.ndarray:
    claims = pd.read_csv(claim_table, usecols=["unit_id", "image"], keep_default_na=False)
    lookup = dict(zip(claims["unit_id"].astype(str), claims["image"].astype(str)))
    return np.array([lookup.get(str(u), "") for u in frame["unit_id"]], dtype=object)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="artifacts/evidence_bundle.npz")
    parser.add_argument("--raw", default="artifacts/raw")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    bundle = np.load(args.bundle, allow_pickle=False)
    reference_names = [str(x) for x in bundle["lm_names"]] + [str(x) for x in bundle["mechanical_names"]]
    score_train_images = set(str(x) for x in bundle["score_train_images"])

    print("=" * 88)
    print("Verifying the ported feature engineering against the frozen matrix")
    print("=" * 88)
    print(f"  reference matrix : {bundle['dev_matrix'].shape} (dev), {bundle['test_matrix'].shape} (test)")
    print(f"  feature names    : {len(reference_names)}")
    print()

    print("[1/3] assembling development frame from raw tables ...")
    dev = build_frame(raw_dir, "dev")
    test = build_frame(raw_dir, "test")
    print(f"      dev {dev.shape}  test {test.shape}")

    claim_image = image_of_claim(dev, raw_dir / "dev_claims.csv.gz")
    train_mask = np.array([img in score_train_images for img in claim_image], dtype=bool)
    print(f"      score-training claim rows: {int(train_mask.sum())}/{len(train_mask)}")

    print("[2/3] selecting features and fitting the imputer on score-training rows ...")
    dev_aligned = add_direction_aligned_features(dev)
    test_aligned = add_direction_aligned_features(test)
    lm_names, mech_names = select_feature_names(dev_aligned, test_aligned, train_mask)
    names = lm_names + mech_names
    print(f"      selected {len(lm_names)} LM + {len(mech_names)} mechanical = {len(names)}")
    print(f"      reference has {len(reference_names)}")

    if names != reference_names:
        first = next((i for i, (a, b) in enumerate(zip(names, reference_names)) if a != b), None)
        print()
        print("  !! feature NAME/ORDER mismatch")
        if first is not None:
            print(f"     first divergence at index {first}: ported={names[first]!r} reference={reference_names[first]!r}")
        only_port = sorted(set(names) - set(reference_names))
        only_ref = sorted(set(reference_names) - set(names))
        if only_port:
            print(f"     only in port   ({len(only_port)}): {only_port[:6]}")
        if only_ref:
            print(f"     only in ref    ({len(only_ref)}): {only_ref[:6]}")
        return 1

    medians = fit_imputer(dev_aligned, names, train_mask)
    reference_medians = bundle["imputer_medians"]
    median_diff = np.max(np.abs(np.array([medians[n] for n in names]) - reference_medians))

    print("[3/3] comparing matrices element-wise ...")
    results = {}
    for split, frame in (("dev", dev_aligned), ("test", test_aligned)):
        mine = apply_imputer(frame, names, medians)
        theirs = bundle[f"{split}_matrix"]
        if mine.shape != theirs.shape:
            print(f"  !! {split}: shape {mine.shape} != reference {theirs.shape}")
            return 1
        diff = np.abs(mine - theirs)
        worst = int(np.argmax(diff))
        row, col = np.unravel_index(worst, diff.shape)
        results[split] = {
            "max_abs_diff": float(diff.max()),
            "n_mismatch": int((diff > args.tolerance).sum()),
            "worst_feature": names[col],
            "worst_row": int(row),
            "worst_mine": float(mine[row, col]),
            "worst_ref": float(theirs[row, col]),
        }
        flag = "OK  " if diff.max() <= args.tolerance else "FAIL"
        print(
            f"  [{flag}] {split:<5} max|diff| = {diff.max():.3e}   "
            f"cells over tol: {results[split]['n_mismatch']}"
        )
        if diff.max() > args.tolerance:
            print(
                f"         worst cell: feature={names[col]!r} row={row} "
                f"ported={mine[row, col]!r} reference={theirs[row, col]!r}"
            )

    print()
    print(f"  imputer medians max|diff| = {median_diff:.3e}")
    ok = all(r["max_abs_diff"] <= args.tolerance for r in results.values()) and median_diff <= args.tolerance
    print()
    print("=" * 88)
    print("VERDICT:", "port reproduces the frozen feature matrix" if ok else "PORT DIVERGES")
    print("=" * 88)

    print(
        json.dumps(
            {
                "results": results,
                "median_max_abs_diff": float(median_diff),
                "ok": bool(ok),
            },
            indent=2,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
