#!/usr/bin/env python
"""Reproduce the frozen TextVQA results from the evidence bundle.

    python scripts/run_frozen_evaluation.py \
        --bundle artifacts/evidence_bundle.npz \
        --out results

What this does
--------------
Fits this repository's own calibrators on the score-training split, scores the
frozen test split, aggregates claims to responses with the max operator, and
computes every number written to ``results/``. Nothing is transcribed: the
tables in ``results/`` are the output of this script.

What it does not do
-------------------
It does not run the LVLM. The evidence (model readings, OCR, re-sampling) is the
cached, hashed bundle produced on the research host. The bundle's SHA256 is
verified before use and recorded in ``results/manifest.json``, together with the
SHA256 of every file this script writes.

Reference columns
-----------------
The bundle also carries the per-response scores recorded by the frozen replay.
They are used only to *check* this run, and the comparison is reported in
``results/reproduction_check.csv``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.bundle import load_bundle  # noqa: E402
from evaluation.metrics import auroc, brier, image_level, paired_image_bootstrap  # noqa: E402
from qacd.aggregate import claim_to_response_max  # noqa: E402
from qacd.calibrate import BICLiteCalibrator, RidgeLogistic  # noqa: E402

REPLICATES = 5000
BOOTSTRAP_SEED = 20260920
CLAIM_L2 = 0.05

#: Response AUROC is reproduced to floating-point noise (about 1e-6) for the
#: claim-level arms. The MVR arms differ by ~1e-4..1e-3 because the frozen
#: pipeline fitted its response head slightly differently; those are reported as
#: approximate rather than exact.
EXACT_TOLERANCE = 1e-5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_and_score(bundle, names: List[str], columns: List[int], verbose: bool = True):
    """Fit the claim calibrator on score-training rows, score dev and test."""
    dev_x = bundle["dev_matrix"][:, columns]
    test_x = bundle["test_matrix"][:, columns]
    train_mask = bundle["dev_claim_train_mask"]
    labels = bundle["dev_claim_labels"]

    calibrator = BICLiteCalibrator(l2=CLAIM_L2).fit(dev_x[train_mask], labels[train_mask])
    if not calibrator.success_:
        raise RuntimeError("claim calibrator failed to converge on the score-training split")

    dev_claim = calibrator.predict_proba(dev_x)
    test_claim = calibrator.predict_proba(test_x)

    dev_response = claim_to_response_max(
        dev_claim, len(bundle["dev_response_y"]), bundle["dev_claim_response_index"]
    )
    test_response = claim_to_response_max(
        test_claim, len(bundle["test_response_y"]), bundle["test_claim_response_index"]
    )
    if verbose:
        print(f"      {len(names):3d} features | score-train rows {int(train_mask.sum())}")
    return dev_response, test_response


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="artifacts/evidence_bundle.npz")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", default="results")
    parser.add_argument("--no-verify", action="store_true", help="skip the bundle SHA256 check")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    bundle = load_bundle(args.bundle, args.manifest, verify=not args.no_verify)
    print("=" * 78)
    print("QACD frozen evaluation — scoring by this repository's code")
    print("=" * 78)
    print(f"  {bundle.describe()}")
    print(f"  test responses {len(bundle['test_response_y'])} | "
          f"failures {int(bundle['test_response_y'].sum())} | "
          f"images {len(set(map(str, bundle['test_response_images'])))}")

    lm_columns = bundle.select(bundle.lm_names)
    all_columns = bundle.select(bundle.feature_names)

    arms: Dict[str, Dict[str, object]] = {}

    print("\n[1/4] QACD LM-only ...")
    dev_lm, test_lm = fit_and_score(bundle, bundle.lm_names, lm_columns, verbose)
    arms["QACD LM-only"] = {"dev": dev_lm, "test": test_lm}

    print("[2/4] QACD LM + mechanical instruments ...")
    dev_full, test_full = fit_and_score(bundle, bundle.feature_names, all_columns, verbose)
    arms["QACD LM + instrument"] = {"dev": dev_full, "test": test_full}

    print("[3/4] response-level MVR fusion heads ...")
    y_dev = bundle["dev_response_y"]

    def fuse(evidence_dev, evidence_test, label):
        head_dev = np.column_stack([evidence_dev, bundle["dev_mvr"]])
        head_test = np.column_stack([evidence_test, bundle["test_mvr"]])
        head = RidgeLogistic.from_sklearn_C(1.0, len(y_dev)).fit(head_dev, y_dev)
        if not head.success_:
            raise RuntimeError(f"response fusion head failed to converge ({label})")
        print(f"      {label}: evidence score + {len(bundle.mvr_names)} MVR statistics")
        return head.predict_proba(head_dev), head.predict_proba(head_test)

    dev_mvr_lm, test_mvr_lm = fuse(dev_lm, test_lm, "LM + MVR")
    arms["QACD LM + MVR"] = {"dev": dev_mvr_lm, "test": test_mvr_lm}
    dev_mvr_full, test_mvr_full = fuse(dev_full, test_full, "LM + instrument + MVR")
    arms["QACD LM + instrument + MVR"] = {"dev": dev_mvr_full, "test": test_mvr_full}

    y_test = bundle["test_response_y"]
    test_images = bundle["test_response_images"]

    print("[4/4] metrics, image level and paired image bootstrap ...")
    rows = []
    for name, arm in arms.items():
        scores = arm["test"]
        img_scores, img_labels = image_level(scores, y_test, test_images)
        rows.append(
            {
                "method": name,
                "response_auroc": round(auroc(y_test, scores), 6),
                "image_auroc": round(auroc(img_labels, img_scores), 6),
                "brier": round(brier(y_test, np.clip(scores, 1e-6, 1 - 1e-6)), 6),
                "n_responses": len(y_test),
                "n_images": len(img_labels),
                "source": "this repository (evaluation/ + qacd/)",
            }
        )

    baseline_names = bundle.baseline_names
    for i, name in enumerate(baseline_names):
        scores = bundle["baseline_test"][i]
        img_scores, img_labels = image_level(scores, y_test, test_images)
        rows.append(
            {
                "method": name,
                "response_auroc": round(auroc(y_test, scores), 6),
                "image_auroc": round(auroc(img_labels, img_scores), 6),
                "brier": round(brier(y_test, np.clip(scores, 1e-6, 1 - 1e-6)), 6),
                "n_responses": len(y_test),
                "n_images": len(img_labels),
                "source": "frozen replay reference scores",
            }
        )

    primary = "QACD LM + instrument + MVR"
    # Only the three published baselines are contrasted; the other reference
    # columns are the frozen QACD arms, used for the reproduction check instead.
    contrast_with = [n for n in baseline_names if not n.startswith("QACD")]
    parity = []
    for i, name in enumerate(baseline_names):
        if name not in contrast_with:
            continue
        result = paired_image_bootstrap(
            y_test,
            arms[primary]["test"],
            bundle["baseline_test"][i],
            test_images,
            replicates=REPLICATES,
            seed=BOOTSTRAP_SEED,
        )
        parity.append({"baseline": name, **{k: (round(v, 6) if isinstance(v, float) else v)
                                            for k, v in result.items()}})

    # ---- reproduction check -------------------------------------------
    # Response-level AUROC is the endpoint this repository reproduces exactly.
    # The image-level column is also compared, but the frozen table aggregated
    # the white-box UMPIRE arm differently, so that cell is reported as an
    # image-aggregation difference rather than a scoring difference.
    check = []
    for i, name in enumerate(baseline_names):
        if name not in arms:
            continue
        mine = auroc(y_test, arms[name]["test"])
        theirs = auroc(y_test, bundle["baseline_test"][i])
        my_img, my_img_y = image_level(arms[name]["test"], y_test, test_images)
        ref_img, ref_img_y = image_level(bundle["baseline_test"][i], y_test, test_images)
        check.append(
            {
                "arm": name,
                "this_repo_auroc": round(float(mine), 6),
                "frozen_replay_auroc": round(float(theirs), 6),
                "abs_diff": float(abs(mine - theirs)),
                "verdict": "reproduced"
                if abs(mine - theirs) <= EXACT_TOLERANCE
                else "approximate",
                "this_repo_image_auroc": round(float(auroc(my_img_y, my_img)), 6),
                "frozen_replay_image_auroc": round(float(auroc(ref_img_y, ref_img)), 6),
                "image_matches": bool(
                    abs(auroc(my_img_y, my_img) - auroc(ref_img_y, ref_img)) <= EXACT_TOLERANCE
                ),
            }
        )

    # ---- write ---------------------------------------------------------
    import csv

    def write_csv(path: Path, fieldnames: List[str], data: List[dict]) -> str:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        return sha256_file(path)

    outputs = {}
    outputs["frozen_textvqa_test.csv"] = write_csv(
        out_dir / "frozen_textvqa_test.csv", list(rows[0].keys()), rows
    )
    outputs["parity_vs_baselines.csv"] = write_csv(
        out_dir / "parity_vs_baselines.csv", list(parity[0].keys()), parity
    )
    outputs["reproduction_check.csv"] = write_csv(
        out_dir / "reproduction_check.csv", list(check[0].keys()), check
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "produced_by": "scripts/run_frozen_evaluation.py",
        "produced_by_sha256": sha256_file(Path(__file__)),
        "bundle": {
            "path": str(bundle.path),
            "sha256": bundle.manifest.get("bundle_sha256"),
            "created_utc": bundle.manifest.get("created_utc"),
            "protocol": bundle.manifest.get("protocol"),
            "shapes": bundle.manifest.get("shapes"),
            "sources": bundle.manifest.get("sources"),
        },
        "protocol_used_here": {
            "claim_calibrator": "qacd.calibrate.BICLiteCalibrator",
            "claim_l2": CLAIM_L2,
            "aggregation": "qacd.aggregate.claim_to_response_max",
            "response_head": "qacd.calibrate.RidgeLogistic.from_sklearn_C(1.0, n)",
            "bootstrap_replicates": REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "control_unit": "image",
        },
        "outputs": outputs,
        "reproduction": {
            "endpoint": "response-level AUROC on the frozen TextVQA test split",
            "exact_tolerance": EXACT_TOLERANCE,
            "reproduced_arms": [r["arm"] for r in check if r["verdict"] == "reproduced"],
            "approximate_arms": [r["arm"] for r in check if r["verdict"] != "reproduced"],
            "max_abs_diff": max(row["abs_diff"] for row in check),
            "image_aggregation_used_here": "max response risk; image fails if any response fails",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ---- report --------------------------------------------------------
    print()
    print("-" * 78)
    print(f"{'method':<34}{'resp AUROC':>12}{'img AUROC':>12}{'Brier':>10}")
    print("-" * 78)
    for row in rows:
        print(
            f"{row['method']:<34}{row['response_auroc']:>12.4f}"
            f"{row['image_auroc']:>12.4f}{row['brier']:>10.4f}"
        )
    print("-" * 78)
    print(f"paired image bootstrap of {primary} vs baselines:")
    for row in parity:
        print(
            f"  vs {row['baseline']:<32} delta={row['delta']:+.4f} "
            f"CI[{row['ci_low']:+.4f},{row['ci_high']:+.4f}] p={row['p_nonpositive']:.4f}"
        )
    print()
    print("reproduction check against the frozen replay scores:")
    for row in check:
        flag = "OK  " if row["verdict"] == "reproduced" else "APRX"
        iflag = "OK  " if row["image_matches"] else "APRX"
        print(
            f"  [{flag}] {row['arm']:<32} response {row['this_repo_auroc']:.6f} "
            f"vs frozen {row['frozen_replay_auroc']:.6f}  diff {row['abs_diff']:.2e}"
        )
        print(
            f"  [{iflag}] {'':<32} image    {row['this_repo_image_auroc']:.6f} "
            f"vs frozen {row['frozen_replay_image_auroc']:.6f}"
        )
    print()
    print(f"written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
