#!/usr/bin/env python
"""Measure the calibration gain the 指标 2.2 indicator asks for.

The task book measures "校准性能提升" with ECE. This script computes it on the
frozen TextVQA test split, before and after QACD's calibration stage, so the
indicator has a number behind it rather than an assertion.

What "before" means here
------------------------

QACD's job is to turn a raw uncertainty signal into a calibrated risk. The raw
signal is the frozen model's own multi-view confidence: it is what you have before
the risk-uncertainty mapping is fitted. "Before" is therefore that confidence
expressed as a risk and aggregated to the response with the same operator QACD
uses, so the only thing that differs between the two sides is the calibration.

Both sides are scored on the same responses, with the same binning, because ECE is
binning-dependent and comparing across different settings would be meaningless.

    python scripts/measure_calibration_gain.py
    python scripts/measure_calibration_gain.py --n-bins 20 --strategy quantile
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qacd.aggregate import claim_to_response_max  # noqa: E402

#: The headline configuration, by its name in the frozen bundle.
TARGET = "QACD LM + instrument + MVR"

#: The raw uncertainty signal: the frozen model's own multi-view confidence.
RAW_CONFIDENCE_COLUMN = "bcm_multiview_confidence_mean"


def response_level_max(claim_values: np.ndarray, index: np.ndarray, n: int) -> np.ndarray:
    return claim_to_response_max(claim_values, n, index, default=0.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bundle", default="artifacts/evidence_bundle.npz")
    parser.add_argument("--raw", default="artifacts/raw")
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument(
        "--strategy", default="uniform", choices=["uniform", "quantile"]
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    from evaluation.data import require_data
    from evaluation.metrics import brier, ece

    raw_dir = Path(args.raw)
    absent = require_data(
        [args.bundle, raw_dir / "test_features.csv.gz"],
        "measure the calibration gain",
    )
    if absent is not None:
        return absent

    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a documented dependency
        raise SystemExit("pandas is required to read the raw evidence tables")

    bundle = np.load(args.bundle, allow_pickle=False)

    names = [str(n) for n in bundle["baseline_names"]]
    if TARGET not in names:
        raise SystemExit(f"{TARGET!r} not in the bundle; found {names}")

    y = np.asarray(bundle["test_response_y"], dtype=int)
    calibrated = np.asarray(bundle["baseline_test"][names.index(TARGET)], dtype=float)

    frame = pd.read_csv(raw_dir / "test_features.csv.gz")
    if RAW_CONFIDENCE_COLUMN not in frame.columns:
        raise SystemExit(
            f"{RAW_CONFIDENCE_COLUMN!r} missing from {raw_dir / 'test_features.csv.gz'}; "
            "cannot reconstruct the pre-calibration signal"
        )

    n_responses = len(y)
    index = np.asarray(bundle["test_claim_response_index"], dtype=int)
    if len(frame) != len(index):
        raise SystemExit(
            f"the raw table has {len(frame)} claim rows but the bundle indexes "
            f"{len(index)}; they are not the same run"
        )

    # Before: the model's own confidence, as a risk, aggregated the same way.
    confidence = frame[RAW_CONFIDENCE_COLUMN].to_numpy(dtype=float)
    raw_risk_claims = np.clip(1.0 - confidence, 0.0, 1.0)
    uncalibrated = response_level_max(raw_risk_claims, index, n_responses)

    # The demanding baseline: take the same raw signal but give it its own
    # calibrator, fitted on the development split. This isolates what QACD's
    # features add over simply calibrating the one signal it starts from. Beating
    # an uncalibrated score is easy and would prove little.
    from qacd.calibrate import BICLiteCalibrator

    dev_features = raw_dir / "dev_features.csv.gz"
    dev_frame = pd.read_csv(dev_features)
    y_dev = np.asarray(bundle["dev_response_y"], dtype=int)
    dev_index = np.asarray(bundle["dev_claim_response_index"], dtype=int)
    dev_confidence = dev_frame[RAW_CONFIDENCE_COLUMN].to_numpy(dtype=float)
    dev_raw = response_level_max(np.clip(1.0 - dev_confidence, 0.0, 1.0), dev_index, len(y_dev))
    raw_calibrator = BICLiteCalibrator(l2=0.05).fit(
        dev_raw.reshape(-1, 1), y_dev.reshape(-1)
    )
    raw_only_calibrated = raw_calibrator.predict_proba(uncalibrated.reshape(-1, 1)).reshape(-1)

    print("=" * 78)
    print("指标 2.2 —— 校准性能（ECE）")
    print("=" * 78)
    print(f"  测试集           : {n_responses} 条回答 / {len(set(bundle['test_response_images']))} 张图像")
    print(f"  失败率           : {y.mean():.3%}")
    print(f"  方法             : {TARGET}")
    print(f"  分箱             : n_bins={args.n_bins}, strategy={args.strategy}")
    print()

    results = {}
    for label, scores in (
        ("校准前：模型自报置信度的补（未校准）", uncalibrated),
        ("对照：只把该信号校准一遍（最强基线）", raw_only_calibrated),
        ("QACD：完整特征 + 校准", calibrated),
    ):
        value = ece(y, scores, n_bins=args.n_bins, strategy=args.strategy)
        results[label] = value
        print(f"  {label}")
        print(f"      ECE   = {value:.6f}")
        print(f"      Brier = {brier(y, scores):.6f}")
        print()

    before = results["校准前：模型自报置信度的补（未校准）"]
    after = results["QACD：完整特征 + 校准"]
    strongest = results["对照：只把该信号校准一遍（最强基线）"]
    gain = (before - after) / before if before > 0 else float("nan")
    gain_vs_strongest = (strongest - after) / strongest if strongest > 0 else float("nan")

    print("=" * 78)
    print(f"  vs 未校准   : ({before:.6f} - {after:.6f}) / {before:.6f} = {gain:+.2%}")
    print(f"  vs 最强基线 : ({strongest:.6f} - {after:.6f}) / {strongest:.6f} = {gain_vs_strongest:+.2%}")
    print("=" * 78)
    for bar, name in ((0.10, "中期 ≥10%"), (0.20, "验收 ≥20%")):
        mark = "达标" if gain >= bar else "未达"
        print(f"  {name}: {mark}   (相对未校准口径)")
    print()
    if gain_vs_strongest > 0:
        print("  即使把原始信号单独校准一遍再比，QACD 仍然更低，说明增益来自特征而不只是")
        print("  「做了一次校准」。")
    else:
        print("  注意：相对最强基线的改进不为正 —— 这说明 QACD 的校准增益主要来自「做了")
        print("  一次校准」，而不是特征本身。引用指标时应当用未校准口径并说明这一点。")
    print()

    print("  分箱敏感性（引用数值时必须连同设置一起写）:")
    sensitivity = {}
    for n_bins in (5, 10, 15, 20):
        row = {}
        for strategy in ("uniform", "quantile"):
            b = ece(y, uncalibrated, n_bins=n_bins, strategy=strategy)
            a = ece(y, calibrated, n_bins=n_bins, strategy=strategy)
            row[strategy] = (b - a) / b if b > 0 else float("nan")
        sensitivity[f"n_bins={n_bins}"] = row
        print(
            f"      n_bins={n_bins:3}  uniform {row['uniform']:+.2%}   "
            f"quantile {row['quantile']:+.2%}"
        )
    print()

    if args.json:
        print(
            json.dumps(
                {
                    "method": TARGET,
                    "n_responses": int(n_responses),
                    "n_bins": args.n_bins,
                    "strategy": args.strategy,
                    "ece_before": before,
                    "ece_after": after,
                    "brier_before": brier(y, uncalibrated),
                    "brier_after": brier(y, calibrated),
                    "relative_gain": gain,
                    "sensitivity": sensitivity,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
