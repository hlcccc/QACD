#!/usr/bin/env python
"""Close the loop: QACDPipeline scoring the frozen test split end to end.

Chain exercised here, all with this repository's code:

    raw evidence tables
      -> frozen.assemble.FrozenFeatureAssembler   (112-column vector per claim)
      -> QACDPipeline.fit_matrix                  (calibrator fitted on score-training)
      -> QACDPipeline.score_evidence_frame        (claim risk)
      -> qacd.aggregate.claim_to_response_max     (response risk)
      -> evaluation.metrics                       (AUROC / image level)

Checks performed:

1. the assembler's matrix equals the frozen matrix element-wise;
2. assembling row by row equals assembling the whole table;
3. the pipeline's response AUROC equals the batch path's, and matches the
   frozen replay record.

    python scripts/verify_pipeline_frozen.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import auroc, image_level  # noqa: E402
from frozen.assemble import FrozenFeatureAssembler  # noqa: E402
from frozen.build import build_matrices, claim_images, load_split  # noqa: E402
from qacd.aggregate import claim_to_response_max  # noqa: E402
from qacd.pipeline import QACDPipeline  # noqa: E402

TOLERANCE = 1e-12


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="artifacts/evidence_bundle.npz")
    parser.add_argument("--raw", default="artifacts/raw")
    parser.add_argument(
        "--row-check-n",
        type=int,
        default=200,
        help="how many claims to re-assemble one at a time (the property is "
             "structural; a sample keeps the check fast)",
    )
    args = parser.parse_args()

    from evaluation.data import require_data

    raw = Path(args.raw)
    absent = require_data(
        [args.bundle, raw / "dev_features.csv.gz", raw / "test_features.csv.gz"],
        "verify the pipeline over the frozen feature set",
    )
    if absent is not None:
        return absent

    bundle = np.load(args.bundle, allow_pickle=False)

    print("=" * 90)
    print("QACDPipeline over the frozen feature set")
    print("=" * 90)

    print("\n[1/5] building matrices from raw evidence ...")
    dev = load_split(raw, "dev")
    test = load_split(raw, "test")
    matrices = build_matrices(dev, test, claim_images(dev, raw / "dev_claims.csv.gz"))
    print(f"      {matrices.n_features} features, dev {matrices.dev_matrix.shape}, "
          f"test {matrices.test_matrix.shape}")

    print("\n[2/5] assembling through frozen.assemble.FrozenFeatureAssembler ...")
    assembler = FrozenFeatureAssembler.from_matrices(matrices)
    assert assembler.n_features == len(matrices.names)
    mine_dev = assembler.transform_frame(dev)
    mine_test = assembler.transform_frame(test)
    checks = {}
    for label, mine, ref in (
        ("dev", mine_dev, bundle["dev_matrix"]),
        ("test", mine_test, bundle["test_matrix"]),
    ):
        diff = float(np.max(np.abs(mine - ref))) if mine.shape == ref.shape else float("inf")
        checks[f"assembler_matrix_{label}"] = diff
        print(f"      [{'OK  ' if diff <= TOLERANCE else 'FAIL'}] {label} "
              f"assembler vs frozen max|diff| = {diff:.3e}")

    print(f"\n[3/5] assembling claim by claim (first {args.row_check_n} dev rows) ...")
    n = min(args.row_check_n, len(dev))
    row_dev = assembler.transform_rows(dev.head(n).to_dict("records"))
    row_diff = float(np.max(np.abs(row_dev - mine_dev[:n])))
    checks["rowwise_vs_table_dev"] = row_diff
    print(f"      [{'OK  ' if row_diff <= TOLERANCE else 'FAIL'}] dev row-by-row vs table "
          f"max|diff| = {row_diff:.3e}  (n={n})")

    print("\n[4/5] fitting QACDPipeline on the score-training split ...")
    pipeline = QACDPipeline(assembler=assembler)
    pipeline.fit_matrix(
        matrices.dev_matrix, bundle["dev_claim_labels"], matrices.train_mask
    )
    print(f"      fitted={pipeline.fitted_}  feature_dim={len(pipeline.feature_names)}")
    if not pipeline.fitted_:
        print("      !! calibrator did not converge")
        return 1

    print("\n[5/5] scoring the frozen test split through the pipeline ...")
    claim_scores = pipeline.score_evidence_frame(test)
    sampled = test.head(n)
    claim_scores_row = pipeline.score_evidence_frame_by_row(sampled)
    score_diff = float(np.max(np.abs(claim_scores[:n] - claim_scores_row)))
    checks["pipeline_scores_table_vs_rowwise"] = score_diff
    print(f"      [{'OK  ' if score_diff <= TOLERANCE else 'FAIL'}] claim scores, table vs "
          f"row-by-row max|diff| = {score_diff:.3e}  (n={n})")

    y_test = bundle["test_response_y"]
    images = np.asarray(bundle["test_response_images"]).astype(str)
    idx = bundle["test_claim_response_index"]
    response = claim_to_response_max(claim_scores, len(y_test), idx)
    img_scores, img_labels = image_level(response, y_test, images)

    pipeline_auroc = auroc(y_test, response)
    image_auroc = auroc(img_labels, img_scores)
    print(f"      response AUROC = {pipeline_auroc:.6f}   image AUROC = {image_auroc:.6f}")

    # Reference: the frozen replay record for the same arm.
    index = {str(n): i for i, n in enumerate(bundle["baseline_names"])}
    reference = None
    for name in ("QACD LM + instrument",):
        if name in index:
            reference = float(auroc(y_test, bundle["baseline_test"][index[name]]))
    if reference is not None:
        delta = abs(pipeline_auroc - reference)
        checks["auroc_vs_frozen_replay"] = delta
        print(f"      [{'OK  ' if delta <= 1e-5 else 'APRX'}] vs frozen replay "
              f"{reference:.6f}  diff {delta:.2e}")

    ok = all(v <= TOLERANCE for k, v in checks.items() if "auroc" not in k)
    ok = ok and checks.get("auroc_vs_frozen_replay", 0.0) <= 1e-5

    print()
    print("=" * 90)
    print("VERDICT:", "pipeline closes the loop on the frozen feature set" if ok else "FAILED")
    print("=" * 90)
    print(json.dumps({"ok": bool(ok), "checks": checks,
                      "response_auroc": float(pipeline_auroc),
                      "image_auroc": float(image_auroc)}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
