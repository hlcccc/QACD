#!/usr/bin/env python
"""Offline end-to-end demo — no models, no GPU, no network.

Run from the repository root::

    python scripts/demo_offline.py

It fits a scorer on a tiny synthetic development set, then scores a correct and
a wrong answer and prints the four platform-facing payload shapes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qacd.conformal import VALIDATED, conformal_select  # noqa: E402
from qacd.mechanical import mechanical_ocr_features, mvr_features  # noqa: E402
from qacd.pipeline import QACDConfig, QACDPipeline  # noqa: E402
from qacd.providers import MockProvider  # noqa: E402

IMAGE_TEXT = "Dakota Digital; open 24 days; 3 bicycles"
QUESTION = "What brand is the camera?"


def dev_records(n: int = 120):
    records = []
    for i in range(n):
        correct = i % 2 == 0
        records.append(
            {
                "question": QUESTION,
                "answer": "Dakota Digital" if correct else "Nikon Coolpix",
                "image": IMAGE_TEXT,
                "failed": 0 if correct else 1,
                "group": f"img{i:04d}",
            }
        )
    return records


def provider():
    return MockProvider(
        image_text=IMAGE_TEXT,
        sampled_pool=["Dakota Digital", "Dakota Digital", "Nikon Coolpix"],
    )


def main() -> int:
    print("=" * 78)
    print("QACD offline demo — synthetic development set, no model weights")
    print("=" * 78)

    pipeline = QACDPipeline(provider=provider(), config=QACDConfig(k=3))
    pipeline.fit(dev_records(), verbose=True)

    for label, answer in [("correct answer", "Dakota Digital"), ("wrong answer", "Nikon Coolpix")]:
        result = pipeline.score(QUESTION, answer, "demo.jpg")
        print(f"\n--- {label}: {answer!r} ---")
        print(f"  risk_score   : {result.risk_score:.4f}   high_risk={result.is_high_risk}")
        print(f"  num_claims   : {result.num_claims}   feature_dim={result.feature_dim}")
        print(f"  channels     : {json.dumps(result.channels, ensure_ascii=False)}")
        for claim in result.claims:
            print(f"    [{claim.claim_type:>16s}] {claim.risk:.4f}  {claim.claim_text}")

    # 技术点 3 —— 机械 OCR 通道
    ocr = provider().ocr("demo.jpg")
    print("\n--- 技术点3 mechanical OCR channel ---")
    print(json.dumps(mechanical_ocr_features("Dakota Digital", ocr.texts, ocr.scores), indent=2))

    # 技术点 2 —— MVR 通道
    print("\n--- 技术点2 MVR channel (K=3) ---")
    print(json.dumps(mvr_features(["Dakota Digital", "Dakota Digital", "Nikon Coolpix"], ocr.texts, k=3), indent=2))

    # 技术点 4 —— 保形选择性预测
    print("\n--- 技术点4 conformal selective prediction ---")
    selection = conformal_select(
        calibration_null_scores=[0.05, 0.08, 0.11, 0.15, 0.20, 0.24, 0.30],
        test_scores=[0.02, 0.06, 0.9],
        alpha=0.10,
        procedure="BY",
        test_labels=[0, 0, 1],
    )
    print(json.dumps(selection.to_dict(), indent=2, ensure_ascii=False))
    print(f"\nconformal validated flag: {VALIDATED}")
    print("\nNOTE: this demo uses a synthetic dev set. It exercises the plumbing,")
    print("      it does not reproduce the reported research numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
