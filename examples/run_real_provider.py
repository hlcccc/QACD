#!/usr/bin/env python
"""Run QACD against a real LLaVA checkpoint — the complete working path.

This is the walkthrough the README quick start does *not* cover on its own,
because it needs hardware the repository cannot ship:

    a frozen LLaVA-1.5-13B checkpoint   (~26 GB in fp16, ≈40 GB card recommended)
    a labelled development set          (>=1,200 responses advised; see docs/03)
    the image files the development set refers to

Everything else is in the repository. The mock provider that ``qacd demo`` uses
needs none of this, but it reads no model and no image — a scorer fitted on it
carries no model evidence and its scores mean nothing. This script exists so that
the distance between "it runs" and "it works" is one file rather than a reading
of ``qacd/providers.py``.

Usage
-----

    # 1. fit a scorer on your own development set
    python examples/run_real_provider.py fit \
        --data dev.jsonl \
        --model-path /models/llava-1.5-13b-hf \
        --out scorer.json \
        --k 3

    # 2. score a single answer with it
    python examples/run_real_provider.py score \
        --model-path /models/llava-1.5-13b-hf \
        --scorer scorer.json \
        --question "what is the website that host this photo?" \
        --answer "Flickr" \
        --image /data/textvqa/val/eb38600d8a5ade9a.jpg

``dev.jsonl`` holds one JSON object per line::

    {"question": "...", "answer": "...", "image": "/path/to.jpg",
     "failed": 0, "group": "/path/to.jpg"}

``failed`` is 1 when the frozen model's answer was wrong, 0 when it was right.
``group`` is the image path; the calibration split is taken by image, so responses
about the same image never straddle the split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_provider(model_path: str, with_ocr: bool, temperature: float):
    """LLaVA evidence provider, optionally with the RapidOCR instrument channel.

    The OCR channel is what lets the mechanical features check a claim against
    text actually visible in the image. Without it the pipeline still runs, but
    roughly a fifth of the frozen feature set goes unused.
    """
    from qacd.providers import LLaVAProvider

    ocr_provider = None
    if with_ocr:
        try:
            from qacd.providers import RapidOCRProvider

            ocr_provider = RapidOCRProvider()
            print("[example] OCR channel: RapidOCR")
        except ImportError:
            print(
                "[example] rapidocr-onnxruntime is not installed; continuing without the "
                "mechanical OCR channel.\n"
                "          pip install rapidocr-onnxruntime",
                file=sys.stderr,
            )
    else:
        print("[example] OCR channel: disabled (--no-ocr)")

    return LLaVAProvider(
        model_path=model_path,
        ocr_provider=ocr_provider,
        temperature=temperature,
    )


def cmd_fit(args) -> int:
    from qacd.pipeline import QACDConfig, QACDPipeline

    records = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise SystemExit(f"{args.data} holds no records")

    missing = [k for k in ("question", "answer", "image", "failed") if k not in records[0]]
    if missing:
        raise SystemExit(
            f"{args.data} records are missing {missing}. Each line needs "
            '{"question", "answer", "image", "failed", "group"}.'
        )
    groups = {str(r.get("group", r["image"])) for r in records}
    print(
        f"[example] {len(records)} responses over {len(groups)} images\n"
        f"[example] loading {args.model_path} (first call downloads nothing; the "
        "checkpoint must already be on disk)"
    )

    provider = build_provider(args.model_path, not args.no_ocr, args.temperature)
    pipeline = QACDPipeline(provider=provider, config=QACDConfig(k=args.k, l2=args.l2))
    pipeline.fit(records, verbose=True)

    for warning in pipeline.fit_warnings:
        print(f"[example] !! {warning}", file=sys.stderr)
    if not pipeline.fitted_:
        raise SystemExit("the calibrator did not converge; try a larger development set")

    path = pipeline.save(args.out)
    print(f"\n[example] scorer written to {path}")
    print(f"[example] provenance: {json.dumps(pipeline.provenance, ensure_ascii=False)}")
    if not pipeline.provenance.get("carries_model_evidence", False):
        print(
            "[example] !! this scorer carries no model evidence; do not use it to score",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_score(args) -> int:
    from qacd.pipeline import QACDPipeline

    provider = build_provider(args.model_path, not args.no_ocr, args.temperature)
    pipeline = QACDPipeline.load(args.scorer, provider=provider)
    for warning in pipeline.fit_warnings:
        print(f"[example] !! {warning}", file=sys.stderr)

    result = pipeline.score(question=args.question, answer=args.answer, image=args.image)
    print()
    print(f"risk_score            : {result.risk_score:.4f}")
    print(f"calibrated_confidence : {result.calibrated_confidence:.4f}")
    print(f"is_high_risk          : {result.is_high_risk}  (threshold {result.threshold})")
    print(f"model_calls           : {result.model_calls}")
    print(f"latency_ms            : {result.latency_ms}")
    for claim in result.claims:
        print(f"  - [{claim.claim_type:>16s}] {claim.risk:.4f}  {claim.claim_text}")
    if result.channels:
        print(f"channels              : {result.channels}")
    for warning in result.warnings:
        print(f"  ! {warning}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-path", required=True, help="frozen LLaVA checkpoint directory")
    common.add_argument("--no-ocr", action="store_true", help="skip the RapidOCR channel")
    common.add_argument("--temperature", type=float, default=0.7, help="sampling temperature")

    p_fit = sub.add_parser("fit", parents=[common], help="fit a scorer on a development set")
    p_fit.add_argument("--data", required=True, help="JSONL development set")
    p_fit.add_argument("--out", required=True, help="destination scorer JSON")
    p_fit.add_argument("--k", type=int, default=3, help="MVR depth (3 is the reported knee)")
    p_fit.add_argument("--l2", type=float, default=0.05)

    p_score = sub.add_parser("score", parents=[common], help="score one answer")
    p_score.add_argument("--scorer", required=True, help="scorer JSON from `fit`")
    p_score.add_argument("--question", required=True)
    p_score.add_argument("--answer", required=True)
    p_score.add_argument("--image", required=True)

    args = parser.parse_args()
    return {"fit": cmd_fit, "score": cmd_score}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
