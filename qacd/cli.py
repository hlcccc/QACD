"""Command line entry point.

    qacd score   --question "..." --answer "..." [--image PATH] [--scorer FILE]
    qacd demo    [--json]
    qacd fit     --data dev.jsonl --out scorer.json [--k 3]
    qacd serve   [--scorer FILE] [--host 0.0.0.0] [--port 8080]
    qacd version
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

from qacd import __version__
from qacd.conformal import VALIDATED
from qacd.mechanical import mechanical_ocr_features, mvr_features
from qacd.pipeline import QACDConfig, QACDPipeline
from qacd.providers import MockProvider

__all__ = ["main", "build_parser"]

DEMO_IMAGE_TEXT = "Dakota Digital; open 24 days"
DEMO_QUESTION = "What brand is the camera?"
DEMO_ANSWER = "Dakota digital"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qacd", description="QACD answer-error risk scorer")
    parser.add_argument("--version", action="version", version=f"qacd {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="score one answer")
    p_score.add_argument("--question", required=True)
    p_score.add_argument("--answer", required=True)
    p_score.add_argument("--image", default="")
    p_score.add_argument("--scorer", default=None, help="JSON scorer exported by `qacd fit`")
    p_score.add_argument("--ocr-text", action="append", default=[], help="repeatable; OCR strings for the image")
    p_score.add_argument("--k", type=int, default=None, help="enable the MVR channel with this depth")
    p_score.add_argument("--json", action="store_true", help="emit raw JSON")

    p_demo = sub.add_parser("demo", help="run the built-in offline example")
    p_demo.add_argument("--json", action="store_true")

    p_fit = sub.add_parser("fit", help="fit a scorer from a JSONL development set")
    p_fit.add_argument("--data", required=True, help="JSONL with question/answer/image/failed")
    p_fit.add_argument("--out", required=True, help="destination JSON scorer")
    p_fit.add_argument("--k", type=int, default=None)
    p_fit.add_argument("--l2", type=float, default=0.05)

    p_serve = sub.add_parser("serve", help="run the HTTP service")
    p_serve.add_argument("--scorer", default=None)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)

    sub.add_parser("version", help="print the version")
    return parser


def _load_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _pipeline_from_args(args) -> QACDPipeline:
    config = QACDConfig(k=getattr(args, "k", None))
    ocr_texts = list(getattr(args, "ocr_text", []) or [])
    provider = MockProvider(image_text="; ".join(ocr_texts), sampled_pool=ocr_texts)
    if getattr(args, "scorer", None):
        return QACDPipeline.load(args.scorer, provider=provider)
    return QACDPipeline(provider=provider, config=config)


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "version":
        print(f"qacd {__version__}")
        return 0

    if args.command == "score":
        pipeline = _pipeline_from_args(args)
        result = pipeline.score(args.question, args.answer, args.image)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(f"risk_score   : {result.risk_score:.4f}  (threshold {result.threshold})")
            print(f"is_high_risk : {result.is_high_risk}   claims: {result.num_claims}")
            for claim in result.claims:
                print(f"  - [{claim.claim_type:>16s}] {claim.risk:.4f}  {claim.claim_text}")
            for warning in result.warnings:
                print(f"  ! {warning}")
        return 0

    if args.command == "demo":
        provider = MockProvider(image_text=DEMO_IMAGE_TEXT, sampled_pool=[DEMO_ANSWER, DEMO_ANSWER, "Nikon"])
        pipeline = QACDPipeline(provider=provider, config=QACDConfig(k=3))
        result = pipeline.score(DEMO_QUESTION, DEMO_ANSWER, "demo.jpg")
        ocr = provider.ocr("demo.jpg")
        payload = {
            "risk": result.to_dict(),
            "mechanical_features": mechanical_ocr_features(DEMO_ANSWER, ocr.texts, ocr.scores),
            "mvr_features": mvr_features([DEMO_ANSWER, DEMO_ANSWER, "Nikon"], ocr.texts, k=3),
            "conformal_validated": VALIDATED,
        }
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print("QACD offline demo")
            print(f"  question : {DEMO_QUESTION}")
            print(f"  answer   : {DEMO_ANSWER}")
            print(f"  ocr      : {ocr.texts}")
            print(f"  risk     : {result.risk_score:.4f}  (high_risk={result.is_high_risk})")
            print(f"  channels : {result.channels}")
            print(f"  warnings : {result.warnings}")
            print(f"  conformal selective prediction validated: {VALIDATED}")
        return 0

    if args.command == "fit":
        records = _load_records(args.data)
        pipeline = QACDPipeline(config=QACDConfig(k=args.k, l2=args.l2))
        pipeline.fit(records, verbose=True)
        path = pipeline.save(args.out)
        print(f"[qacd] scorer written to {path}")
        return 0

    if args.command == "serve":
        try:
            import uvicorn
        except ImportError:
            print('service mode needs uvicorn: pip install "uvicorn>=0.27"', file=sys.stderr)
            return 2
        from qacd.service import create_app

        app = create_app(scorer_path=args.scorer)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
