"""Command line entry point.

    qacd score   --question "..." --answer "..." --provider llava --model-path DIR
    qacd demo    [--json]
    qacd fit     --data dev.jsonl --out scorer.json --provider llava --model-path DIR
    qacd serve   --scorer scorer.json [--provider llava --model-path DIR]
    qacd version

``--provider`` is **required** for ``score`` and ``fit``. The mock provider is
still available, but it has to be asked for by name: it derives every reading
from lexical overlap rather than from a model, so a scorer fitted on it is a
plumbing artefact whose score has no relationship to answer correctness. Making
that the silent default is how a first-time user ends up with a confident,
backwards score and no indication that anything is wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from qacd import __version__
from qacd.conformal import VALIDATED
from qacd.console import make_console_safe
from qacd.mechanical import mechanical_ocr_features, mvr_features
from qacd.pipeline import QACDConfig, QACDPipeline
from qacd.providers import MockProvider

__all__ = ["main", "build_parser"]

DEMO_IMAGE_TEXT = "Dakota Digital; open 24 days"
DEMO_QUESTION = "What brand is the camera?"
DEMO_ANSWER = "Dakota digital"

PROVIDER_CHOICES = ("mock", "llava")

_MOCK_BANNER = (
    "!! --provider mock: this provider reads no model and no image. It exists to\n"
    "!! exercise the plumbing. Any score it produces says nothing about whether an\n"
    "!! answer is correct. Pass --provider llava --model-path DIR for real evidence."
)


def _add_provider_args(parser: argparse.ArgumentParser) -> None:
    """Attach the evidence-provider options shared by score / fit / serve."""
    parser.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        required=True,
        help="where evidence comes from. 'llava' is the real one; 'mock' is a "
             "model-free stand-in for plumbing tests only",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="frozen LLaVA checkpoint directory; required by --provider llava",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="do not attach the RapidOCR instrument channel (LLaVA provider only)",
    )


def build_provider(args) -> Any:
    """Construct the evidence provider named on the command line."""
    kind = getattr(args, "provider", None)
    ocr_texts = list(getattr(args, "ocr_text", []) or [])

    if kind == "mock":
        print(_MOCK_BANNER, file=sys.stderr)
        return MockProvider(image_text="; ".join(ocr_texts), sampled_pool=ocr_texts)

    if kind == "llava":
        model_path = getattr(args, "model_path", None)
        if not model_path:
            raise SystemExit(
                "--provider llava needs --model-path DIR pointing at a frozen "
                "LLaVA-1.5-13B checkpoint (HuggingFace format).\n"
                "  The checkpoint is ~26 GB in fp16 and is not distributed with this "
                "repository."
            )
        from qacd.providers import LLaVAProvider

        ocr_provider = None
        if not getattr(args, "no_ocr", False):
            try:
                from qacd.providers import RapidOCRProvider

                ocr_provider = RapidOCRProvider()
            except ImportError as exc:  # pragma: no cover - depends on the install
                raise SystemExit(
                    "the mechanical OCR channel needs rapidocr-onnxruntime:\n"
                    '  pip install rapidocr-onnxruntime\n'
                    "Pass --no-ocr to run without it (accuracy will be lower)."
                ) from exc
        return LLaVAProvider(model_path=model_path, ocr_provider=ocr_provider)

    raise SystemExit(f"unknown provider: {kind!r}")


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
    _add_provider_args(p_score)

    p_demo = sub.add_parser("demo", help="run the built-in offline example")
    p_demo.add_argument("--json", action="store_true")

    p_fit = sub.add_parser("fit", help="fit a scorer from a JSONL development set")
    p_fit.add_argument("--data", required=True, help="JSONL with question/answer/image/failed")
    p_fit.add_argument("--out", required=True, help="destination JSON scorer")
    p_fit.add_argument("--k", type=int, default=None)
    p_fit.add_argument("--l2", type=float, default=0.05)
    _add_provider_args(p_fit)

    p_serve = sub.add_parser("serve", help="run the HTTP service")
    p_serve.add_argument("--scorer", default=None)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        default=None,
        help="evidence provider for live scoring; defaults to the one recorded in "
             "the scorer file",
    )
    p_serve.add_argument("--model-path", default=None)
    p_serve.add_argument("--no-ocr", action="store_true")

    sub.add_parser("version", help="print the version")
    return parser


def _require_file(path: str, what: str, hint: str = "") -> Path:
    """Resolve a path the user named, or exit with something readable.

    Without this a typo produced a raw ``FileNotFoundError`` traceback, which
    tells a first-time user nothing about which argument was wrong.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        message = f"{what} not found: {path}"
        if hint:
            message += f"\n  {hint}"
        raise SystemExit(message)
    return resolved


def _load_records(path: str) -> List[Dict[str, Any]]:
    source = _require_file(
        path,
        "development set",
        'Expected JSONL, one object per line:\n'
        '  {"question": "...", "answer": "...", "image": "...", '
        '"failed": 0|1, "group": "..."}',
    )
    records: List[Dict[str, Any]] = []
    # utf-8-sig, not utf-8: on Windows both PowerShell's `Out-File -Encoding utf8`
    # and Excel's CSV export prepend a BOM, and the resulting JSONL is otherwise
    # rejected on its first line with a message about UTF-8 that reads like a
    # file-corruption problem rather than a byte-order mark.
    with source.open(encoding="utf-8-sig") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno} is not valid JSON: {exc.msg}") from exc
    if not records:
        raise SystemExit(f"{path} holds no records")
    return records


def _pipeline_from_args(args) -> QACDPipeline:
    config = QACDConfig(k=getattr(args, "k", None))
    provider = build_provider(args)
    if getattr(args, "scorer", None):
        scorer = _require_file(
            args.scorer,
            "scorer file",
            "`--scorer` takes the JSON written by `qacd fit --out`.",
        )
        return QACDPipeline.load(scorer, provider=provider)
    return QACDPipeline(provider=provider, config=config)


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # `qacd demo` prints Chinese labels, and a cp1252 console raises on them
    # instead of printing. One call here covers every subcommand.
    make_console_safe()

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
        pipeline = QACDPipeline(
            provider=build_provider(args), config=QACDConfig(k=args.k, l2=args.l2)
        )
        pipeline.fit(records, verbose=True)
        path = pipeline.save(args.out)
        for warning in pipeline.fit_warnings:
            print(f"[qacd] !! {warning}", file=sys.stderr)
        print(f"[qacd] scorer written to {path}")
        print(f"[qacd] fitted on provider: {pipeline.provenance.get('provider_kind')}")
        return 0

    if args.command == "serve":
        try:
            import uvicorn
        except ImportError:
            print('service mode needs uvicorn: pip install "uvicorn>=0.27"', file=sys.stderr)
            return 2
        from qacd.service import create_app

        pipeline = None
        if args.scorer:
            scorer = _require_file(
                args.scorer,
                "scorer file",
                "`--scorer` takes the JSON written by `qacd fit --out`.",
            )
            provider = build_provider(args) if args.provider else None
            pipeline = QACDPipeline.load(scorer, provider=provider)
            for warning in pipeline.fit_warnings:
                print(f"[qacd] !! {warning}", file=sys.stderr)
        elif args.provider:
            pipeline = QACDPipeline(provider=build_provider(args))
        elif args.provider is None and args.scorer is None:
            args.provider = "mock"
            pipeline = QACDPipeline(provider=build_provider(args))

        app = create_app(pipeline=pipeline, scorer_path=None if pipeline else args.scorer)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
