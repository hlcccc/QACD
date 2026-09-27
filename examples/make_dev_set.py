#!/usr/bin/env python
"""Build the labelled development set `qacd fit` needs, from your own VQA data.

Why this exists
---------------

QACD scores answers produced by a *frozen* model, and its calibrator is fitted on
your data. So the one thing every integrator has to do before anything works is
produce a development set with the fields `qacd fit` reads:

    {"question": ..., "answer": ..., "image": ..., "failed": 0|1, "group": ...}

The README documented the field list but never said where the labels come from.
This script does that step: it takes the answers your model produced together with
the accepted answers for each question, decides which ones are wrong, and writes
the JSONL.

Input
-----

JSONL, one object per line. Either of these shapes:

    {"question": "...", "answer": "...", "image": "/path/a.jpg",
     "gold": "Dakota Digital"}

    {"question": "...", "answer": "...", "image": "/path/a.jpg",
     "answers": ["Dakota Digital", "Dakota Digital", "Nikon", ...]}

`answers` is the VQA convention: the ten answers human annotators gave. When it is
present the standard VQA rule is used -- an answer scores `min(matches / 3, 1)` --
and `--accuracy-threshold` decides the binary label. With a single `gold` string
the rule is an exact match after normalisation.

**The label definition is a platform decision, not a property of QACD.** If your
product cares about, say, brand-name errors more than count errors, replace
`decide_label` with your own rule. What matters is that it is written down: the
calibrator can only be as good as the labels it is fitted on.

On `group`
----------

`group` is the image. QACD's calibration split is taken **by image**, so several
questions about the same photograph must never straddle the train/calibration
boundary -- otherwise the calibration set contains near-duplicates of the training
set and the reported confidence is optimistic. Passing the image path as `group`
is what makes that hold; this script does it for you.

Usage
-----

    python examples/make_dev_set.py --data raw.jsonl --out dev.jsonl
    python examples/make_dev_set.py --data raw.jsonl --out dev.jsonl \\
        --accuracy-threshold 1.0

Then:

    qacd fit --data dev.jsonl --out scorer.json --k 3 \\
             --provider llava --model-path /models/llava-1.5-13b-hf
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: How many VQA annotators must agree before an answer counts as fully correct.
#: The official VQA metric divides by three, so three agreeing humans score 1.0.
VQA_AGREEMENT_DIVISOR = 3


def decide_label(model_answer: str, record: dict, threshold: float) -> tuple[int, float]:
    """Return ``(failed, score)`` for one record.

    Replace this function if your definition of "wrong" differs -- see the module
    docstring. It is deliberately the only place the labelling rule lives.
    """
    from qacd.mechanical import normalize_text

    predicted = normalize_text(model_answer)

    answers = record.get("answers")
    if answers is None:
        gold = record.get("gold")
        if gold is None:
            raise ValueError("record has neither 'answers' nor 'gold'")
        gold_list = [gold] if isinstance(gold, str) else list(gold)
        score = 1.0 if any(normalize_text(g) == predicted for g in gold_list) else 0.0
    else:
        gold_list = [str(a) for a in answers]
        matches = sum(1 for g in gold_list if normalize_text(g) == predicted)
        score = min(matches / VQA_AGREEMENT_DIVISOR, 1.0)

    return (0 if score >= threshold else 1), score


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, help="input JSONL, see the docstring")
    parser.add_argument("--out", required=True, help="output dev.jsonl")
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=0.5,
        help="an answer scoring below this is labelled failed (default 0.5; pass 1.0 "
             "to require every matching annotator to agree)",
    )
    parser.add_argument(
        "--group-by",
        default="image",
        choices=["image", "none"],
        help="what to put in 'group'. The calibration split is taken by image, so "
             "'image' is almost always right; 'none' groups everything together "
             "and is only useful for a quick look.",
    )
    parser.add_argument("--seed", type=int, default=20260920, help="for the summary sample")
    args = parser.parse_args()

    source = Path(args.data).expanduser()
    if not source.is_file():
        raise SystemExit(f"input not found: {args.data}")

    records = []
    skipped = 0
    for lineno, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{args.data}:{lineno} is not valid JSON: {exc.msg}") from exc

        for field in ("question", "answer", "image"):
            if field not in raw:
                raise SystemExit(f"{args.data}:{lineno} has no {field!r} field")

        try:
            failed, _ = decide_label(str(raw["answer"]), raw, args.accuracy_threshold)
        except ValueError as exc:
            raise SystemExit(f"{args.data}:{lineno}: {exc}") from exc

        records.append(
            {
                "question": str(raw["question"]),
                "answer": str(raw["answer"]),
                "image": str(raw["image"]),
                "failed": failed,
                "group": str(raw["image"]) if args.group_by == "image" else "all",
            }
        )

    if not records:
        raise SystemExit(f"{args.data} produced no records")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    failures = sum(r["failed"] for r in records)
    groups = Counter(r["group"] for r in records)
    rate = failures / len(records)

    print(f"[make_dev_set] {len(records)} responses over {len(groups)} images -> {out}")
    print(f"[make_dev_set] failed: {failures} ({rate:.1%})")
    if skipped:
        print(f"[make_dev_set] skipped {skipped} malformed lines")
    print()

    if len(records) < 300:
        print(
            f"[make_dev_set] !! {len(records)} responses is below the 300 floor. The "
            "calibrator will fit, but docs/03 section 4 shows it has not saturated.",
            file=sys.stderr,
        )
    elif len(records) < 1200:
        print(
            f"[make_dev_set] note: {len(records)} responses. Most of the gain is in by "
            "about 1,200; see docs/03 section 4.",
            file=sys.stderr,
        )
    if failures == 0:
        print(
            "[make_dev_set] !! every answer was labelled correct. A calibrator fitted "
            "on one class learns nothing. Check the labelling rule and the gold answers.",
            file=sys.stderr,
        )
    elif failures == len(records):
        print(
            "[make_dev_set] !! every answer was labelled wrong; same problem inverted.",
            file=sys.stderr,
        )
    if args.group_by == "image" and len(groups) < len(records) / 2:
        print(
            f"[make_dev_set] note: {len(records)} responses share only {len(groups)} "
            "images, so the effective sample size is smaller than it looks.",
            file=sys.stderr,
        )

    print("[make_dev_set] a few rows, to check the labelling looks sane:")
    sample = random.Random(args.seed).sample(records, min(3, len(records)))
    for record in sample:
        mark = "WRONG" if record["failed"] else "ok   "
        print(f"    [{mark}] Q: {record['question'][:52]}")
        print(f"            model: {record['answer'][:52]}")
    print()
    print("[make_dev_set] next:")
    print(f"    qacd fit --data {out} --out scorer.json --k 3 \\")
    print("             --provider llava --model-path /models/llava-1.5-13b-hf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
