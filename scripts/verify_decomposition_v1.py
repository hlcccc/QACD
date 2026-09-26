#!/usr/bin/env python
"""Check that :mod:`qacd.decompose_v1` reproduces the frozen claim tables.

The frozen TextVQA claim tables were built by the v1 decomposer. Every row whose
``decomposition_method`` is ``rule_fallback_qacd_v1`` is deterministic given
``(question, response_text)``, so it can be re-derived here and compared field
by field. This is what lets the repository state which decomposition a set of
results came from instead of guessing.

    python scripts/verify_decomposition_v1.py

Exit status is 0 only when every fallback claim matches.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qacd.decompose_v1 import (  # noqa: E402
    V1_METHOD_FALLBACK,
    V1_METHOD_LLM,
    fallback_decompose,
)

#: Fields compared per claim. Strings must match exactly; floats to this tol.
STRING_FIELDS = ("claim_text", "claim_type", "source_span", "is_visual_verifiable",
                 "verification_prompt")
BOOL_FIELDS = ("is_atomic", "requires_external_knowledge", "parser_added_claim")
FLOAT_FIELDS = ("atomicity_score", "faithfulness_score", "decomposition_confidence")
FLOAT_TOL = 1e-12


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def check_split(frame: pd.DataFrame, label: str) -> dict:
    mixed = 0
    compared = 0
    skipped_llm = 0
    length_mismatch = 0
    field_hits = Counter()
    field_total = Counter()
    examples = []

    for sample_id, group in frame.groupby("sample_id", sort=False):
        methods = set(group["decomposition_method"].astype(str))
        if methods == {V1_METHOD_LLM}:
            skipped_llm += 1
            continue
        if len(methods) > 1:
            mixed += 1

        frozen = group[group["decomposition_method"].astype(str) == V1_METHOD_FALLBACK]
        if frozen.empty:
            continue

        question = str(group.iloc[0]["question"])
        response = str(group.iloc[0]["response_text"])
        mine = fallback_decompose(question, response)

        compared += 1
        if len(mine) != len(frozen):
            length_mismatch += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "sample_id": int(sample_id),
                        "kind": "length",
                        "question": question[:70],
                        "response": response[:70],
                        "mine": [c.claim_text for c in mine][:4],
                        "frozen": [str(t)[:60] for t in frozen["claim_text"].tolist()][:4],
                    }
                )
            continue

        rows = frozen.sort_values("claim_id").to_dict("records")
        for got, want in zip(mine, rows):
            for field in STRING_FIELDS:
                field_total[field] += 1
                if str(got.__dict__.get(field, "")) == str(want.get(field, "")):
                    field_hits[field] += 1
                elif len(examples) < 8:
                    examples.append(
                        {
                            "sample_id": int(sample_id),
                            "kind": field,
                            "question": question[:60],
                            "mine": str(got.__dict__.get(field))[:90],
                            "frozen": str(want.get(field))[:90],
                        }
                    )
            for field in BOOL_FIELDS:
                field_total[field] += 1
                if bool(got.__dict__.get(field)) == _as_bool(want.get(field)):
                    field_hits[field] += 1
            for field in FLOAT_FIELDS:
                field_total[field] += 1
                try:
                    same = abs(float(got.__dict__.get(field)) - float(want.get(field))) <= FLOAT_TOL
                except (TypeError, ValueError):
                    same = False
                if same:
                    field_hits[field] += 1

    return {
        "label": label,
        "responses_total": int(frame["sample_id"].nunique()),
        "responses_on_llm_path": skipped_llm,
        "responses_with_mixed_paths": mixed,
        "responses_compared": compared,
        "responses_with_length_mismatch": length_mismatch,
        "field_match_rate": {
            f: (field_hits[f] / field_total[f] if field_total[f] else None) for f in field_total
        },
        "field_compared": dict(field_total),
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="artifacts/raw")
    parser.add_argument("--splits", nargs="*", default=["dev", "test"])
    args = parser.parse_args()

    from evaluation.data import DATA_ABSENT_EXIT, require_data

    raw = Path(args.raw)
    absent = require_data(
        [raw / f"{split}_claims.csv.gz" for split in args.splits],
        "verify qacd.decompose_v1 against the frozen claim tables",
    )
    if absent is not None:
        return absent

    print("=" * 88)
    print("Verifying qacd.decompose_v1 against the frozen claim tables")
    print("=" * 88)

    reports = []
    ok = True
    for split in args.splits:
        path = raw / f"{split}_claims.csv.gz"
        if not path.exists():
            print(f"  skip {split}: {path} not found")
            continue
        frame = pd.read_csv(path, keep_default_na=False)
        report = check_split(frame, split)
        reports.append(report)

        print()
        print(f"--- {split} ---")
        print(f"  responses                : {report['responses_total']}")
        print(f"    on the LLM path (skip) : {report['responses_on_llm_path']}")
        print(f"    mixed paths            : {report['responses_with_mixed_paths']}")
        print(f"    compared (fallback)    : {report['responses_compared']}")
        print(f"    claim-count mismatch   : {report['responses_with_length_mismatch']}")

        rates = report["field_match_rate"]
        for field, rate in rates.items():
            if rate is None:
                continue
            flag = "OK  " if rate == 1.0 else "DIFF"
            if rate != 1.0:
                ok = False
            print(
                f"    [{flag}] {field:<28} {rate * 100:6.2f}%  "
                f"({int(rate * report['field_compared'][field])}/{report['field_compared'][field]})"
            )

        if report["responses_with_length_mismatch"]:
            ok = False
            print("    first mismatches:")
            for ex in report["examples"][:3]:
                print(f"      q={ex.get('question')!r}")
                print(f"        mine  ={ex.get('mine')}")
                print(f"        frozen={ex.get('frozen')}")
        else:
            for ex in report["examples"][:3]:
                print(f"    diff in {ex['kind']}: mine={ex.get('mine')!r} frozen={ex.get('frozen')!r}")

    print()
    print("=" * 88)
    print("VERDICT:", "v1 reproduces the frozen fallback claims" if ok else "V1 DIVERGES")
    print("=" * 88)
    print(json.dumps({"ok": bool(ok), "reports": [
        {k: v for k, v in r.items() if k != "examples"} for r in reports
    ]}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
