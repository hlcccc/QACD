"""QACD decomposition v1 — the version behind the frozen results.

v1 and v2 differ in ways that are easy to conflate, so each documented
difference gets its own test. The artifact-backed test re-derives every fallback
claim of the frozen tables and requires an exact match.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from qacd import decompose as v2
from qacd import decompose_v1 as v1
from qacd.decompose_v1 import (
    V1_ATOMIC_THRESHOLD,
    V1_METHOD_FALLBACK,
    V1_METHOD_LLM,
)

RAW = Path(__file__).resolve().parents[1] / "artifacts" / "raw"

QUESTION = "Answer this question in only a word or a phrase. What brand is the camera?"
ANSWER = "Dakota Digital"


# -- version labels ------------------------------------------------------

def test_version_labels_are_distinct():
    assert V1_METHOD_FALLBACK == "rule_fallback_qacd_v1"
    assert V1_METHOD_LLM == "llm_qacd_v1_checked"
    assert v2.__dict__.get("DEFAULT_MAX_CLAIMS") is not None
    a = v1.fallback_decompose(QUESTION, ANSWER)[0]
    b = v2.fallback_decompose(QUESTION, ANSWER)[0]
    assert a.decomposition_method != b.decomposition_method


# -- split behaviour -----------------------------------------------------

def test_v1_splits_only_on_sentences_and_clauses():
    # v2 subdivides further (secondary clauses, commas, 22-word chunks) and
    # deduplicates; v1 does not.
    answer = "The sign says OPEN 24 DAYS; there are three bicycles outside"
    assert len(v1.split_fallback_claims(answer)) == 2
    assert len(v2.split_fallback_claims(answer)) >= 2


def test_v1_does_not_deduplicate_sentences():
    answer = "a cat. a cat"
    v1_spans = v1.split_fallback_claims(answer)
    v2_spans = v2.split_fallback_claims(answer)
    assert len(v1_spans) == 2
    assert len(v2_spans) == 1


def test_v1_long_spans_are_not_chunked():
    answer = " ".join(f"word{i}" for i in range(60))
    assert len(v1.split_fallback_claims(answer)) == 1


# -- claim typing --------------------------------------------------------

def test_v1_types_from_the_conditioned_claim():
    """v1 classified `question + claim_text`, v2 classifies the source span."""
    claims = v1.fallback_decompose("How many cats are there?", "three")
    assert claims[0].claim_type == "counting"
    assert "The answer count for the question" in claims[0].claim_text


def test_v1_number_rule_beats_ocr_rule():
    # "number" is an OCR keyword, but v1 tests numbers first.
    assert v1.infer_claim_type("What is it?", "number 3") == "counting"
    assert v2.infer_claim_type("What is it?", "number 3") == "OCR_text"


# -- atomicity -----------------------------------------------------------

def test_v1_atomicity_has_no_quoted_question_exemption():
    """v2 strips the question scaffold before the length penalty; v1 does not.

    The scaffold only exceeds 22 words for a long question, so the difference
    shows up there and nowhere else.
    """
    long_question = (
        "What is the name of the tall grey building standing next to the river "
        "bridge in the old town district"
    )
    conditioned = v1.question_conditioned_claim(long_question, "Empire", "Empire")
    assert len(conditioned.split()) > 22
    assert v1.atomicity_score(conditioned) == pytest.approx(0.85)
    assert v2.atomicity_score(conditioned) == pytest.approx(1.0)


def test_v1_uses_a_lower_atomicity_threshold():
    assert V1_ATOMIC_THRESHOLD == 0.75
    # A claim scoring between the two thresholds is atomic under v1, not v2.
    claim = " ".join(["word"] * 25)
    score = v2.atomicity_score(claim)
    assert 0.75 <= score < 0.9
    assert v2.atomicity_score(claim) < 0.9


# -- claim-set validation ------------------------------------------------

def test_v1_fallback_performs_no_claim_set_validation():
    claims = v1.fallback_decompose(QUESTION, ANSWER)
    for claim in claims:
        assert claim.claim_set_size == 0
        assert claim.claim_set_validated is False
        assert claim.claim_set_validation_errors == ""


def test_v2_fallback_attaches_claim_set_validation():
    claims = v2.fallback_decompose(QUESTION, ANSWER)
    assert all(c.claim_set_size == len(claims) for c in claims)


def test_v1_fallback_ignores_a_max_claims_hint():
    """v1 had no cap in the decomposer; truncation happened in the table builder."""
    answer = ". ".join(f"sentence {i}" for i in range(6))
    assert len(v1.fallback_decompose("Describe.", answer)) == 6


# -- LLM path ------------------------------------------------------------

def test_v1_llm_sanitiser_labels_and_validates_differently():
    raw_claims = [{"claim_text": "The sign reads OPEN.", "source_span": "OPEN",
                   "claim_type": "OCR_text"}]
    a = v1.sanitize_llm_claims(QUESTION, "The sign says OPEN", raw_claims)[0]
    b = v2.sanitize_llm_claims(QUESTION, "The sign says OPEN", raw_claims)[0]

    assert a.decomposition_method == V1_METHOD_LLM
    assert b.decomposition_method == "llm_qacd_v2_multiclaim_checked"

    # v1 leaves the claim-set bookkeeping untouched; v2 runs the validator and
    # records its verdict, including the coverage failure this span causes.
    assert a.claim_set_size == 0
    assert a.claim_set_validated is False
    assert a.claim_set_validation_errors == ""

    assert b.claim_set_size == 1
    assert b.response_coverage_score > 0.0
    assert "incomplete_response_coverage" in b.claim_set_validation_errors


# -- artifact-backed -----------------------------------------------------

requires_artifacts = pytest.mark.skipif(
    not (RAW / "test_claims.csv.gz").exists(), reason="raw evidence tables not present"
)

V1_STRING_FIELDS = ("claim_text", "claim_type", "source_span", "is_visual_verifiable",
                    "verification_prompt")


@requires_artifacts
@pytest.mark.parametrize("split", ["dev", "test"])
def test_v1_reproduces_the_frozen_fallback_claims(split):
    frame = pd.read_csv(RAW / f"{split}_claims.csv.gz", keep_default_na=False)
    compared = 0
    for _, group in frame.groupby("sample_id", sort=False):
        frozen = group[group["decomposition_method"].astype(str) == V1_METHOD_FALLBACK]
        if frozen.empty:
            continue
        mine = v1.fallback_decompose(str(group.iloc[0]["question"]), str(group.iloc[0]["response_text"]))
        rows = frozen.sort_values("claim_id").to_dict("records")
        assert len(mine) == len(rows), f"claim count differs for sample {group.iloc[0]['sample_id']}"
        for got, want in zip(mine, rows):
            for field in V1_STRING_FIELDS:
                assert str(got.__dict__[field]) == str(want[field]), (
                    f"{split} sample {group.iloc[0]['sample_id']} field {field}"
                )
            assert abs(got.atomicity_score - float(want["atomicity_score"])) < 1e-12
            assert abs(got.faithfulness_score - float(want["faithfulness_score"])) < 1e-12
        compared += len(rows)
    assert compared > 0
    assert compared == int((frame["decomposition_method"].astype(str) == V1_METHOD_FALLBACK).sum())


@requires_artifacts
def test_frozen_tables_contain_no_v2_rows():
    """The reported results sit entirely on v1 labels."""
    for split in ("dev", "test"):
        frame = pd.read_csv(RAW / f"{split}_claims.csv.gz", keep_default_na=False)
        methods = set(frame["decomposition_method"].astype(str))
        assert methods == {V1_METHOD_FALLBACK, V1_METHOD_LLM}, methods
