"""Decomposition behaviour: question conditioning, typing, coverage."""

from qacd.decompose import (
    atomicity_score,
    decompose_claims,
    fallback_decompose,
    infer_claim_type,
    lexical_faithfulness,
    split_fallback_claims,
    strip_instruction_prefix,
    validate_claim_set,
)

INSTRUCTION = "Answer this question in only a word or a phrase."


def test_instruction_prefix_is_stripped():
    assert strip_instruction_prefix(f"{INSTRUCTION} What brand is this?") == "What brand is this?"
    assert strip_instruction_prefix("What brand is this?") == "What brand is this?"


def test_short_answer_becomes_self_contained_proposition():
    claims = fallback_decompose("What brand is the camera?", "Dakota digital")
    assert len(claims) == 1
    text = claims[0].claim_text
    # The bare noun phrase must gain the question as context.
    assert "Dakota digital" in text
    assert "camera" in text
    assert text.endswith(".")


def test_short_answer_with_instruction_prefix_is_still_conditioned():
    claims = fallback_decompose(f"{INSTRUCTION} What brand is the camera?", "Dakota digital")
    text = claims[0].claim_text
    assert "camera" in text, "instruction prefix must not defeat question conditioning"
    assert not text.startswith(INSTRUCTION)


def test_compound_answer_splits_into_multiple_claims():
    answer = "The sign says OPEN 24 DAYS and there are three bicycles outside."
    claims = fallback_decompose("What does the sign say, and how many bicycles are outside?", answer)
    assert len(claims) >= 2
    types = {c.claim_type for c in claims}
    assert "counting" in types or "OCR_text" in types


def test_claim_types_follow_the_claim_not_the_question():
    assert infer_claim_type("How many cats?", "three") == "counting"
    assert infer_claim_type("What brand is it?", "this is the brand Dakota") == "OCR_text"
    assert infer_claim_type("What color is the car?", "red") == "attribute"


def test_coverage_validation_flags_incomplete_sets():
    answer = "A red bicycle leaning against a wooden fence"
    claims = fallback_decompose("What is there?", answer)
    verdict = validate_claim_set(answer, claims)
    assert verdict["response_coverage_score"] >= 0.8
    assert verdict["valid"] or "non_atomic_claim" in verdict["reasons"]


def test_empty_answer_is_handled():
    claims = fallback_decompose("What is this?", "")
    assert len(claims) == 1
    assert claims[0].claim_text


def test_max_claims_cap_is_enforced():
    answer = ". ".join(f"sentence number {i}" for i in range(20))
    claims = fallback_decompose("Describe.", answer, max_claims=3)
    assert len(claims) <= 3


def test_atomicity_and_faithfulness_are_bounded():
    assert 0.0 <= atomicity_score("The sign reads OPEN.") <= 1.0
    assert 0.0 <= lexical_faithfulness("a red car", "a red car", "The car is red.") <= 1.0
    # A source span copied verbatim from the answer is maximally faithful.
    assert lexical_faithfulness("a red car", "a red car", "The car is red.") == 1.0
    # A claim introducing content the answer never contained must score low.
    # The source span is deliberately *not* a substring of the answer, which is
    # the case the overlap branch is responsible for.
    assert lexical_faithfulness("a red car", "engine block", "The engine is a V8.") < 0.5


def test_llm_path_is_used_when_it_validates():
    completion = (
        '{"claims":[{"claim_id":"c1","claim_text":"The sign reads OPEN 24 DAYS.",'
        '"source_span":"The sign says OPEN 24 DAYS","claim_type":"OCR_text",'
        '"verification_prompt":"Does the sign read OPEN 24 DAYS?"}]}'
    )
    result = decompose_claims(
        "What does the sign say?", "The sign says OPEN 24 DAYS", llm_completion=completion
    )
    assert result.method == "llm_checked"
    assert result.claims[0].claim_type == "OCR_text"


def test_llm_failure_falls_back_without_raising():
    for bad in ["not json at all", "", '{"claims": []}', "```json\n{oops\n```"]:
        result = decompose_claims("What color?", "red", llm_completion=bad)
        assert result.method == "rule_fallback"
        assert len(result.claims) >= 1


def test_span_splitting_is_deduplicated():
    spans = split_fallback_claims("a cat. a cat. a dog")
    assert len(spans) == len({s.lower().rstrip(".!?") for s in spans})
