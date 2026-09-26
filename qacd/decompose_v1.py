"""QACD decomposition, **v1** — the version that produced the frozen results.

The reported TextVQA numbers were computed on claim tables whose
``decomposition_method`` is ``rule_fallback_qacd_v1`` / ``llm_qacd_v1_checked``.
Commit ``f03b27a`` ("Upgrade QACD to validated multi-claim decomposition")
replaced this with v2, which is what :mod:`qacd.decompose` implements.

Why both live here
------------------
So that the repository can state, and verify, which decomposition a given set of
results came from. ``scripts/verify_decomposition_v1.py`` re-derives the fallback
claims of the frozen tables with this module and compares them element-wise.

What differs from v2
--------------------
* :func:`split_fallback_claims` splits on sentences and one level of clause
  boundaries only — no secondary-clause split, no comma split, no 22-word
  chunking, no deduplication.
* :func:`infer_claim_type` classifies from ``question + claim_text`` (the
  *conditioned* claim), tests numbers before OCR text, and reads the combined
  string rather than the claim alone.
* :func:`atomicity_score` has no quoted-question handling, so the
  question-conditioning scaffold counts towards the length penalty.
* ``is_atomic`` uses a ``0.75`` threshold, not ``0.9``.
* The fallback path performs no claim-set validation: there is no coverage
  check, no ``claim_set_*`` bookkeeping and no ``max_claims`` truncation here.
  Truncation happened in the table builder, not in this function.

Ported verbatim from ``hallucination_calibration/methods/qacd.py`` at
``f03b27a^``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from qacd.types import CLAIM_TYPES, EMPTY_RESPONSE, Claim

__all__ = [
    "V1_METHOD_FALLBACK",
    "V1_METHOD_LLM",
    "split_fallback_claims",
    "infer_claim_type",
    "question_conditioned_claim",
    "lexical_faithfulness",
    "atomicity_score",
    "visual_verifiability",
    "requires_external_knowledge",
    "verification_prompt_for",
    "fallback_decompose",
    "sanitize_llm_claims",
]

#: ``decomposition_method`` labels this version emits.
V1_METHOD_FALLBACK = "rule_fallback_qacd_v1"
V1_METHOD_LLM = "llm_qacd_v1_checked"

#: v1 accepted an atomicity of 0.75; v2 raised it to 0.9.
V1_ATOMIC_THRESHOLD = 0.75

_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")
_CLAUSE_RE = re.compile(r"\s*(?:;|；|, and | and |, but | but | because | while | with )\s*", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b", re.IGNORECASE)
_COLOR_RE = re.compile(r"\b(red|blue|green|yellow|black|white|brown|orange|purple|pink|gray|grey)\b", re.IGNORECASE)
_SPATIAL_RE = re.compile(r"\b(left|right|above|below|behind|front|near|next to|beside|under|over|between)\b", re.IGNORECASE)
_ACTION_RE = re.compile(r"\b(holding|standing|sitting|walking|running|playing|riding|wearing|eating|drinking|looking)\b", re.IGNORECASE)
_OCR_RE = re.compile(r"\b(text|word|letter|sign|logo|brand|number|phone|written|says|label)\b", re.IGNORECASE)
_ENTITY_RE = re.compile(r"\b(named|called|brand|company|city|country|mount|mountain|person|team|species)\b", re.IGNORECASE)

_INSTRUCTION_PREFIXES = (
    "Answer this question in only a word or a phrase.",
    "Answer this question in a word or phrase.",
    "Answer the question using a single word or phrase.",
)


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def strip_instruction_prefix(question: str) -> str:
    question = clean_text(question)
    lower = question.lower()
    for prefix in _INSTRUCTION_PREFIXES:
        if lower.startswith(prefix.lower()):
            return clean_text(question[len(prefix):])
    return question


def split_fallback_claims(response: str) -> List[str]:
    """Sentence- then clause-level split, with no further subdivision."""
    response = clean_text(response)
    if not response:
        return [EMPTY_RESPONSE]
    pieces: List[str] = []
    for sentence in _SENTENCE_RE.split(response):
        sentence = clean_text(sentence)
        if not sentence:
            continue
        clauses = [clean_text(p) for p in _CLAUSE_RE.split(sentence) if clean_text(p)]
        pieces.extend(clauses or [sentence])
    return pieces or [response]


def infer_claim_type(question: str, claim: str) -> str:
    """Classify from ``question + claim``; numbers win over OCR keywords."""
    text = f"{question} {claim}"
    if _NUMBER_RE.search(text) or "how many" in question.lower():
        return "counting"
    if _OCR_RE.search(text):
        return "OCR_text"
    if _SPATIAL_RE.search(text):
        return "spatial_relation"
    if _ACTION_RE.search(text):
        return "action"
    if _COLOR_RE.search(text):
        return "attribute"
    if _ENTITY_RE.search(text) or any(w in question.lower() for w in ["who", "where", "what brand", "what is the name"]):
        return "entity_identity"
    if any(w in question.lower() for w in ["is there", "are there", "do you see"]):
        return "object_presence"
    if len(claim.split()) <= 5:
        return "answer_identity"
    return "other"


def question_conditioned_claim(question: str, response: str, span: str) -> str:
    question = clean_text(question)
    response = clean_text(response)
    span = clean_text(span)
    if not span:
        return EMPTY_RESPONSE
    lower_q = question.lower().rstrip("?")
    if len(span.split()) <= 5 and question:
        if lower_q.startswith("what ") or lower_q.startswith("which ") or lower_q.startswith("who ") or lower_q.startswith("where "):
            return f"The answer to the question '{question}' is {span}."
        if lower_q.startswith("how many"):
            return f"The answer count for the question '{question}' is {span}."
        if lower_q.startswith("is ") or lower_q.startswith("are ") or lower_q.startswith("does "):
            return f"The answer to the yes/no question '{question}' is {span}."
    if span.endswith("."):
        return span
    return f"{span}."


def lexical_faithfulness(response: str, source_span: str, claim: str) -> float:
    response = clean_text(response).lower()
    source_span = clean_text(source_span).lower()
    claim = clean_text(claim).lower()
    if not claim or claim == EMPTY_RESPONSE.lower():
        return 0.0
    if source_span and source_span in response:
        return 1.0
    claim_tokens = [t for t in re.findall(r"[a-z0-9]+", claim) if len(t) > 2]
    response_tokens = set(re.findall(r"[a-z0-9]+", response))
    if not claim_tokens:
        return 0.5
    overlap = sum(1 for t in claim_tokens if t in response_tokens) / len(claim_tokens)
    return float(max(0.0, min(1.0, overlap)))


def atomicity_score(claim: str) -> float:
    """v1 scoring: no quoted-question exemption."""
    claim = clean_text(claim)
    if not claim:
        return 0.0
    penalty = 0.0
    lower = claim.lower()
    for marker in [" and ", ";", " but ", " because ", " while "]:
        if marker in lower:
            penalty += 0.18
    if len(claim.split()) > 22:
        penalty += 0.15
    return float(max(0.2, 1.0 - penalty))


def visual_verifiability(question: str, claim_type: str) -> str:
    if claim_type in {"object_presence", "attribute", "counting", "spatial_relation", "action", "scene_global"}:
        return "direct"
    if claim_type == "OCR_text":
        return "ocr"
    if claim_type in {"entity_identity", "external_knowledge"}:
        return "hard"
    if claim_type == "answer_identity":
        q = question.lower()
        if any(word in q for word in ["brand", "name", "where", "who"]):
            return "hard"
        return "indirect"
    return "indirect"


def requires_external_knowledge(question: str, claim_type: str) -> bool:
    q = question.lower()
    return bool(
        claim_type in {"entity_identity", "external_knowledge"}
        or any(w in q for w in ["what country", "what city", "what brand", "who is", "what is the name"])
    )


def verification_prompt_for(claim: str, claim_type: str) -> str:
    if claim_type == "counting":
        return f"Verify the count claim: {claim} Is the stated number correct?"
    if claim_type == "OCR_text":
        return f"Verify the text/OCR claim: {claim} Is the text visible and correctly read?"
    if claim_type == "entity_identity":
        return f"Verify the entity identity claim: {claim} Is this identity supported by the image/question context?"
    return f"Verify this claim from the image and question context: {claim}"


def fallback_decompose(question: str, response: str) -> List[Claim]:
    """v1 rule decomposition.

    Note the argument order of :func:`infer_claim_type`: v1 classified the
    *conditioned* claim text, which is why the question scaffold influences the
    claim type.
    """
    rows: List[Claim] = []
    spans = split_fallback_claims(response)
    for i, span in enumerate(spans):
        claim = question_conditioned_claim(question, response, span)
        claim_type = infer_claim_type(question, claim)
        faith = lexical_faithfulness(response, span, claim)
        atom = atomicity_score(claim)
        rows.append(
            Claim(
                claim_id=f"c{i + 1}",
                claim_text=claim,
                source_span=span,
                claim_type=claim_type,
                is_atomic=atom >= V1_ATOMIC_THRESHOLD,
                is_visual_verifiable=visual_verifiability(question, claim_type),
                requires_external_knowledge=requires_external_knowledge(question, claim_type),
                decomposition_confidence=min(atom, faith),
                atomicity_score=atom,
                faithfulness_score=faith,
                verification_prompt=verification_prompt_for(claim, claim_type),
                parser_added_claim=False,
                decomposition_method=V1_METHOD_FALLBACK,
            )
        )
    return rows


def sanitize_llm_claims(question: str, response: str, raw_claims) -> List[Claim]:
    """v1 checked-LLM path: no claim-set validation, no span recovery."""
    rows: List[Claim] = []
    for i, item in enumerate(raw_claims):
        if not isinstance(item, dict):
            continue
        claim = clean_text(item.get("claim_text", ""))
        span = clean_text(item.get("source_span", ""))
        if not claim:
            continue
        claim_type = clean_text(item.get("claim_type", "other")) or "other"
        if claim_type not in CLAIM_TYPES:
            claim_type = infer_claim_type(question, claim)
        atom = atomicity_score(claim)
        faith = lexical_faithfulness(response, span, claim)
        rows.append(
            Claim(
                claim_id=f"c{i + 1}",
                claim_text=claim,
                source_span=span,
                claim_type=claim_type,
                is_atomic=atom >= V1_ATOMIC_THRESHOLD,
                is_visual_verifiable=visual_verifiability(question, claim_type),
                requires_external_knowledge=requires_external_knowledge(question, claim_type),
                decomposition_confidence=min(atom, faith),
                atomicity_score=atom,
                faithfulness_score=faith,
                verification_prompt=clean_text(item.get("verification_prompt", ""))
                or verification_prompt_for(claim, claim_type),
                parser_added_claim=False,
                decomposition_method=V1_METHOD_LLM,
            )
        )
    return rows
