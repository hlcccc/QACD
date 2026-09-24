"""Question-conditioned atomic claim decomposition.

This is the front end of QACD. A frozen LVLM answer to a VQA question is
rewritten into one or more *question-conditioned atomic claims*: self-contained
propositions that can be verified against the image without access to the
original question.

Two paths are supported and both are part of the deployed system:

1. **Checked LLM decomposition** — the prompt in :func:`build_decomposition_prompt`
   asks for a constrained JSON object; :func:`sanitize_llm_claims` validates it
   (atomicity, lexical faithfulness, exact source spans, response coverage).
2. **Deterministic rule fallback** — :func:`fallback_decompose` splits the answer
   on sentence/clause boundaries and re-attaches the question context.

The research pipeline reports both paths in its results; the fallback is not a
degraded mode that can be ignored, it carried roughly two thirds of the claim
rows in the reported TextVQA experiments.

Ported from the research implementation
``hallucination_calibration/methods/qacd.py`` (behaviour preserved; the module
has been made import-clean — no pandas dependency at import time).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from qacd.types import (
    CLAIM_TYPES,
    DEFAULT_MAX_CLAIMS,
    EMPTY_RESPONSE,
    MIN_RESPONSE_COVERAGE,
    Claim,
    DecompositionResult,
)

__all__ = [
    "CLAIM_TYPES",
    "clean_text",
    "normalize_max_claims",
    "strip_instruction_prefix",
    "split_fallback_claims",
    "infer_claim_type",
    "question_conditioned_claim",
    "lexical_faithfulness",
    "atomicity_score",
    "visual_verifiability",
    "requires_external_knowledge",
    "verification_prompt_for",
    "fallback_decompose",
    "build_decomposition_prompt",
    "parse_json_claims",
    "sanitize_llm_claims",
    "validate_claim_set",
    "response_coverage",
    "decompose_claims",
]

_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")
_CLAUSE_RE = re.compile(r"\s*(?:;|；|, and | and |, but | but | because | while | with )\s*", re.IGNORECASE)
_SECONDARY_CLAUSE_RE = re.compile(
    r"\s*(?::|：|—|–|\bas well as\b|\balong with\b|\bin addition to\b)\s*", re.IGNORECASE
)
_NUMBER_RE = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b", re.IGNORECASE)
_COLOR_RE = re.compile(r"\b(red|blue|green|yellow|black|white|brown|orange|purple|pink|gray|grey)\b", re.IGNORECASE)
_SPATIAL_RE = re.compile(r"\b(left|right|above|below|behind|front|near|next to|beside|under|over|between)\b", re.IGNORECASE)
_ACTION_RE = re.compile(r"\b(holding|standing|sitting|walking|running|playing|riding|wearing|eating|drinking|looking)\b", re.IGNORECASE)
_OCR_RE = re.compile(r"\b(text|word|letter|sign|logo|brand|number|phone|written|says|label)\b", re.IGNORECASE)
_ENTITY_RE = re.compile(r"\b(named|called|brand|company|city|country|mount|mountain|person|team|species)\b", re.IGNORECASE)
_CONTENT_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CONTENT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "but", "by", "for",
    "from", "has", "have", "in", "is", "it", "of", "on", "or", "that", "the",
    "there", "these", "this", "those", "to", "was", "were", "while", "with",
}

# Instruction prefixes injected by TextVQA-style benchmark harnesses. They carry
# no factual content and must be stripped before a short answer is turned into a
# self-contained proposition.
_INSTRUCTION_PREFIXES = (
    "Answer this question in only a word or a phrase.",
    "Answer this question in a word or phrase.",
    "Answer the question using a single word or phrase.",
)


def clean_text(value: Any) -> str:
    """Collapse all whitespace runs and coerce to ``str``."""
    return " ".join(str(value or "").strip().split())


def normalize_max_claims(value: Any) -> int:
    """Validate the implementation cap on claims per response."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"max_claims must be a positive integer, got {value!r}") from exc
    if not numeric.is_integer() or numeric < 1:
        raise ValueError(f"max_claims must be a positive integer, got {value!r}")
    return int(numeric)


def strip_instruction_prefix(question: str) -> str:
    """Remove a benchmark instruction prefix from the question."""
    question = clean_text(question)
    lower = question.lower()
    for prefix in _INSTRUCTION_PREFIXES:
        if lower.startswith(prefix.lower()):
            return clean_text(question[len(prefix):])
    return question


# --------------------------------------------------------------------------
# Rule-based decomposition
# --------------------------------------------------------------------------

def split_fallback_claims(response: str) -> List[str]:
    """Split an answer into maximal clause-level spans.

    Long spans are chunked at comma boundaries and then into 22-word blocks so
    that the atomicity validator does not reject open-form answers outright.
    """
    response = clean_text(response)
    if not response:
        return [EMPTY_RESPONSE]
    pieces: List[str] = []
    for sentence in _SENTENCE_RE.split(response):
        sentence = clean_text(sentence)
        if not sentence:
            continue
        clauses = [clean_text(p) for p in _CLAUSE_RE.split(sentence) if clean_text(p)]
        for clause in clauses or [sentence]:
            secondary = [clean_text(p) for p in _SECONDARY_CLAUSE_RE.split(clause) if clean_text(p)]
            for span in secondary or [clause]:
                comma_parts = [clean_text(p) for p in re.split(r"\s*,\s*", span) if clean_text(p)]
                for part in comma_parts or [span]:
                    words = part.split()
                    if len(words) <= 22:
                        pieces.append(part)
                    else:
                        pieces.extend(" ".join(words[i:i + 22]) for i in range(0, len(words), 22))
    deduplicated: List[str] = []
    seen = set()
    for piece in pieces:
        key = clean_text(piece).lower().rstrip(".!?")
        if key and key not in seen:
            seen.add(key)
            deduplicated.append(piece)
    return deduplicated or [response]


def infer_claim_type(question: str, claim: str) -> str:
    """Assign a verification type to a claim.

    The individual claim is inspected before the question so that compound
    questions do not make every child claim inherit the same type.
    """
    question = clean_text(question)
    claim = clean_text(claim)
    text = f"{question} {claim}"
    if _OCR_RE.search(claim):
        return "OCR_text"
    if _NUMBER_RE.search(claim):
        return "counting"
    if _SPATIAL_RE.search(claim):
        return "spatial_relation"
    if _ACTION_RE.search(claim):
        return "action"
    if _COLOR_RE.search(claim):
        return "attribute"
    if _ENTITY_RE.search(claim):
        return "entity_identity"
    if "how many" in question.lower():
        return "counting"
    if _OCR_RE.search(text):
        return "OCR_text"
    if _SPATIAL_RE.search(text):
        return "spatial_relation"
    if _ACTION_RE.search(text):
        return "action"
    if _COLOR_RE.search(text):
        return "attribute"
    if any(w in question.lower() for w in ["who", "where", "what brand", "what is the name"]):
        return "entity_identity"
    if any(w in question.lower() for w in ["is there", "are there", "do you see"]):
        return "object_presence"
    if len(claim.split()) <= 5:
        return "answer_identity"
    return "other"


def question_conditioned_claim(question: str, response: str, span: str) -> str:
    """Turn a bare short answer into a self-contained proposition."""
    question = clean_text(question)
    response = clean_text(response)
    span = clean_text(span)
    if not span:
        return EMPTY_RESPONSE
    lower_q = question.lower().rstrip("?")
    if len(span.split()) <= 5 and question:
        if lower_q.startswith(("what ", "which ", "who ", "where ")):
            return f"The answer to the question '{question}' is {span}."
        if lower_q.startswith("how many"):
            return f"The answer count for the question '{question}' is {span}."
        if lower_q.startswith(("is ", "are ", "does ")):
            return f"The answer to the yes/no question '{question}' is {span}."
    if span.endswith("."):
        return span
    return f"{span}."


def lexical_faithfulness(response: str, source_span: str, claim: str) -> float:
    """Fraction of claim content words that occur in the answer."""
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
    """Heuristic single-fact score in ``[0.2, 1.0]``.

    Conjunctions and over-long unquoted spans are penalised. The quoted question
    scaffold inserted by :func:`question_conditioned_claim` is removed first: it
    is context, not an additional assertion.
    """
    claim = clean_text(claim)
    if not claim:
        return 0.0
    penalty = 0.0
    lower = claim.lower()
    unquoted = re.sub(r"(['\"]).*?\1", "", lower)
    question_quote = lower.find("question '")
    answer_separator = lower.rfind("' is ")
    if question_quote >= 0 and answer_separator > question_quote:
        quoted_start = question_quote + len("question ")
        unquoted = lower[:quoted_start] + lower[answer_separator + 1:]
    for marker in [" and ", ";", " but ", " because ", " while "]:
        if marker in unquoted:
            penalty += 0.18
    if len(unquoted.split()) > 22:
        penalty += 0.15
    return float(max(0.2, 1.0 - penalty))


def visual_verifiability(question: str, claim_type: str) -> str:
    """Route a claim type to the evidence channel that can check it."""
    if claim_type in {"object_presence", "attribute", "counting", "spatial_relation", "action", "scene_global"}:
        return "direct"
    if claim_type == "OCR_text":
        return "ocr"
    if claim_type in {"entity_identity", "external_knowledge"}:
        return "hard"
    if claim_type == "answer_identity":
        q = question.lower()
        return "hard" if any(w in q for w in ["brand", "name", "where", "who"]) else "indirect"
    return "indirect"


def requires_external_knowledge(question: str, claim_type: str) -> bool:
    """Flag claims that cannot be settled from the image alone."""
    q = question.lower()
    return bool(
        claim_type in {"entity_identity", "external_knowledge"}
        or any(w in q for w in ["what country", "what city", "what brand", "who is", "what is the name"])
    )


def verification_prompt_for(claim: str, claim_type: str) -> str:
    """Type-routed yes/no verification prompt."""
    if claim_type == "counting":
        return f"Verify the count claim: {claim} Is the stated number correct?"
    if claim_type == "OCR_text":
        return f"Verify the text/OCR claim: {claim} Is the text visible and correctly read?"
    if claim_type == "entity_identity":
        return f"Verify the entity identity claim: {claim} Is this identity supported by the image/question context?"
    return f"Verify this claim from the image and question context: {claim}"


def _claim_from_fallback_span(question: str, span: str, span_count: int) -> str:
    if span_count == 1:
        return question_conditioned_claim(question, span, span)
    claim = clean_text(span)
    if claim and claim[0].islower():
        claim = claim[0].upper() + claim[1:]
    return claim if claim.endswith((".", "!", "?")) else f"{claim}."


def fallback_decompose(question: str, response: str, max_claims: int = DEFAULT_MAX_CLAIMS) -> List[Claim]:
    """Deterministic rule decomposition — the deployed safety net."""
    max_claims = normalize_max_claims(max_claims)
    spans = split_fallback_claims(response)
    truncated = len(spans) > max_claims
    spans = spans[:max_claims]
    conditioned_question = strip_instruction_prefix(question)
    rows: List[Claim] = []
    for i, span in enumerate(spans):
        claim_text = _claim_from_fallback_span(conditioned_question, span, len(spans))
        claim_type = infer_claim_type(question, span)
        faith = lexical_faithfulness(response, span, claim_text)
        atom = atomicity_score(claim_text)
        rows.append(
            Claim(
                claim_id=f"c{i + 1}",
                claim_text=claim_text,
                source_span=span,
                claim_type=claim_type,
                is_atomic=atom >= 0.9,
                is_visual_verifiable=visual_verifiability(question, claim_type),
                requires_external_knowledge=requires_external_knowledge(question, claim_type),
                verification_prompt=verification_prompt_for(claim_text, claim_type),
                decomposition_confidence=min(atom, faith),
                atomicity_score=atom,
                faithfulness_score=faith,
                fallback_truncated=truncated,
                decomposition_method="rule_fallback_qacd_v2_multiclaim",
            )
        )
    return attach_claim_set_validation(rows, response, max_claims)


# --------------------------------------------------------------------------
# LLM decomposition
# --------------------------------------------------------------------------

def build_decomposition_prompt(question: str, response: str, max_claims: int = DEFAULT_MAX_CLAIMS) -> str:
    """Constrained JSON decomposition prompt (answer-preserving, no correction)."""
    question = strip_instruction_prefix(question)
    max_claims = normalize_max_claims(max_claims)
    type_list = ", ".join(CLAIM_TYPES)
    return (
        "Decompose the VQA answer into a complete list of faithful atomic claims.\n"
        f"Return between 1 and {max_claims} claims. Use the fewest claims that cover every factual assertion in the Answer.\n"
        "Each claim must contain exactly one independently verifiable fact. Split facts joined by and, but, while, or separate clauses.\n"
        "Use the Question to make short answers understandable, but do not add facts. Preserve mistakes in the Answer; do not correct them from the image.\n"
        "For every claim, source_span must be an exact contiguous span copied from the Answer. "
        "The source spans and claim texts must collectively cover all factual content in the Answer.\n"
        "Use sequential claim_id values c1, c2, ... and do not duplicate claims.\n"
        f"Allowed TYPE values: {type_list}.\n"
        "Return valid JSON only, without Markdown or explanation, using this schema:\n"
        '{"claims":[{"claim_id":"c1","claim_text":"...","source_span":"exact Answer span",'
        '"claim_type":"one allowed TYPE","verification_prompt":"short yes/no question"}]}\n\n'
        "Compound example:\n"
        "Question: What does the sign say, and how many bicycles are outside?\n"
        "Answer: The sign says OPEN 24 DAYS and there are three bicycles outside.\n"
        'Output: {"claims":[{"claim_id":"c1","claim_text":"The sign reads OPEN 24 DAYS.",'
        '"source_span":"The sign says OPEN 24 DAYS","claim_type":"OCR_text",'
        '"verification_prompt":"Does the sign read OPEN 24 DAYS?"},'
        '{"claim_id":"c2","claim_text":"There are three bicycles outside.",'
        '"source_span":"there are three bicycles outside.","claim_type":"counting",'
        '"verification_prompt":"Are there three bicycles outside?"}]}\n\n'
        "Now decompose this input:\n"
        f"Question: {question}\n"
        f"Answer: {response}"
    )


def parse_json_claims(text: str) -> List[Dict[str, Any]]:
    """Tolerantly pull the claims list out of an LLM completion."""
    text = str(text or "").strip()
    if not text:
        return []
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    obj: Any = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for start, char in enumerate(text):
            if char not in "[{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[start:])
                break
            except json.JSONDecodeError:
                continue
        if obj is None:
            return []
    if isinstance(obj, dict):
        claims = obj.get("claims", [])
    elif isinstance(obj, list):
        claims = obj
    else:
        return []
    return claims if isinstance(claims, list) else []


def _content_tokens(value: str) -> set:
    return {
        token
        for token in _CONTENT_TOKEN_RE.findall(clean_text(value).lower())
        if token not in _CONTENT_STOPWORDS
    }


def _best_source_span(response: str, claim: str) -> str:
    """Recover a source span when the model hallucinated one."""
    response = clean_text(response)
    candidates = split_fallback_claims(response)
    if response not in candidates:
        candidates.append(response)
    claim_tokens = _content_tokens(claim)
    best_span, best_key = "", (-1.0, 0)
    for candidate in candidates:
        candidate_tokens = _content_tokens(candidate)
        if not candidate_tokens:
            continue
        overlap = len(claim_tokens & candidate_tokens)
        recall = overlap / max(len(claim_tokens), 1)
        precision = overlap / len(candidate_tokens)
        key = (recall + precision, -len(candidate_tokens))
        if key > best_key:
            best_key, best_span = key, candidate
    return best_span


def response_coverage(response: str, claims: Sequence[Claim]) -> Tuple[float, List[str]]:
    """Fraction of answer content words covered by the claim set."""
    response_tokens = _content_tokens(response)
    if not response_tokens:
        return 1.0, []
    claim_tokens: set = set()
    for claim in claims:
        claim_tokens.update(_content_tokens(claim.claim_text))
    uncovered = sorted(response_tokens - claim_tokens)
    score = 1.0 - len(uncovered) / len(response_tokens)
    return float(max(0.0, min(1.0, score))), uncovered


def validate_claim_set(response: str, claims: Sequence[Claim], max_claims: int = DEFAULT_MAX_CLAIMS) -> Dict[str, Any]:
    """Reject decompositions that are incomplete, compound or duplicated."""
    max_claims = normalize_max_claims(max_claims)
    reasons: List[str] = []
    if not claims:
        reasons.append("no_claims")
    if len(claims) > max_claims:
        reasons.append("too_many_claims")
    normalized = [clean_text(c.claim_text).lower() for c in claims]
    if len(normalized) != len(set(normalized)):
        reasons.append("duplicate_claims")
    if any(not bool(c.is_atomic) for c in claims):
        reasons.append("non_atomic_claim")
    if any(bool(c.fallback_truncated) for c in claims):
        reasons.append("fallback_truncated")
    coverage, uncovered = response_coverage(response, claims)
    if coverage < MIN_RESPONSE_COVERAGE:
        reasons.append("incomplete_response_coverage")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "response_coverage_score": coverage,
        "uncovered_response_terms": uncovered,
    }


def attach_claim_set_validation(claims: Sequence[Claim], response: str, max_claims: int = DEFAULT_MAX_CLAIMS) -> List[Claim]:
    """Copy the claim-set verdict onto every claim of the response."""
    validation = validate_claim_set(response, claims, max_claims)
    errors = "|".join(validation["reasons"])
    uncovered = "|".join(validation["uncovered_response_terms"])
    for claim in claims:
        claim.claim_set_size = len(claims)
        claim.response_coverage_score = validation["response_coverage_score"]
        claim.coverage_complete = validation["response_coverage_score"] >= MIN_RESPONSE_COVERAGE
        claim.claim_set_validated = validation["valid"]
        claim.claim_set_validation_errors = errors
    return list(claims)


def sanitize_llm_claims(
    question: str,
    response: str,
    raw_claims: Iterable[Dict[str, Any]],
    max_claims: int = DEFAULT_MAX_CLAIMS,
) -> List[Claim]:
    """Validate and normalise the checked-LLM decomposition path."""
    max_claims = normalize_max_claims(max_claims)
    rows: List[Claim] = []
    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        claim_text = clean_text(item.get("claim_text", ""))
        span = clean_text(item.get("source_span", ""))
        if not claim_text:
            continue
        span_in_response = bool(span and span.lower() in clean_text(response).lower())
        source_span_inferred = not span_in_response
        if source_span_inferred:
            span = _best_source_span(response, claim_text)
        claim_type = clean_text(item.get("claim_type", "other"))
        if claim_type not in CLAIM_TYPES:
            claim_type = infer_claim_type(question, claim_text)
        atom = atomicity_score(claim_text)
        faith = lexical_faithfulness(response, span, claim_text)
        try:
            llm_conf = float(item.get("decomposition_confidence", 0.8))
        except (TypeError, ValueError):
            llm_conf = 0.8
        declared_atomic = item.get("is_atomic", True)
        if isinstance(declared_atomic, str):
            declared_atomic = declared_atomic.strip().lower() in {"1", "true", "yes", "y"}
        rows.append(
            Claim(
                claim_id=f"c{len(rows) + 1}",
                claim_text=claim_text,
                source_span=span,
                claim_type=claim_type,
                is_atomic=bool(declared_atomic) and atom >= 0.9,
                is_visual_verifiable=clean_text(item.get("is_visual_verifiable", "")) or visual_verifiability(question, claim_type),
                requires_external_knowledge=bool(
                    item.get("requires_external_knowledge", requires_external_knowledge(question, claim_type))
                ),
                verification_prompt=clean_text(item.get("verification_prompt", ""))
                or verification_prompt_for(claim_text, claim_type),
                decomposition_confidence=float(max(0.0, min(1.0, min(llm_conf, max(atom, 0.1), max(faith, 0.1))))),
                atomicity_score=atom,
                faithfulness_score=faith,
                parser_added_claim=faith < 0.35,
                source_span_inferred=source_span_inferred,
                decomposition_method="llm_qacd_v2_multiclaim_checked",
            )
        )
    return attach_claim_set_validation(rows, response, max_claims)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def decompose_claims(
    question: str,
    response: str,
    max_claims: int = DEFAULT_MAX_CLAIMS,
    llm_completion: Any = None,
) -> DecompositionResult:
    """Decompose one answer, preferring the checked LLM path when available.

    Parameters
    ----------
    llm_completion:
        Either ``None`` (rule fallback only), a raw JSON string, or an already
        parsed list of claim dicts. Any failure of the LLM path falls through to
        :func:`fallback_decompose`; the deployed system is the hybrid, and the
        returned ``method`` records which path produced the claims.
    """
    question = clean_text(question)
    response = clean_text(response)

    if llm_completion is not None:
        raw = llm_completion if isinstance(llm_completion, list) else parse_json_claims(str(llm_completion))
        claims = sanitize_llm_claims(question, response, raw, max_claims)
        validation = validate_claim_set(response, claims, max_claims)
        if claims and validation["valid"]:
            return DecompositionResult(
                claims=claims,
                valid=True,
                reasons=[],
                response_coverage_score=validation["response_coverage_score"],
                uncovered_response_terms=validation["uncovered_response_terms"],
                method="llm_checked",
            )

    claims = fallback_decompose(question, response, max_claims)
    validation = validate_claim_set(response, claims, max_claims)
    return DecompositionResult(
        claims=claims,
        valid=validation["valid"],
        reasons=validation["reasons"],
        response_coverage_score=validation["response_coverage_score"],
        uncovered_response_terms=validation["uncovered_response_terms"],
        method="rule_fallback",
    )
