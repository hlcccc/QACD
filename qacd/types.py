"""Typed containers shared across the QACD decision layer.

Everything here is plain Python + dataclasses so that the package remains
importable with NumPy alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# Ordered claim-type vocabulary. Kept identical to the research pipeline so that
# frozen calibration coefficients stay interpretable across versions.
CLAIM_TYPES: List[str] = [
    "object_presence",
    "attribute",
    "counting",
    "spatial_relation",
    "action",
    "OCR_text",
    "entity_identity",
    "scene_global",
    "answer_identity",
    "external_knowledge",
    "other",
]

# Implementation cap on the number of claims per response. This bounds parse
# length and downstream verification cost. It is NOT a tuned hyper-parameter and
# the project makes no claim that 8 is optimal.
DEFAULT_MAX_CLAIMS = 8

# Minimum fraction of response content words that the claim set must cover for
# the decomposition to be accepted as complete.
MIN_RESPONSE_COVERAGE = 0.8

EMPTY_RESPONSE = "[EMPTY_RESPONSE]"


@dataclass
class Claim:
    """One question-conditioned atomic claim extracted from a VQA answer."""

    claim_id: str
    claim_text: str
    source_span: str = ""
    claim_type: str = "other"
    is_atomic: bool = False
    is_visual_verifiable: str = "indirect"
    requires_external_knowledge: bool = False
    verification_prompt: str = ""
    decomposition_confidence: float = 0.0
    atomicity_score: float = 0.0
    faithfulness_score: float = 0.0
    decomposition_method: str = "rule_fallback"
    parser_added_claim: bool = False
    source_span_inferred: bool = False
    fallback_truncated: bool = False
    # Claim-set level bookkeeping, copied onto every claim of the response.
    claim_set_size: int = 0
    response_coverage_score: float = 0.0
    coverage_complete: bool = False
    claim_set_validated: bool = False
    claim_set_validation_errors: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Claim":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class ClaimRisk:
    """Intermediate per-claim risk produced by the calibrated scorer."""

    claim_id: str
    claim_text: str
    claim_type: str
    risk: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResponseRisk:
    """Final answer-level risk object — this is the platform-facing payload."""

    risk_score: float
    is_high_risk: bool
    threshold: float
    num_claims: int
    claims: List[ClaimRisk] = field(default_factory=list)
    feature_dim: int = 0
    model_calls: int = 0
    latency_ms: int = 0
    version: str = ""
    warnings: List[str] = field(default_factory=list)
    # Individual evidence channels, exposed for downstream routing/audit.
    channels: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["claims"] = [c.to_dict() for c in self.claims]
        return payload


@dataclass
class DecompositionResult:
    """Claims plus the validation verdict for one response."""

    claims: List[Claim]
    valid: bool
    reasons: Sequence[str]
    response_coverage_score: float
    uncovered_response_terms: Sequence[str]
    method: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claims": [c.to_dict() for c in self.claims],
            "valid": self.valid,
            "reasons": list(self.reasons),
            "response_coverage_score": self.response_coverage_score,
            "uncovered_response_terms": list(self.uncovered_response_terms),
            "method": self.method,
        }


@dataclass
class OCRRecord:
    """Raw OCR output for one image."""

    image: str
    texts: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RiskRequest:
    """Canonical inference request for the platform-facing interfaces."""

    question: str
    answer: str
    image: str = ""
    max_claims: int = DEFAULT_MAX_CLAIMS
    threshold: Optional[float] = None
    k: Optional[int] = None
    return_claims: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
