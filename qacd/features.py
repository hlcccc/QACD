"""Feature assembly.

Turns (claim, provider readings, OCR text) into a *named, ordered* feature
vector. Two properties matter for integration:

1. **Names are stable.** The frozen calibration coefficients are bound to a
   feature order. Any change to :data:`LM_FEATURE_NAMES` invalidates a frozen
   scorer and must bump :data:`FEATURE_SCHEMA_VERSION`.
2. **Defined for every input.** Missing readings become explicit default values
   rather than NaNs, so the platform never has to decide what to do with a
   partial payload.

Scope note: this module produces the core LM-evidence sub-family plus the
alignment and interaction terms. The research pipeline's frozen TextVQA scorer
consumes a 95-feature LM vector assembled from the same families with additional
per-view and per-routing columns. :data:`RESEARCH_LM_FEATURE_COUNT` records that
number so the two are never confused.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Sequence

from qacd.align import alignment_features
from qacd.mechanical import MECHANICAL_FEATURES, MVR_FEATURES, mechanical_ocr_features
from qacd.providers import BELIEF_VIEWS, DirectVerification, VerificationView

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "RESEARCH_LM_FEATURE_COUNT",
    "LM_FEATURE_NAMES",
    "ALL_FEATURE_NAMES",
    "belief_features",
    "direct_features",
    "interaction_features",
    "assemble_claim_features",
    "build_matrix",
]

#: Bump whenever the meaning or order of a feature changes.
FEATURE_SCHEMA_VERSION = "1.0.0"

#: Feature count of the frozen research LM channel (TextVQA-3K, 95 columns).
RESEARCH_LM_FEATURE_COUNT = 95

#: Feature count of the complete research mechanical instrument family
#: (19 OCR + CLIP probes + deterministic counting probes).
RESEARCH_MECHANICAL_FEATURE_COUNT = 28


def belief_features(views: Sequence[VerificationView]) -> Dict[str, float]:
    """Summarise the belief views for one claim."""
    if not views:
        views = [VerificationView(view=v, support=0.5, confidence=0.5, uncertainty=0.5) for v in BELIEF_VIEWS]

    support = [float(v.support) for v in views]
    confidence = [float(v.confidence) for v in views]
    uncertainty = [float(v.uncertainty) for v in views]

    out: Dict[str, float] = {
        "belief_support_mean": statistics.fmean(support),
        "belief_support_min": min(support),
        "belief_support_max": max(support),
        "belief_support_spread": max(support) - min(support),
        "belief_confidence_mean": statistics.fmean(confidence),
        "belief_uncertainty_mean": statistics.fmean(uncertainty),
        # Risk direction: low support across views is the warning signal.
        "belief_low_support_risk": 1.0 - statistics.fmean(support),
        "belief_min_support_risk": 1.0 - min(support),
        "belief_num_views": float(len(views)),
    }
    for view in BELIEF_VIEWS:
        match = [v for v in views if v.view == view]
        out[f"belief_support_{view}"] = float(match[0].support) if match else 0.5
    return out


def direct_features(direct: DirectVerification) -> Dict[str, float]:
    """Summarise the type-routed direct verification for one claim."""
    support = float(direct.support)
    contradiction = float(direct.contradiction)
    uncertainty = float(direct.uncertainty)
    # Interpretable composite, as described in the method write-up. It is an
    # input feature, not the final calibrated score.
    composite_risk = max(0.0, min(1.0, 0.5 * (1.0 - support) + 0.3 * contradiction + 0.2 * uncertainty))
    return {
        "direct_verifier_support": support,
        "direct_verifier_contradiction": contradiction,
        "direct_verifier_evidence_present": float(direct.evidence_present),
        "direct_verifier_uncertainty": uncertainty,
        "direct_verifier_composite_risk": composite_risk,
        "direct_verifier_visual_evidence_support": float(direct.visual_evidence_support),
        "direct_verifier_ocr_read_support": float(direct.ocr_read_support),
        "direct_verifier_number_check_support": float(direct.number_check_support),
        "direct_verifier_low_support_risk": 1.0 - support,
    }


def interaction_features(belief: Dict[str, float], direct: Dict[str, float], align: Dict[str, float]) -> Dict[str, float]:
    """Cross-family terms carried by the frozen feature vector."""
    return {
        "direct_verifier_visual_x_contradiction": (
            direct["direct_verifier_visual_evidence_support"] * direct["direct_verifier_contradiction"]
        ),
        "direct_verifier_low_support_x_qacd_quality": (
            direct["direct_verifier_low_support_risk"] * belief["belief_low_support_risk"]
        ),
        "align_x_low_support_risk": align["align_unsupported_risk"] * belief["belief_low_support_risk"],
    }


#: Ordered LM-channel feature names produced by this module.
LM_FEATURE_NAMES: List[str] = [
    "belief_support_mean",
    "belief_support_min",
    "belief_support_max",
    "belief_support_spread",
    "belief_confidence_mean",
    "belief_uncertainty_mean",
    "belief_low_support_risk",
    "belief_min_support_risk",
    "belief_num_views",
    "belief_support_independent",
    "belief_support_visual",
    "belief_support_minus_claim",
    "belief_support_answer_match",
    "direct_verifier_support",
    "direct_verifier_contradiction",
    "direct_verifier_evidence_present",
    "direct_verifier_uncertainty",
    "direct_verifier_composite_risk",
    "direct_verifier_visual_evidence_support",
    "direct_verifier_ocr_read_support",
    "direct_verifier_number_check_support",
    "direct_verifier_low_support_risk",
    "align_char_similarity",
    "align_content_coverage",
    "align_score",
    "align_unsupported_risk",
    "direct_verifier_visual_x_contradiction",
    "direct_verifier_low_support_x_qacd_quality",
    "align_x_low_support_risk",
]

#: Full order used by the reference pipeline: LM, then mechanical, then MVR.
ALL_FEATURE_NAMES: List[str] = LM_FEATURE_NAMES + MECHANICAL_FEATURES + MVR_FEATURES

#: Non-feature columns carried alongside the claim for audit.
CLAIM_META_COLUMNS: List[str] = [
    "claim_id",
    "claim_text",
    "claim_type",
    "decomposition_method",
    "claim_set_validated",
    "response_coverage_score",
    "atomicity_score",
    "faithfulness_score",
]


def assemble_claim_features(
    claim_text: str,
    views: Sequence[VerificationView],
    direct: DirectVerification,
    ocr_texts: Sequence[str],
    sampled_answers: Sequence[str] | None = None,
    k: int | None = None,
) -> Dict[str, float]:
    """Assemble the complete reference feature vector for one claim.

    MVR features are only added when ``sampled_answers`` is supplied; they are a
    response-level signal in the frozen configuration, so
    :func:`qacd.pipeline.QACDPipeline` normally attaches them after aggregation
    rather than per claim.
    """
    evidence = direct.evidence_phrase or ""
    align = alignment_features(claim_text, evidence)
    belief = belief_features(views)
    directf = direct_features(direct)
    mech = mechanical_ocr_features(claim_text, ocr_texts)

    features: Dict[str, float] = {}
    features.update(belief)
    features.update(directf)
    features.update(align)
    features.update(interaction_features(belief, directf, align))
    features.update(mech)

    if sampled_answers is not None:
        from qacd.mechanical import mvr_features  # local import keeps the cycle out

        features.update(mvr_features(sampled_answers, ocr_texts, k=k))

    # Guarantee the contract: every declared name is present.
    for name in ALL_FEATURE_NAMES:
        features.setdefault(name, 0.0)
    return features


def build_matrix(rows: Sequence[Dict[str, float]], feature_names: Sequence[str]):
    """Stack feature dicts into a float matrix in a fixed column order."""
    import numpy as np

    matrix = np.zeros((len(rows), len(feature_names)), dtype=float)
    for i, row in enumerate(rows):
        for j, name in enumerate(feature_names):
            try:
                matrix[i, j] = float(row.get(name, 0.0))
            except (TypeError, ValueError):
                matrix[i, j] = 0.0
    return matrix
