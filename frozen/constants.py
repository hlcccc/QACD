"""Feature-group constants of the frozen research pipeline.

Transcribed verbatim from the research modules so that the ported feature
engineering is bit-identical to what produced the frozen evidence:

    SUPPORT_OR_CONFIDENCE_COLUMNS / RISK_COLUMNS / BASE_EVIDENCE / INTERACTIONS
        hallucination_calibration.experiments.run_belief_weighted_calibration
    VIEW_SUPPORT_COLUMNS
        hallucination_calibration.experiments.run_bew_bcm_v2
    DIRECT_VERIFIER_FEATURES
        hallucination_calibration.experiments.run_qacd_direct_verifier_benchmark
    QACD_RISK_FEATURES / QACD_TYPE_FEATURES / QACD_FEATURES / VERIFIABILITY
        hallucination_calibration.experiments.run_qacd_aware_bew_bcm_v2_benchmark
    V2_EVIDENCE
        hallucination_calibration.experiments.run_bew_bcm_v2

Order matters: `select_feature_names` walks these lists in order, and the frozen
calibrator's coefficients are bound to that resulting column order.
"""

from __future__ import annotations

from typing import List, Tuple

__all__ = [
    "SUPPORT_OR_CONFIDENCE_COLUMNS",
    "RISK_COLUMNS",
    "BASE_EVIDENCE",
    "INTERACTIONS",
    "VIEW_SUPPORT_COLUMNS",
    "DIRECT_VERIFIER_FEATURES",
    "QACD_RISK_FEATURES",
    "QACD_TYPE_FEATURES",
    "QACD_FEATURES",
    "V2_EVIDENCE",
    "VERIFIABILITY",
    "CLAIM_TYPES",
    "CLAIM_METADATA_COLUMNS",
]

SUPPORT_OR_CONFIDENCE_COLUMNS: List[str] = [
    "bcm_multiview_support_mean",
    "bcm_multiview_top_support",
    "bcm_multiview_compare_consensus",
    "bcm_multiview_confidence_mean",
    "bcm_multiview_answer_similarity_mean",
    "bcm_multiview_answer_similarity_max",
    "bcm_multiview_claim_support_rate",
    "claim_bcm_independent_answer_support",
    "claim_bcm_independent_answer_confidence",
    "claim_bcm_independent_answer_similarity",
    "claim_bcm_visual_answer_support",
    "claim_bcm_visual_answer_confidence",
    "claim_bcm_visual_answer_similarity",
    "claim_bcm_answer_without_claim_support",
    "claim_bcm_answer_without_claim_confidence",
    "claim_bcm_answer_without_claim_similarity",
    "claim_bcm_claim_match_after_answer_support",
    "claim_bcm_claim_match_after_answer_confidence",
    "claim_bcm_claim_match_after_answer_similarity",
    "claim_bcm_direct_verify_support",
    "claim_bcm_counterfactual_verify_support",
    "claim_bcm_claim_compare_support",
    "claim_bcm_evidence_explain_support",
]

RISK_COLUMNS: List[str] = [
    "bcm_multiview_support_std",
    "bcm_multiview_support_margin",
    "bcm_multiview_disagreement",
    "bcm_multiview_uncertainty",
    "bcm_multiview_unique_answer_ratio",
    "claim_bcm_independent_answer_uncertain",
    "claim_bcm_visual_answer_uncertain",
    "claim_bcm_answer_without_claim_uncertain",
    "claim_bcm_claim_match_after_answer_uncertain",
]

BASE_EVIDENCE: List[str] = [
    "risk_bcm_multiview_support_mean",
    "risk_bcm_multiview_support_std",
    "risk_bcm_multiview_disagreement",
    "risk_bcm_multiview_answer_similarity_mean",
    "risk_bcm_multiview_claim_support_rate",
    "risk_bcm_multiview_unique_answer_ratio",
    "risk_claim_bcm_visual_answer_similarity",
    "risk_claim_bcm_answer_without_claim_similarity",
    "risk_claim_bcm_direct_verify_support",
    "risk_claim_bcm_counterfactual_verify_support",
    "risk_claim_bcm_claim_compare_support",
    "risk_claim_bcm_evidence_explain_support",
]

INTERACTIONS: List[Tuple[str, str, str]] = [
    ("risk_bcm_multiview_support_mean", "risk_bcm_multiview_disagreement", "ix_support_disagreement"),
    ("risk_bcm_multiview_answer_similarity_mean", "risk_bcm_multiview_unique_answer_ratio", "ix_similarity_unique"),
    ("risk_bcm_multiview_claim_support_rate", "risk_bcm_multiview_disagreement", "ix_claim_mismatch_disagreement"),
    ("risk_claim_bcm_visual_answer_similarity", "risk_bcm_multiview_disagreement", "ix_visual_similarity_disagreement"),
    ("risk_claim_bcm_direct_verify_support", "risk_bcm_multiview_disagreement", "ix_direct_verify_disagreement"),
]

VIEW_SUPPORT_COLUMNS: List[str] = [
    "claim_bcm_independent_answer_support",
    "claim_bcm_visual_answer_support",
    "claim_bcm_answer_without_claim_support",
    "claim_bcm_claim_match_after_answer_support",
    "claim_bcm_direct_verify_support",
    "claim_bcm_counterfactual_verify_support",
    "claim_bcm_claim_compare_support",
    "claim_bcm_evidence_explain_support",
]

DIRECT_VERIFIER_FEATURES: List[str] = [
    "direct_verifier_risk",
    "direct_verifier_low_support_mean",
    "direct_verifier_low_support_min",
    "direct_verifier_contradict_mean",
    "direct_verifier_low_match_mean",
    "direct_verifier_low_evidence_presence",
    "direct_verifier_confidence_mean",
    "direct_verifier_confidence_weighted_risk",
    "direct_verifier_support_uncertainty",
    "direct_verifier_support_spread",
    "direct_verifier_direct_support_risk",
    "direct_verifier_visual_evidence_risk",
    "direct_verifier_ocr_read_risk",
    "direct_verifier_number_check_risk",
    "direct_verifier_ocr_requested",
    "direct_verifier_number_requested",
    "direct_verifier_ocr_or_number_risk",
    "direct_verifier_visual_x_contradiction",
    "direct_verifier_low_support_x_qacd_quality",
    "direct_verifier_risk_prob",
    "direct_verifier_low_support_prob_mean",
    "direct_verifier_low_support_prob_min",
    "direct_verifier_low_match_prob_mean",
    "direct_verifier_contradict_prob_mean",
    "direct_verifier_contradict_prob_max",
    "direct_verifier_low_evidence_prob_mean",
    "direct_verifier_uncertainty_prob_mean",
    "direct_verifier_confident_low_support_prob",
    "direct_verifier_confident_contradict_prob",
    "direct_verifier_support_prob_spread",
    "direct_verifier_direct_support_prob_risk",
    "direct_verifier_visual_evidence_prob_risk",
    "direct_verifier_ocr_read_prob_risk",
    "direct_verifier_number_check_prob_risk",
    "direct_verifier_prob_visual_x_contradiction",
    "direct_verifier_extracted_text_alignment_risk",
    "direct_verifier_evidence_text_alignment_risk",
    "direct_verifier_ocr_text_alignment_risk",
    "direct_verifier_number_alignment_risk",
    "direct_verifier_number_missing_risk",
    "direct_verifier_extracted_text_present_mean",
    "direct_verifier_extracted_text_similarity_mean",
    "direct_verifier_extracted_text_similarity_max",
    "direct_verifier_extracted_text_coverage_mean",
    "direct_verifier_extracted_text_alignment_max",
    "direct_verifier_evidence_text_alignment_max",
    "direct_verifier_ocr_text_alignment_max",
    "direct_verifier_claim_number_present",
    "direct_verifier_extracted_number_present",
    "direct_verifier_number_claim_match_max",
    "direct_verifier_prob_low_support_x_qacd_quality",
]

QACD_RISK_FEATURES: List[str] = [
    "qacd_low_decomposition_confidence",
    "qacd_low_atomicity",
    "qacd_low_faithfulness",
    "qacd_parser_added_claim",
    "qacd_external_knowledge_risk",
    "qacd_visual_verifiability_hard",
    "qacd_visual_verifiability_ocr",
    "qacd_visual_verifiability_indirect",
    "qacd_quality_x_contradiction",
    "qacd_quality_x_visual_risk",
]

VERIFIABILITY: List[str] = ["direct", "ocr", "hard", "indirect"]

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

QACD_TYPE_FEATURES: List[str] = [f"qacd_type_{name}" for name in CLAIM_TYPES]

QACD_FEATURES: List[str] = QACD_RISK_FEATURES + QACD_TYPE_FEATURES

V2_EVIDENCE: List[str] = [
    "v2_contradiction_rate",
    "v2_uncertain_rate",
    "v2_support_absence_rate",
    "v2_view_reliability",
    "v2_low_reliability_risk",
    "v2_visual_grounded_risk",
    "v2_contradiction_pressure_mean",
    "v2_contradiction_pressure_max",
    "v2_reliable_disagreement",
    "v2_reliable_contradiction",
    "v2_uncertain_contradiction",
    "v2_visual_contradiction_interaction",
    "v2_direct_counterfactual_conflict",
]

#: Claim metadata copied out of the claim table (target columns excluded).
CLAIM_METADATA_COLUMNS: Tuple[str, ...] = (
    "unit_id",
    "claim_type",
    "is_atomic",
    "is_visual_verifiable",
    "requires_external_knowledge",
    "decomposition_confidence",
    "atomicity_score",
    "faithfulness_score",
    "parser_added_claim",
    "decomposition_method",
)
