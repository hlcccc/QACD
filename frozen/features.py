"""Feature engineering of the frozen research pipeline, ported verbatim.

This module is the layer that was previously missing from this repository. It
turns raw evidence tables into the 112-column claim-level feature matrix that
the frozen calibrator consumes:

    raw evidence tables
      -> add_direct_verifier_features
           -> add_qacd_aware_features
                -> add_contradiction_aware_features
                     -> add_direction_aligned_features  (risk_* and ix_* columns)
      -> select_feature_names + fit_imputer + apply_imputer
      -> matrix

Fidelity
--------
Every function body below is transcribed from the research modules named in
:mod:`frozen.constants`; :func:`verify_against_bundle` in
``scripts/verify_feature_port.py`` checks the result element-wise against the
feature matrix recorded in ``artifacts/evidence_bundle.npz``.

The only deliberate deviation: the research pipeline had **two different**
helpers both named ``_num``, in different modules.

* ``run_bew_bcm_v2._num(df, col, default=nan)`` returns the column untouched
  (no fill, no clip). ``add_contradiction_aware_features`` relies on that — it
  calls ``.notna()`` on the result.
* ``run_qacd_direct_verifier_benchmark._num(df, col, default=0.5)`` fills and
  clips to ``[0, 1]``.

Merged into one module they would collide, so they are kept separate here as
:func:`_num_bew` and :func:`_num_dvb`, and each ported function calls the one it
called originally.

Dependency note: this module needs pandas. The decision layer (:mod:`qacd`) and
the metrics (:mod:`evaluation`) remain NumPy-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frozen.constants import (  # noqa: E402
    BASE_EVIDENCE,
    CLAIM_METADATA_COLUMNS,
    CLAIM_TYPES,
    DIRECT_VERIFIER_FEATURES,
    INTERACTIONS,
    QACD_FEATURES,
    RISK_COLUMNS,
    SUPPORT_OR_CONFIDENCE_COLUMNS,
    V2_EVIDENCE,
    VERIFIABILITY,
    VIEW_SUPPORT_COLUMNS,
)

__all__ = [
    "add_direction_aligned_features",
    "add_contradiction_aware_features",
    "add_qacd_aware_features",
    "add_direct_verifier_features",
    "merge_claim_metadata",
    "load_direct_features",
    "available_feature_names",
    "usable_features",
    "dev_only_names",
    "select_feature_names",
    "fit_imputer",
    "apply_imputer",
    "read_csv_without_targets",
    "FORBIDDEN_TARGET_COLUMNS",
]

#: Target columns that must never be read while scoring (label gate).
FORBIDDEN_TARGET_COLUMNS = {"correct", "hallucination_label", "label", "gold_letter", "answers"}


# ---------------------------------------------------------------------------
# run_belief_weighted_calibration
# ---------------------------------------------------------------------------

def add_direction_aligned_features(df):
    out = df.copy()
    for col in SUPPORT_OR_CONFIDENCE_COLUMNS:
        if col in out.columns:
            out[f"risk_{col}"] = 1.0 - pd.to_numeric(out[col], errors="coerce")
    for col in RISK_COLUMNS:
        if col in out.columns:
            out[f"risk_{col}"] = pd.to_numeric(out[col], errors="coerce")
    for left, right, name in INTERACTIONS:
        if left in out.columns and right in out.columns:
            out[name] = pd.to_numeric(out[left], errors="coerce") * pd.to_numeric(out[right], errors="coerce")
    return out


def available_feature_names(source, target):
    names = [name for name in BASE_EVIDENCE if name in source.columns and name in target.columns]
    names += [name for _, _, name in INTERACTIONS if name in source.columns and name in target.columns]
    usable = []
    for name in names:
        s = pd.to_numeric(source[name], errors="coerce")
        t = pd.to_numeric(target[name], errors="coerce")
        if s.notna().sum() > 10 and t.notna().sum() > 10 and (s.std() > 1e-10 or t.std() > 1e-10):
            usable.append(name)
    return usable


# ---------------------------------------------------------------------------
# run_bew_bcm_v2  (the non-filling _num)
# ---------------------------------------------------------------------------

def _num_bew(df, col, default=np.nan):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _view_matrix(df):
    usable = [pd.to_numeric(df[col], errors="coerce").clip(0.0, 1.0) for col in VIEW_SUPPORT_COLUMNS if col in df.columns]
    if not usable:
        return pd.DataFrame({"fallback_support": np.full(len(df), 0.5)}, index=df.index)
    return pd.concat(usable, axis=1).fillna(0.5)


def add_contradiction_aware_features(df):
    """Add BEW-BCM v2 evidence.

    v1 uses generic support/disagreement/similarity risk features. v2 adds an
    explicit contradiction layer: per-view contradiction rate, visual grounding
    mismatch, and reliability-gated contradiction/disagreement terms.
    """
    out = add_direction_aligned_features(df)
    views = _view_matrix(out)

    contradiction = (views <= 0.25).astype(float)
    uncertain = ((views > 0.25) & (views < 0.75)).astype(float)
    supported = (views >= 0.75).astype(float)
    reliability = (np.abs(views - 0.5) * 2.0).clip(0.0, 1.0)

    out["v2_contradiction_rate"] = contradiction.mean(axis=1)
    out["v2_uncertain_rate"] = uncertain.mean(axis=1)
    out["v2_support_absence_rate"] = 1.0 - supported.mean(axis=1)
    out["v2_view_reliability"] = reliability.mean(axis=1)
    out["v2_low_reliability_risk"] = 1.0 - out["v2_view_reliability"]

    visual_risk_terms = []
    for col in [
        "claim_bcm_visual_answer_similarity",
        "claim_bcm_visual_answer_support",
        "claim_bcm_direct_verify_support",
        "claim_bcm_evidence_explain_support",
    ]:
        if col in out.columns:
            visual_risk_terms.append(1.0 - pd.to_numeric(out[col], errors="coerce"))
    if visual_risk_terms:
        out["v2_visual_grounded_risk"] = pd.concat(visual_risk_terms, axis=1).mean(axis=1, skipna=True).fillna(0.5).clip(0.0, 1.0)
    else:
        out["v2_visual_grounded_risk"] = _num_bew(out, "risk_bcm_multiview_support_mean", 0.5).fillna(0.5).clip(0.0, 1.0)

    contradiction_terms = []
    for col in [
        "risk_bcm_multiview_support_mean",
        "risk_bcm_multiview_claim_support_rate",
        "risk_claim_bcm_visual_answer_support",
        "risk_claim_bcm_visual_answer_similarity",
        "risk_claim_bcm_claim_match_after_answer_support",
        "risk_claim_bcm_direct_verify_support",
        "risk_claim_bcm_counterfactual_verify_support",
        "risk_claim_bcm_claim_compare_support",
        "risk_claim_bcm_evidence_explain_support",
    ]:
        if col in out.columns:
            contradiction_terms.append(pd.to_numeric(out[col], errors="coerce"))
    contradiction_terms.append(out["v2_contradiction_rate"])
    contradiction_terms.append(out["v2_visual_grounded_risk"])
    pressure = pd.concat(contradiction_terms, axis=1)
    out["v2_contradiction_pressure_mean"] = pressure.mean(axis=1, skipna=True).fillna(0.5).clip(0.0, 1.0)
    out["v2_contradiction_pressure_max"] = pressure.max(axis=1, skipna=True).fillna(0.5).clip(0.0, 1.0)

    disagreement = _num_bew(out, "risk_bcm_multiview_disagreement", 0.0).fillna(0.0).clip(0.0, 1.0)
    out["v2_reliable_disagreement"] = (disagreement * out["v2_view_reliability"]).clip(0.0, 1.0)
    out["v2_reliable_contradiction"] = (out["v2_contradiction_pressure_mean"] * out["v2_view_reliability"]).clip(0.0, 1.0)
    out["v2_uncertain_contradiction"] = (out["v2_contradiction_pressure_mean"] * out["v2_uncertain_rate"]).clip(0.0, 1.0)
    out["v2_visual_contradiction_interaction"] = (out["v2_visual_grounded_risk"] * out["v2_contradiction_pressure_mean"]).clip(0.0, 1.0)

    direct_support = _num_bew(out, "claim_bcm_direct_verify_support", np.nan)
    counter_support = _num_bew(out, "claim_bcm_counterfactual_verify_support", np.nan)
    old_claim_match = _num_bew(out, "claim_bcm_claim_match_after_answer_support", np.nan)
    if direct_support.notna().any() or counter_support.notna().any():
        direct_risk = (1.0 - direct_support).fillna(out["v2_contradiction_pressure_mean"])
        counter_risk = (1.0 - counter_support).fillna(out["v2_contradiction_pressure_mean"])
        out["v2_direct_counterfactual_conflict"] = np.maximum(direct_risk, counter_risk).clip(0.0, 1.0)
    elif old_claim_match.notna().any():
        out["v2_direct_counterfactual_conflict"] = (1.0 - old_claim_match).fillna(out["v2_contradiction_pressure_mean"]).clip(0.0, 1.0)
    else:
        out["v2_direct_counterfactual_conflict"] = out["v2_contradiction_pressure_mean"]
    return out


# ---------------------------------------------------------------------------
# run_qacd_aware_bew_bcm_v2_benchmark
# ---------------------------------------------------------------------------

def _bool_to_float(series):
    if series.dtype == bool:
        return series.astype(float)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin(["true", "1", "yes", "y"]).astype(float)


def add_qacd_aware_features(df):
    out = add_contradiction_aware_features(df)
    conf = pd.to_numeric(out.get("decomposition_confidence", 1.0), errors="coerce").fillna(1.0).clip(0.0, 1.0)
    atomic = pd.to_numeric(out.get("atomicity_score", 1.0), errors="coerce").fillna(1.0).clip(0.0, 1.0)
    faithful = pd.to_numeric(out.get("faithfulness_score", 1.0), errors="coerce").fillna(1.0).clip(0.0, 1.0)
    out["qacd_low_decomposition_confidence"] = 1.0 - conf
    out["qacd_low_atomicity"] = 1.0 - atomic
    out["qacd_low_faithfulness"] = 1.0 - faithful

    if "parser_added_claim" in out.columns:
        out["qacd_parser_added_claim"] = _bool_to_float(out["parser_added_claim"])
    else:
        out["qacd_parser_added_claim"] = 0.0
    if "requires_external_knowledge" in out.columns:
        out["qacd_external_knowledge_risk"] = _bool_to_float(out["requires_external_knowledge"])
    else:
        out["qacd_external_knowledge_risk"] = 0.0

    ver = out.get("is_visual_verifiable", pd.Series("indirect", index=out.index)).astype(str)
    for name in VERIFIABILITY:
        out[f"qacd_visual_verifiability_{name}"] = ver.str.lower().eq(name).astype(float)
    # Direct visual claims are usually easier, so keep them out of the risk feature list.
    out["qacd_visual_verifiability_hard"] = out.get("qacd_visual_verifiability_hard", 0.0)
    out["qacd_visual_verifiability_ocr"] = out.get("qacd_visual_verifiability_ocr", 0.0)
    out["qacd_visual_verifiability_indirect"] = out.get("qacd_visual_verifiability_indirect", 0.0)

    ctype = out.get("claim_type", pd.Series("other", index=out.index)).astype(str)
    for name in CLAIM_TYPES:
        out[f"qacd_type_{name}"] = ctype.eq(name).astype(float)

    quality_risk = out[["qacd_low_decomposition_confidence", "qacd_low_atomicity", "qacd_low_faithfulness"]].mean(axis=1).clip(0.0, 1.0)
    contradiction = pd.to_numeric(out.get("v2_contradiction_pressure_mean", 0.5), errors="coerce").fillna(0.5).clip(0.0, 1.0)
    visual = pd.to_numeric(out.get("v2_visual_grounded_risk", 0.5), errors="coerce").fillna(0.5).clip(0.0, 1.0)
    out["qacd_quality_x_contradiction"] = (quality_risk * contradiction).clip(0.0, 1.0)
    out["qacd_quality_x_visual_risk"] = (quality_risk * visual).clip(0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# run_qacd_direct_verifier_benchmark  (the filling _num)
# ---------------------------------------------------------------------------

def _num_dvb(df, column, default=0.5):
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).clip(0.0, 1.0)


def add_direct_verifier_features(df):
    out = add_qacd_aware_features(df)
    support_mean = _num_dvb(out, "direct_verifier_support_mean")
    support_min = _num_dvb(out, "direct_verifier_support_min")
    contradict = _num_dvb(out, "direct_verifier_contradict_mean")
    match = _num_dvb(out, "direct_verifier_match_mean")
    confidence = _num_dvb(out, "direct_verifier_confidence_mean")
    evidence = _num_dvb(out, "direct_verifier_evidence_presence", default=0.0)
    base_risk = _num_dvb(out, "direct_verifier_risk")

    support_prob_mean = _num_dvb(out, "direct_verifier_support_prob_mean")
    support_prob_min = _num_dvb(out, "direct_verifier_support_prob_min")
    support_prob_max = _num_dvb(out, "direct_verifier_support_prob_max")
    contradict_prob = _num_dvb(out, "direct_verifier_contradict_prob_mean")
    contradict_prob_max = _num_dvb(out, "direct_verifier_contradict_prob_max")
    match_prob = _num_dvb(out, "direct_verifier_match_prob_mean")
    evidence_prob = _num_dvb(out, "direct_verifier_evidence_prob_mean", default=0.0)

    out["direct_verifier_low_support_mean"] = 1.0 - support_mean
    out["direct_verifier_low_support_min"] = 1.0 - support_min
    out["direct_verifier_contradict_mean"] = contradict
    out["direct_verifier_low_match_mean"] = 1.0 - match
    out["direct_verifier_low_evidence_presence"] = 1.0 - evidence
    out["direct_verifier_confidence_mean"] = confidence
    out["direct_verifier_confidence_weighted_risk"] = (base_risk * confidence).clip(0.0, 1.0)
    out["direct_verifier_support_uncertainty"] = (1.0 - 2.0 * (support_mean - 0.5).abs()).clip(0.0, 1.0)

    out["direct_verifier_low_support_prob_mean"] = 1.0 - support_prob_mean
    out["direct_verifier_low_support_prob_min"] = 1.0 - support_prob_min
    out["direct_verifier_low_match_prob_mean"] = 1.0 - match_prob
    out["direct_verifier_contradict_prob_mean"] = contradict_prob
    out["direct_verifier_contradict_prob_max"] = contradict_prob_max
    out["direct_verifier_low_evidence_prob_mean"] = 1.0 - evidence_prob
    out["direct_verifier_uncertainty_prob_mean"] = _num_dvb(out, "direct_verifier_uncertainty_prob_mean", default=1.0)
    out["direct_verifier_confident_low_support_prob"] = _num_dvb(out, "direct_verifier_confident_low_support_prob")
    out["direct_verifier_confident_contradict_prob"] = _num_dvb(out, "direct_verifier_confident_contradict_prob")
    out["direct_verifier_risk_prob"] = _num_dvb(out, "direct_verifier_risk_prob")
    out["direct_verifier_support_prob_spread"] = (support_prob_max - support_prob_min).clip(0.0, 1.0)

    view_cols = []
    prob_view_cols = []
    for view in ["direct_support", "visual_evidence", "ocr_read", "number_check"]:
        col = f"direct_verifier_{view}_support"
        view_support = _num_dvb(out, col)
        risk_col = f"direct_verifier_{view}_risk"
        out[risk_col] = 1.0 - view_support
        view_cols.append(col)

        prob_col = f"direct_verifier_{view}_support_prob"
        view_support_prob = _num_dvb(out, prob_col)
        prob_risk_col = f"direct_verifier_{view}_prob_risk"
        out[prob_risk_col] = 1.0 - view_support_prob
        prob_view_cols.append(prob_col)

    view_values = pd.concat([_num_dvb(out, c) for c in view_cols], axis=1)
    out["direct_verifier_support_spread"] = (view_values.max(axis=1) - view_values.min(axis=1)).clip(0.0, 1.0)
    out["direct_verifier_ocr_requested"] = (out.get("direct_verifier_ocr_read_support", pd.Series(0.5, index=out.index)) != 0.5).astype(float)
    out["direct_verifier_number_requested"] = (out.get("direct_verifier_number_check_support", pd.Series(0.5, index=out.index)) != 0.5).astype(float)
    out["direct_verifier_ocr_or_number_risk"] = (
        out[["direct_verifier_ocr_read_risk", "direct_verifier_number_check_risk"]].max(axis=1)
        * out[["direct_verifier_ocr_requested", "direct_verifier_number_requested"]].max(axis=1)
    ).clip(0.0, 1.0)
    out["direct_verifier_visual_x_contradiction"] = (
        out["direct_verifier_visual_evidence_risk"] * contradict
    ).clip(0.0, 1.0)
    out["direct_verifier_prob_visual_x_contradiction"] = (
        out["direct_verifier_visual_evidence_prob_risk"] * contradict_prob
    ).clip(0.0, 1.0)
    quality_cols = [
        c
        for c in ["qacd_low_decomposition_confidence", "qacd_low_atomicity", "qacd_low_faithfulness"]
        if c in out.columns
    ]
    quality_risk = out[quality_cols].mean(axis=1).fillna(0.0) if quality_cols else pd.Series(0.0, index=out.index)
    out["direct_verifier_low_support_x_qacd_quality"] = (
        out["direct_verifier_low_support_mean"] * quality_risk
    ).clip(0.0, 1.0)
    out["direct_verifier_prob_low_support_x_qacd_quality"] = (
        out["direct_verifier_low_support_prob_mean"] * quality_risk
    ).clip(0.0, 1.0)
    return out


def merge_claim_metadata(features_path, claim_table_path):
    features = pd.read_csv(features_path)
    claims = pd.read_csv(claim_table_path)
    keep = [
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
    ]
    keep = [col for col in keep if col in claims.columns]
    merged = features.merge(claims[keep].drop_duplicates("unit_id"), on="unit_id", how="left")
    return merged


def load_direct_features(features_path, claim_table_path, direct_features_path):
    base = merge_claim_metadata(features_path, claim_table_path)
    direct = pd.read_csv(direct_features_path)
    direct = direct.drop_duplicates("unit_id")
    return add_direct_verifier_features(base.merge(direct, on="unit_id", how="left", suffixes=("", "_direct")))


# ---------------------------------------------------------------------------
# run_qacd_textvqa_heldout_evaluation
# ---------------------------------------------------------------------------

def usable_features(source, target, include_direct=True):
    names = list(BASE_EVIDENCE)
    names += [name for _, _, name in INTERACTIONS]
    names += list(V2_EVIDENCE)
    names += list(QACD_FEATURES)
    if include_direct:
        names += list(DIRECT_VERIFIER_FEATURES)
    usable = []
    for name in names:
        if name not in source.columns or name not in target.columns:
            continue
        s = pd.to_numeric(source[name], errors="coerce")
        t = pd.to_numeric(target[name], errors="coerce")
        if s.notna().sum() > 10 and t.notna().sum() > 10 and (s.std() > 1e-10 or t.std() > 1e-10):
            usable.append(name)
    return usable


def dev_only_names(dev, test, qacd=True):
    names = usable_features(dev, dev, include_direct=True) if qacd else available_feature_names(dev, dev)
    missing = [name for name in names if name not in test.columns]
    if missing:
        raise ValueError(
            "Frozen held-out table is missing development-selected features: "
            + ", ".join(missing)
        )
    return names


# ---------------------------------------------------------------------------
# Score-training-only selection and imputation (from the frozen protocol script
# run_conformal_selective_prediction_strict)
# ---------------------------------------------------------------------------

def read_csv_without_targets(path, **kwargs) -> pd.DataFrame:
    """Read a CSV after asserting that it carries no target column."""
    header = pd.read_csv(path, nrows=0, **{k: v for k, v in kwargs.items() if k != "usecols"}).columns
    offenders = sorted(FORBIDDEN_TARGET_COLUMNS.intersection(header))
    if offenders:
        raise ValueError(f"refusing to read target columns {offenders} from {path}")
    return pd.read_csv(path, **kwargs)


def select_feature_names(dev_aligned, test_aligned, claim_train_mask):
    """Select LM and mechanical features using score-training rows only."""
    train = dev_aligned.loc[claim_train_mask].copy()
    lm_names = usable_features(train, train, include_direct=True)
    lm_names = [name for name in lm_names if not name.startswith("mech_")]
    missing_test = [name for name in lm_names if name not in test_aligned.columns]
    if missing_test:
        raise ValueError(f"test table missing score-training-selected features: {missing_test}")
    mechanical_names = []
    for name in dev_aligned.columns:
        if not name.startswith("mech_") or name not in test_aligned.columns:
            continue
        values = pd.to_numeric(train[name], errors="coerce")
        if values.notna().sum() > 10 and float(values.std()) > 1e-10:
            mechanical_names.append(name)
    if not lm_names or not mechanical_names:
        raise ValueError(
            f"invalid feature selection: {len(lm_names)} LM, {len(mechanical_names)} mechanical"
        )
    return lm_names, mechanical_names


def fit_imputer(frame, names, mask):
    """Median imputation fitted on score-training rows only."""
    medians = {}
    for name in names:
        values = pd.to_numeric(frame.loc[mask, name], errors="coerce")
        medians[name] = float(values.median()) if values.notna().any() else 0.0
    return medians


def apply_imputer(frame, names, medians):
    columns = []
    for name in names:
        values = pd.to_numeric(frame[name], errors="coerce").fillna(medians[name])
        columns.append(values.to_numpy(dtype=np.float64))
    matrix = np.stack(columns, axis=1)
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite matrix after score-training-only imputation")
    return matrix
