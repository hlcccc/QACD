"""Ported frozen feature engineering.

Two kinds of test here:

* **unit** — each derivation rule, including the two different ``_num`` helpers
  whose behaviour must not be conflated;
* **end-to-end** — the whole port rebuilt from the committed raw evidence tables
  and compared element-wise against the frozen matrix. That is the test that
  would catch a mis-transcribed line.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandas")

from frozen import constants as C  # noqa: E402
from frozen.build import build_matrices, claim_images, image_split, load_split  # noqa: E402
from frozen.features import (  # noqa: E402
    _num_bew,
    _num_dvb,
    add_contradiction_aware_features,
    add_direction_aligned_features,
    add_direct_verifier_features,
    add_qacd_aware_features,
    apply_imputer,
    fit_imputer,
    select_feature_names,
)

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "artifacts" / "raw"
BUNDLE = REPO / "artifacts" / "evidence_bundle.npz"

#: ``DIRECT_VERIFIER_FEATURES`` mixes two origins: these arrive pre-computed in
#: the raw direct-verifier table, the remaining 27 are derived by
#: :func:`add_direct_verifier_features`.
RAW_DIRECT_COLUMNS = (
    "direct_verifier_risk",
    "direct_verifier_confidence_mean",
    "direct_verifier_contradict_mean",
    "direct_verifier_risk_prob",
    "direct_verifier_uncertainty_prob_mean",
    "direct_verifier_contradict_prob_mean",
    "direct_verifier_contradict_prob_max",
    "direct_verifier_confident_low_support_prob",
    "direct_verifier_confident_contradict_prob",
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
)
DERIVED_DIRECT_COUNT = len(C.DIRECT_VERIFIER_FEATURES) - len(RAW_DIRECT_COLUMNS)


def _frame(n: int = 40) -> pd.DataFrame:
    """A synthetic claim frame carrying the raw columns the derivations read."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "unit_id": [f"u{i}" for i in range(n)],
            "image": [f"img{i // 2}" for i in range(n)],
        }
    )
    for col in C.SUPPORT_OR_CONFIDENCE_COLUMNS:
        frame[col] = rng.random(n)
    for col in C.RISK_COLUMNS:
        frame[col] = rng.random(n)
    frame["claim_type"] = rng.choice(C.CLAIM_TYPES, n)
    frame["is_visual_verifiable"] = rng.choice(C.VERIFIABILITY, n)
    frame["is_atomic"] = rng.choice([True, False], n)
    frame["requires_external_knowledge"] = rng.choice([True, False], n)
    frame["parser_added_claim"] = rng.choice([True, False], n)
    frame["decomposition_confidence"] = rng.random(n)
    frame["atomicity_score"] = rng.random(n)
    frame["faithfulness_score"] = rng.random(n)
    for view in ("direct_support", "visual_evidence", "ocr_read", "number_check"):
        frame[f"direct_verifier_{view}_support"] = rng.random(n)
        frame[f"direct_verifier_{view}_support_prob"] = rng.random(n)
    for col in RAW_DIRECT_COLUMNS:
        frame[col] = rng.random(n)
    frame["direct_verifier_evidence_presence"] = rng.random(n)
    frame["direct_verifier_match_mean"] = rng.random(n)
    for col in (
        "direct_verifier_support_mean", "direct_verifier_support_min",
        "direct_verifier_support_prob_mean", "direct_verifier_support_prob_min",
        "direct_verifier_support_prob_max", "direct_verifier_match_prob_mean",
        "direct_verifier_evidence_prob_mean",
    ):
        frame[col] = rng.random(n)
    return frame


# -- constants -----------------------------------------------------------

def test_feature_groups_are_consistent():
    assert len(C.SUPPORT_OR_CONFIDENCE_COLUMNS) == 23
    assert len(C.RISK_COLUMNS) == 9
    assert len(C.BASE_EVIDENCE) == 12
    assert len(C.INTERACTIONS) == 5
    assert len(C.VIEW_SUPPORT_COLUMNS) == 8
    assert len(C.DIRECT_VERIFIER_FEATURES) == 51
    assert len(C.QACD_RISK_FEATURES) == 10
    assert len(C.QACD_TYPE_FEATURES) == 11
    assert len(C.V2_EVIDENCE) == 13
    assert C.QACD_FEATURES == C.QACD_RISK_FEATURES + C.QACD_TYPE_FEATURES


def test_no_duplicate_names_within_a_group():
    for name in ("SUPPORT_OR_CONFIDENCE_COLUMNS", "RISK_COLUMNS", "BASE_EVIDENCE",
                 "VIEW_SUPPORT_COLUMNS", "DIRECT_VERIFIER_FEATURES", "V2_EVIDENCE",
                 "QACD_FEATURES"):
        values = getattr(C, name)
        assert len(values) == len(set(values)), name


def test_interactions_reference_declared_risk_columns():
    declared = set(C.BASE_EVIDENCE)
    for left, right, name in C.INTERACTIONS:
        assert name.startswith("ix_")
        assert left in declared and right in declared


# -- the two _num helpers must stay distinct -----------------------------

def test_num_bew_leaves_missing_columns_as_nan():
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    out = _num_bew(frame, "absent", np.nan)
    assert out.isna().all(), "the BEW helper must not fill"
    assert out.dtype == "float64"


def test_num_bew_does_not_clip():
    frame = pd.DataFrame({"a": [5.0, -3.0]})
    assert _num_bew(frame, "a").tolist() == [5.0, -3.0]


def test_num_dvb_fills_and_clips():
    frame = pd.DataFrame({"a": [5.0, -3.0]})
    assert _num_dvb(frame, "a").tolist() == [1.0, 0.0]
    assert _num_dvb(frame, "absent", default=0.5).tolist() == [0.5, 0.5]


def test_conflating_the_two_helpers_would_change_the_result():
    """The contradiction layer depends on _num_bew NOT filling."""
    frame = _frame()
    frame.loc[: len(frame) // 2, "claim_bcm_direct_verify_support"] = np.nan
    out = add_contradiction_aware_features(frame)
    assert out["v2_direct_counterfactual_conflict"].notna().all()
    assert out["v2_direct_counterfactual_conflict"].between(0.0, 1.0).all()


# -- derivation rules ----------------------------------------------------

def test_direction_alignment_creates_risk_and_interaction_columns():
    out = add_direction_aligned_features(_frame())
    for col in C.SUPPORT_OR_CONFIDENCE_COLUMNS:
        assert f"risk_{col}" in out.columns
    for col in C.RISK_COLUMNS:
        assert f"risk_{col}" in out.columns
    for _, _, name in C.INTERACTIONS:
        assert name in out.columns


def test_direction_alignment_inverts_support_columns():
    frame = _frame(4)
    frame["bcm_multiview_support_mean"] = [0.0, 0.25, 0.75, 1.0]
    out = add_direction_aligned_features(frame)
    assert out["risk_bcm_multiview_support_mean"].tolist() == [1.0, 0.75, 0.25, 0.0]


def test_direction_alignment_passes_risk_columns_through():
    frame = _frame(4)
    frame["bcm_multiview_disagreement"] = [0.1, 0.2, 0.3, 0.4]
    out = add_direction_aligned_features(frame)
    assert out["risk_bcm_multiview_disagreement"].tolist() == [0.1, 0.2, 0.3, 0.4]


def test_contradiction_layer_produces_every_v2_feature():
    out = add_contradiction_aware_features(_frame())
    for name in C.V2_EVIDENCE:
        assert name in out.columns, name
        values = out[name]
        assert values.between(0.0, 1.0).all(), name


def test_qacd_layer_produces_every_qacd_feature():
    out = add_qacd_aware_features(_frame())
    for name in C.QACD_FEATURES:
        assert name in out.columns, name
        assert out[name].between(0.0, 1.0).all(), name


def test_claim_type_one_hot_has_exactly_one_hit_per_row():
    out = add_qacd_aware_features(_frame())
    hits = out[C.QACD_TYPE_FEATURES].sum(axis=1)
    assert (hits == 1.0).all()


def test_direct_verifier_layer_produces_every_declared_feature():
    out = add_direct_verifier_features(_frame())
    for name in C.DIRECT_VERIFIER_FEATURES:
        assert name in out.columns, name
        assert out[name].notna().all(), name
        assert out[name].between(0.0, 1.0).all(), name


def test_direct_verifier_layer_derives_the_expected_number_of_columns():
    """Guards against a derivation being silently dropped."""
    before = set(_frame().columns)
    after = set(add_direct_verifier_features(_frame()).columns)
    newly_created = after - before

    derived_direct = set(C.DIRECT_VERIFIER_FEATURES) - set(RAW_DIRECT_COLUMNS)
    assert len(derived_direct) == DERIVED_DIRECT_COUNT == 27
    assert derived_direct <= newly_created
    # The chain also materialises the v2 and qacd families.
    assert set(C.V2_EVIDENCE) <= newly_created
    assert set(C.QACD_FEATURES) <= newly_created


def test_direct_verifier_layer_tolerates_missing_raw_readings():
    frame = _frame().drop(columns=["direct_verifier_match_mean", "direct_verifier_evidence_presence"])
    out = add_direct_verifier_features(frame)
    assert out["direct_verifier_low_match_mean"].notna().all()


# -- split ---------------------------------------------------------------

def test_image_split_is_disjoint_and_complete():
    images = [f"img{i}" for i in range(200)] * 2
    train, calib = image_split(images, seed=20260920, calibration_fraction=0.5)
    assert not (train & calib)
    assert train | calib == set(images)


def test_image_split_is_deterministic():
    images = [f"img{i}" for i in range(100)]
    assert image_split(images) == image_split(images)


def test_image_split_honours_the_calibration_fraction():
    images = [f"img{i}" for i in range(100)]
    train, calib = image_split(images, calibration_fraction=0.5)
    assert len(calib) == 50 and len(train) == 50
    train, calib = image_split(images, calibration_fraction=0.25)
    assert len(calib) == 25 and len(train) == 75


# -- end-to-end against the committed artifacts --------------------------

requires_artifacts = pytest.mark.skipif(
    not (RAW / "dev_features.csv.gz").exists() or not BUNDLE.exists(),
    reason="raw evidence tables or the evidence bundle are not present",
)


@requires_artifacts
def test_port_reproduces_the_frozen_matrix_exactly():
    """The regression test that matters: rebuild every cell from raw tables."""
    bundle = np.load(BUNDLE, allow_pickle=False)
    dev = load_split(RAW, "dev")
    test = load_split(RAW, "test")
    built = build_matrices(dev, test, claim_images(dev, RAW / "dev_claims.csv.gz"))

    reference_names = [str(x) for x in bundle["lm_names"]] + [str(x) for x in bundle["mechanical_names"]]
    assert built.names == reference_names
    assert built.n_features == 112

    assert np.array_equal(built.dev_matrix, bundle["dev_matrix"])
    assert np.array_equal(built.test_matrix, bundle["test_matrix"])

    medians = np.array([built.medians[n] for n in built.names])
    assert np.array_equal(medians, bundle["imputer_medians"])


@requires_artifacts
def test_real_frames_carry_every_declared_feature():
    """On real evidence, all 51 direct-verifier names must resolve."""
    dev = load_split(RAW, "dev")
    for name in C.DIRECT_VERIFIER_FEATURES:
        assert name in dev.columns, name


@requires_artifacts
def test_split_matches_the_frozen_split():
    bundle = np.load(BUNDLE, allow_pickle=False)
    dev = load_split(RAW, "dev")
    images = claim_images(dev, RAW / "dev_claims.csv.gz")
    train, calib = image_split(images)
    assert sorted(train) == sorted(str(x) for x in bundle["score_train_images"])
    assert sorted(calib) == sorted(str(x) for x in bundle["calibration_images"])
