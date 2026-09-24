"""Mechanical OCR channel and MVR statistics."""

import math

from qacd.features import MECHANICAL_FEATURES, MVR_FEATURES
from qacd.mechanical import (
    MECHANICAL_FAMILY_SIZE,
    mechanical_ocr_features,
    mvr_features,
    normalize_text,
)

OCR = ["Dakota Digital", "open 24 days"]


def test_every_declared_feature_is_produced():
    feats = mechanical_ocr_features("Dakota Digital", OCR)
    for name in MECHANICAL_FEATURES:
        assert name in feats, name
        assert isinstance(feats[name], float)
    assert len(MECHANICAL_FEATURES) < MECHANICAL_FAMILY_SIZE  # OCR sub-family only


def test_exact_hit_has_no_absence_risk():
    feats = mechanical_ocr_features("Dakota Digital", OCR)
    assert feats["mech_ocr_exact_present"] == 1.0
    assert feats["mech_ocr_absent_risk"] == 0.0
    assert feats["mech_ocr_low_similarity_risk"] == 0.0


def test_claims_absent_from_the_image_carry_risk():
    feats = mechanical_ocr_features("Nikon Coolpix", OCR)
    assert feats["mech_ocr_exact_present"] == 0.0
    assert feats["mech_ocr_absent_risk"] == 1.0
    assert feats["mech_ocr_low_similarity_risk"] > 0.4


def test_numbers_are_checked_explicitly():
    present = mechanical_ocr_features("open 24 days", OCR)
    missing = mechanical_ocr_features("open 48 days", OCR)
    assert present["mech_ocr_has_number"] == 1.0
    assert present["mech_ocr_number_present"] == 1.0
    assert present["mech_ocr_number_missing_risk"] == 0.0
    assert missing["mech_ocr_number_missing_risk"] == 1.0


def test_empty_ocr_is_flagged_not_crashed():
    feats = mechanical_ocr_features("anything", [])
    assert feats["mech_ocr_empty"] == 1.0
    assert feats["mech_ocr_absent_risk"] == 1.0


def test_normalisation_drops_articles_and_punctuation():
    assert normalize_text("The Cat, sat!") == "cat sat"


# -- MVR -----------------------------------------------------------------

def test_mvr_all_supported():
    feats = mvr_features(["Dakota Digital"] * 3, OCR, k=3)
    assert feats["mvr_all_supported"] == 1.0
    assert feats["mvr_unsupported_rate"] == 0.0
    assert feats["mvr_fuzzy_min"] == 1.0


def test_mvr_all_unsupported():
    feats = mvr_features(["Nikon"] * 3, OCR, k=3)
    assert feats["mvr_all_unsupported"] == 1.0
    assert feats["mvr_unsupported_rate"] == 1.0


def test_mvr_mixed_samples_sit_in_between():
    feats = mvr_features(["Dakota Digital", "Nikon", "Dakota Digital"], OCR, k=3)
    assert 0.0 < feats["mvr_unsupported_rate"] < 1.0
    assert feats["mvr_all_supported"] == 0.0
    assert feats["mvr_all_unsupported"] == 0.0
    assert feats["mvr_fuzzy_best_of_k"] >= feats["mvr_fuzzy_mean"] >= feats["mvr_fuzzy_min"]
    assert feats["mvr_fuzzy_std"] >= 0.0


def test_mvr_k_truncates_and_only_k_feature_is_k_dependent():
    pool = ["Dakota Digital", "Nikon", "Canon", "Dakota Digital", "Dakota Digital"]
    k2 = mvr_features(pool, OCR, k=2)
    k5 = mvr_features(pool, OCR, k=5)
    assert k2["mvr_unsupported_rate"] != k5["mvr_unsupported_rate"]
    assert set(k2) == set(k5) == set(MVR_FEATURES)


def test_mvr_requires_at_least_one_sample():
    try:
        mvr_features([], OCR, k=1)
    except ValueError as exc:
        assert "sampled answer" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_mvr_std_matches_manual_computation():
    import difflib

    pool = ["Dakota Digital", "Nikon"]
    feats = mvr_features(pool, OCR, k=2)
    ocr_norm = [normalize_text(t) for t in OCR]
    sims = [
        max(difflib.SequenceMatcher(None, normalize_text(s), o).ratio() for o in ocr_norm)
        for s in pool
    ]
    mean = sum(sims) / len(sims)
    expected = math.sqrt(sum((s - mean) ** 2 for s in sims) / len(sims))
    assert feats["mvr_fuzzy_mean"] == mean
    assert feats["mvr_fuzzy_min"] == min(sims)
    assert feats["mvr_fuzzy_best_of_k"] == max(sims)
    assert feats["mvr_fuzzy_std"] == expected
