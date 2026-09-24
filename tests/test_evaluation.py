"""Evaluation layer: metrics, image aggregation, bundle validation."""

import json

import numpy as np
import pytest

from evaluation.bundle import load_bundle, sha256_file
from evaluation.metrics import (
    auroc,
    brier,
    image_level,
    paired_image_bootstrap,
    weighted_auroc,
)


# -- metrics -------------------------------------------------------------

def test_auroc_is_one_for_a_perfect_ranking():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_auroc_is_zero_for_an_inverted_ranking():
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)


def test_auroc_is_one_half_for_a_constant_score():
    assert auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_auroc_counts_ties_as_one_half():
    # One positive tied with one negative: half a concordant pair out of two.
    assert auroc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)


def test_auroc_matches_a_brute_force_count():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    if y.min() == y.max():
        y[0], y[1] = 0, 1
    s = rng.random(60)
    brute = np.mean(
        [(s[i] > s[j]) + 0.5 * (s[i] == s[j]) for i in range(60) for j in range(60) if y[i] == 1 and y[j] == 0]
    )
    assert auroc(y, s) == pytest.approx(float(brute))


def test_weighted_auroc_equals_plain_auroc_under_unit_weights():
    y = [0, 1, 0, 1, 1]
    s = [0.1, 0.4, 0.35, 0.8, 0.2]
    assert weighted_auroc(y, s, np.ones(5)) == pytest.approx(auroc(y, s))


def test_weighted_auroc_ignores_zero_weight_rows():
    y = [0, 1, 0, 1]
    s = [0.1, 0.9, 0.95, 0.2]
    w = [1.0, 1.0, 0.0, 1.0]
    assert weighted_auroc(y, s, w) == pytest.approx(auroc([0, 1, 1], [0.1, 0.9, 0.2]))


def test_auroc_rejects_non_overlapping_mismatched_inputs():
    with pytest.raises(ValueError):
        weighted_auroc([0, 1], [0.1, 0.2], [1.0])


def test_brier_is_zero_for_perfect_probabilities():
    assert brier([0, 1], [0.0, 1.0]) == pytest.approx(0.0)


def test_brier_matches_the_definition():
    assert brier([0, 1, 1], [0.2, 0.7, 0.9]) == pytest.approx((0.04 + 0.09 + 0.01) / 3)


# -- image aggregation ---------------------------------------------------

def test_image_level_takes_the_max_and_any_failure():
    scores = [0.2, 0.7, 0.1]
    labels = [0, 0, 1]
    images = ["a", "a", "b"]
    s, y = image_level(scores, labels, images)
    assert sorted(s.tolist()) == [0.1, 0.7]
    assert sorted(y.tolist()) == [0, 1]


def test_image_level_does_not_clip_negative_scores():
    """Regression: a zero initialiser turned every negative score into 0.0,
    which understated UMPIRE's image-level ranking."""
    scores = [-5.0, -3.0, -9.0, -1.0]
    labels = [0, 0, 1, 1]
    images = ["a", "a", "b", "b"]
    s, y = image_level(scores, labels, images)
    # max per image: a -> -3.0, b -> -1.0. Both stay negative.
    assert s.tolist() == [-3.0, -1.0]
    assert (s < 0).all(), "negative scores must survive aggregation"


def test_image_level_agrees_between_negative_and_shifted_positive_scores():
    scores = [-5.0, -3.0, -9.0, -1.0]
    labels = [0, 0, 1, 1]
    images = ["a", "a", "b", "b"]
    neg_scores, neg_labels = image_level(scores, labels, images)
    pos_scores, pos_labels = image_level(np.asarray(scores) + 100.0, labels, images)
    assert auroc(neg_labels, neg_scores) == pytest.approx(auroc(pos_labels, pos_scores))


def test_image_level_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        image_level([0.1, 0.2], [0, 1], ["a"])


# -- paired bootstrap ----------------------------------------------------

def test_paired_bootstrap_reports_a_positive_delta_for_a_better_scorer():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    good = y + rng.normal(0, 0.4, 400)
    bad = rng.normal(0, 1.0, 400)
    images = [f"img{i // 2}" for i in range(400)]
    result = paired_image_bootstrap(y, good, bad, images, replicates=300, seed=7)
    assert result["delta"] > 0
    assert result["ci_low"] <= result["delta"] <= result["ci_high"] or result["ci_low"] < result["ci_high"]


def test_paired_bootstrap_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200)
    a = rng.random(200)
    b = rng.random(200)
    images = [f"img{i // 4}" for i in range(200)]
    first = paired_image_bootstrap(y, a, b, images, replicates=200, seed=11)
    second = paired_image_bootstrap(y, a, b, images, replicates=200, seed=11)
    assert first == second


def test_paired_bootstrap_of_a_scorer_against_itself_centres_on_zero():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    a = rng.random(300)
    images = [f"img{i // 3}" for i in range(300)]
    result = paired_image_bootstrap(y, a, a, images, replicates=200, seed=5)
    assert result["delta"] == pytest.approx(0.0)
    assert result["ci_low"] <= 0.0 <= result["ci_high"]


# -- bundle loading ------------------------------------------------------

def _make_bundle(tmp_path, n_dev=40, n_test=20, n_features=3, tamper=False):
    rng = np.random.default_rng(0)
    lm_names = [f"lm_{i}" for i in range(n_features - 1)]
    mech_names = ["mech_0"]
    dev_matrix = rng.normal(size=(n_dev, n_features))
    test_matrix = rng.normal(size=(n_test, n_features))
    dev_response_y = rng.integers(0, 2, n_dev).astype(np.int64)
    test_response_y = rng.integers(0, 2, n_test).astype(np.int64)
    dev_images = np.array([f"d{i // 2}" for i in range(n_dev)])
    test_images = np.array([f"t{i // 2}" for i in range(n_test)])

    path = tmp_path / "evidence_bundle.npz"
    np.savez_compressed(
        path,
        dev_matrix=dev_matrix,
        test_matrix=test_matrix,
        dev_claim_labels=np.tile(dev_response_y, 1).astype(np.int64),
        dev_claim_train_mask=np.ones(n_dev, dtype=bool),
        dev_claim_response_index=np.arange(n_dev, dtype=np.int64),
        dev_response_ids=np.arange(n_dev, dtype=np.int64),
        dev_response_y=dev_response_y,
        dev_response_images=dev_images,
        test_claim_response_index=np.arange(n_test, dtype=np.int64),
        test_response_ids=np.arange(n_test, dtype=np.int64),
        test_response_y=test_response_y,
        test_response_images=test_images,
        dev_mvr=rng.normal(size=(n_dev, 7)),
        test_mvr=rng.normal(size=(n_test, 7)),
        lm_names=np.array(lm_names),
        mechanical_names=np.array(mech_names),
        mvr_names=np.array([f"mvr_{i}" for i in range(7)]),
        baseline_names=np.array(["UMPIRE K=5"]),
        baseline_dev=rng.random((1, n_dev)),
        baseline_test=rng.random((1, n_test)),
        seed=np.array([20260920], dtype=np.int64),
    )
    manifest = {"bundle_sha256": "0" * 64 if tamper else sha256_file(path)}
    (tmp_path / "bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_bundle_loads_and_exposes_feature_names(tmp_path):
    path = _make_bundle(tmp_path)
    bundle = load_bundle(path)
    assert bundle.feature_names == bundle.lm_names + bundle.mechanical_names
    assert len(bundle.feature_names) == 3
    assert bundle.select(["mech_0", "lm_0"]) == [2, 0]


def test_bundle_hash_mismatch_is_rejected(tmp_path):
    path = _make_bundle(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_bundle(path)


def test_bundle_can_skip_verification_explicitly(tmp_path):
    path = _make_bundle(tmp_path, tamper=True)
    assert load_bundle(path, verify=False) is not None


def test_bundle_rejects_an_unknown_feature_name(tmp_path):
    bundle = load_bundle(_make_bundle(tmp_path))
    with pytest.raises(KeyError):
        bundle.select(["does_not_exist"])


def test_bundle_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "nope.npz")


def test_bundle_rejects_inconsistent_shapes(tmp_path):
    path = _make_bundle(tmp_path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["dev_claim_labels"] = arrays["dev_claim_labels"][:-1]
    np.savez_compressed(path, **arrays)
    (tmp_path / "bundle_manifest.json").write_text(
        json.dumps({"bundle_sha256": sha256_file(path)}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="disagree on claim count"):
        load_bundle(path)
