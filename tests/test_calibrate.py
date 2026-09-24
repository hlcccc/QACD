"""Calibrator: correctness, determinism, and the research penalty convention."""

import numpy as np
import pytest

from qacd.calibrate import BICLiteCalibrator, RidgeLogistic, logit, sigmoid


def _separable(n=200, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    # Risk increases with the first two coordinates.
    logits = 1.6 * X[:, 0] + 1.1 * X[:, 1] - 0.2
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < p).astype(int)
    return X, y


def _auc(y, scores):
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def test_sigmoid_and_logit_are_inverse():
    z = np.array([-6.0, -1.0, 0.0, 1.0, 6.0])
    assert np.allclose(sigmoid(z), [1 / (1 + np.exp(6)), 1 / (1 + np.exp(1)), 0.5, 1 / (1 + np.exp(-1)), 1 / (1 + np.exp(-6))])
    assert sigmoid(np.array([1000.0]))[0] == pytest.approx(1.0)
    assert sigmoid(np.array([-1000.0]))[0] == pytest.approx(0.0)
    assert logit(sigmoid(np.array([0.7]))[0]) == pytest.approx(0.7, abs=1e-6)


def test_learns_the_signal():
    X, y = _separable()
    model = BICLiteCalibrator(l2=0.05).fit(X, y)
    assert model.success_
    assert _auc(y, model.predict(X)) > 0.8
    # The first two coordinates carry the signal, so they must get the weight.
    assert abs(model.coef_[1]) > abs(model.coef_[3])
    assert abs(model.coef_[2]) > abs(model.coef_[3])


def test_fit_is_deterministic():
    X, y = _separable()
    a = BICLiteCalibrator(l2=0.05).fit(X, y).coef_
    b = BICLiteCalibrator(l2=0.05).fit(X, y).coef_
    assert np.allclose(a, b)


def test_single_class_falls_back_to_the_base_rate():
    X = np.random.default_rng(0).normal(size=(30, 2))
    y = np.ones(30, dtype=int)
    model = BICLiteCalibrator().fit(X, y)
    assert not model.success_
    preds = model.predict(X)
    assert np.allclose(preds, np.clip(1.0, 1e-6, 1 - 1e-6))


def test_predictions_are_strictly_inside_the_unit_interval():
    X, y = _separable()
    preds = BICLiteCalibrator().fit(X, y).predict(X)
    assert preds.min() > 0.0 and preds.max() < 1.0


def test_unfitted_predict_is_the_fallback_constant():
    X = np.zeros((5, 2))
    preds = RidgeLogistic().predict(X)
    assert np.allclose(preds, 0.5)


def test_nan_features_are_imputed_not_propagated():
    X, y = _separable()
    X = X.copy()
    X[::7, 0] = np.nan
    model = BICLiteCalibrator().fit(X, y)
    preds = model.predict(X)
    assert np.isfinite(preds).all()


def test_standardisation_makes_predictions_scale_invariant():
    X, y = _separable()
    a = BICLiteCalibrator(l2=0.05).fit(X, y)
    b = BICLiteCalibrator(l2=0.05).fit(X * 1000.0, y)
    # z-scoring removes the input scale, so both models must produce the same
    # probabilities for the same physical rows.
    assert np.allclose(a.coef_, b.coef_, atol=1e-9)
    assert np.allclose(a.predict(X), b.predict(X * 1000.0), atol=1e-9)


def test_stronger_l2_shrinks_weights():
    X, y = _separable()
    weak = BICLiteCalibrator(l2=0.001).fit(X, y)
    strong = BICLiteCalibrator(l2=1.0).fit(X, y)
    assert np.linalg.norm(strong.coef_[1:]) < np.linalg.norm(weak.coef_[1:])


def test_sklearn_penalty_conversion():
    # LogisticRegression(C=1.0) on n samples == mean-loss l2 of 1/(2n).
    n = 500
    model = RidgeLogistic.from_sklearn_C(1.0, n)
    assert model.l2 == pytest.approx(1.0 / (2 * n))


def test_weights_are_sorted_by_magnitude():
    X, y = _separable()
    names = ["a", "b", "c"]
    rows = BICLiteCalibrator().fit(X, y).weights(names)
    mags = [abs(r["weight_standardized"]) for r in rows]
    assert mags == sorted(mags, reverse=True)
    assert {r["feature"] for r in rows} == set(names)
