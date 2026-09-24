"""Risk calibration.

The calibrator maps a direction-aligned claim feature vector to a claim-level
error probability. QACD deliberately separates *evidence construction* from
*probability mapping*: the calibrator is a small, L2-regularised logistic model
fitted only on development labels.

``BICLiteCalibrator``
    The frozen TextVQA configuration (``l2 = 0.05``). Numerically this minimises

        mean( log(1 + exp(z)) - y * z )  +  l2 * ||w||^2 ,   z = b + X w

    over median-imputed, z-scored features. Note that the penalty is added to the
    *mean* negative log-likelihood, not the sum.

``RidgeLogistic``
    The same optimiser exposed for the response-level MVR fusion head, where the
    research pipeline used ``sklearn.linear_model.LogisticRegression(C=1.0)``.
    Use :meth:`RidgeLogistic.from_sklearn_C` to reproduce that penalty exactly.

Why not scikit-learn? The decision layer has to be installable and unit-testable
on the platform's CPU nodes with no research stack. The optimiser below is a
damped Newton (IRLS) solver over ``d + 1`` parameters; with d <= a few hundred
that is exact, deterministic and takes milliseconds.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

__all__ = ["sigmoid", "logit", "BICLiteCalibrator", "RidgeLogistic"]


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logit(p: float) -> float:
    """Inverse logistic, clipped away from the poles."""
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def _as_2d(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D feature matrix.")
    return arr


class RidgeLogistic:
    """L2-regularised logistic regression on z-scored features (Newton solver).

    Parameters
    ----------
    l2:
        Ridge coefficient added to the mean negative log-likelihood.
    max_iter, tol:
        Newton iteration budget and gradient-norm stopping threshold.
    """

    def __init__(self, l2: float = 0.05, max_iter: int = 200, tol: float = 1e-9):
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.success_: bool = False
        self.fallback_: float = 0.5
        self.n_iter_: int = 0

    # -- standardisation ---------------------------------------------------
    @staticmethod
    def _standardize_fit(features: np.ndarray):
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(features, axis=0)
            std = np.nanstd(features, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
        return mean, std

    @staticmethod
    def _standardize_apply(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return (clean - mean) / std

    # -- objective ---------------------------------------------------------
    def _objective(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
        z = X @ w + b
        nll = float(np.mean(np.logaddexp(0.0, z) - y * z))
        return nll + self.l2 * float(np.sum(w ** 2))

    def fit(self, features, labels) -> "RidgeLogistic":
        X = _as_2d(features)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]} labels")

        self.fallback_ = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6)) if y.size else 0.5
        if y.size == 0 or np.unique(y.astype(int)).size < 2:
            self.success_ = False
            self.coef_ = None
            return self

        self.mean_, self.std_ = self._standardize_fit(X)
        Xs = self._standardize_apply(X, self.mean_, self.std_)
        n, d = Xs.shape

        w = np.zeros(d, dtype=np.float64)
        b = logit(self.fallback_)
        obj = self._objective(Xs, y, w, b)
        eye = np.eye(d + 1)

        self.success_ = False
        for it in range(self.max_iter):
            self.n_iter_ = it + 1
            z = Xs @ w + b
            p = sigmoid(z)
            resid = p - y
            W = p * (1.0 - p)

            g_w = Xs.T @ resid / n + 2.0 * self.l2 * w
            g_b = float(np.mean(resid))
            grad = np.concatenate([g_w, [g_b]])

            if np.max(np.abs(grad)) < self.tol:
                self.success_ = True
                break

            H_ww = (Xs * W[:, None]).T @ Xs / n + 2.0 * self.l2 * np.eye(d)
            H_wb = Xs.T @ W / n
            H_bb = float(np.mean(W))
            H = np.empty((d + 1, d + 1), dtype=np.float64)
            H[:d, :d] = H_ww
            H[:d, d] = H_wb
            H[d, :d] = H_wb
            H[d, d] = H_bb

            try:
                step = np.linalg.solve(H + 1e-10 * eye, grad)
            except np.linalg.LinAlgError:
                step = grad

            # Backtracking line search on the penalised objective.
            t = 1.0
            improved = False
            for _ in range(60):
                w_new = w - t * step[:d]
                b_new = b - t * step[d]
                cand = self._objective(Xs, y, w_new, b_new)
                if cand <= obj - 1e-4 * t * float(grad @ step) or cand < obj:
                    w, b, obj = w_new, b_new, cand
                    improved = True
                    break
                t *= 0.5
            if not improved:
                self.success_ = True
                break

        self.coef_ = np.concatenate([[b], w])
        return self

    def predict_proba(self, features) -> np.ndarray:
        X = _as_2d(features)
        if self.coef_ is None or self.mean_ is None or self.std_ is None:
            return np.full(X.shape[0], self.fallback_, dtype=np.float64)
        Xs = self._standardize_apply(X, self.mean_, self.std_)
        return np.clip(sigmoid(Xs @ self.coef_[1:] + self.coef_[0]), 1e-6, 1.0 - 1e-6)

    # Alias kept so the object is a drop-in for the research code path.
    predict = predict_proba

    def weights(self, feature_names: Sequence[str]) -> List[Dict[str, float]]:
        """Standardised coefficients, largest magnitude first."""
        if self.coef_ is None:
            return []
        rows = [
            {"feature": str(name), "weight_standardized": float(w)}
            for name, w in zip(feature_names, self.coef_[1:])
        ]
        return sorted(rows, key=lambda r: -abs(r["weight_standardized"]))

    @classmethod
    def from_sklearn_C(cls, C: float, n_samples: int, **kwargs) -> "RidgeLogistic":
        """Match ``sklearn.linear_model.LogisticRegression(C=C)``.

        scikit-learn minimises ``0.5 * ||w||^2 + C * sum(loss_i)``. Dividing by
        ``n`` puts that in the mean-loss convention used here, giving
        ``l2 = 1 / (2 * C * n)``.
        """
        if C <= 0:
            raise ValueError("C must be positive")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        return cls(l2=1.0 / (2.0 * float(C) * float(n_samples)), **kwargs)


class BICLiteCalibrator(RidgeLogistic):
    """The frozen QACD claim-risk calibrator (``l2 = 0.05``)."""

    def __init__(self, l2: float = 0.05, max_iter: int = 200, tol: float = 1e-9):
        super().__init__(l2=l2, max_iter=max_iter, tol=tol)
