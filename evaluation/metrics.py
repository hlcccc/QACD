"""Ranking and calibration metrics — NumPy only.

Implemented here rather than pulled from scikit-learn so that the evaluation
path has the same dependency footprint as the decision layer: the reported
numbers can be recomputed on any machine with NumPy installed.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

__all__ = [
    "auroc",
    "weighted_auroc",
    "brier",
    "ece",
    "reliability_curve",
    "image_level",
    "paired_image_bootstrap",
]


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve, with correct handling of tied scores."""
    return weighted_auroc(labels, scores, np.ones(len(np.asarray(scores).reshape(-1))))


def weighted_auroc(labels: Sequence[int], scores: Sequence[float], weights: Sequence[float]) -> float:
    """Weighted AUROC, used by the clustered bootstrap.

    Vectorised: the clustered bootstrap calls this tens of thousands of times,
    so the tie handling is expressed as cumulative sums over the sorted score
    order rather than as a Python loop. Ties contribute one half, matching the
    Mann-Whitney formulation.
    """
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not (len(y) == len(s) == len(w)):
        raise ValueError("labels, scores and weights must have the same length")

    pos = y == 1
    neg = ~pos
    w_pos = float(w[pos].sum())
    w_neg = float(w[neg].sum())
    if w_pos <= 0.0 or w_neg <= 0.0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    w_sorted = w[order]

    neg_weight = np.where(y_sorted == 0, w_sorted, 0.0)
    # Negatives strictly before each position in sorted order.
    cum_before = np.cumsum(neg_weight) - neg_weight

    # Collapse tie blocks: every member of a block shares the same baseline and
    # the same half-weight of the negatives inside the block.
    n = len(s_sorted)
    is_start = np.empty(n, dtype=bool)
    is_start[0] = True
    if n > 1:
        is_start[1:] = s_sorted[1:] != s_sorted[:-1]
    block_id = np.cumsum(is_start) - 1
    n_blocks = int(block_id[-1]) + 1
    block_base = cum_before[is_start]
    neg_in_block = np.bincount(block_id, weights=neg_weight, minlength=n_blocks)

    contribution = np.where(
        y_sorted == 1,
        w_sorted * (block_base[block_id] + 0.5 * neg_in_block[block_id]),
        0.0,
    )
    return float(contribution.sum() / (w_pos * w_neg))


def brier(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Brier score (mean squared error of the probability)."""
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(y) != len(p):
        raise ValueError("labels and probabilities must have the same length")
    return float(np.mean((p - y) ** 2))


def reliability_curve(
    labels: Sequence[int],
    probabilities: Sequence[float],
    n_bins: int = 15,
    strategy: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin ``(mean_confidence, empirical_accuracy, count)``.

    Empty bins are dropped rather than reported as zero, so the returned arrays
    can be plotted directly without inventing points.

    Parameters
    ----------
    n_bins:
        Number of bins. 15 equal-width bins is the convention used in the
        calibration literature (Guo et al., 2017); equal-mass bins are available
        because a score distribution concentrated in a narrow range -- which is
        what a saturated risk map looks like -- leaves most equal-width bins
        empty and makes the estimate noisy.
    strategy:
        ``"uniform"`` for equal-width bins over ``[0, 1]``, ``"quantile"`` for
        equal-count bins.
    """
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(y) != len(p):
        raise ValueError("labels and probabilities must have the same length")
    if len(y) == 0:
        raise ValueError("no observations")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if strategy not in ("uniform", "quantile"):
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    if float(p.min()) < 0.0 or float(p.max()) > 1.0:
        raise ValueError(
            "probabilities must lie in [0, 1]; ECE is undefined for raw scores. "
            "Map them through a calibrator first."
        )

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        # np.digitize puts p == 1.0 out of range; clamp it into the last bin.
        index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    else:
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(p, quantiles)
        index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    confidences, accuracies, counts = [], [], []
    for b in range(n_bins):
        mask = index == b
        n = int(mask.sum())
        if n == 0:
            continue
        confidences.append(float(p[mask].mean()))
        accuracies.append(float(y[mask].mean()))
        counts.append(n)
    return np.asarray(confidences), np.asarray(accuracies), np.asarray(counts)


def ece(
    labels: Sequence[int],
    probabilities: Sequence[float],
    n_bins: int = 15,
    strategy: str = "uniform",
) -> float:
    """Expected Calibration Error: the count-weighted mean gap between the
    predicted probability and the observed frequency.

    ``ECE = sum_b (n_b / N) * |acc(b) - conf(b)|``

    This is the metric the task book's "校准性能" indicator is measured with, so
    it is the number that decides whether a calibration gain can be claimed. Two
    properties matter when quoting it:

    * it is only defined for genuine probabilities in ``[0, 1]``. A raw score
      (UMPIRE's, for instance, which ranges over roughly ``[-7, 6]``) has to be
      mapped through a calibrator first, or the comparison is meaningless;
    * it is **binning-dependent**. ``n_bins`` and ``strategy`` change the value,
      so any reported gain must state both, and the same settings must be used on
      both sides of the comparison. Defaults here are 15 equal-width bins.
    """
    confidences, accuracies, counts = reliability_curve(
        labels, probabilities, n_bins, strategy
    )
    if counts.sum() == 0:
        raise ValueError("no observations")
    return float(np.sum(counts / counts.sum() * np.abs(accuracies - confidences)))


def image_level(
    response_scores: Sequence[float],
    response_labels: Sequence[int],
    response_images: Sequence[str],
    default: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse responses to images: score = max, label = any failure.

    This is the primary control unit of the frozen conformal protocol.
    ``default`` is only used for an image that owns no responses.
    """
    scores = np.asarray(response_scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(response_labels, dtype=np.int64).reshape(-1)
    images = np.asarray(response_images).astype(str).reshape(-1)
    if not (len(scores) == len(labels) == len(images)):
        raise ValueError("scores, labels and images must have the same length")

    unique = np.unique(images)
    index = {name: i for i, name in enumerate(unique)}
    mapped = np.array([index[name] for name in images], dtype=np.int64)

    # Initialise with the identity for ``max``/``any``, NOT with ``default``:
    # a zero initialiser silently clips negative scores (UMPIRE's raw scores can
    # be negative) and would understate the image-level ranking.
    out_scores = np.full(len(unique), -np.inf, dtype=np.float64)
    out_labels = np.zeros(len(unique), dtype=np.int64)
    np.maximum.at(out_scores, mapped, scores)
    np.maximum.at(out_labels, mapped, labels)

    empty = ~np.isfinite(out_scores)
    if empty.any():
        out_scores[empty] = float(default)
    return out_scores, out_labels


def paired_image_bootstrap(
    labels: Sequence[int],
    candidate: Sequence[float],
    baseline: Sequence[float],
    images: Sequence[str],
    replicates: int = 5000,
    seed: int = 20260920,
) -> dict:
    """Paired image-clustered bootstrap of the AUROC difference.

    Whole images are resampled with replacement, so responses sharing an image
    move together — the dependence structure the frozen protocol controls for.
    """
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    a = np.asarray(candidate, dtype=np.float64).reshape(-1)
    b = np.asarray(baseline, dtype=np.float64).reshape(-1)
    img = np.asarray(images).astype(str).reshape(-1)
    if not (len(y) == len(a) == len(b) == len(img)):
        raise ValueError("labels, candidate, baseline and images must have the same length")

    unique, inverse = np.unique(img, return_inverse=True)
    n_groups = len(unique)
    rng = np.random.default_rng(int(seed))

    deltas = []
    for _ in range(int(replicates)):
        counts = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
        w = counts[inverse].astype(np.float64)
        if w[y == 0].sum() == 0 or w[y == 1].sum() == 0:
            continue
        deltas.append(weighted_auroc(y, a, w) - weighted_auroc(y, b, w))

    d = np.asarray(deltas, dtype=np.float64)
    if d.size == 0:
        raise RuntimeError("no valid bootstrap replicate")
    return {
        "delta": float(auroc(y, a) - auroc(y, b)),
        "ci_low": float(np.quantile(d, 0.025)),
        "ci_high": float(np.quantile(d, 0.975)),
        "p_nonpositive": float(np.mean(d <= 0.0)),
        "replicates_valid": int(d.size),
        "replicates_requested": int(replicates),
    }
