"""Split-conformal selective prediction and abstention.

Given a calibration set of risk scores for *known-correct* responses, a
conformal p-value can be attached to every test score:

    p_j = (1 + #{ i in calibration : s_i >= s_j }) / (n_cal + 1)

A selection procedure then chooses which responses to **accept** (keep) while
controlling a chosen error rate, using either

* ``BH`` — Benjamini-Hochberg, valid under independence or PRDS, or
* ``BY`` — Benjamini-Yekutieli, valid under arbitrary dependence. This is the
  procedure QACD uses for its primary analysis because responses cluster within
  images.

--------------------------------------------------------------------------
This module is a *reference implementation* of the procedure. It exposes the
p-value construction and the multiplicity-control machinery so that a caller can
run the selection on its own calibration batch and read coverage, retention and
realised FDP directly off the returned object.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

__all__ = [
    "conformal_pvalues",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "conformal_select",
    "SelectionResult",
    "VALIDATED",
]

#: Status flag carried on every result so that callers can gate on it in code.
VALIDATED = False


def conformal_pvalues(calibration_null_scores: Sequence[float], test_scores: Sequence[float]) -> np.ndarray:
    """Split-conformal p-values against a calibration set of null scores.

    Parameters
    ----------
    calibration_null_scores:
        Risk scores of calibration items known to be *correct*. Higher score
        means higher risk, so a null item scoring high is the conservative case.
    test_scores:
        Risk scores of the items being tested.
    """
    cal = np.asarray(calibration_null_scores, dtype=np.float64).reshape(-1)
    if cal.size == 0:
        raise ValueError("calibration_null_scores must be non-empty")
    cal = np.sort(cal)
    test = np.asarray(test_scores, dtype=np.float64).reshape(-1)

    # Number of calibration scores >= each test score, computed by searchsorted
    # on the ascending array: #{s_i >= t} = n - searchsorted(cal, t, side='left').
    n = cal.size
    ge = n - np.searchsorted(cal, test, side="left")
    return (1.0 + ge) / (n + 1.0)


def benjamini_hochberg(pvalues: Sequence[float], alpha: float) -> np.ndarray:
    """BH step-up. Returns a boolean accept mask (``p <= threshold``)."""
    p = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    thresholds = alpha * (np.arange(1, m + 1) / m)
    below = ranked <= thresholds
    mask = np.zeros(m, dtype=bool)
    if below.any():
        k = int(np.max(np.nonzero(below)[0]))
        cutoff = ranked[k]
        mask = p <= cutoff
    return mask


def benjamini_yekutieli(pvalues: Sequence[float], alpha: float) -> np.ndarray:
    """BY step-up — BH with the harmonic correction ``H_m``."""
    p = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    thresholds = alpha * (np.arange(1, m + 1) / (m * harmonic))
    below = ranked <= thresholds
    mask = np.zeros(m, dtype=bool)
    if below.any():
        k = int(np.max(np.nonzero(below)[0]))
        cutoff = ranked[k]
        mask = p <= cutoff
    return mask


@dataclass
class SelectionResult:
    """Outcome of one selective-prediction run."""

    accepted: List[bool]
    num_accepted: int
    num_items: int
    coverage: float
    procedure: str
    alpha: float
    realized_fdp: float | None = None
    validated: bool = VALIDATED
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "accepted": [bool(a) for a in self.accepted],
            "num_accepted": self.num_accepted,
            "num_items": self.num_items,
            "coverage": self.coverage,
            "procedure": self.procedure,
            "alpha": self.alpha,
            "realized_fdp": self.realized_fdp,
            "validated": self.validated,
            "notes": list(self.notes),
        }


def conformal_select(
    calibration_null_scores: Sequence[float],
    test_scores: Sequence[float],
    alpha: float = 0.10,
    procedure: str = "BY",
    test_labels: Sequence[int] | None = None,
) -> SelectionResult:
    """Accept a subset of test items at a target error rate.

    Parameters
    ----------
    test_labels:
        Optional ground truth, used only to *report* realised FDP. Passing labels
        never changes the selection, which is what makes the procedure
        label-blind at decision time.
    """
    procedure = str(procedure).upper()
    if procedure not in {"BH", "BY"}:
        raise ValueError("procedure must be 'BH' or 'BY'")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0, 1)")

    p = conformal_pvalues(calibration_null_scores, test_scores)
    mask = benjamini_hochberg(p, alpha) if procedure == "BH" else benjamini_yekutieli(p, alpha)

    num_items = int(p.size)
    num_accepted = int(mask.sum())
    coverage = float(num_accepted / num_items) if num_items else 0.0

    realized_fdp = None
    if test_labels is not None and num_accepted:
        y = np.asarray(test_labels, dtype=int).reshape(-1)
        if y.size != num_items:
            raise ValueError("test_labels length must match test_scores")
        realized_fdp = float((y[mask] == 1).sum() / num_accepted)

    notes = []
    if num_accepted == 0:
        notes.append(
            f"No item was accepted at alpha={float(alpha):g} under {procedure}. "
            "The supplied calibration set supports no selection at this error rate."
        )

    return SelectionResult(
        accepted=[bool(v) for v in mask],
        num_accepted=num_accepted,
        num_items=num_items,
        coverage=coverage,
        procedure=procedure,
        alpha=float(alpha),
        realized_fdp=realized_fdp,
        notes=notes,
    )
