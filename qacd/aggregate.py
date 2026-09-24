"""Claim-to-response aggregation.

A response may carry several claims. QACD projects the per-claim risks back to a
single answer-level score with a **maximum operator**:

    r_i = max_j r_hat_ij

which implements the conservative "warn if any claim looks unsupported" rule.
The maximum is not a tuned choice between alternatives; the research pipeline
fixed it in advance. It is nevertheless a real modelling decision with a known
cost: one badly parsed claim can lift the score of an otherwise correct answer.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

__all__ = ["claim_to_response_max", "response_risk", "DEFAULT_THRESHOLD"]

#: Default warning threshold. 0.5 is the natural probability cut and is used for
#: reporting only — the research pipeline evaluates ranking (AUROC/AUPRC) and
#: probability quality (Brier/NLL), never a tuned operating point.
DEFAULT_THRESHOLD = 0.5


def claim_to_response_max(
    claim_scores: Sequence[float],
    num_responses: int,
    claim_response_index: Sequence[int],
    default: float = 0.0,
) -> np.ndarray:
    """Reduce per-claim risks to per-response risks with ``max``.

    Parameters
    ----------
    claim_scores:
        One risk per claim row, in claim-table order.
    num_responses:
        Number of responses the claims belong to.
    claim_response_index:
        For each claim row, the index of its response in ``[0, num_responses)``.
    default:
        Value used for a response that owns no claim rows.
    """
    scores = np.asarray(claim_scores, dtype=np.float64).reshape(-1)
    index = np.asarray(claim_response_index, dtype=np.int64).reshape(-1)
    if scores.shape[0] != index.shape[0]:
        raise ValueError(
            f"claim_scores has {scores.shape[0]} entries but claim_response_index has {index.shape[0]}"
        )
    if num_responses < 0:
        raise ValueError("num_responses must be non-negative")
    if index.size and (index.min() < 0 or index.max() >= num_responses):
        raise ValueError("claim_response_index contains an out-of-range response index")

    out = np.full(int(num_responses), float(default), dtype=np.float64)
    if index.size:
        np.maximum.at(out, index, scores)
    return out


def response_risk(
    claim_scores: Sequence[float],
    claim_response_index: Sequence[int],
    num_responses: int,
    threshold: float = DEFAULT_THRESHOLD,
    default: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(risk_scores, is_high_risk)`` for every response."""
    scores = claim_to_response_max(claim_scores, num_responses, claim_response_index, default=default)
    return scores, scores >= float(threshold)


def channel_summary(names: Sequence[str], values: Sequence[float]) -> Dict[str, float]:
    """Small helper for the audit payload returned next to the risk score."""
    return {str(n): round(float(v), 6) for n, v in zip(names, values)}
