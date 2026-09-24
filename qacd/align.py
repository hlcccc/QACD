"""Evidence-claim alignment features.

The frozen LM verifier can return a confident verdict together with an evidence
string that never mentions the claim. QACD therefore measures how well the
extracted evidence actually covers the claim, combining

* ``S`` — normalised character-level similarity between evidence and claim, and
* ``T`` — content-word coverage of the claim by the evidence.

Both signals are on the same ``[0, 1]`` scale and are combined with equal weight;
no weight was fitted. The feature family is part of the frozen vector and is
exposed so the platform can route on it.
"""

from __future__ import annotations

import difflib
import re
from typing import Dict, List, Sequence

__all__ = ["ALIGNMENT_FEATURES", "content_words", "alignment_features"]

ALIGNMENT_FEATURES: List[str] = [
    "align_char_similarity",
    "align_content_coverage",
    "align_score",
    "align_unsupported_risk",
]

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "but", "by", "for",
    "from", "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "there", "these", "this", "those", "to", "was", "were", "while", "with",
}


def content_words(text: str) -> List[str]:
    """Lower-cased content words (stopwords removed)."""
    return [w for w in _WORD_RE.findall(str(text or "").lower()) if w not in _STOPWORDS]


def _normalize(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def alignment_features(claim: str, evidence: str) -> Dict[str, float]:
    """Alignment features between one claim and one evidence string.

    Returns both an alignment direction (higher = better aligned) and its
    risk-oriented complement, so downstream direction alignment is a no-op.
    """
    c = _normalize(claim)
    e = _normalize(evidence)
    if not c or not e:
        return {
            "align_char_similarity": 0.0,
            "align_content_coverage": 0.0,
            "align_score": 0.0,
            "align_unsupported_risk": 1.0,
        }

    similarity = float(difflib.SequenceMatcher(None, c, e).ratio())

    cw = set(content_words(c))
    ew = set(content_words(e))
    coverage = float(len(cw & ew) / len(cw)) if cw else 0.0

    # Equal-weight combination, matching the reported feature construction: the
    # two signals are on the same [0, 1] scale and no weight was fitted.
    score = 0.5 * (similarity + coverage)
    return {
        "align_char_similarity": similarity,
        "align_content_coverage": coverage,
        "align_score": score,
        "align_unsupported_risk": 1.0 - score,
    }


def mean_alignment(claim: str, evidences: Sequence[str]) -> Dict[str, float]:
    """Average alignment over the available verification views."""
    rows = [alignment_features(claim, e) for e in evidences if str(e or "").strip()]
    if not rows:
        return alignment_features(claim, "")
    keys = rows[0].keys()
    n = float(len(rows))
    return {k: float(sum(r[k] for r in rows) / n) for k in keys}
