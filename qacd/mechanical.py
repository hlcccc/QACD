"""Non-LM evidence channels: mechanical OCR instruments and MVR.

Both channels exist because the frozen LM verifier is over-confident and its
residual signal is largely lexical. They replace an LM judgement with a
*deterministic comparison* between the claim string and text actually read off
the image.

``mechanical_ocr_features``
    Claim-versus-OCR-text overlap statistics (28 features in the research
    pipeline, of which 20 are produced here directly; the remaining instrument
    families are CLIP and counting probes and live behind the same interface).

``mvr_features``
    Multi-View Resampling. K sampled generations from the *same* prompt are each
    checked against the image OCR text; the spread of the resulting support
    indicators is the stability signal.

Ported from ``build_mechanical_features.py`` and ``own_component_full.py`` in the
research repository.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Dict, Iterable, List, Sequence

__all__ = [
    "MVR_FEATURES",
    "MECHANICAL_FEATURES",
    "MECHANICAL_FAMILY_SIZE",
    "normalize_text",
    "tokens",
    "mechanical_ocr_features",
    "mvr_features",
]

_PUNCT = re.compile(r"[^a-z0-9]+")
_ART = re.compile(r"\b(a|an|the)\b")
_NUM = re.compile(r"\d+(?:[.:]\d+)?")

#: Feature order consumed by the frozen calibration coefficients.
MECHANICAL_FEATURES: List[str] = [
    "mech_ocr_n_strings",
    "mech_ocr_text_density",
    "mech_ocr_empty",
    "mech_ocr_exact_present",
    "mech_ocr_containment_present",
    "mech_ocr_all_words_present",
    "mech_ocr_best_similarity",
    "mech_ocr_best_similarity_conf",
    "mech_ocr_best_token_recall",
    "mech_ocr_best_token_precision",
    "mech_ocr_best_match_confidence",
    "mech_ocr_char_coverage",
    "mech_ocr_has_number",
    "mech_ocr_number_present",
    "mech_ocr_number_missing_risk",
    "mech_ocr_absent_risk",
    "mech_ocr_low_similarity_risk",
    "mech_ocr_low_recall_risk",
    "mech_ocr_low_char_coverage_risk",
]

#: Size of the *complete* mechanical instrument family used by the frozen
#: TextVQA configuration. This module implements the OCR sub-family (19
#: features, listed above); the remainder are CLIP probes and deterministic
#: counting probes, which plug in through the same feature-dict contract.
MECHANICAL_FAMILY_SIZE = 28

#: Feature order of the MVR channel.
MVR_FEATURES: List[str] = [
    "mvr_unsupported_rate",
    "mvr_all_unsupported",
    "mvr_all_supported",
    "mvr_fuzzy_mean",
    "mvr_fuzzy_min",
    "mvr_fuzzy_std",
    "mvr_fuzzy_best_of_k",
]

#: Motivation for the channel, as measured in the research diagnostics: the
#: claim text is present in the image for 64.8% of correct answers versus 37.1%
#: of wrong ones, while the frozen LM verifier answers "supported" on ~94% of
#: responses and therefore carries little discriminative signal on its own.
_OCR_PRESENCE_CORRECT = 0.648
_OCR_PRESENCE_WRONG = 0.371


def normalize_text(value: Any) -> str:
    """Lower-case, drop articles and punctuation, collapse whitespace."""
    text = str(value or "").strip().lower()
    text = _ART.sub(" ", text)
    return " ".join(_PUNCT.sub(" ", text).split())


def tokens(value: Any) -> List[str]:
    """Token list of :func:`normalize_text`."""
    return [w for w in normalize_text(value).split() if w]


def mechanical_ocr_features(claim: str, ocr_texts: Sequence[str], ocr_scores: Sequence[float] | None = None) -> Dict[str, float]:
    """Compare one claim against the OCR strings read from its image.

    Parameters
    ----------
    claim:
        Claim text (or the raw answer when no decomposition is wanted).
    ocr_texts:
        Text boxes returned by the OCR engine for the image.
    ocr_scores:
        Optional per-box confidence. When omitted every box is weighted 1.0.

    Returns
    -------
    dict
        Support-oriented and risk-oriented features. Both directions are emitted
        so that downstream direction-alignment code works unchanged.
    """
    texts = [str(t) for t in (ocr_texts or [])]
    scores = [float(s) for s in (ocr_scores or [])]
    if len(scores) != len(texts):
        scores = [1.0] * len(texts)

    c = normalize_text(claim)
    ctoks = tokens(claim)
    ntexts = [t for t in (normalize_text(t) for t in texts) if t]
    n_numbers = set(_NUM.findall(c))

    exact = 1.0 if c and c in ntexts else 0.0
    contain = 0.0
    best_sim = 0.0
    best_sim_conf = 0.0
    best_recall = 0.0
    best_prec = 0.0
    best_conf = 0.0

    for raw, nt, sc in zip(texts, (normalize_text(t) for t in texts), scores):
        if not nt:
            continue
        if c and (c in nt or nt in c):
            contain = 1.0
        sim = difflib.SequenceMatcher(None, c, nt).ratio() if c else 0.0
        if sim > best_sim:
            best_sim, best_sim_conf, best_conf = sim, sim * sc, sc
        nt_toks = set(nt.split())
        if ctoks:
            inter = len(set(ctoks) & nt_toks)
            rec = inter / len(set(ctoks))
            if rec > best_recall:
                best_recall, best_prec = rec, inter / max(len(nt_toks), 1)

    all_words = 0.0
    if ctoks:
        joined = " ".join(ntexts)
        all_words = float(all(w in joined for w in ctoks))

    cchars = set(c.replace(" ", ""))
    ochars = set("".join(ntexts).replace(" ", ""))
    char_cov = float(len(cchars & ochars) / len(cchars)) if cchars else 0.0

    joined_text = " ".join(ntexts)
    num_present = 1.0 if n_numbers and all(n in joined_text for n in n_numbers) else 0.0
    num_missing = 1.0 if n_numbers and not num_present else 0.0

    features = {
        "mech_ocr_n_strings": float(len(ntexts)),
        "mech_ocr_text_density": float(len(joined_text.split())),
        "mech_ocr_empty": 1.0 if not ntexts else 0.0,
        "mech_ocr_exact_present": exact,
        "mech_ocr_containment_present": contain,
        "mech_ocr_all_words_present": all_words,
        "mech_ocr_best_similarity": best_sim,
        "mech_ocr_best_similarity_conf": best_sim_conf,
        "mech_ocr_best_token_recall": best_recall,
        "mech_ocr_best_token_precision": best_prec,
        "mech_ocr_best_match_confidence": best_conf,
        "mech_ocr_char_coverage": char_cov,
        "mech_ocr_has_number": 1.0 if n_numbers else 0.0,
        "mech_ocr_number_present": num_present,
        "mech_ocr_number_missing_risk": num_missing,
        "mech_ocr_absent_risk": 1.0 - max(exact, contain, all_words),
        "mech_ocr_low_similarity_risk": 1.0 - best_sim,
        "mech_ocr_low_recall_risk": 1.0 - best_recall,
        "mech_ocr_low_char_coverage_risk": 1.0 - char_cov,
    }
    return features


def _mechanical_support_bit(sampled: str, ocr_norm: Iterable[str]) -> float:
    """Decide whether one sampled answer string is backed by the image text."""
    toks = [t for t in ocr_norm if t]
    c = normalize_text(sampled)
    if not c or not toks:
        return 0.0
    if c in toks or any(c == o or c in o or o in c for o in toks):
        return 1.0
    words = [x for x in c.split() if len(x) > 1]
    joined = " ".join(toks)
    return 1.0 if words and all(x in joined for x in words) else 0.0


def mvr_features(sampled_answers: Sequence[str], ocr_texts: Sequence[str], k: int | None = None) -> Dict[str, float]:
    """Multi-View Resampling statistics for one response.

    Parameters
    ----------
    sampled_answers:
        K sampled generations for the *same* prompt, in sampling order.
    ocr_texts:
        OCR strings read from the corresponding image.
    k:
        Number of samples to use. Defaults to all provided. ``k`` is exposed to
        the platform because it is the cost/accuracy dial: the research pipeline
        reports K = 1, 2, 3, 5 with K = 5 best and K = 3 the knee.
    """
    selected = list(sampled_answers)[: (k if k is not None else len(sampled_answers))]
    ocr_norm = [normalize_text(t) for t in (ocr_texts or [])]
    if not selected:
        raise ValueError("mvr_features requires at least one sampled answer")

    bits = [_mechanical_support_bit(s, ocr_norm) for s in selected]
    fz = [
        max([difflib.SequenceMatcher(None, normalize_text(s), o).ratio() for o in ocr_norm] or [0.0])
        for s in selected
    ]

    n = len(bits)
    return {
        "mvr_unsupported_rate": float(1.0 - sum(bits) / n),
        "mvr_all_unsupported": float(all(b == 0 for b in bits)),
        "mvr_all_supported": float(all(b == 1 for b in bits)),
        "mvr_fuzzy_mean": float(sum(fz) / n),
        "mvr_fuzzy_min": float(min(fz)),
        "mvr_fuzzy_std": float((sum((x - sum(fz) / n) ** 2 for x in fz) / n) ** 0.5),
        "mvr_fuzzy_best_of_k": float(max(fz)),
    }
