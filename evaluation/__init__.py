"""Reproducible evaluation of the frozen QACD results.

This package owns the *scoring and evaluation* stage:

    frozen evidence  ->  this repository's calibrator  ->  response risk

It deliberately does not own the LVLM evidence extraction, which lives behind
:mod:`qacd.providers`. See ``scripts/run_frozen_evaluation.py``.
"""

from evaluation.bundle import EvidenceBundle, load_bundle
from evaluation.metrics import (
    auroc,
    brier,
    image_level,
    paired_image_bootstrap,
    weighted_auroc,
)

__all__ = [
    "EvidenceBundle",
    "load_bundle",
    "auroc",
    "weighted_auroc",
    "brier",
    "image_level",
    "paired_image_bootstrap",
]
