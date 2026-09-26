"""Assemble the frozen feature matrices from raw evidence tables.

Ties the ported feature engineering (:mod:`frozen.features`) to the frozen
protocol's split, selection and imputation rules, producing the exact 112-column
matrices the calibrator was fitted on.

Everything here runs on CPU with pandas + NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from frozen.features import (
    add_direction_aligned_features,
    apply_imputer,
    fit_imputer,
    load_direct_features,
    select_feature_names,
)

__all__ = ["FrozenMatrices", "load_split", "image_split", "build_matrices", "claim_images"]

#: Split parameters of the frozen protocol.
SEED = 20260920
CALIBRATION_FRACTION = 0.5


@dataclass
class FrozenMatrices:
    """Everything the scoring stage needs, plus the frames for inspection."""

    names: List[str]
    lm_names: List[str]
    mechanical_names: List[str]
    medians: Dict[str, float]
    dev_matrix: np.ndarray
    test_matrix: np.ndarray
    train_mask: np.ndarray
    score_train_images: List[str] = field(default_factory=list)
    calibration_images: List[str] = field(default_factory=list)

    @property
    def n_features(self) -> int:
        return len(self.names)


def load_split(raw_dir: str | Path, split: str) -> pd.DataFrame:
    """Assemble one split's claim frame from its raw evidence tables."""
    raw_dir = Path(raw_dir)
    required = [
        raw_dir / f"{split}_features.csv.gz",
        raw_dir / f"{split}_claims.csv.gz",
        raw_dir / f"{split}_direct.csv.gz",
        raw_dir / f"{split}_mechanical.csv.gz",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing raw evidence tables: {missing}")

    return load_direct_features(required[0], required[1], required[2]).merge(
        pd.read_csv(required[3], keep_default_na=False).drop_duplicates("unit_id"),
        on="unit_id",
        how="left",
        suffixes=("", "_mechanical"),
    )


def claim_images(frame: pd.DataFrame, claim_table: str | Path) -> np.ndarray:
    """Image id per claim row, taken from the claim table."""
    claims = pd.read_csv(claim_table, usecols=["unit_id", "image"], keep_default_na=False)
    lookup = dict(zip(claims["unit_id"].astype(str), claims["image"].astype(str)))
    return np.array([lookup.get(str(u), "") for u in frame["unit_id"]], dtype=object)


def image_split(
    images: Sequence[str],
    seed: int = SEED,
    calibration_fraction: float = CALIBRATION_FRACTION,
) -> Tuple[set, set]:
    """Image-disjoint score-training / calibration split (ported verbatim).

    The permutation depends only on the seed and the sorted image list, so the
    split is reproducible without any label.
    """
    unique_images = np.asarray(sorted(set(map(str, images))), dtype=object)
    permutation = np.random.default_rng(int(seed)).permutation(unique_images)
    calibration_count = int(round(float(calibration_fraction) * len(permutation)))
    calibration = set(map(str, permutation[:calibration_count]))
    score_training = set(map(str, permutation[calibration_count:]))
    if calibration & score_training or calibration | score_training != set(unique_images):
        raise AssertionError("image split is not a disjoint partition")
    return score_training, calibration


def build_matrices(
    dev_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    dev_claim_images: np.ndarray,
    seed: int = SEED,
    calibration_fraction: float = CALIBRATION_FRACTION,
) -> FrozenMatrices:
    """Derive the frozen feature matrices from two raw claim frames."""
    score_train_images, calibration_images = image_split(dev_claim_images, seed, calibration_fraction)
    train_mask = np.array([str(img) in score_train_images for img in dev_claim_images], dtype=bool)
    if not train_mask.any():
        raise ValueError("no claim rows fall in the score-training images")

    dev_aligned = add_direction_aligned_features(dev_frame)
    test_aligned = add_direction_aligned_features(test_frame)
    lm_names, mechanical_names = select_feature_names(dev_aligned, test_aligned, train_mask)
    names = lm_names + mechanical_names

    medians = fit_imputer(dev_aligned, names, train_mask)
    return FrozenMatrices(
        names=names,
        lm_names=lm_names,
        mechanical_names=mechanical_names,
        medians=medians,
        dev_matrix=apply_imputer(dev_aligned, names, medians),
        test_matrix=apply_imputer(test_aligned, names, medians),
        train_mask=train_mask,
        score_train_images=sorted(score_train_images),
        calibration_images=sorted(calibration_images),
    )
