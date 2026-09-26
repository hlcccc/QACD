"""Per-claim assembly of the frozen 112-column feature vector.

Every derivation in :mod:`frozen.features` is row-wise, so a single claim can be
assembled on its own — which is what makes the frozen feature set usable at
scoring time rather than only in batch. Verified by
``scripts/verify_pipeline_frozen.py``: assembling row by row reproduces the
frozen matrix exactly, and scoring through :class:`qacd.pipeline.QACDPipeline`
reproduces the batch result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from frozen.features import add_direct_verifier_features

__all__ = ["FrozenFeatureAssembler"]


@dataclass
class FrozenFeatureAssembler:
    """Turns one claim's raw evidence into the frozen feature vector.

    Parameters
    ----------
    feature_names:
        The 112 names, in the frozen order.
    medians:
        Score-training medians, applied to any feature that comes out missing.
        This mirrors ``apply_imputer`` and must come from the same split the
        calibrator was fitted on.
    """

    feature_names: List[str]
    medians: Dict[str, float]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @classmethod
    def from_matrices(cls, matrices) -> "FrozenFeatureAssembler":
        """Build from a :class:`frozen.build.FrozenMatrices`."""
        return cls(feature_names=list(matrices.names), medians=dict(matrices.medians))

    # -- framing -----------------------------------------------------------
    def _derive(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            raise ValueError("cannot assemble features from an empty frame")
        return add_direct_verifier_features(frame)

    def _select(self, derived: pd.DataFrame) -> np.ndarray:
        columns = []
        for name in self.feature_names:
            if name in derived.columns:
                values = pd.to_numeric(derived[name], errors="coerce")
            else:
                values = pd.Series(np.nan, index=derived.index, dtype="float64")
            columns.append(values.fillna(self.medians.get(name, 0.0)).to_numpy(dtype=np.float64))
        matrix = np.stack(columns, axis=1)
        if not np.isfinite(matrix).all():
            raise ValueError("non-finite feature vector after imputation")
        return matrix

    # -- public API --------------------------------------------------------
    def transform_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Assemble a whole table: ``(n_claims, n_features)``."""
        return self._select(self._derive(frame))

    def transform_one(self, row: Mapping[str, Any]) -> np.ndarray:
        """Assemble a single claim: ``(n_features,)``."""
        frame = pd.DataFrame([dict(row)])
        return self._select(self._derive(frame))[0]

    def transform_rows(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Assemble row by row. Equal to :meth:`transform_frame` by construction."""
        if not rows:
            raise ValueError("no rows supplied")
        return np.stack([self.transform_one(row) for row in rows], axis=0)
