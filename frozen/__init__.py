"""Feature engineering of the frozen research pipeline.

This package is the bridge that was previously missing: it turns raw evidence
tables into the 112-column claim-level feature matrix the frozen calibrator
consumes, so that :class:`qacd.pipeline.QACDPipeline` can build the same
features the reported results were computed from.

    frozen.constants   feature-group name lists (verbatim from the research code)
    frozen.features    the ported derivation functions

Requires pandas. The decision layer (:mod:`qacd`) and the metrics
(:mod:`evaluation`) stay NumPy-only.
"""

from frozen.constants import (
    BASE_EVIDENCE,
    CLAIM_TYPES,
    DIRECT_VERIFIER_FEATURES,
    QACD_FEATURES,
    V2_EVIDENCE,
)
from frozen.features import (
    add_contradiction_aware_features,
    add_direction_aligned_features,
    add_direct_verifier_features,
    add_qacd_aware_features,
    apply_imputer,
    dev_only_names,
    fit_imputer,
    load_direct_features,
    select_feature_names,
    usable_features,
)

__all__ = [
    "BASE_EVIDENCE",
    "CLAIM_TYPES",
    "DIRECT_VERIFIER_FEATURES",
    "QACD_FEATURES",
    "V2_EVIDENCE",
    "add_direction_aligned_features",
    "add_contradiction_aware_features",
    "add_qacd_aware_features",
    "add_direct_verifier_features",
    "load_direct_features",
    "usable_features",
    "dev_only_names",
    "select_feature_names",
    "fit_imputer",
    "apply_imputer",
]
