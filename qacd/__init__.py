"""QACD — Question-Conditioned Atomic Claim Decomposition.

Post-hoc answer-error risk scoring for frozen LVLM visual question answering.

Public API
----------
``QACDPipeline``
    End-to-end orchestrator: decompose -> evidence -> calibrate -> aggregate.
``BICLiteCalibrator``
    L2-regularised logistic risk calibrator over aligned claim features.
``decompose_claims``
    Question-conditioned atomic claim decomposition with a deterministic
    rule fallback.
``mvr_features`` / ``mechanical_ocr_features``
    The two non-LM evidence channels.
``conformal_select``
    Split-conformal selective prediction with BH/BY multiplicity control.

See ``docs/`` for the method write-up, the platform interface contract and the
declared evidence status.
"""

from qacd.aggregate import claim_to_response_max, response_risk
from qacd.calibrate import BICLiteCalibrator, RidgeLogistic
from qacd.conformal import conformal_pvalues, conformal_select
from qacd.decompose import CLAIM_TYPES, decompose_claims, fallback_decompose
from qacd.mechanical import MVR_FEATURES, mechanical_ocr_features, mvr_features
from qacd.pipeline import QACDPipeline
from qacd.types import Claim, ClaimRisk, ResponseRisk

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "QACDPipeline",
    "BICLiteCalibrator",
    "RidgeLogistic",
    "Claim",
    "ClaimRisk",
    "ResponseRisk",
    "CLAIM_TYPES",
    "decompose_claims",
    "fallback_decompose",
    "mechanical_ocr_features",
    "mvr_features",
    "MVR_FEATURES",
    "claim_to_response_max",
    "response_risk",
    "conformal_pvalues",
    "conformal_select",
]
