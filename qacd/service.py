"""HTTP service exposing the four QACD integration points.

    POST /v1/qacd/risk        技术点 1 — full answer-error risk score
    POST /v1/qacd/mvr         技术点 2 — multi-view resampling consistency
    POST /v1/qacd/mechanical  技术点 3 — mechanical OCR instrument channel
    POST /v1/qacd/select      技术点 4 — conformal selective prediction
    GET  /health

Run with::

    pip install "fastapi>=0.110" "uvicorn>=0.27" "pydantic>=2.0"
    qacd serve --scorer configs/reference_scorer.json --port 8080

FastAPI is an optional dependency: importing :mod:`qacd` does not import it, and
this module raises a clear error if the extra is missing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from qacd.conformal import VALIDATED, conformal_select
from qacd.mechanical import MECHANICAL_FAMILY_SIZE, mechanical_ocr_features, mvr_features
from qacd.pipeline import QACDPipeline
from qacd.providers import MockProvider

__all__ = ["create_app", "SERVICE_VERSION"]

SERVICE_VERSION = "1.0.0"


def _require_fastapi():
    try:
        from fastapi import Body, FastAPI, HTTPException  # noqa: F401
        from pydantic import BaseModel, Field  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Service mode needs the service extra: "
            'pip install "fastapi>=0.110" "uvicorn>=0.27" "pydantic>=2.0"'
        ) from exc
    from fastapi import Body, FastAPI, HTTPException
    from pydantic import BaseModel, Field

    return FastAPI, Body, HTTPException, BaseModel, Field


def create_app(pipeline: Optional[QACDPipeline] = None, scorer_path: str | None = None):
    """Build the FastAPI application.

    Parameters
    ----------
    pipeline:
        A ready pipeline. Mutually exclusive with ``scorer_path``.
    scorer_path:
        Path to a JSON scorer exported by :meth:`QACDPipeline.save`.
    """
    FastAPI, Body, HTTPException, BaseModel, Field = _require_fastapi()

    if pipeline is None:
        if scorer_path:
            pipeline = QACDPipeline.load(scorer_path, provider=MockProvider())
        else:
            pipeline = QACDPipeline(provider=MockProvider())

    # ---- request / response models ------------------------------------
    class RiskRequest(BaseModel):
        question: str = Field(..., description="VQA question text")
        answer: str = Field(..., description="Frozen LVLM answer to be scored")
        image: str = Field("", description="Image path, URL or opaque identifier")
        threshold: Optional[float] = Field(None, description="Override the frozen warning threshold")
        return_claims: bool = Field(True, description="Include per-claim risks in the response")

    class ClaimOut(BaseModel):
        claim_id: str
        claim_text: str
        claim_type: str
        risk: float

    class RiskResponse(BaseModel):
        risk_score: float
        is_high_risk: bool
        threshold: float
        num_claims: int
        claims: List[ClaimOut] = []
        feature_dim: int
        model_calls: int
        latency_ms: int
        version: str
        warnings: List[str] = []
        channels: Dict[str, float] = {}

    class MVRRequest(BaseModel):
        sampled_answers: List[str] = Field(..., description="K same-prompt generations, in order")
        ocr_texts: List[str] = Field(default_factory=list, description="OCR strings read from the image")
        k: Optional[int] = Field(None, description="How many samples to use; default all")

    class MechanicalRequest(BaseModel):
        claim_text: str
        ocr_texts: List[str] = Field(default_factory=list)
        ocr_scores: List[float] = Field(default_factory=list)

    class SelectRequest(BaseModel):
        calibration_null_scores: List[float] = Field(..., description="Risk scores of known-correct calibration items")
        test_scores: List[float]
        alpha: float = 0.10
        procedure: str = Field("BY", description="'BH' (independence/PRDS) or 'BY' (arbitrary dependence)")
        test_labels: Optional[List[int]] = Field(None, description="Optional; used only to report realised FDP")

    app = FastAPI(
        title="QACD integration service",
        version=SERVICE_VERSION,
        description=(
            "Question-Conditioned Atomic Claim Decomposition — post-hoc answer-error "
            "risk scoring for LVLM visual question answering."
        ),
    )

    # ---- endpoints -----------------------------------------------------
    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "service_version": SERVICE_VERSION,
            "scorer_version": pipeline.config.version,
            "scorer_fitted": pipeline.fitted_,
            "feature_dim": len(pipeline.feature_names),
            "mvr_enabled": bool(pipeline.config.k),
            "conformal_validated": VALIDATED,
        }

    @app.post("/v1/qacd/risk", response_model=RiskResponse)
    def risk(request: RiskRequest) -> RiskResponse:
        if request.threshold is not None:
            pipeline.config.threshold = float(request.threshold)
        try:
            result = pipeline.score(request.question, request.answer, request.image)
        except Exception as exc:  # noqa: BLE001 - surface a clean 400 to the platform
            raise HTTPException(status_code=400, detail=f"scoring failed: {exc}") from exc
        payload = result.to_dict()
        if not request.return_claims:
            payload["claims"] = []
        return RiskResponse(**payload)

    @app.post("/v1/qacd/mvr")
    def mvr(request: MVRRequest) -> Dict[str, Any]:
        try:
            features = mvr_features(request.sampled_answers, request.ocr_texts, k=request.k)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "k_used": int(request.k) if request.k else len(request.sampled_answers),
            "features": features,
            "note": (
                "Stability channel only. In the frozen configuration these features are "
                "consumed by a response-level logistic fusion head together with the "
                "evidence score; they are not a standalone detector."
            ),
        }

    @app.post("/v1/qacd/mechanical")
    def mechanical(request: MechanicalRequest) -> Dict[str, Any]:
        features = mechanical_ocr_features(request.claim_text, request.ocr_texts, request.ocr_scores)
        return {
            "features": features,
            "family_size_in_frozen_config": MECHANICAL_FAMILY_SIZE,
            "note": (
                "This endpoint implements the OCR sub-family. The frozen 28-feature "
                "mechanical family also contains CLIP and counting probes."
            ),
        }

    @app.post("/v1/qacd/select")
    def select(request: SelectRequest) -> Dict[str, Any]:
        try:
            result = conformal_select(
                calibration_null_scores=request.calibration_null_scores,
                test_scores=request.test_scores,
                alpha=request.alpha,
                procedure=request.procedure,
                test_labels=request.test_labels,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.to_dict()

    return app
