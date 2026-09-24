"""End-to-end QACD orchestration.

    decompose -> evidence -> align -> calibrate -> aggregate -> (MVR fuse) -> risk

The pipeline object holds a *frozen* configuration: a feature order, a fitted
calibrator and an optional response-level fusion head. Nothing in :meth:`score`
updates any parameter, which is what makes an exported scorer reproducible.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from qacd.aggregate import DEFAULT_THRESHOLD, claim_to_response_max
from qacd.calibrate import BICLiteCalibrator, RidgeLogistic
from qacd.decompose import decompose_claims
from qacd.features import (
    ALL_FEATURE_NAMES,
    LM_FEATURE_NAMES,
    build_matrix,
    assemble_claim_features,
)
from qacd.mechanical import MECHANICAL_FEATURES, MVR_FEATURES, mvr_features
from qacd.providers import DirectVerification, MockProvider
from qacd.types import Claim, ClaimRisk, ResponseRisk

__all__ = ["QACDConfig", "QACDPipeline", "REFERENCE_FEATURE_NAMES"]

#: Features used by the reference pipeline's claim-level calibrator.
REFERENCE_FEATURE_NAMES: List[str] = LM_FEATURE_NAMES + MECHANICAL_FEATURES


@dataclass
class QACDConfig:
    """Frozen, serialisable configuration of a QACD scorer."""

    version: str = "1.0.0"
    max_claims: int = 8
    l2: float = 0.05
    threshold: float = DEFAULT_THRESHOLD

    #: Resampling depth for the MVR channel. ``None`` disables MVR; 3 is the
    #: reported cost/accuracy knee, 5 the best measured setting.
    k: Optional[int] = None

    #: Response-level fusion penalty, in the mean-loss convention. The research
    #: head used ``LogisticRegression(C=1.0)``; see
    #: :meth:`RidgeLogistic.from_sklearn_C`.
    fusion_l2: float = 0.02

    feature_names: List[str] = field(default_factory=lambda: list(REFERENCE_FEATURE_NAMES))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "QACDConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


class QACDPipeline:
    """Score VQA answers for error risk.

    Parameters
    ----------
    provider:
        Anything satisfying :class:`qacd.providers.EvidenceProvider`. Defaults to
        :class:`~qacd.providers.MockProvider`, which lets the pipeline run with no
        models attached.
    config:
        A :class:`QACDConfig`. Defaults to the reference configuration.
    calibrator, fusion:
        Pre-fitted heads. When absent, :meth:`fit` must be called before the
        scores mean anything; until then the pipeline returns an explicitly
        flagged uncalibrated prior.
    """

    def __init__(
        self,
        provider: Any = None,
        config: Optional[QACDConfig] = None,
        calibrator: Optional[BICLiteCalibrator] = None,
        fusion: Optional[RidgeLogistic] = None,
    ):
        self.provider = provider if provider is not None else MockProvider()
        self.config = config or QACDConfig()
        self.calibrator = calibrator
        self.fusion = fusion
        self.feature_names = list(self.config.feature_names)
        self.fitted_ = bool(calibrator is not None and getattr(calibrator, "coef_", None) is not None)
        self.fusion_fitted_ = bool(fusion is not None and getattr(fusion, "coef_", None) is not None)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _claim_rows(
        self,
        question: str,
        answer: str,
        image: str = "",
        ocr_texts: Optional[Sequence[str]] = None,
    ) -> tuple[List[Claim], Sequence[str], List[Dict[str, float]], int]:
        """Build claims and their feature rows. Returns calls made.

        ``ocr_texts`` is passed in by the callers so the image is read exactly
        once per request; passing ``None`` falls back to reading it here.
        """
        calls = 0
        completion = None
        if hasattr(self.provider, "decompose"):
            completion = self.provider.decompose(question, answer, self.config.max_claims)
            calls += 1

        decomposition = decompose_claims(
            question, answer, max_claims=self.config.max_claims, llm_completion=completion
        )
        claims = decomposition.claims

        if ocr_texts is None:
            ocr_texts = list(self.provider.ocr(image).texts)
        ocr_texts = list(ocr_texts)

        rows: List[Dict[str, float]] = []
        for claim in claims:
            views = self.provider.belief_views(question, answer, claim.claim_text)
            direct = self.provider.direct_verification(question, answer, claim.claim_text, claim.claim_type)
            calls += 1
            rows.append(
                assemble_claim_features(
                    claim_text=claim.claim_text,
                    views=views,
                    direct=direct,
                    ocr_texts=ocr_texts,
                )
            )
        return claims, decomposition.reasons, rows, calls

    def _uncalibrated_prior(self, rows: Sequence[Dict[str, float]]) -> np.ndarray:
        """Transparent stand-in used only before :meth:`fit` has been called.

        It averages two documented risk-direction features. It is deliberately
        crude: the point is that an unfitted pipeline still returns something
        monotone and explainable, with a warning attached, instead of silently
        pretending to be calibrated.
        """
        out = np.zeros(len(rows), dtype=float)
        for i, row in enumerate(rows):
            out[i] = 0.5 * (
                float(row.get("direct_verifier_composite_risk", 0.5))
                + float(row.get("mech_ocr_absent_risk", 0.5))
            )
        return np.clip(out, 1e-6, 1.0 - 1e-6)

    def score(self, question: str, answer: str, image: str = "") -> ResponseRisk:
        """Score a single answer."""
        started = time.perf_counter()
        warnings: List[str] = []

        # The image is read exactly once and reused by both the claim-level
        # mechanical features and the response-level MVR channel.
        ocr = self.provider.ocr(image)
        ocr_texts = list(ocr.texts)

        claims, reasons, rows, calls = self._claim_rows(question, answer, image, ocr_texts)
        if not claims:
            warnings.append("decomposition produced no claims")
        if reasons:
            warnings.append("claim set validation: " + "|".join(reasons))

        if rows:
            matrix = build_matrix(rows, self.feature_names)
            if self.fitted_:
                claim_scores = self.calibrator.predict_proba(matrix)
            else:
                claim_scores = self._uncalibrated_prior(rows)
                warnings.append(
                    "scorer is not fitted: scores are an uncalibrated prior, not QACD probabilities"
                )
        else:
            claim_scores = np.zeros(0, dtype=float)

        evidence_score = float(
            claim_to_response_max(claim_scores, 1, np.zeros(len(claim_scores), dtype=int), default=0.0)[0]
        )

        # ---- MVR channel --------------------------------------------------
        channels: Dict[str, float] = {"evidence_score": evidence_score}
        final_score = evidence_score
        if self.config.k:
            sampled = self.provider.sample_answers(question, int(self.config.k))
            calls += len(sampled)
            mvr = mvr_features(sampled, ocr_texts, k=int(self.config.k))
            channels.update({k: float(v) for k, v in mvr.items()})
            if self.fusion_fitted_:
                vector = np.array([[evidence_score] + [float(mvr[f]) for f in MVR_FEATURES]], dtype=float)
                final_score = float(self.fusion.predict_proba(vector)[0])
                channels["fused_score"] = final_score
            else:
                warnings.append("MVR features computed but the fusion head is not fitted; returning the evidence score")
            channels["mvr_unsupported_rate"] = float(mvr["mvr_unsupported_rate"])

        claim_risks = [
            ClaimRisk(
                claim_id=c.claim_id,
                claim_text=c.claim_text,
                claim_type=c.claim_type,
                risk=float(s),
            )
            for c, s in zip(claims, claim_scores)
        ]

        return ResponseRisk(
            risk_score=round(final_score, 6),
            is_high_risk=bool(final_score >= self.config.threshold),
            threshold=float(self.config.threshold),
            num_claims=len(claims),
            claims=claim_risks,
            feature_dim=len(self.feature_names),
            model_calls=int(calls),
            latency_ms=int((time.perf_counter() - started) * 1000),
            version=self.config.version,
            warnings=warnings,
            channels={k: round(float(v), 6) for k, v in channels.items()},
        )

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, records: Sequence[Dict[str, Any]], verbose: bool = False) -> "QACDPipeline":
        """Fit the claim calibrator (and the MVR fusion head when ``k`` is set).

        ``records`` is a development set. Each record needs ``question``,
        ``answer``, ``image`` and ``failed`` (1 when the answer is wrong). The
        response-level label is propagated to every claim row, matching the
        training-table construction used to produce the reported scorers.
        """
        claim_rows: List[Dict[str, float]] = []
        claim_labels: List[int] = []
        response_evidence: List[float] = []
        response_mvr: List[List[float]] = []
        response_labels: List[int] = []
        response_groups: List[str] = []

        for record in records:
            question = str(record.get("question", ""))
            answer = str(record.get("answer", ""))
            image = str(record.get("image", ""))
            label = int(record.get("failed", 0))

            claims, _, rows, _ = self._claim_rows(question, answer, image)
            if not rows:
                continue
            matrix = build_matrix(rows, self.feature_names)
            # Claim rows inherit the response label.
            claim_rows.extend(rows)
            claim_labels.extend([label] * len(rows))

            # Out-of-sample-ish evidence score for the fusion stage: use the
            # current deterministic prior if the calibrator is not yet fitted,
            # otherwise the fitted model.
            scores = (
                self.calibrator.predict_proba(matrix)
                if self.fitted_
                else self._uncalibrated_prior(rows)
            )
            evidence = float(claim_to_response_max(scores, 1, np.zeros(len(scores), dtype=int), default=0.0)[0])
            response_evidence.append(evidence)
            response_labels.append(label)
            response_groups.append(str(record.get("group", image or question)))

            if self.config.k:
                ocr = self.provider.ocr(image)
                sampled = self.provider.sample_answers(question, int(self.config.k))
                mvr = mvr_features(sampled, list(ocr.texts), k=int(self.config.k))
                response_mvr.append([float(mvr[f]) for f in MVR_FEATURES])
        if not claim_rows:
            raise ValueError("no claim rows were produced from the supplied records")

        X = build_matrix(claim_rows, self.feature_names)
        self.calibrator = BICLiteCalibrator(l2=self.config.l2).fit(X, np.asarray(claim_labels))
        self.fitted_ = bool(self.calibrator.success_)

        if self.config.k and response_mvr:
            head = np.column_stack([np.asarray(response_evidence), np.asarray(response_mvr, dtype=float)])
            self.fusion = RidgeLogistic(l2=self.config.fusion_l2).fit(head, np.asarray(response_labels))
            self.fusion_fitted_ = bool(self.fusion.success_)

        if verbose:
            print(
                f"[qacd] fitted: {len(claim_rows)} claim rows, {len(response_labels)} responses, "
                f"feature_dim={X.shape[1]}, calibrator_converged={self.fitted_}"
            )
        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def state(self) -> Dict[str, Any]:
        """Serialisable scorer state — JSON only, no pickle."""
        return {
            "format": "qacd-scorer",
            "format_version": 1,
            "config": self.config.to_dict(),
            "feature_names": self.feature_names,
            "calibrator": None
            if not self.fitted_
            else {
                "kind": "BICLiteCalibrator",
                "l2": self.calibrator.l2,
                "mean": self.calibrator.mean_.tolist(),
                "std": self.calibrator.std_.tolist(),
                "coef": self.calibrator.coef_.tolist(),
            },
            "fusion": None
            if not self.fusion_fitted_
            else {
                "kind": "RidgeLogistic",
                "l2": self.fusion.l2,
                "mean": self.fusion.mean_.tolist(),
                "std": self.fusion.std_.tolist(),
                "coef": self.fusion.coef_.tolist(),
                "feature_order": ["evidence_score"] + list(MVR_FEATURES),
            },
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path, provider: Any = None) -> "QACDPipeline":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "qacd-scorer":
            raise ValueError("not a QACD scorer file")
        config = QACDConfig.from_dict(payload.get("config", {}))

        calibrator = None
        blob = payload.get("calibrator")
        if blob:
            calibrator = BICLiteCalibrator(l2=blob.get("l2", config.l2))
            calibrator.mean_ = np.asarray(blob["mean"], dtype=float)
            calibrator.std_ = np.asarray(blob["std"], dtype=float)
            calibrator.coef_ = np.asarray(blob["coef"], dtype=float)

        fusion = None
        fblob = payload.get("fusion")
        if fblob:
            fusion = RidgeLogistic(l2=fblob.get("l2", config.fusion_l2))
            fusion.mean_ = np.asarray(fblob["mean"], dtype=float)
            fusion.std_ = np.asarray(fblob["std"], dtype=float)
            fusion.coef_ = np.asarray(fblob["coef"], dtype=float)

        return cls(provider=provider, config=config, calibrator=calibrator, fusion=fusion)
