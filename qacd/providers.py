"""Evidence providers — everything that needs a model, behind one interface.

The QACD decision layer never talks to a model directly. It asks a provider for
four things and receives plain Python objects back:

======================  =====================================================
``decompose``           Checked LLM decomposition of an answer (optional).
``belief_views``        Support/confidence readings for the four belief views.
``direct_verification`` Type-routed direct evidence (support, contradiction,
                        evidence phrase, OCR read, numeric check).
``ocr``                 Raw text read off the image.
``sample_answers``      K same-prompt generations, for the MVR channel.
======================  =====================================================

Three implementations ship with the package:

``MockProvider``
    Fully deterministic, no models, no network. Used by the test suite and by
    ``scripts/demo_offline.py`` so that the whole pipeline can be exercised on a
    laptop.
``LLaVAProvider``
    Adapter for a frozen LLaVA-1.5-13B checkpoint. Requires ``torch`` and the
    model weights; not importable without them, by design.
``RapidOCRProvider``
    Adapter for the RapidOCR engine used for the mechanical instrument channel.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Sequence, runtime_checkable

from qacd.types import OCRRecord

__all__ = [
    "EvidenceProvider",
    "MockProvider",
    "LLaVAProvider",
    "RapidOCRProvider",
    "VerificationView",
]

# The four belief views. Kept as data so the platform can see what is requested.
BELIEF_VIEWS = ("independent", "visual", "minus_claim", "answer_match")


@dataclass
class VerificationView:
    """One (claim, view) verification reading."""

    view: str
    support: float = 0.5
    confidence: float = 0.5
    evidence: str = ""
    uncertainty: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "view": self.view,
            "support": float(self.support),
            "confidence": float(self.confidence),
            "evidence": self.evidence,
            "uncertainty": float(self.uncertainty),
        }


@dataclass
class DirectVerification:
    """Type-routed direct evidence for one claim."""

    support: float = 0.5
    contradiction: float = 0.5
    evidence_present: float = 0.5
    uncertainty: float = 0.5
    evidence_phrase: str = ""
    ocr_read_support: float = 0.5
    number_check_support: float = 0.5
    visual_evidence_support: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "support": float(self.support),
            "contradiction": float(self.contradiction),
            "evidence_present": float(self.evidence_present),
            "uncertainty": float(self.uncertainty),
            "evidence_phrase": self.evidence_phrase,
            "ocr_read_support": float(self.ocr_read_support),
            "number_check_support": float(self.number_check_support),
            "visual_evidence_support": float(self.visual_evidence_support),
        }


@runtime_checkable
class EvidenceProvider(Protocol):
    """What the pipeline needs from the outside world."""

    name: str

    def ocr(self, image: str) -> OCRRecord:
        """Text read from the image."""

    def belief_views(self, question: str, answer: str, claim_text: str) -> List[VerificationView]:
        """Support/confidence readings across the belief views."""

    def direct_verification(self, question: str, answer: str, claim_text: str, claim_type: str) -> DirectVerification:
        """Direct evidence for one typed claim."""

    def sample_answers(self, question: str, k: int) -> List[str]:
        """K same-prompt generations (MVR channel)."""

    def decompose(self, question: str, answer: str, max_claims: int) -> Any:
        """Optional checked LLM decomposition; return ``None`` to force the fallback."""


# ---------------------------------------------------------------------------
# Deterministic offline provider
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has",
    "have", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was",
    "were", "with", "this", "these", "those", "there",
}


def _digest(*parts: str) -> float:
    """Stable pseudo-random value in [0, 1) derived from the inputs.

    Determinism matters: the demo and the tests must produce identical numbers on
    every machine, and nothing here may depend on wall-clock time or ``hash()``.
    """
    h = hashlib.sha256("\u241f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:6], "big") / float(1 << 48)


class MockProvider:
    """Deterministic stand-in for the LVLM + OCR stack.

    It is *not* a hallucination detector. It derives every reading from lexical
    overlap between the claim and the image text, which is enough to exercise the
    plumbing and to give the platform team stable, reproducible payloads while
    the real provider is wired up.
    """

    name = "mock"

    def __init__(self, image_text: str = "", sampled_pool: Sequence[str] | None = None):
        self.image_text = str(image_text or "")
        self.sampled_pool = list(sampled_pool or [])
        #: OCR result of the most recent :meth:`ocr` call. A real provider holds
        #: the image in context; the mock mirrors that so verification does not
        #: have to re-read (and cannot invent) an image path.
        self._last_ocr: OCRRecord | None = None

    # -- OCR ---------------------------------------------------------------
    def ocr(self, image: str) -> OCRRecord:
        text = self.image_text or image
        parts = [p.strip() for p in re.split(r"[;\n]", str(text)) if p.strip()]
        record = OCRRecord(image=str(image), texts=parts, scores=[1.0] * len(parts))
        self._last_ocr = record
        return record

    def _current_ocr(self) -> OCRRecord:
        return self._last_ocr if self._last_ocr is not None else self.ocr("")

    # -- belief views ------------------------------------------------------
    def belief_views(self, question: str, answer: str, claim_text: str) -> List[VerificationView]:
        claim_words = {w for w in _WORD_RE.findall(claim_text.lower()) if w not in _STOP}
        answer_words = {w for w in _WORD_RE.findall(answer.lower()) if w not in _STOP}
        overlap = len(claim_words & answer_words) / max(len(claim_words), 1)

        views: List[VerificationView] = []
        for view in BELIEF_VIEWS:
            jitter = _digest(self.name, view, question, answer, claim_text)
            support = max(0.0, min(1.0, 0.35 + 0.5 * overlap + 0.15 * (jitter - 0.5)))
            views.append(
                VerificationView(
                    view=view,
                    support=support,
                    confidence=max(0.0, min(1.0, 0.6 + 0.3 * (jitter - 0.5))),
                    evidence=answer,
                    uncertainty=max(0.0, min(1.0, 1.0 - support)),
                )
            )
        return views

    # -- direct verification ----------------------------------------------
    def direct_verification(self, question: str, answer: str, claim_text: str, claim_type: str) -> DirectVerification:
        ocr = self._current_ocr()
        ocr_text = " ".join(ocr.texts).lower()
        claim_words = {w for w in _WORD_RE.findall(claim_text.lower()) if w not in _STOP}
        hit = len([w for w in claim_words if w in ocr_text]) / max(len(claim_words), 1)
        jitter = _digest(self.name, "direct", claim_type, claim_text)

        support = max(0.0, min(1.0, 0.25 + 0.6 * hit + 0.15 * (jitter - 0.5)))
        return DirectVerification(
            support=support,
            contradiction=max(0.0, min(1.0, 0.55 - 0.5 * hit)),
            evidence_present=1.0 if ocr.texts else 0.0,
            uncertainty=max(0.0, min(1.0, 1.0 - support)),
            evidence_phrase=" ".join(ocr.texts[:2]),
            ocr_read_support=hit,
            number_check_support=hit,
            visual_evidence_support=max(0.0, min(1.0, 0.3 + 0.5 * hit)),
        )

    # -- MVR ---------------------------------------------------------------
    def sample_answers(self, question: str, k: int) -> List[str]:
        if self.sampled_pool:
            out = list(self.sampled_pool[:k])
            while len(out) < k:
                out.append(out[-1] if out else "")
            return out
        # Without a pool, emit the question echo: a deliberately weak but stable
        # stand-in so callers can see the shape of the payload.
        return [str(question or "")] * int(k)

    # -- decomposition -----------------------------------------------------
    def decompose(self, question: str, answer: str, max_claims: int) -> Any:
        return None  # force the deterministic fallback


# ---------------------------------------------------------------------------
# Real providers
# ---------------------------------------------------------------------------

class RapidOCRProvider:
    """OCR half of the evidence stack, backed by RapidOCR (ONNX runtime).

    The research pipeline read text off every image once and cached it as JSONL;
    this adapter exposes the same engine for online use.
    """

    name = "rapidocr"

    def __init__(self, engine: Any = None):
        if engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "RapidOCRProvider needs the OCR extra: pip install rapidocr-onnxruntime"
                ) from exc
            engine = RapidOCR()
        self.engine = engine

    def ocr(self, image: str) -> OCRRecord:
        result, _ = self.engine(image)
        texts: List[str] = []
        scores: List[float] = []
        for row in result or []:
            try:
                texts.append(str(row[1]))
                scores.append(float(row[2]))
            except (IndexError, TypeError, ValueError):
                continue
        return OCRRecord(image=str(image), texts=texts, scores=scores)


class LLaVAProvider:
    """Frozen LLaVA-1.5-13B evidence provider.

    This adapter is intentionally thin. It performs three kinds of call against a
    frozen checkpoint:

    * decomposition (``decompose``) — the constrained JSON prompt from
      :func:`qacd.decompose.build_decomposition_prompt`;
    * belief/direct verification — short yes/no probes with a constrained answer
      grammar, parsed into support/confidence readings;
    * resampling (``sample_answers``) — K stochastic generations of the *same*
      prompt, used by the MVR channel.

    Resource envelope measured in the research runs (2 x A100-SXM4-80GB host,
    one process per GPU): the 13B checkpoint in fp16 occupies roughly 26 GB of
    device memory, leaving headroom for a batch of 4-8 probes on a 40 GB card.
    ``calls_per_response`` is the cost dial: 7.65 model calls for the LM channel
    alone, plus K for resampling (K = 3 is the reported knee, K = 5 the best
    measured setting).
    """

    name = "llava-1.5-13b"

    def __init__(
        self,
        model_path: str,
        ocr_provider: Any = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 256,
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.ocr_provider = ocr_provider
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)
        self.device = device
        self._model = None
        self._processor = None

    # -- lazy loading ------------------------------------------------------
    def load(self) -> None:
        """Load the checkpoint. Kept lazy so importing this module is free."""
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "LLaVAProvider needs the model extra: pip install torch transformers accelerate pillow"
            ) from exc
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype="float16", device_map=self.device
        )
        self._model.eval()

    def _generate(self, image: str, prompt: str, do_sample: bool) -> str:  # pragma: no cover
        raise NotImplementedError(
            "Wire this to your checkpoint's generate() call. The reference "
            "implementation stops at the interface so that no unverified "
            "modelling code is presented as tested."
        )

    # -- EvidenceProvider surface -----------------------------------------
    def ocr(self, image: str) -> OCRRecord:
        if self.ocr_provider is None:
            return OCRRecord(image=str(image), texts=[], scores=[])
        return self.ocr_provider.ocr(image)

    def belief_views(self, question: str, answer: str, claim_text: str) -> List[VerificationView]:  # pragma: no cover
        raise NotImplementedError

    def direct_verification(self, question: str, answer: str, claim_text: str, claim_type: str) -> DirectVerification:  # pragma: no cover
        raise NotImplementedError

    def sample_answers(self, question: str, k: int) -> List[str]:  # pragma: no cover
        raise NotImplementedError

    def decompose(self, question: str, answer: str, max_claims: int) -> Any:  # pragma: no cover
        raise NotImplementedError
