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

import difflib
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from qacd.decompose import (
    build_decomposition_prompt,
    parse_json_claims,
    verification_prompt_for,
)
from qacd.mechanical import normalize_text
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

    Performs four kinds of call against a frozen checkpoint:

    * decomposition — the constrained JSON prompt from
      :func:`qacd.decompose.build_decomposition_prompt`;
    * belief views — one generation per view (independent / visual /
      minus-claim) plus one yes-no match probe;
    * direct verification — a type-routed yes-no probe per claim, with OCR and
      numeric sub-probes;
    * resampling — K stochastic generations of the *same* prompt, consumed by
      the MVR channel.

    Resource envelope measured in the research runs (2 x A100-SXM4-80GB host,
    one process per GPU): the 13B checkpoint in fp16 occupies roughly 26 GB of
    device memory, leaving headroom for a batch of 4-8 short probes on a 40 GB
    card. Cost per response: 7.65 model calls for the LM channel, plus K for
    resampling (K = 3 is the reported knee, K = 5 the best measured setting).

    Testability
    -----------
    Only :meth:`_generate_with_scores` touches the model. Everything else —
    prompt construction and answer parsing — is pure and unit-tested, so the
    provider can be verified end to end by injecting a fake ``_generate``
    (see ``tests/test_provider.py``). A ``model_fn`` may be passed directly to
    the constructor for that purpose.
    """

    name = "llava-1.5-13b"

    #: Answer grammar for the yes/no probes.
    AFFIRMATIVE = ("yes", "true", "correct", "supported", "support")
    NEGATIVE = ("no", "false", "incorrect", "unsupported", "contradict")

    def __init__(
        self,
        model_path: str = "",
        ocr_provider: Any = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 64,
        device: str = "cuda",
        dtype: str = "float16",
        seed: Optional[int] = None,
        model_fn: Any = None,
    ):
        self.model_path = model_path
        self.ocr_provider = ocr_provider
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self._model = None
        self._processor = None
        self._torch = None
        #: Injectable ``(image, prompt, do_sample) -> (text, confidence)``.
        self._model_fn = model_fn
        self.calls = 0

    # -- model plumbing ----------------------------------------------------
    def load(self) -> None:
        """Load the checkpoint. Kept lazy so importing this module is free."""
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "LLaVAProvider needs the model extra: "
                "pip install torch transformers accelerate pillow"
            ) from exc
        if not self.model_path:
            raise ValueError("model_path is required to load the checkpoint")
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=self.dtype, device_map=self.device
        )
        self._model.eval()

    def _generate_with_scores(self, image: str, prompt: str, do_sample: bool) -> tuple[str, float]:
        """Run one generation. Returns ``(text, mean_token_probability)``.

        This is the only method that touches the model. Override it, or pass
        ``model_fn`` to the constructor, to test the provider without weights.
        """
        if self._model_fn is not None:
            self.calls += 1
            return self._model_fn(image, prompt, do_sample)

        self.load()  # pragma: no cover - requires weights
        torch = self._torch
        from PIL import Image

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(conversation, add_generation_prompt=True)
        pil = Image.open(image).convert("RGB") if isinstance(image, str) and image else None
        inputs = self._processor(images=pil, text=text, return_tensors="pt").to(self.device)

        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": bool(do_sample),
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        if do_sample:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
        if self.seed is not None:
            torch.manual_seed(int(self.seed))

        with torch.inference_mode():
            output = self._model.generate(**inputs, **kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        gen_ids = output.sequences[0][prompt_len:]
        decoded = self._processor.decode(gen_ids, skip_special_tokens=True).strip()

        confidence = 1.0
        if getattr(output, "scores", None):
            probs = []
            for step, step_scores in enumerate(output.scores):
                if step >= len(gen_ids):
                    break
                step_probs = torch.softmax(step_scores[0].float(), dim=-1)
                probs.append(float(step_probs[int(gen_ids[step])]))
            if probs:
                confidence = float(sum(probs) / len(probs))
        self.calls += 1
        return decoded, confidence

    def _generate(self, image: str, prompt: str, do_sample: bool = False) -> str:
        """Text-only convenience wrapper."""
        return self._generate_with_scores(image, prompt, do_sample)[0]

    # -- prompt construction (pure) ---------------------------------------
    @staticmethod
    def build_belief_prompt(view: str, question: str, answer: str, claim_text: str) -> str:
        """Prompt for one belief view."""
        if view == "independent":
            return (
                "Answer the question from scratch using the image.\n"
                f"Question: {question}\n"
                "Answer with a short phrase only."
            )
        if view == "visual":
            return (
                "Answer using only what is visible in the image, ignoring any prior assumption.\n"
                f"Question: {question}\n"
                "Answer with a short phrase only."
            )
        if view == "minus_claim":
            return (
                "Answer the question as if the following statement were not asserted.\n"
                f"Question: {question}\n"
                f"Statement to set aside: {claim_text}\n"
                "Answer with a short phrase only."
            )
        if view == "answer_match":
            return LLaVAProvider.build_match_prompt(claim_text, answer)
        raise ValueError(f"unknown belief view: {view}")

    @staticmethod
    def build_match_prompt(claim_text: str, answer: str) -> str:
        return (
            "Decide whether the ANSWER supports the CLAIM.\n"
            f"CLAIM: {claim_text}\n"
            f"ANSWER: {answer}\n"
            'Reply with exactly one word: "yes" or "no".'
        )

    @staticmethod
    def build_direct_prompt(claim_text: str, claim_type: str) -> str:
        """Type-routed verification probe."""
        probe = verification_prompt_for(claim_text, claim_type)
        return f"{probe}\nReply with exactly one word: \"yes\" or \"no\"."

    @staticmethod
    def build_ocr_prompt() -> str:
        return (
            "Read all text visible in the image, verbatim.\n"
            "If there is no text, reply with exactly: none"
        )

    @staticmethod
    def build_number_prompt(claim_text: str) -> str:
        return (
            "Check the numeric claim against the image.\n"
            f"CLAIM: {claim_text}\n"
            'Reply with exactly one word: "yes" if the number is consistent with the image, otherwise "no".'
        )

    # -- parsing (pure) ----------------------------------------------------
    @classmethod
    def parse_support(cls, text: str) -> float:
        """Map a yes/no completion to a support value in ``[0, 1]``.

        Returns 0.5 when the completion is neither clearly affirmative nor
        clearly negative — the same "uncertain" convention the rest of the
        evidence layer uses.
        """
        lowered = str(text or "").strip().lower()
        if not lowered:
            return 0.5
        head = lowered.split("\n")[0]
        for token in re.findall(r"[a-z']+", head):
            if token in cls.AFFIRMATIVE:
                return 1.0
            if token in cls.NEGATIVE:
                return 0.0
        return 0.5

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        """Normalised similarity used to turn a generated answer into support."""
        a = normalize_text(left)
        b = normalize_text(right)
        if not a or not b:
            return 0.0
        if a in b or b in a:
            return 1.0
        return float(difflib.SequenceMatcher(None, a, b).ratio())

    # -- EvidenceProvider surface -----------------------------------------
    def ocr(self, image: str) -> OCRRecord:
        if self.ocr_provider is None:
            return OCRRecord(image=str(image), texts=[], scores=[])
        return self.ocr_provider.ocr(image)

    def decompose(self, question: str, answer: str, max_claims: int) -> Any:
        """Checked LLM decomposition; ``None`` makes the caller use the fallback."""
        prompt = build_decomposition_prompt(question, answer, max_claims)
        text, _ = self._generate_with_scores("", prompt, False)
        claims = parse_json_claims(text)
        return claims or None

    def belief_views(self, question: str, answer: str, claim_text: str) -> List[VerificationView]:
        """Four claim-specific belief readings."""
        views: List[VerificationView] = []
        for view in ("independent", "visual", "minus_claim"):
            prompt = self.build_belief_prompt(view, question, answer, claim_text)
            generated, confidence = self._generate_with_scores("", prompt, False)
            support = self._similarity(generated, claim_text)
            views.append(
                VerificationView(
                    view=view,
                    support=support,
                    confidence=confidence,
                    evidence=generated,
                    uncertainty=1.0 - support,
                )
            )

        match_prompt = self.build_belief_prompt("answer_match", question, answer, claim_text)
        match_text, match_confidence = self._generate_with_scores("", match_prompt, False)
        match_support = self.parse_support(match_text)
        views.append(
            VerificationView(
                view="answer_match",
                support=match_support,
                confidence=match_confidence if match_support != 0.5 else 0.5,
                evidence=match_text,
                uncertainty=1.0 if match_support == 0.5 else 0.0,
            )
        )
        return views

    def direct_verification(
        self, question: str, answer: str, claim_text: str, claim_type: str
    ) -> DirectVerification:
        """Type-routed direct probe plus OCR and numeric sub-probes."""
        direct_text, direct_confidence = self._generate_with_scores(
            "", self.build_direct_prompt(claim_text, claim_type), False
        )
        support = self.parse_support(direct_text)
        contradiction = 1.0 - support if support != 0.5 else 0.5
        uncertainty = 1.0 if support == 0.5 else 0.0

        # OCR probe: only meaningful for text claims, but cheap enough to always
        # run and route on afterwards.
        ocr_text, _ = self._generate_with_scores("", self.build_ocr_prompt(), False)
        ocr_support = self._similarity(ocr_text, claim_text)

        number_text, _ = self._generate_with_scores("", self.build_number_prompt(claim_text), False)
        number_support = self.parse_support(number_text)

        return DirectVerification(
            support=support,
            contradiction=contradiction,
            evidence_present=1.0 if direct_text.strip() else 0.0,
            uncertainty=uncertainty,
            evidence_phrase=direct_text.strip(),
            ocr_read_support=ocr_support,
            number_check_support=number_support,
            visual_evidence_support=support,
        )

    def sample_answers(self, question: str, k: int) -> List[str]:
        """K stochastic generations of the identical prompt (MVR input)."""
        k = int(k)
        if k < 1:
            raise ValueError("k must be >= 1")
        prompt = (
            "Answer the question in a short phrase.\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return [self._generate_with_scores("", prompt, True)[0].strip() for _ in range(k)]
