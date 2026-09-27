"""Regression: the image must reach the model on every provider call.

The real-machine run caught this. ``LLaVAProvider``'s four evidence methods were
calling ``self._generate_with_scores("", prompt, ...)`` — a hardcoded empty
image — so every probe ran against no picture and produced degenerate output:
empty strings, repeated tokens, and an echoed instruction. Nothing in the test
suite noticed, because the injected model was a stub that ignored the image.

These tests assert the image is forwarded, which is the property that was
missing. They need no weights.
"""

from __future__ import annotations

import pytest

from qacd.providers import LLaVAProvider

IMAGE = "/data/textvqa/eb38600d8a5ade9a.jpg"


class RecordingModel:
    """Returns a scripted reply and records every call it receives."""

    def __init__(self, reply="Yes"):
        self.reply = reply
        self.calls = []

    def __call__(self, image, prompt, do_sample):
        self.calls.append({"image": image, "prompt": prompt, "do_sample": do_sample})
        return self.reply, 0.8


def provider(reply="Yes"):
    model = RecordingModel(reply)
    return LLaVAProvider(model_fn=model), model


def test_raw_generation_receives_the_image():
    p, model = provider("a red car")
    p._generate_with_scores(IMAGE, "Describe this image.", False)
    assert model.calls[0]["image"] == IMAGE


def test_decompose_forwards_the_image():
    p, model = provider(
        '{"claims":[{"claim_id":"c1","claim_text":"The sign reads OPEN.",'
        '"source_span":"OPEN","claim_type":"OCR_text"}]}'
    )
    p.decompose("What does the sign say?", "OPEN", 8, IMAGE)
    assert model.calls, "decompose made no model call"
    assert all(c["image"] == IMAGE for c in model.calls), (
        "decompose dropped the image; the model saw no picture"
    )


def test_belief_views_forwards_the_image_to_every_view():
    p, model = provider("Yes")
    views = p.belief_views("What does the sign say?", "OPEN", "The sign reads OPEN.", IMAGE)
    assert len(views) == 4
    assert len(model.calls) == 4
    assert all(c["image"] == IMAGE for c in model.calls), (
        "a belief view ran without the image"
    )


def test_direct_verification_forwards_the_image_to_every_probe():
    p, model = provider("Yes")
    p.direct_verification("What does the sign say?", "OPEN", "The sign reads OPEN.", "OCR_text", IMAGE)
    assert len(model.calls) == 3, "expected direct probe + OCR probe + number probe"
    assert all(c["image"] == IMAGE for c in model.calls), (
        "a direct-verification probe ran without the image"
    )


def test_sample_answers_forwards_the_image_and_samples():
    p, model = provider("OPEN")
    samples = p.sample_answers("What does the sign say?", 3, IMAGE)
    assert samples == ["OPEN"] * 3
    assert len(model.calls) == 3
    assert all(c["image"] == IMAGE for c in model.calls), "MVR sampling ran without the image"
    assert all(c["do_sample"] is True for c in model.calls), "MVR must sample, not decode greedily"


def test_omitting_the_image_is_visible_as_an_empty_string():
    """The old behaviour, kept explicit so the failure mode is documented."""
    p, model = provider("Yes")
    p.belief_views("q", "a", "c")  # no image
    assert all(c["image"] == "" for c in model.calls), (
        "with no image the provider passes an empty string, which is what made "
        "the first real-machine run degenerate"
    )


def test_the_pipeline_passes_the_image_through_to_the_provider():
    """The caller side of the same contract."""
    from qacd.pipeline import QACDConfig, QACDPipeline

    p, model = provider("Yes")
    pipeline = QACDPipeline(provider=p, config=QACDConfig(k=2))
    pipeline.score("What does the sign say?", "OPEN", IMAGE)
    assert model.calls, "the pipeline made no model call"
    assert all(c["image"] == IMAGE for c in model.calls), (
        "QACDPipeline dropped the image before calling the provider"
    )


def test_every_evidence_method_declares_an_image_parameter():
    """The protocol must expose the image, not hide it behind provider state."""
    import inspect

    from qacd.providers import EvidenceProvider

    for name in ("belief_views", "direct_verification", "sample_answers", "decompose"):
        signature = inspect.signature(getattr(EvidenceProvider, name))
        assert "image" in signature.parameters, f"{name} has no image parameter"
        assert signature.parameters["image"].default == "", (
            f"{name}.image should default to an empty string so older callers still run"
        )
