"""Regression tests for the three ways a first-time user got a wrong number.

Each of these reproduces a real failure found by walking the documented workflow
as a stranger. They are grouped here because they share a theme: the pipeline used
to answer confidently when it had no basis for an answer.

1. ``score()`` with a frozen assembler attached silently fed 93 zero columns to a
   calibrator fitted on real values.
2. The MVR fusion head, fitted on a development set where the resampling channel
   carried no signal, put a *negative* weight on the calibrated evidence score and
   inverted the response ranking -- a confident, backwards score.
3. The CLI defaulted to the mock provider, so following the documentation
   produced a scorer with no model evidence behind it.
"""

from __future__ import annotations

import numpy as np
import pytest

from qacd.cli import build_parser
from qacd.features import assemble_claim_features
from qacd.mechanical import MVR_FEATURES
from qacd.pipeline import (
    NON_EVIDENCE_PROVIDERS,
    QACDConfig,
    QACDPipeline,
    provider_kind,
)
from qacd.providers import DirectVerification, MockProvider, VerificationView


# ---------------------------------------------------------------------------
# 1. score() must refuse to mix the live and frozen feature paths
# ---------------------------------------------------------------------------

def test_score_refuses_a_frozen_assembler():
    """The live path builds 48 columns; the assembler expects 112.

    ``build_matrix`` fills whatever it cannot find with 0.0, so mixing them
    produced a plausible score computed mostly from zeros.
    """

    class FakeAssembler:
        feature_names = [f"frozen_{i}" for i in range(112)]

    pipeline = QACDPipeline(provider=MockProvider(), assembler=FakeAssembler())
    assert len(pipeline.feature_names) == 112

    with pytest.raises(RuntimeError) as excinfo:
        pipeline.score("What brand?", "Dakota Digital", "demo.jpg")
    message = str(excinfo.value)
    assert "score_evidence_frame" in message
    assert "112" in message and "48" in message


def test_mixing_the_paths_would_really_have_zeroed_the_columns():
    """Guard against the test above passing for the wrong reason.

    Asserts the hazard is real rather than hypothetical: with 112 frozen column
    names in play, the live row cannot fill more than the 48 it knows about.
    """
    provider = MockProvider()
    row = assemble_claim_features(
        claim_text="Dakota Digital",
        views=[VerificationView(view="independent", support=0.9)],
        direct=DirectVerification(support=0.9, evidence_phrase="Dakota Digital"),
        ocr_texts=["Dakota Digital; open 24 days"],
    )
    from qacd.features import build_matrix

    matrix = build_matrix([row], [f"frozen_{i}" for i in range(112)])
    assert matrix.shape == (1, 112)
    assert int((matrix[0] != 0.0).sum()) == 0  # every column silently zero
    assert provider is not None


# ---------------------------------------------------------------------------
# 2. The fusion head must not contradict the model it sharpens
# ---------------------------------------------------------------------------

def _inverting_dev_set(n=40):
    """The development set that produced the inversion.

    A mock provider with no OCR text returns the *question* for every resample,
    so the MVR columns vary only with the question and carry nothing about
    correctness. Eight distinct questions, alternating labels.
    """
    questions = [
        ("What brand is the camera?", "Dakota Digital", "Nikon Coolpix"),
        ("What is the website?", "Flickr", "Pinterest"),
        ("What color is the shirt?", "blue", "red"),
        ("How many people are there?", "two", "five"),
        ("What sport is being played?", "tennis", "soccer"),
        ("What is on the table?", "a laptop", "a vase"),
        ("What kind of animal is this?", "a dog", "a cat"),
        ("What is the man holding?", "a phone", "a book"),
    ]
    records = []
    for i in range(n):
        question, right, wrong = questions[i % len(questions)]
        failed = i % 2
        records.append(
            {
                "question": question,
                "answer": wrong if failed else right,
                "image": "/data/textvqa/val/eb38600d8a5ade9a.jpg",
                "failed": failed,
                "group": f"img{i}",
            }
        )
    return records


def test_pipeline_fitted_on_a_signal_free_dev_set_keeps_the_right_direction():
    """The exact walkthrough that returned 1.0000 for the correct answer.

    Before the guard, ``k=3`` fitted a fusion head with a negative evidence
    weight and the correct answer scored 1.0000 against the wrong answer's 0.0000.
    """
    pipeline = QACDPipeline(
        provider=MockProvider(image_text="", sampled_pool=[]), config=QACDConfig(k=3)
    )
    pipeline.fit(_inverting_dev_set())

    good = pipeline.score("What brand is the camera?", "Dakota Digital", "/data/x.jpg")
    bad = pipeline.score("What brand is the camera?", "Nikon Coolpix", "/data/x.jpg")

    assert good.risk_score < bad.risk_score, (
        f"ranking inverted: correct={good.risk_score} wrong={bad.risk_score}"
    )


def test_fusion_head_is_skipped_when_it_would_invert_the_ranking(monkeypatch):
    """Unit-level: a negative evidence weight is refused, whatever caused it."""
    from qacd import pipeline as pipeline_module

    class InvertingHead:
        success_ = True
        coef_ = np.array([-0.5] + [0.0] * len(MVR_FEATURES))

        def __init__(self, **kwargs):
            pass

        def fit(self, features, labels):
            return self

    monkeypatch.setattr(pipeline_module, "RidgeLogistic", InvertingHead)

    # The dev set where the MVR columns vary but carry no signal: this reaches
    # the fitting branch rather than the all-constant branch above.
    pipeline = QACDPipeline(
        provider=MockProvider(image_text="", sampled_pool=[]),
        config=QACDConfig(k=3),
    )
    pipeline.fit(_inverting_dev_set())
    assert pipeline.fusion_fitted_ is False
    assert any("negative weight" in w for w in pipeline.fit_warnings)


def test_fusion_head_is_skipped_on_constant_mvr_features():
    """One question, so every MVR column is constant: nothing to learn from."""
    records = [
        {
            "question": "What brand is the camera?",
            "answer": "Dakota Digital" if i % 2 == 0 else "Nikon Coolpix",
            "image": "/data/x.jpg",
            "failed": i % 2,
            "group": f"img{i}",
        }
        for i in range(20)
    ]
    pipeline = QACDPipeline(
        provider=MockProvider(image_text="", sampled_pool=[]), config=QACDConfig(k=3)
    )
    pipeline.fit(records)
    assert pipeline.fusion_fitted_ is False
    assert any("constant" in w for w in pipeline.fit_warnings)


# ---------------------------------------------------------------------------
# 3. Provenance: a mock-fitted scorer must say so, forever
# ---------------------------------------------------------------------------

def test_provider_kind_classifies_the_shipped_providers():
    from qacd.providers import LLaVAProvider, RapidOCRProvider

    assert provider_kind(MockProvider()) == "mock"
    assert provider_kind(RapidOCRProvider(engine=object())) == "rapidocr"
    assert provider_kind(LLaVAProvider(model_path="/x")) == "llava"
    assert provider_kind(None) == "none"
    assert "mock" in NON_EVIDENCE_PROVIDERS


def test_scorer_records_what_it_was_fitted_on(tmp_path):
    pipeline = QACDPipeline(provider=MockProvider(image_text="a"))
    pipeline.fit(_inverting_dev_set())
    assert pipeline.provenance["provider_kind"] == "mock"
    assert pipeline.provenance["carries_model_evidence"] is False
    assert any("no model evidence" in w for w in pipeline.fit_warnings)

    path = pipeline.save(tmp_path / "scorer.json")
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["provenance"]["provider_kind"] == "mock"


def test_a_reloaded_mock_scorer_still_warns(tmp_path):
    pipeline = QACDPipeline(provider=MockProvider(image_text="a"))
    pipeline.fit(_inverting_dev_set())
    path = pipeline.save(tmp_path / "scorer.json")

    reloaded = QACDPipeline.load(path, provider=MockProvider(image_text="a"))
    assert any("no model evidence" in w for w in reloaded.fit_warnings)

    result = reloaded.score("What brand is the camera?", "Dakota Digital", "/data/x.jpg")
    assert any("no model evidence" in w for w in result.warnings), (
        "the warning must reach the scored response, not just the pipeline object"
    )


def test_a_scorer_without_provenance_is_flagged(tmp_path):
    """Files written before provenance existed must not pass as trustworthy."""
    import json

    from qacd.pipeline import QACDPipeline as P

    pipeline = P(provider=MockProvider(image_text="a"))
    pipeline.fit(_inverting_dev_set())
    payload = pipeline.state()
    payload.pop("provenance", None)
    path = tmp_path / "old.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    reloaded = P.load(path, provider=MockProvider(image_text="a"))
    assert any("no provenance" in w for w in reloaded.fit_warnings)


# ---------------------------------------------------------------------------
# 4. The CLI must not pick an evidence source on the user's behalf
# ---------------------------------------------------------------------------

def test_fit_and_score_require_an_explicit_provider():
    parser = build_parser()
    for argv in (
        ["fit", "--data", "d.jsonl", "--out", "s.json"],
        ["score", "--question", "q", "--answer", "a"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def test_provider_choices_are_exactly_the_supported_ones():
    args = build_parser().parse_args(
        ["fit", "--data", "d.jsonl", "--out", "s.json", "--provider", "mock"]
    )
    assert args.provider == "mock"
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["fit", "--data", "d.jsonl", "--out", "s.json", "--provider", "nonsense"]
        )


def test_llava_without_a_model_path_explains_what_is_missing(capsys):
    from qacd.cli import build_provider

    args = build_parser().parse_args(
        ["fit", "--data", "d.jsonl", "--out", "s.json", "--provider", "llava"]
    )
    with pytest.raises(SystemExit) as excinfo:
        build_provider(args)
    assert "--model-path" in str(excinfo.value)


def test_mock_selection_is_loud(capsys):
    from qacd.cli import build_provider

    args = build_parser().parse_args(
        ["fit", "--data", "d.jsonl", "--out", "s.json", "--provider", "mock"]
    )
    build_provider(args)
    stderr = capsys.readouterr().err
    assert "plumbing" in stderr and "no model" in stderr


# ---------------------------------------------------------------------------
# 5. Bad input must be reported, not traced back
# ---------------------------------------------------------------------------

def test_fit_reports_a_missing_development_set(capsys):
    from qacd.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["fit", "--data", "definitely_absent.jsonl", "--out", "s.json", "--provider", "mock"])
    message = str(excinfo.value)
    assert "definitely_absent.jsonl" in message
    assert "question" in message, "the message should show the expected record shape"


def test_fit_reports_malformed_jsonl_with_a_line_number(tmp_path):
    from qacd.cli import main

    path = tmp_path / "dev.jsonl"
    path.write_text('{"question": "q"}\nnot json\n', encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        main(["fit", "--data", str(path), "--out", "s.json", "--provider", "mock"])
    assert ":2" in str(excinfo.value), "the offending line number must be named"


def test_fit_accepts_a_utf8_bom(tmp_path):
    """PowerShell's `Out-File -Encoding utf8` writes a BOM; the loader must cope.

    Without this the first line fails with a message about UTF-8 that reads like
    file corruption rather than a byte-order mark.
    """
    import json

    from qacd.cli import _load_records

    path = tmp_path / "dev.jsonl"
    payload = {
        "question": "What brand?",
        "answer": "Dakota Digital",
        "image": "/x.jpg",
        "failed": 0,
        "group": "/x.jpg",
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8-sig")
    assert path.read_bytes()[:3] == b"\xef\xbb\xbf", "fixture must really carry a BOM"
    records = _load_records(str(path))
    assert records[0]["question"] == "What brand?"


def test_score_reports_a_missing_scorer_file():
    from qacd.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(
            ["score", "--question", "q", "--answer", "a", "--scorer", "absent.json",
             "--provider", "mock"]
        )
    assert "absent.json" in str(excinfo.value)


def test_an_empty_development_set_is_rejected(tmp_path):
    from qacd.cli import main

    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        main(["fit", "--data", str(path), "--out", "s.json", "--provider", "mock"])
    assert "no records" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 6. MVR depends on OCR, and that must not be a silent dead channel
# ---------------------------------------------------------------------------

def test_mvr_is_degenerate_without_ocr_text():
    """The coupling that makes `--k 3` without OCR a waste of generations.

    Documents the mechanism the fusion guard is reacting to: support for a
    resample is decided by comparing it against the OCR text, so with no OCR
    every sample is unsupported no matter what it says.
    """
    from qacd.mechanical import MVR_FEATURES, mvr_features

    samples = ["Flickr", "Flickr", "Pinterest"]
    without = mvr_features(samples, [], k=3)
    with_ocr = mvr_features(samples, ["Flickr"], k=3)

    assert without["mvr_unsupported_rate"] == 1.0
    assert without["mvr_fuzzy_mean"] == 0.0
    assert len({without[name] for name in MVR_FEATURES}) == 2, "expected a constant channel"

    assert with_ocr["mvr_unsupported_rate"] < 1.0
    assert with_ocr["mvr_fuzzy_std"] > 0.0, "with OCR the channel carries spread"


def test_ocr_less_deployment_is_caught_at_fit_time():
    """The pipeline must not silently fit a head on a dead MVR channel."""
    pipeline = QACDPipeline(
        provider=MockProvider(image_text="", sampled_pool=[]), config=QACDConfig(k=3)
    )
    pipeline.fit(_inverting_dev_set())
    assert not pipeline.fusion_fitted_
    assert any("no signal" in w or "constant" in w for w in pipeline.fit_warnings)
