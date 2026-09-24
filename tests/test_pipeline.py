"""End-to-end pipeline: fitting, scoring, persistence, interface contract."""

import json

import pytest

from qacd.features import ALL_FEATURE_NAMES, assemble_claim_features, build_matrix
from qacd.pipeline import QACDConfig, QACDPipeline, REFERENCE_FEATURE_NAMES
from qacd.providers import DirectVerification, MockProvider, VerificationView


def _dev_records(n=120):
    """Synthetic development set.

    Answers that share words with the image text are labelled correct; answers
    that do not are labelled wrong. This gives the calibrator a real, learnable
    signal so the test exercises the fitting path rather than a constant.
    """
    records = []
    for i in range(n):
        if i % 2 == 0:
            answer, failed = "Dakota Digital", 0
        else:
            answer, failed = "Nikon Coolpix", 1
        records.append(
            {
                "question": "What brand is the camera?",
                "answer": answer,
                "image": "Dakota Digital; open 24 days",
                "failed": failed,
                "group": f"img{i}",
            }
        )
    return records


def _provider(sampled=None):
    return MockProvider(image_text="Dakota Digital; open 24 days", sampled_pool=sampled or [])


def test_every_declared_feature_name_is_produced():
    row = assemble_claim_features(
        claim_text="The answer to the question 'What brand?' is Dakota digital.",
        views=[VerificationView(view="independent", support=0.9)],
        direct=DirectVerification(support=0.9, evidence_phrase="Dakota Digital"),
        ocr_texts=["Dakota Digital"],
    )
    for name in ALL_FEATURE_NAMES:
        assert name in row, name
    matrix = build_matrix([row], ALL_FEATURE_NAMES)
    assert matrix.shape == (1, len(ALL_FEATURE_NAMES))


def test_scoring_before_fitting_is_flagged_not_silent():
    pipeline = QACDPipeline(provider=_provider())
    result = pipeline.score("What brand is the camera?", "Dakota Digital")
    assert 0.0 <= result.risk_score <= 1.0
    assert any("not fitted" in w for w in result.warnings)
    assert result.feature_dim == len(REFERENCE_FEATURE_NAMES)


def test_fit_then_score_separates_correct_from_wrong():
    pipeline = QACDPipeline(provider=_provider())
    pipeline.fit(_dev_records())
    assert pipeline.fitted_

    good = pipeline.score("What brand is the camera?", "Dakota Digital")
    bad = pipeline.score("What brand is the camera?", "Nikon Coolpix")
    assert good.risk_score < bad.risk_score
    assert not good.warnings or all("not fitted" not in w for w in good.warnings)


def test_response_payload_contract():
    pipeline = QACDPipeline(provider=_provider())
    pipeline.fit(_dev_records())
    result = pipeline.score("What brand is the camera?", "Dakota Digital", "demo.jpg")
    payload = result.to_dict()
    for key in [
        "risk_score",
        "is_high_risk",
        "threshold",
        "num_claims",
        "claims",
        "feature_dim",
        "latency_ms",
        "version",
        "warnings",
        "channels",
    ]:
        assert key in payload, key
    assert isinstance(payload["claims"], list)
    assert isinstance(payload["is_high_risk"], bool)
    assert json.dumps(payload)  # must be JSON-serialisable as-is


def test_mvr_channel_engages_and_is_reported():
    pipeline = QACDPipeline(
        provider=_provider(sampled=["Dakota Digital", "Dakota Digital", "Nikon"]),
        config=QACDConfig(k=3),
    )
    pipeline.fit(_dev_records())
    assert pipeline.fusion_fitted_
    result = pipeline.score("What brand is the camera?", "Dakota Digital")
    assert "mvr_unsupported_rate" in result.channels
    assert "fused_score" in result.channels
    assert result.model_calls >= 3


def test_scorer_roundtrip_is_bit_identical(tmp_path):
    pipeline = QACDPipeline(provider=_provider(), config=QACDConfig(k=3))
    pipeline.fit(_dev_records())
    path = pipeline.save(tmp_path / "scorer.json")

    reloaded = QACDPipeline.load(path, provider=_provider())
    assert reloaded.fitted_
    assert reloaded.fusion_fitted_
    assert reloaded.feature_names == pipeline.feature_names

    a = pipeline.score("What brand is the camera?", "Dakota Digital").risk_score
    b = reloaded.score("What brand is the camera?", "Dakota Digital").risk_score
    assert a == pytest.approx(b, abs=1e-9)


def test_scorer_file_is_json_and_pickle_free(tmp_path):
    pipeline = QACDPipeline(provider=_provider())
    pipeline.fit(_dev_records())
    path = pipeline.save(tmp_path / "scorer.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "qacd-scorer"
    assert payload["calibrator"]["kind"] == "BICLiteCalibrator"
    assert "coef" in payload["calibrator"]


def test_loading_a_foreign_file_is_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": "something-else"}', encoding="utf-8")
    with pytest.raises(ValueError):
        QACDPipeline.load(bad)


def test_fit_rejects_records_that_produce_no_claims():
    with pytest.raises(ValueError):
        QACDPipeline(provider=_provider()).fit([])


def test_image_argument_reaches_the_provider_exactly_once():
    """Regression: score() used to drop the image, so claim-level mechanical
    features were always computed against an empty OCR result."""

    class _RecordingProvider(MockProvider):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ocr_calls = []

        def ocr(self, image):
            self.ocr_calls.append(image)
            return super().ocr(image)

    provider = _RecordingProvider(image_text="Dakota Digital; open 24 days")
    pipeline = QACDPipeline(provider=provider, config=QACDConfig(k=2))
    pipeline.fit(_dev_records())

    provider.ocr_calls.clear()
    pipeline.score("What brand is the camera?", "Dakota Digital", "photo_42.jpg")
    assert provider.ocr_calls == ["photo_42.jpg"], "image must be read once, and passed through"


def test_claim_rows_uses_the_supplied_ocr_texts():
    """Features must be built from the OCR of the image under test.

    The answer carries two spans so the fallback decomposition yields
    unconditional claims whose text can match the image text verbatim.
    """
    pipeline = QACDPipeline(provider=_provider())
    answer = "Dakota Digital; open 24 days"

    _, _, rows_hit, _ = pipeline._claim_rows("Describe.", answer, "img.jpg", ["Dakota Digital"])
    _, _, rows_miss, _ = pipeline._claim_rows("Describe.", answer, "img.jpg", ["completely unrelated"])

    assert rows_hit[0]["mech_ocr_exact_present"] > rows_miss[0]["mech_ocr_exact_present"]
    assert rows_hit[0]["mech_ocr_absent_risk"] < rows_miss[0]["mech_ocr_absent_risk"]


def test_threshold_controls_the_flag():
    pipeline = QACDPipeline(provider=_provider(), config=QACDConfig(threshold=0.999))
    pipeline.fit(_dev_records())
    assert pipeline.score("What brand is the camera?", "Nikon Coolpix").is_high_risk is False
