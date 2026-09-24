"""HTTP service layer: the four platform endpoints, exercised end to end.

Uses FastAPI's TestClient, so every route, request model and response model in
``qacd/service.py`` is actually executed. Skipped when the service extra is not
installed.
"""

import pytest

fastapi = pytest.importorskip("fastapi", reason="service extra not installed")
from fastapi.testclient import TestClient  # noqa: E402

from qacd.pipeline import QACDConfig, QACDPipeline  # noqa: E402
from qacd.providers import MockProvider  # noqa: E402
from qacd.service import create_app  # noqa: E402

IMAGE_TEXT = "Dakota Digital; open 24 days"


def build_client(fit: bool = True, k: int | None = 3) -> TestClient:
    provider = MockProvider(
        image_text=IMAGE_TEXT,
        sampled_pool=["Dakota Digital", "Dakota Digital", "Nikon Coolpix"],
    )
    pipeline = QACDPipeline(provider=provider, config=QACDConfig(k=k))
    if fit:
        records = [
            {
                "question": "What brand is the camera?",
                "answer": "Dakota Digital" if i % 2 == 0 else "Nikon Coolpix",
                "image": IMAGE_TEXT,
                "failed": 0 if i % 2 == 0 else 1,
                "group": f"img{i}",
            }
            for i in range(120)
        ]
        pipeline.fit(records)
    return TestClient(create_app(pipeline))


def test_health_reports_scorer_state():
    client = build_client()
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["scorer_fitted"] is True
    assert body["mvr_enabled"] is True
    assert body["conformal_validated"] is False


def test_health_reports_an_unfitted_scorer():
    body = build_client(fit=False).get("/health").json()
    assert body["scorer_fitted"] is False


def test_risk_endpoint_returns_the_platform_contract():
    client = build_client()
    response = client.post(
        "/v1/qacd/risk",
        json={
            "question": "What brand is the camera?",
            "answer": "Dakota Digital",
            "image": "demo.jpg",
        },
    )
    assert response.status_code == 200
    body = response.json()
    for key in (
        "risk_score",
        "is_high_risk",
        "threshold",
        "num_claims",
        "claims",
        "feature_dim",
        "model_calls",
        "latency_ms",
        "version",
        "warnings",
        "channels",
    ):
        assert key in body, key
    assert 0.0 <= body["risk_score"] <= 1.0
    assert isinstance(body["is_high_risk"], bool)
    assert body["num_claims"] >= 1


def test_risk_endpoint_separates_correct_from_wrong_answers():
    client = build_client()
    ok = client.post(
        "/v1/qacd/risk", json={"question": "What brand?", "answer": "Dakota Digital", "image": "x.jpg"}
    ).json()
    bad = client.post(
        "/v1/qacd/risk", json={"question": "What brand?", "answer": "Nikon Coolpix", "image": "x.jpg"}
    ).json()
    assert ok["risk_score"] < bad["risk_score"]
    assert bad["is_high_risk"] is True


def test_return_claims_false_omits_the_detail():
    client = build_client()
    body = client.post(
        "/v1/qacd/risk",
        json={"question": "What brand?", "answer": "Dakota Digital", "return_claims": False},
    ).json()
    assert body["claims"] == []
    assert body["num_claims"] >= 1


def test_threshold_override_changes_only_the_flag():
    client = build_client()
    payload = {"question": "What brand?", "answer": "Nikon Coolpix", "image": "x.jpg"}
    strict = client.post("/v1/qacd/risk", json={**payload, "threshold": 0.999}).json()
    loose = client.post("/v1/qacd/risk", json={**payload, "threshold": 0.001}).json()
    assert strict["risk_score"] == loose["risk_score"]
    assert strict["is_high_risk"] is False and loose["is_high_risk"] is True


def test_risk_endpoint_rejects_a_missing_field():
    client = build_client()
    assert client.post("/v1/qacd/risk", json={"question": "only a question"}).status_code == 422


def test_mvr_endpoint_returns_the_seven_statistics():
    client = build_client()
    body = client.post(
        "/v1/qacd/mvr",
        json={
            "sampled_answers": ["Dakota Digital", "Nikon Coolpix", "Dakota Digital"],
            "ocr_texts": ["Dakota Digital", "open 24 days"],
            "k": 3,
        },
    ).json()
    assert body["k_used"] == 3
    assert set(body["features"]) == {
        "mvr_unsupported_rate",
        "mvr_all_unsupported",
        "mvr_all_supported",
        "mvr_fuzzy_mean",
        "mvr_fuzzy_min",
        "mvr_fuzzy_std",
        "mvr_fuzzy_best_of_k",
    }
    assert 0.0 <= body["features"]["mvr_unsupported_rate"] <= 1.0


def test_mvr_endpoint_rejects_an_empty_sample_list():
    client = build_client()
    response = client.post("/v1/qacd/mvr", json={"sampled_answers": [], "ocr_texts": []})
    assert response.status_code == 400


def test_mechanical_endpoint_returns_the_ocr_subfamily():
    client = build_client()
    body = client.post(
        "/v1/qacd/mechanical",
        json={"claim_text": "Dakota Digital", "ocr_texts": ["Dakota Digital"]},
    ).json()
    assert body["features"]["mech_ocr_exact_present"] == 1.0
    assert body["features"]["mech_ocr_absent_risk"] == 0.0
    assert body["family_size_in_frozen_config"] == 28


def test_mechanical_endpoint_tolerates_empty_ocr():
    client = build_client()
    body = client.post("/v1/qacd/mechanical", json={"claim_text": "anything"}).json()
    assert body["features"]["mech_ocr_empty"] == 1.0


def test_select_endpoint_runs_the_conformal_procedure():
    client = build_client()
    body = client.post(
        "/v1/qacd/select",
        json={
            "calibration_null_scores": [0.05, 0.1, 0.15, 0.2, 0.25],
            "test_scores": [0.02, 0.9],
            "alpha": 0.2,
            "procedure": "BH",
            "test_labels": [0, 1],
        },
    ).json()
    assert body["num_items"] == 2
    assert len(body["accepted"]) == 2
    assert body["validated"] is False
    assert isinstance(body["notes"], list)


def test_select_endpoint_rejects_a_bad_procedure():
    client = build_client()
    response = client.post(
        "/v1/qacd/select",
        json={"calibration_null_scores": [0.1], "test_scores": [0.2], "procedure": "bonferroni"},
    )
    assert response.status_code == 400


def test_select_endpoint_rejects_an_out_of_range_alpha():
    client = build_client()
    response = client.post(
        "/v1/qacd/select",
        json={"calibration_null_scores": [0.1], "test_scores": [0.2], "alpha": 2.0},
    )
    assert response.status_code == 400
