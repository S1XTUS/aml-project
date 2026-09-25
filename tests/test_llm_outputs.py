from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.llm.sar_generator as sar
import src.models.risk_classifier as rc
from src.llm.kyc_validator import build_prompt

SAMPLE_CASE = {
    "case_id": "CASE_TEST_001",
    "from_account": "123456789",
    "from_bank": "First National Bank",
    "to_account": "987654321",
    "to_bank": "International Bank Ltd",
    "amount": 75000,
    "currency": "USD",
    "transaction_type": "wire_transfer",
    "transaction_time": "2024-01-15T02:30:00Z",
    "recipient_country": "offshore",
    "transaction_frequency": 15,
    "kyc_completeness": 0.6,
}


class FakeLLM:
    """Records the prompt and returns a canned narrative."""

    def __init__(self):
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.prompts.append(kwargs["messages"][-1]["content"])
        message = SimpleNamespace(content="**SUSPICIOUS ACTIVITY:** test narrative")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def fake_llm(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(sar, "get_llm_client", lambda: llm)
    return llm


@pytest.fixture
def generator(trained_classifier, anomaly_model_dir):
    gen = sar.IntegratedSARGenerator()
    gen.anomaly_detector = sar.ModelAnomalyDetector(model_dir=anomaly_model_dir)
    return gen


def test_sar_uses_trained_models(generator, fake_llm):
    output = generator.generate_enhanced_sar(dict(SAMPLE_CASE))
    risk = output["analysis_results"]["risk"]
    anomaly = output["analysis_results"]["anomaly"]
    explanation = output["analysis_results"]["explanation"]

    # Same score the /predict-risk endpoint would return for this case
    expected = rc.classify_transaction(rc.transaction_from_case(SAMPLE_CASE))
    assert risk.scoring_method == "model"
    assert risk.risk_score == expected["risk_score"]
    assert risk.risk_level == expected["risk_level"]

    assert anomaly.anomaly_type in {"ANOMALOUS", "NORMAL"}
    assert anomaly.time_series_anomalies == []  # no fabricated history

    assert explanation.top_features
    assert all(name in rc.get_classifier().feature_columns for name, _ in explanation.top_features)

    assert output["sar_narrative"].startswith("**SUSPICIOUS ACTIVITY:**")
    prompt = fake_llm.prompts[0]
    assert "MODEL DRIVERS" in prompt
    assert "High-risk jurisdiction involvement" in prompt
    assert "SANCTIONS_CHECK" in prompt and "KYC_REVIEW" in prompt


def test_sar_without_anomaly_model_still_reports_patterns(generator, fake_llm, tmp_path):
    generator.anomaly_detector = sar.ModelAnomalyDetector(model_dir=str(tmp_path / "missing"))
    anomaly = generator.analyze_transaction(dict(SAMPLE_CASE))["anomaly"]
    assert anomaly.anomaly_type == "UNAVAILABLE"
    assert "High-risk jurisdiction recipient" in anomaly.detected_patterns


def test_kyc_prompt_contains_fields():
    record = {key: f"<{key}>" for key in ["Customer Name", "Date of Birth", "Nationality",
                                          "Current Address", "Account Opening Date",
                                          "Source of Funds", "Occupation", "Red Flags"]}
    prompt = build_prompt(record)
    assert isinstance(prompt, str)
    assert all(value in prompt for value in record.values())


@pytest.fixture
def api_client(generator, fake_llm, monkeypatch):
    import app.api as api
    monkeypatch.setattr(api, "sar_generator", generator)
    return api, TestClient(api.app)


def test_api_predict_risk(api_client):
    _, client = api_client
    payload = {k: SAMPLE_CASE[k] for k in ["case_id", "from_bank", "from_account", "to_bank",
                                           "to_account", "amount", "currency", "transaction_type"]}
    payload["transaction_time"] = SAMPLE_CASE["transaction_time"]
    response = client.post("/predict-risk", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["scoring_method"] == "model"
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}


def test_api_validate_kyc_parses_fields(api_client, monkeypatch):
    api, client = api_client
    received = {}
    monkeypatch.setattr(api, "validate_kyc", lambda record: received.update(record) or "LOW risk")

    text = b"Customer Name: Jane Doe\nNationality: Panama\nSource of Funds: Consulting\n"
    response = client.post("/validate-kyc", files={"file": ("kyc.txt", text, "text/plain")})

    assert response.status_code == 200
    assert response.json()["validation_result"] == "LOW risk"
    assert received["Customer Name"] == "Jane Doe"


def test_api_validate_kyc_rejects_unstructured_text(api_client):
    _, client = api_client
    response = client.post("/validate-kyc", files={"file": ("kyc.txt", b"hello", "text/plain")})
    assert response.status_code == 400
