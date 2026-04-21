from unittest.mock import patch
import pytest
from tests.conftest import VALID_PAYLOAD, MOCK_RESULT


def test_health_returns_200(client):
    assert client.get("/health").status_code == 200

def test_health_status_ok(client):
    assert client.get("/health").json()["status"] == "ok"

def test_health_dataset_version_ke(client):
    assert client.get("/health").json()["dataset_version"] == "KE"

def test_health_has_version_field(client):
    assert "version" in client.get("/health").json()

def test_predict_valid_returns_200(client):
    with patch("app.main.run_predict", return_value=MOCK_RESULT):
        assert client.post("/predict", json=VALID_PAYLOAD).status_code == 200

def test_predict_response_has_prediction(client):
    with patch("app.main.run_predict", return_value=MOCK_RESULT):
        assert "prediction" in client.post("/predict", json=VALID_PAYLOAD).json()

def test_predict_response_has_probabilities(client):
    with patch("app.main.run_predict", return_value=MOCK_RESULT):
        assert "probabilities" in client.post("/predict", json=VALID_PAYLOAD).json()

def test_predict_probabilities_sum_to_one(client):
    with patch("app.main.run_predict", return_value=MOCK_RESULT):
        probs = client.post("/predict", json=VALID_PAYLOAD).json()["probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 0.01

def test_predict_prediction_is_valid_class(client):
    with patch("app.main.run_predict", return_value=MOCK_RESULT):
        pred = client.post("/predict", json=VALID_PAYLOAD).json()["prediction"]
        assert pred in ["Normal", "Abnormal", "Inconclusive"]

def test_predict_context_is_ke(client):
    with patch("app.main.run_predict", return_value=MOCK_RESULT):
        assert client.post("/predict", json=VALID_PAYLOAD).json().get("context") == "KE"

def test_predict_missing_age_returns_422(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
    assert client.post("/predict", json=payload).status_code == 422

def test_predict_invalid_gender_returns_422(client):
    assert client.post("/predict", json={**VALID_PAYLOAD, "gender": "Other"}).status_code == 422

def test_predict_invalid_blood_type_returns_422(client):
    assert client.post("/predict", json={**VALID_PAYLOAD, "blood_type": "C+"}).status_code == 422

def test_predict_invalid_medical_condition_returns_422(client):
    assert client.post("/predict", json={**VALID_PAYLOAD, "medical_condition": "Ebola"}).status_code == 422

def test_predict_invalid_insurance_provider_returns_422(client):
    assert client.post("/predict", json={**VALID_PAYLOAD, "insurance_provider": "BlueCross"}).status_code == 422

def test_predict_invalid_admission_type_returns_422(client):
    assert client.post("/predict", json={**VALID_PAYLOAD, "admission_type": "Walk-in"}).status_code == 422

def test_predict_invalid_medication_returns_422(client):
    assert client.post("/predict", json={**VALID_PAYLOAD, "medication": "Aspirin"}).status_code == 422

def test_predict_negative_billing_returns_422(client):
    assert client.post("/predict", json={**VALID_PAYLOAD, "billing_amount": -100.0}).status_code == 422

def test_predict_age_over_120_returns_422(client):
    assert client.post("/predict", json={**VALID_PAYLOAD, "age": 150}).status_code == 422

def test_retrain_without_key_returns_422(client):
    assert client.post("/retrain").status_code == 422

def test_retrain_with_wrong_key_returns_401(client):
    assert client.post("/retrain", headers={"x-api-key": "wrong-key"}).status_code == 401

def test_docs_returns_200(client):
    assert client.get("/docs").status_code == 200

def test_openapi_json_returns_200(client):
    assert client.get("/openapi.json").status_code == 200