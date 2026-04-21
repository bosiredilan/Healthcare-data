import os
import sys
from unittest.mock import MagicMock, patch

import pytest

os.environ["DATABASE_URL"] = "sqlite:///./test_afyapredict.db"
os.environ["RETRAIN_API_KEY"] = "test-api-key-ke"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VALID_PAYLOAD = {
    "age": 34,
    "gender": "Female",
    "blood_type": "B+",
    "medical_condition": "Malaria",
    "insurance_provider": "NHIF",
    "billing_amount": 45500.0,
    "admission_type": "Emergency",
    "medication": "Coartem",
}

MOCK_RESULT = {
    "prediction": "Normal",
    "probabilities": {"Normal": 0.60, "Abnormal": 0.25, "Inconclusive": 0.15},
    "model_version": "20240101-120000-KE",
}


@pytest.fixture(scope="session")
def client():
    p_model = patch("app.predict.is_model_loaded", return_value=True)
    p_load = patch("app.predict.load_model", return_value=True)
    p_tables = patch("app.database.create_tables")
    p_model.start()
    p_load.start()
    p_tables.start()

    from app.database import get_db
    from app.main import app
    from fastapi.testclient import TestClient

    mock_db = MagicMock()
    mock_db.add = MagicMock()
    mock_db.commit = MagicMock()

    def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    app.dependency_overrides.clear()
    p_model.stop()
    p_load.stop()
    p_tables.stop()