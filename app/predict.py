from pathlib import Path
import numpy as np
import joblib

MODEL_PATH = Path("models/model.joblib")
ENCODERS_PATH = Path("models/encoders.joblib")

CATEGORICAL_COLS = [
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Insurance Provider",
    "Admission Type",
    "Medication",
]

FEATURE_ORDER = [
    "Age",
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Insurance Provider",
    "Billing Amount",
    "Admission Type",
    "Medication",
]

_model = None
_artifacts: dict | None = None


def load_model() -> bool:
    global _model, _artifacts
    if not MODEL_PATH.exists() or not ENCODERS_PATH.exists():
        return False
    _model = joblib.load(MODEL_PATH)
    _artifacts = joblib.load(ENCODERS_PATH)
    return True


def is_model_loaded() -> bool:
    return _model is not None and _artifacts is not None


def get_model_version() -> str:
    if not is_model_loaded():
        return "unloaded"
    return _artifacts.get("version", "unknown-KE")


def predict(request_data: dict) -> dict:
    if not is_model_loaded():
        raise RuntimeError("Model is not loaded. Run python ml/train.py first.")

    encoders = _artifacts["encoders"]
    scaler = _artifacts["scaler"]
    label_encoder = _artifacts["label_encoder"]

    row = [
        request_data["age"],
        encoders["Gender"].transform([request_data["gender"]])[0],
        encoders["Blood Type"].transform([request_data["blood_type"]])[0],
        encoders["Medical Condition"].transform([request_data["medical_condition"]])[0],
        encoders["Insurance Provider"].transform([request_data["insurance_provider"]])[0],
        request_data["billing_amount"],
        encoders["Admission Type"].transform([request_data["admission_type"]])[0],
        encoders["Medication"].transform([request_data["medication"]])[0],
    ]

    X = np.array(row, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred_encoded = _model.predict(X_scaled)[0]
    proba = _model.predict_proba(X_scaled)[0]
    classes = label_encoder.classes_

    prediction = label_encoder.inverse_transform([pred_encoded])[0]
    probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

    return {
        "prediction": prediction,
        "probabilities": probabilities,
        "model_version": get_model_version(),
    }