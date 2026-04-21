import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

CATEGORICAL_COLS = [
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Insurance Provider",
    "Admission Type",
    "Medication",
]

NUMERICAL_COLS = ["Age", "Billing Amount"]

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

TARGET_COL = "Test Results"


def fit_and_transform(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict, StandardScaler, LabelEncoder]:
    encoders: dict[str, LabelEncoder] = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df[TARGET_COL].astype(str))

    X = df[FEATURE_ORDER].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, encoders, scaler, target_encoder