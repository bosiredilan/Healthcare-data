#!/usr/bin/env python
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from ml.preprocess import TARGET_COL, fit_and_transform

DATA_PATH = Path("data/cleaned_healthcare_ke.csv")
MODEL_DIR = Path("models")


def train() -> tuple[str, float, str]:
    MODEL_DIR.mkdir(exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Kenyan dataset not found at {DATA_PATH}. "
            "Run scripts/kenyanize.py first."
        )

    print(f"📥 Loading Kenyan dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"   {len(df):,} rows × {len(df.columns)} columns")

    if "dataset_version" in df.columns:
        df = df.drop(columns=["dataset_version"])

    df = df.dropna(subset=[TARGET_COL])
    print(f"   Target distribution:\n{df[TARGET_COL].value_counts().to_string()}\n")

    X, y, encoders, scaler, label_encoder = fit_and_transform(df.copy())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🌲 Training RandomForest (n=50, max_depth=15)...")
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_f1 = f1_score(y_test, rf_preds, average="macro")
    print(f"   RandomForest macro F1: {rf_f1:.4f}")

    print("⚡ Training XGBoost (n=100, max_depth=6)...")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_f1 = f1_score(y_test, xgb_preds, average="macro")
    print(f"   XGBoost macro F1:      {xgb_f1:.4f}\n")

    if rf_f1 >= xgb_f1:
        best_model = rf
        best_algo = "RandomForest"
        best_f1 = rf_f1
        best_preds = rf_preds
    else:
        best_model = xgb
        best_algo = "XGBoost"
        best_f1 = xgb_f1
        best_preds = xgb_preds

    print(f"🏆 Best model: {best_algo} (macro F1 = {best_f1:.4f})")
    print("\nClassification Report:")
    print(classification_report(y_test, best_preds, target_names=label_encoder.classes_))

    version_tag = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-KE"

    joblib.dump(best_model, MODEL_DIR / "model.joblib")
    joblib.dump(
        {
            "encoders": encoders,
            "scaler": scaler,
            "label_encoder": label_encoder,
            "algorithm": best_algo,
            "macro_f1": best_f1,
            "version": version_tag,
        },
        MODEL_DIR / "encoders.joblib",
    )

    print(f"\n✅ model.joblib    → {MODEL_DIR}/model.joblib")
    print(f"✅ encoders.joblib → {MODEL_DIR}/encoders.joblib")
    print(f"🎉 Version tag: {version_tag}")

    return best_algo, best_f1, version_tag


if __name__ == "__main__":
    train()