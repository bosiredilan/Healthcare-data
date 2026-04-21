#!/usr/bin/env python
"""
scripts/clean.py
Cleans raw_healthcare.csv → cleaned_healthcare.csv.
Removes duplicates, nulls, and normalises string casing.
Retains only the 8 model features + target column.
"""

from pathlib import Path

import pandas as pd

RAW_CSV = Path("data/raw_healthcare.csv")
CLEANED_CSV = Path("data/cleaned_healthcare.csv")

KEEP_COLS = [
    "Age",
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Insurance Provider",
    "Billing Amount",
    "Admission Type",
    "Medication",
    "Test Results",
]

TITLE_CASE_COLS = [
    "Gender",
    "Medical Condition",
    "Insurance Provider",
    "Admission Type",
    "Medication",
    "Test Results",
]


def clean() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"❌ {RAW_CSV} not found. Run scripts/ingest.py first."
        )

    print(f"📥 Loading raw dataset from {RAW_CSV}...")
    df = pd.read_csv(RAW_CSV)
    print(f"   Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df = df[KEEP_COLS].copy()

    before_dedup = len(df)
    df = df.drop_duplicates()
    removed_dupes = before_dedup - len(df)
    print(f"   Removed {removed_dupes:,} duplicate rows")

    before_null = len(df)
    df = df.dropna()
    removed_nulls = before_null - len(df)
    print(f"   Removed {removed_nulls:,} rows with null values")

    for col in TITLE_CASE_COLS:
        df[col] = df[col].astype(str).str.strip().str.title()

    df["Blood Type"] = df["Blood Type"].astype(str).str.strip().str.upper()
    df["Age"] = df["Age"].astype(int)
    df["Billing Amount"] = df["Billing Amount"].astype(float).round(2)

    CLEANED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_CSV, index=False)

    print(f"\n✅ Cleaned dataset saved → {CLEANED_CSV}")
    print(f"   Final shape: {len(df):,} rows × {len(df.columns)} columns")
    print(f"   Test Results distribution:\n{df['Test Results'].value_counts().to_string()}")


if __name__ == "__main__":
    clean()