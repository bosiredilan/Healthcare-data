#!/usr/bin/env python
"""
scripts/load.py
Idempotently loads cleaned_healthcare_ke.csv into the PostgreSQL patients table.
Uses replace strategy – safe for dev / scheduled retraining runs.
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

KENYAN_CSV = Path("data/cleaned_healthcare_ke.csv")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/afyapredict",
)

COLUMN_RENAME = {
    "Age": "age",
    "Gender": "gender",
    "Blood Type": "blood_type",
    "Medical Condition": "medical_condition",
    "Insurance Provider": "insurance_provider",
    "Billing Amount": "billing_amount",
    "Admission Type": "admission_type",
    "Medication": "medication",
    "Test Results": "test_results",
    "dataset_version": "dataset_version",
}

CREATE_PATIENTS_SQL = """
CREATE TABLE IF NOT EXISTS patients (
    id          SERIAL PRIMARY KEY,
    age         INTEGER        NOT NULL,
    gender      VARCHAR(10)    NOT NULL,
    blood_type  VARCHAR(5)     NOT NULL,
    medical_condition  VARCHAR(60) NOT NULL,
    insurance_provider VARCHAR(60) NOT NULL,
    billing_amount     FLOAT       NOT NULL,
    admission_type     VARCHAR(20) NOT NULL,
    medication         VARCHAR(60) NOT NULL,
    test_results       VARCHAR(20),
    dataset_version    VARCHAR(5)  DEFAULT 'KE',
    created_at  TIMESTAMP      DEFAULT NOW()
);
"""


def load() -> None:
    if not KENYAN_CSV.exists():
        raise FileNotFoundError(
            f"❌ {KENYAN_CSV} not found. Run scripts/kenyanize.py first."
        )

    print(f"📥 Reading {KENYAN_CSV}...")
    df = pd.read_csv(KENYAN_CSV)
    print(f"   {len(df):,} rows loaded")

    df = df.rename(columns=COLUMN_RENAME)
    valid_cols = [c for c in COLUMN_RENAME.values() if c in df.columns]
    df = df[valid_cols]

    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        conn.execute(text(CREATE_PATIENTS_SQL))
        conn.commit()
        print("   patients table ensured ✓")

    df.to_sql(
        "patients",
        engine,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=1000,
    )

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM patients")).scalar()

    print(f"✅ Loaded {count:,} rows into patients table")


if __name__ == "__main__":
    load()