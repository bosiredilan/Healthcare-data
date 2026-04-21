#!/usr/bin/env python
"""
scripts/kenyanize.py
Kenyan adaptation script for AfyaPredict KE.
Transforms cleaned_healthcare.csv into a Kenyan-contextual dataset
while keeping the exact same schema and row count.
"""

from pathlib import Path

import pandas as pd

CLEANED_CSV = Path("data/cleaned_healthcare.csv")
KENYAN_CSV = Path("data/cleaned_healthcare_ke.csv")

INSURANCE_MAP = {
    "Medicare": "NHIF",
    "Aetna": "Jubilee Insurance",
    "Unitedhealthcare": "AAR Insurance",
    "Cigna": "CIC Insurance",
    "Blue Cross": "Britam",
}

MEDICAL_CONDITION_MAP = {
    "Diabetes": "Diabetes",
    "Hypertension": "Hypertension",
    "Asthma": "Pneumonia",
    "Obesity": "Malaria",
    "Arthritis": "Typhoid",
    "Cancer": "HIV/AIDS",
}

MEDICATION_MAP = {
    "Aspirin": "Metformin",
    "Ibuprofen": "Amoxicillin",
    "Paracetamol": "Coartem",
    "Penicillin": "Tenofovir",
    "Lipitor": "Sulphadoxine-Pyrimethamine",
}

USD_TO_KES_RATE = 130.0


def kenyanize_dataset() -> None:
    if not CLEANED_CSV.exists():
        raise FileNotFoundError(
            f"❌ {CLEANED_CSV} not found. Run scripts/clean.py first."
        )

    print("🔄 Loading cleaned dataset...")
    df = pd.read_csv(CLEANED_CSV)
    original_shape = df.shape
    print(f"   Loaded {original_shape[0]:,} rows × {original_shape[1]} columns")

    print("   → Mapping Insurance Provider...")
    df["Insurance Provider"] = df["Insurance Provider"].replace(INSURANCE_MAP)

    print("   → Mapping Medical Condition...")
    df["Medical Condition"] = df["Medical Condition"].replace(MEDICAL_CONDITION_MAP)

    print("   → Mapping Medication...")
    df["Medication"] = df["Medication"].replace(MEDICATION_MAP)

    print("   → Converting Billing Amount to KES...")
    df["Billing Amount"] = (df["Billing Amount"] * USD_TO_KES_RATE).round(2)

    df["dataset_version"] = "KE"

    KENYAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(KENYAN_CSV, index=False)
    print(f"✅ Kenyanised dataset saved → {KENYAN_CSV}")

    print("\n📊 Transformation Summary:")
    original_df = pd.read_csv(CLEANED_CSV)
    for col in ["Insurance Provider", "Medical Condition", "Medication", "Billing Amount"]:
        if col in df.columns:
            if col == "Billing Amount":
                print(
                    f"   {col}:\n"
                    f"      Before → range [{original_df[col].min():.2f}, {original_df[col].max():.2f}] USD\n"
                    f"      After  → range [{df[col].min():.2f}, {df[col].max():.2f}] KES\n"
                )
            else:
                print(
                    f"   {col}:\n"
                    f"      Before → {sorted(original_df[col].unique())}\n"
                    f"      After  → {sorted(df[col].unique())}\n"
                )

    print(f"🎉 Kenyan adaptation complete! {df.shape[0]:,} rows ready for loading.")


if __name__ == "__main__":
    kenyanize_dataset()