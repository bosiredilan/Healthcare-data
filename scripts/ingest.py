#!/usr/bin/env python
"""
scripts/ingest.py
Downloads the raw healthcare dataset from Kaggle into data/raw_healthcare.csv.
Requires KAGGLE_USERNAME and KAGGLE_KEY env vars (or ~/.kaggle/kaggle.json).
"""

import shutil
from pathlib import Path

RAW_CSV = Path("data/raw_healthcare.csv")
KAGGLE_DATASET = "prasad22/healthcare-dataset"
KAGGLE_FILENAME = "healthcare_dataset.csv"


def ingest() -> None:
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)

    if RAW_CSV.exists():
        print(f"✅ Raw dataset already present at {RAW_CSV} – skipping download.")
        return

    try:
        import kaggle

        print(f"📥 Downloading dataset '{KAGGLE_DATASET}' from Kaggle...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(RAW_CSV.parent),
            unzip=True,
        )

        downloaded = RAW_CSV.parent / KAGGLE_FILENAME
        if downloaded.exists():
            downloaded.rename(RAW_CSV)
        else:
            candidates = list(RAW_CSV.parent.glob("*.csv"))
            if not candidates:
                raise FileNotFoundError("No CSV found after Kaggle download.")
            candidates[0].rename(RAW_CSV)

        print(f"✅ Raw dataset saved → {RAW_CSV}")

    except ImportError:
        print("❌ kaggle package not installed. Run: pip install kaggle")
        raise SystemExit(1)
    except Exception as exc:
        print(f"❌ Kaggle download failed: {exc}")
        print(
            "   Manually place the CSV at data/raw_healthcare.csv and re-run.\n"
            "   Dataset URL: https://www.kaggle.com/datasets/prasad22/healthcare-dataset"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    ingest()