"""Download the Kaggle Medical Cost Personal Dataset into data/raw/."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import kagglehub


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TARGET = RAW_DIR / "insurance.csv"


def download_dataset() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = kagglehub.dataset_download("mirichoi0218/insurance")
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV found in downloaded Kaggle dataset.")
    source = Path(path) / csv_files[0]
    shutil.copy2(source, TARGET)
    print(f"Dataset saved to: {TARGET}")
    return TARGET


if __name__ == "__main__":
    download_dataset()
