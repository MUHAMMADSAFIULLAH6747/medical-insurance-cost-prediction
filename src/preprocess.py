"""Shared preprocessing helpers for training and inference."""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region_northwest",
    "region_southeast",
    "region_southwest",
]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and encode categorical features for modeling."""
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    cleaned["sex"] = cleaned["sex"].map({"female": 0, "male": 1})
    cleaned["smoker"] = cleaned["smoker"].map({"no": 0, "yes": 1})
    cleaned = pd.get_dummies(cleaned, columns=["region"], drop_first=True, dtype=int)

    # Guarantee expected one-hot columns exist even if a region is missing.
    for col in ["region_northwest", "region_southeast", "region_southwest"]:
        if col not in cleaned.columns:
            cleaned[col] = 0

    return cleaned


def encode_single_input(
    age: int,
    sex: str,
    bmi: float,
    children: int,
    smoker: str,
    region: str,
) -> pd.DataFrame:
    """Encode one user profile into the model feature matrix."""
    row = {
        "age": age,
        "sex": 1 if sex.lower() == "male" else 0,
        "bmi": bmi,
        "children": children,
        "smoker": 1 if smoker.lower() == "yes" else 0,
        "region_northwest": 1 if region.lower() == "northwest" else 0,
        "region_southeast": 1 if region.lower() == "southeast" else 0,
        "region_southwest": 1 if region.lower() == "southwest" else 0,
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
