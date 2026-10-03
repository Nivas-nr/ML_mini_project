"""Loading and splitting the UCI Polish companies bankruptcy data."""
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split

from config import DATA_DIR, SEED, TEST_SIZE


def load_year(year: int) -> tuple[pd.DataFrame, pd.Series]:
    """Load one ARFF file and return (X, y). y = 1 means the firm went bankrupt."""
    raw, _ = arff.loadarff(DATA_DIR / f"{year}year.arff")
    df = pd.DataFrame(raw)
    df["class"] = df["class"].astype(str).str.contains("1").astype(int)

    # Exact duplicate rows would leak between train and test, so drop them first
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns="class").astype(float)
    y = df["class"]
    return X, y


def split(X: pd.DataFrame, y: pd.Series):
    """Stratified 80/20 split so both sets keep the same bankrupt share."""
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)


def describe(X: pd.DataFrame, y: pd.Series) -> dict:
    return {
        "rows": len(X),
        "features": X.shape[1],
        "bankrupt": int(y.sum()),
        "bankrupt_pct": round(100 * y.mean(), 2),
        "missing_cells": int(X.isna().sum().sum()),
        "rows_with_missing": int(X.isna().any(axis=1).sum()),
    }
