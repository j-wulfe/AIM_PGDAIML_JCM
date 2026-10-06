"""
data_prep.py
------------
Reproducible data loading and preprocessing for the fraud-detection pipeline.

Design choices (see reports/Final_Report.md for full justification):
- Drop exact duplicate rows (they inflate validation scores).
- Engineer behavioural features (Hour, Log_Amount, Amount_bin).
- Split BEFORE scaling and fit the scaler on train only -> no data leakage.
- Keep outliers: in fraud detection the tails are the signal.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42
SCALE_COLS = ["Time", "Amount", "Log_Amount", "Hour"]


def load_data(path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Load the dataset and drop exact duplicate rows."""
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-derived features. Pure function: returns a new frame."""
    df = df.copy()
    df["Hour"] = (df["Time"] // 3600) % 24          # hour of day
    df["Log_Amount"] = np.log1p(df["Amount"])       # tame right skew
    df["Amount_bin"] = pd.qcut(df["Amount"], q=5, labels=False, duplicates="drop")
    return df


def split_and_scale(df: pd.DataFrame, test_size: float = 0.20,
                    random_state: int = RANDOM_STATE):
    """
    Stratified split, then RobustScaler fit on train only.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, amount_test
        amount_test is the UNSCALED Amount for the test rows (used by fairness.py).
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]
    amount_raw = df["Amount"]

    X_train, X_test, y_train, y_test, amt_train, amt_test = train_test_split(
        X, y, amount_raw, test_size=test_size, stratify=y, random_state=random_state)

    scaler = RobustScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])  # fit on TRAIN only
    X_test[SCALE_COLS] = scaler.transform(X_test[SCALE_COLS])

    return X_train, X_test, y_train, y_test, scaler, amt_test


def prepare(path: str = "data/creditcard.csv"):
    """Convenience: load -> features -> split/scale."""
    return split_and_scale(add_features(load_data(path)))


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, _, _ = prepare()
    print(f"Train {X_tr.shape} | Test {X_te.shape}")
    print(f"Train fraud %: {y_tr.mean()*100:.3f} | Test fraud %: {y_te.mean()*100:.3f}")
