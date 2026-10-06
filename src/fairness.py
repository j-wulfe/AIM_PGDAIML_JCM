"""
fairness.py
-----------
Fairness metrics. IMPORTANT: this dataset has NO sensitive attributes
(no gender/race/age). A true demographic audit is impossible here - that is
itself an ethical finding. These functions demonstrate the methodology on a
DECLARED PROXY (transaction Amount as a rough socioeconomic stand-in).
Treat results as method demonstration, not proof of demographic fairness.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def amount_proxy_groups(amount_test: pd.Series) -> np.ndarray:
    """Split into high/low value groups at the median Amount (declared proxy)."""
    med = amount_test.median()
    return np.where(amount_test.values > med, "high_value", "low_value")


def group_metrics(y_true, y_pred, groups) -> pd.DataFrame:
    """Per-group flag rate (selection rate), TPR (recall), and FPR."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = []
    for g in np.unique(groups):
        m = groups == g
        yt, yp = y_true[m], y_pred[m]
        tp = int(((yp == 1) & (yt == 1)).sum()); fn = int(((yp == 0) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum()); tn = int(((yp == 0) & (yt == 0)).sum())
        rows.append({"group": g, "n": int(m.sum()),
                     "flag_rate": yp.mean(),
                     "TPR": tp / (tp + fn) if tp + fn else np.nan,
                     "FPR": fp / (fp + tn) if fp + tn else np.nan})
    return pd.DataFrame(rows)


def disparate_impact(fair: pd.DataFrame) -> float:
    """min/max flag rate. < 0.80 is the classic 80%-rule red flag."""
    return fair["flag_rate"].min() / fair["flag_rate"].max()


def fairness_report(y_true, y_pred, amount_test) -> dict:
    """Run the full proxy audit and return a summary dict."""
    groups = amount_proxy_groups(amount_test)
    fair = group_metrics(y_true, y_pred, groups)
    return {
        "table": fair,
        "disparate_impact": disparate_impact(fair),
        "equal_opportunity_gap": abs(fair["TPR"].diff().iloc[-1]),
        "equalised_odds_gap": abs(fair["FPR"].diff().iloc[-1]),
    }
