"""
train.py
--------
Train the model slate, tune each threshold on train cross-validation (no leakage),
evaluate on the held-out test set, and save artifacts to models/.

Usage:
    python -m src.train --data data/creditcard.csv --out models
"""
from __future__ import annotations
import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from .data_prep import prepare, RANDOM_STATE
from .evaluate import best_threshold, evaluate


def build_models(y_train):
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=12, n_jobs=-1,
            class_weight="balanced", random_state=RANDOM_STATE),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, eval_metric="aucpr",
            scale_pos_weight=neg / pos, n_jobs=-1, random_state=RANDOM_STATE)
    except ImportError:
        print("[info] xgboost not installed - skipping (pip install xgboost)")
    return models


def main(data_path: str, out_dir: str, cv: int = 5):
    os.makedirs(out_dir, exist_ok=True)
    X_tr, X_te, y_tr, y_te, scaler, _ = prepare(data_path)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    results, fitted = [], {}
    for name, model in build_models(y_tr).items():
        print(f"[train] {name}")
        oof = cross_val_predict(model, X_tr, y_tr, cv=skf,
                                method="predict_proba", n_jobs=-1)[:, 1]
        thr = best_threshold(y_tr, oof, beta=2.0)
        model.fit(X_tr, y_tr)
        scores = model.predict_proba(X_te)[:, 1]
        results.append(evaluate(name, y_te, scores, thr))
        fitted[name] = model

    # unsupervised baseline
    print("[train] IsolationForest")
    iso = IsolationForest(n_estimators=200, contamination=float(y_tr.mean()),
                          random_state=RANDOM_STATE, n_jobs=-1).fit(X_tr)
    iso_scores = -iso.score_samples(X_te)
    iso_thr = float(np.quantile(iso_scores, 1 - y_tr.mean()))
    results.append(evaluate("IsolationForest", y_te, iso_scores, iso_thr))
    fitted["IsolationForest"] = iso

    # persist artifacts
    for name, model in fitted.items():
        joblib.dump(model, os.path.join(out_dir, f"{name}.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))

    comparison = (pd.DataFrame(results)
                  .sort_values("AUC_PR", ascending=False).reset_index(drop=True))
    comparison.to_csv(os.path.join(out_dir, "metrics_comparison.csv"), index=False)

    config = {"random_state": RANDOM_STATE, "cv": cv,
              "features": list(X_tr.columns),
              "thresholds": {r["model"]: r["threshold"] for r in results}}
    with open(os.path.join(out_dir, "configs.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\n=== Model comparison (ranked by AUC-PR) ===")
    print(comparison.to_string(index=False))
    print(f"\nArtifacts saved to {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--cv", type=int, default=5)
    args = ap.parse_args()
    main(args.data, args.out, args.cv)
