# Credit Card Fraud Detection

A reproducible machine-learning pipeline for detecting fraudulent credit-card
transactions under **extreme class imbalance** (~0.17% fraud). Built as a data-science
capstone, structured like an open-source project.

> **Status:** educational / capstone project. Metrics in the report and decks are
> illustrative placeholders — run the pipeline to produce your own.

---

## Problem

Fraud costs an institution twice: a missed fraud is a direct loss, and a wrongly blocked
legitimate transaction loses a customer. This project frames the task as **supervised binary
classification** with a strong anomaly-detection character, and optimises for the trade-off
between fraud caught and false alarms.

- **Task:** binary classification (fraud vs. legitimate)
- **Primary metric:** **AUC-PR** (precision-recall AUC) — the right summary for rare positives
- **Secondary:** recall, precision, F1/F2, ROC-AUC. *Accuracy is deliberately not used.*

## Dataset

[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(ULB / Worldline). 284,807 transactions, ~0.17% fraud. Features `V1`–`V28` are anonymised PCA
components; only `Time` and `Amount` are raw. **The data is not committed** — see
[`data/README.md`](data/README.md) to download it.

## Repository structure

```
fraud-detection/
├── data/           # dataset goes here (not committed) — see data/README.md
├── notebooks/      # step-by-step analysis
│   ├── 01_eda_feature_engineering.ipynb
│   ├── 02_model_implementation.ipynb
│   └── 03_bias_fairness_audit.ipynb
├── src/            # reproducible pipeline modules
│   ├── data_prep.py    # load, feature-engineer, split, scale (leakage-safe)
│   ├── train.py        # train models, tune thresholds, save artifacts
│   ├── evaluate.py     # metrics + F2 threshold selection
│   └── fairness.py     # proxy-based fairness audit
├── models/         # trained artifacts (generated, not committed)
├── reports/
│   ├── Final_Report.md
│   ├── Bias_Fairness_Analysis.md
│   └── presentations/  # technical + business decks (.pptx)
├── requirements.txt
├── LICENSE
└── README.md
```

## Quickstart

```bash
# 1. Clone and enter
git clone https://github.com/j-wulfe/fraud-detection.git
cd fraud-detection

# 2. Environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Get the data (see data/README.md), place creditcard.csv in data/

# 4. Train all models, tune thresholds, save artifacts to models/
python -m src.train --data data/creditcard.csv --out models

# 5. Or explore interactively
jupyter notebook notebooks/
```

`src.train` prints a comparison table ranked by AUC-PR and writes
`models/metrics_comparison.csv`, `models/configs.json`, and one `.joblib` per model.

## Approach (highlights)

- **Leakage prevention** — split before scaling; all transforms fit on train only; thresholds
  tuned on train cross-validation; any SMOTE confined to CV folds.
- **Imbalance** — balanced class weights / `scale_pos_weight`; decision threshold tuned for
  **F2** (recall-leaning) rather than the default 0.5.
- **Models** — Logistic Regression, Decision Tree, Random Forest, XGBoost, plus an
  Isolation Forest unsupervised baseline. SVM excluded (scale); deep learning omitted
  (trees dominate tabular data).
- **Explainability** — SHAP, PDP, ICE, and LIME.

## Ethics & fairness

This dataset contains **no sensitive attributes**, so a true demographic fairness audit is
impossible — which is itself an ethical finding we report openly. We demonstrate the fairness
*methodology* on a declared proxy (transaction `Amount`) and set out the data-governance steps
a real deployment would need. See [`reports/Bias_Fairness_Analysis.md`](reports/Bias_Fairness_Analysis.md).

## License

Released under the [MIT License](LICENSE). The dataset is subject to its own
[Kaggle license/terms](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
