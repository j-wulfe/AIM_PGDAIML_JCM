# Models

Trained artifacts are **generated, not committed**. Produce them with:

```bash
python -m src.train --data data/creditcard.csv --out models
```

This writes:

| File | Description |
|---|---|
| `<ModelName>.joblib` | One fitted model per algorithm (LogisticRegression, DecisionTree, RandomForest, XGBoost, IsolationForest) |
| `scaler.joblib` | The fitted `RobustScaler` (apply to new data before prediction) |
| `metrics_comparison.csv` | All models ranked by AUC-PR |
| `configs.json` | Random seed, feature list, and tuned decision thresholds |

## Loading a saved model

```python
import joblib
model  = joblib.load("models/RandomForest.joblib")
scaler = joblib.load("models/scaler.joblib")
# scale Time/Amount/Log_Amount/Hour with `scaler` before calling model.predict_proba(...)
```
