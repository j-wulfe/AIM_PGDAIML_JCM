# Data

The dataset is **not committed** to this repository (it is large and governed by Kaggle's
terms). Download it and place `creditcard.csv` in this folder.

## Get the data

**Option A — Kaggle website**
1. Visit https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download and unzip `creditcard.csv` into this `data/` directory.

**Option B — Kaggle CLI**
```bash
pip install kaggle                       # configure ~/.kaggle/kaggle.json first
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```

Expected result:
```
data/creditcard.csv      # 284,807 rows x 31 columns
```

## About the dataset

- Real credit-card transactions by European cardholders, September 2013 (ULB / Worldline).
- `V1`–`V28` are anonymised PCA components; `Time` and `Amount` are raw; `Class` is the label
  (0 = legitimate, 1 = fraud).
- ~0.17% of transactions are fraud. No missing values. Contains some exact duplicate rows,
  which the pipeline removes.

See the data dictionary in [`../reports/Final_Report.md`](../reports/Final_Report.md).
