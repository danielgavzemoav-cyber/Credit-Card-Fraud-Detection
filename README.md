# Fraud Detection — Unsupervised vs Supervised Approach

A comparative study of two strategies for detecting fraudulent credit-card applications:

| Approach | Model | Labels needed? | ROC-AUC |
|---|---|---|---|
| Unsupervised | Isolation Forest | No | 0.524 |
| Supervised | Logistic Regression | Yes | **0.872** |

## Dataset

This project uses the [**Bank Account Fraud Dataset Suite (NeurIPS 2022)**](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) from Kaggle — a set of realistic, synthetic tabular datasets designed for evaluating ML models on biased, imbalanced, and dynamic fraud-detection tasks.

## Problem

Given a dataset of credit-card applications with behavioural, demographic, and device features, flag the ones that are likely fraudulent. Fraud is rare (~1.1 % of applications), making this a heavily imbalanced classification problem.

## Project Structure

```
├── README.md
├── Rubrik_test.ipynb        # Main analysis notebook
├── Base_unlabeled.csv       # Raw application data (no fraud labels)
└── Base.csv                 # Same data with ground-truth fraud_bool column
```

## Notebook Walkthrough

### 1 · Setup & Imports
Standard data-science stack: pandas, numpy, matplotlib, seaborn, scikit-learn.

### 2 · Exploratory Data Analysis
- Inspect sentinel values (`-1` used as a missing indicator in several columns).
- Analyse pairwise co-occurrence and lift of missing values.
- Identify columns with excessive missingness (e.g. `prev_address_months_count` at 71 %).

### 3 · Data Cleaning & Feature Engineering
- Drop `prev_address_months_count` (too sparse, near-zero correlation with other features).
- Engineer `balance_to_income_ratio`.
- Replace `-1` sentinels with `NaN` for proper imputation.
- Build a `ColumnTransformer` pipeline: median imputation + standard scaling for numerics, one-hot encoding for categoricals, mode imputation for binary flags.

### 3a · KMeans Clustering
- Elbow and silhouette analysis show no meaningful cluster structure (silhouette ≈ 0.07).
- Dimensionality reduction does not help — KMeans is the wrong tool for sparse-anomaly data.

### 3b · Isolation Forest (Unsupervised)
- Fit an Isolation Forest on the full unlabeled dataset.
- Extract anomaly scores; inspect the 1 000 most anomalous rows.
- PCA projection shows anomalies concentrating in one region, but explained variance is low (~18 %).

### 4 · Logistic Regression (Supervised)
- Re-load data with ground-truth labels.
- GridSearchCV over regularisation strength; best model uses L2 with C = 0.001.
- **ROC-AUC 0.872** vs Isolation Forest's 0.524 — supervision makes a dramatic difference.
- At the default 0.5 threshold: 78.8 % recall but only 4.3 % precision (flags ~20 % of traffic).

### 5 · Next Steps
- Train a gradient-boosted model (LightGBM) for stronger performance.
- Optimise the decision threshold to match a realistic alerting budget.

## Key Takeaways

- **Isolation Forest alone is not enough** for this fraud dataset — the anomaly signal is too weak (AUC barely above random).
- **Logistic Regression with balanced class weights** provides a solid supervised baseline, though precision at the default threshold is low due to extreme class imbalance.
- **Feature engineering matters**: the `balance_to_income_ratio` and velocity features contribute the most discriminative power.

## Requirements

```
python >= 3.10
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

```bash
jupyter notebook Rubrik_test.ipynb
```

## License

This project is for educational and evaluation purposes.
