# Fraud Detection — XGBoost + SHAP + FastAPI + Claude Agent

A full end-to-end fraud detection system: model training, explainability, a REST API, Docker deployment, and an AI agent that explains predictions in natural language.

---

## Results

| Approach | Model | Labels needed? | ROC-AUC |
|---|---|---|---|
| Unsupervised | Isolation Forest | No | 0.524 |
| Supervised | Logistic Regression | Yes | 0.872 |
| Supervised | **XGBoost** | Yes | **0.882** |

XGBoost was tuned with `scale_pos_weight` to handle the 1.1% class imbalance, and evaluated on `aucpr` (average precision) rather than plain accuracy, since the latter is misleading on imbalanced data.

---

## Dataset

This project uses the [**Bank Account Fraud Dataset Suite (NeurIPS 2022)**](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) from Kaggle — realistic, synthetic tabular datasets for evaluating ML models on imbalanced, biased fraud-detection tasks.

Fraud rate: ~1.1% (heavily imbalanced), ~1M rows.

---

## Project Structure

```
├── Credit_Card_Fraud_DS_Project.ipynb   # EDA, feature engineering, model comparison
├── data/
│   ├── Base.csv                         # Labeled dataset (not tracked in git — see below)
│   └── Base_unlabeled.csv               # Unlabeled variant
├── src/
│   ├── engineering.py                   # create_fraud_features() — feature engineering logic
│   └── models.py                        # Model training utilities
├── tests/
│   └── test_engineering.py              # pytest suite for feature engineering edge cases
├── API/
│   ├── api.py                           # FastAPI app: /predict, /explain, /explain/plot, /agent, /chat
│   ├── train_xgboost.py                 # XGBoost training script
│   ├── fraud_model.joblib               # Saved sklearn Pipeline (preprocessor + XGBoost)
│   ├── xgb_booster.json                 # Version-clean XGBoost export, used by SHAP
│   ├── Dockerfile
│   └── requirements.txt
└── README.md
```

> **Note on the data files:** `Base.csv` and `Base_unlabeled.csv` are excluded from version control (>100MB, exceeds GitHub's limit). Download them from the [Kaggle dataset page](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) and place them in `data/`.

---

## Notebook Walkthrough

### 1 · EDA
- Inspect sentinel values (`-1` used as a missing-data indicator across several columns)
- Identify columns with excessive missingness (`prev_address_months_count` at 71%)
- Analyse feature correlations and distributions

### 2 · Data Cleaning & Feature Engineering
- Drop `prev_address_months_count` (too sparse)
- Engineer `balance_to_income_ratio`, with safe handling of zero-income rows
- Replace `-1` sentinels with `NaN` for proper imputation
- Build a `ColumnTransformer` pipeline: median imputation + scaling for numerics, one-hot encoding for categoricals

### 3 · Unsupervised — KMeans & Isolation Forest
- KMeans: no meaningful cluster structure emerges (low silhouette score)
- Isolation Forest: ROC-AUC 0.524 — weak signal without labels

### 4 · Supervised — Logistic Regression
- ROC-AUC: **0.872**

### 5 · Supervised — XGBoost
- `scale_pos_weight` tuned for class imbalance
- ROC-AUC: **0.882**
- Saved as a full sklearn `Pipeline` (`preprocessor` + `model` steps) via `joblib`

---

## Tests

```bash
pytest tests/ -v
```

Run from the repo root — `src` is imported as `from src.engineering import create_fraud_features`, which needs the repo root on the path.

Covers three edge cases in `create_fraud_features()` (in `src/engineering.py`):
- **Zero-income division** — ensures `balance_to_income_ratio` never produces `inf`; it becomes `NaN` and is median-imputed downstream, matching training-time behavior
- **Sentinel conversion** — verifies `-1` is converted to `NaN` in actual sentinel columns (`current_address_months_count`, `bank_months_count`, `session_length_in_minutes`, `device_distinct_emails_8w`)
- **Non-sentinel preservation** — verifies `-1` outside those columns (e.g. `intended_balcon_amount`) is left alone, since it's a real value there, not a missing-data marker

`API/api.py`'s `_prepare_features()` calls this same function, so the live API is covered by these tests too — no more separate untested duplicate.

---

## API

### Run with Docker

```bash
# from the repo root — build context must include src/, which api.py imports from
docker build -t fraud-api -f API/Dockerfile .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your_key fraud-api
```

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Fraud probability from structured transaction data |
| `/explain` | POST | `/predict` + top SHAP-contributing features (JSON) |
| `/explain/plot` | POST | Same as `/explain`, returned as a PNG bar chart |
| `/agent` | POST | Natural-language transaction analysis, SHAP-grounded, includes an embedded plot |
| `/chat` | GET | Web chat UI for the agent — shows both the explanation and the SHAP chart |
| `/docs` | GET | Auto-generated OpenAPI docs |
| `/health` | GET | Health check |

### Example — `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"income": 0.9, "customer_age": 60, "credit_risk_score": 131, ...}'
```

```json
{
  "fraud_probability": 0.7919,
  "is_fraud": true,
  "risk_level": "HIGH"
}
```

### Example — `/explain`

Same input as `/predict`, but the response also includes:

```json
{
  "fraud_probability": 0.7919,
  "is_fraud": true,
  "risk_level": "HIGH",
  "top_contributing_features": [
    {"feature": "housing_status_BA", "impact": 0.83, "direction": "increases risk"},
    {"feature": "device_os_windows", "impact": 0.67, "direction": "increases risk"},
    {"feature": "phone_home_valid", "impact": -0.64, "direction": "decreases risk"}
  ]
}
```

### Example — `/agent` (natural language)

```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"message": "Customer age 20, very low income, no valid phones, device has 2 prior fraud cases..."}'
```

```json
{
  "analysis": "HIGH RISK (79%). Key factors: device fraud history, 17 distinct DOB-linked emails, no valid contact info...",
  "model": "XGBoost + SHAP + Claude",
  "plot_base64": "iVBORw0KGgo..."
}
```

`plot_base64` is a PNG chart of the same SHAP features Claude's explanation is grounded in — rendered automatically in `/chat`, or decodable manually from the raw API response.

---

## Claude Agent

The `/agent` endpoint and `/chat` UI let a non-technical user describe a transaction in plain English and get back:

1. A fraud probability and risk level
2. A plain-language explanation, **grounded only in real SHAP values** — the system prompt explicitly forbids Claude from inventing plausible-sounding reasons not backed by the model's actual feature attributions
3. A bar chart visualizing those same contributing features

This closes the gap between "a model that scores well" and "a tool a non-technical stakeholder can actually use and trust."

---

## Why SHAP is loaded from a separate `xgb_booster.json`

`fraud_model.joblib` contains the full sklearn pipeline used for serving predictions (`/predict`, `/agent`). SHAP's `TreeExplainer`, however, is sensitive to the exact XGBoost version used when a model was originally trained and pickled — loading an XGBoost model pickled under one version into a different installed version can produce a malformed internal `base_score` field that crashes SHAP.

`xgb_booster.json` is a version-clean re-export of the *same trained model* (`booster.save_model()` / `xgb.Booster().load_model()`), used **only** to build the SHAP explainer. It does not affect `/predict` — predictions still come from the original pipeline. Both are verified to produce identical probabilities.

---

## Key Takeaways

- **Isolation Forest alone is not enough** for this dataset — anomaly signal is barely above random (0.524 AUC)
- **XGBoost with balanced class weights** outperforms Logistic Regression (0.882 vs 0.872 ROC-AUC)
- **Feature engineering matters**: `balance_to_income_ratio` and velocity features are among the most discriminative
- **Explainability is not optional for a usable fraud tool** — SHAP grounding prevents the LLM layer from generating plausible-but-fabricated explanations
- **Cross-version model serialization is a real production hazard** — the `xgb_booster.json` workaround exists because of a genuine XGBoost/SHAP version-compatibility bug encountered while deploying

---

## Requirements

```
python >= 3.12
pandas, numpy, scikit-learn, xgboost, shap, matplotlib
fastapi, uvicorn, pydantic
anthropic
joblib
pytest (for tests/)
```

Install:
```bash
pip install -r API/requirements.txt
```

---

## Roadmap / Not Yet Implemented

- Request logging / prediction history for monitoring model drift
- Rate limiting and API key auth on the deployed endpoint
- Cost/latency A/B test: Claude Haiku (parameter extraction) vs Sonnet (final explanation)
- Batch analysis endpoint (CSV upload → summary report across many transactions)
- Stricter input validation bounds (e.g. `customer_age` range checks) beyond basic Pydantic typing

---

## License

Educational and evaluation purposes.
