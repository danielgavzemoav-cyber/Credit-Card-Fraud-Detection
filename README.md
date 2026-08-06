# Fraud Detection — XGBoost + FastAPI + Claude Agent

A full end-to-end fraud detection system combining machine learning, a REST API, Docker, and an AI agent.

---

## Results

| Approach | Model | Labels needed? | ROC-AUC |
|---|---|---|---|
| Unsupervised | Isolation Forest | No | 0.524 |
| Supervised | Logistic Regression | Yes | 0.872 |
| Supervised | **XGBoost** | Yes | **0.8818** |

---

## Dataset

This project uses the [**Bank Account Fraud Dataset Suite (NeurIPS 2022)**](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) from Kaggle — realistic, synthetic tabular datasets for evaluating ML models on imbalanced fraud-detection tasks.

Fraud rate: ~1.1% (heavily imbalanced)

---

## Project Structure

```
├── Credit_Card_Fraud_DS_Project.ipynb   # Main analysis notebook
├── API/
│   ├── api.py                           # FastAPI app + Claude Agent endpoint
│   ├── agent.py                         # Standalone Claude Agent
│   ├── train_xgboost.py                 # XGBoost training script
│   ├── fraud_model.joblib               # Saved model
│   ├── Dockerfile                       # Docker container
│   └── requirements.txt                 # Dependencies
└── README.md
```

---

## Notebook Walkthrough

### 1 · EDA
- Inspect sentinel values (`-1` used as missing indicator)
- Identify columns with excessive missingness (`prev_address_months_count` at 71%)
- Analyse feature correlations and distributions

### 2 · Data Cleaning & Feature Engineering
- Drop `prev_address_months_count` (too sparse)
- Engineer `balance_to_income_ratio`
- Replace `-1` sentinels with `NaN`
- Build `ColumnTransformer` pipeline: median imputation + scaling for numerics, one-hot encoding for categoricals

### 3 · Unsupervised — KMeans & Isolation Forest
- KMeans: silhouette ≈ 0.07 — no meaningful cluster structure
- Isolation Forest: ROC-AUC 0.524 — weak signal without labels

### 4 · Supervised — Logistic Regression
- GridSearchCV over regularisation strength
- ROC-AUC: **0.872**

### 5 · Supervised — XGBoost
- `scale_pos_weight=89.7` to handle class imbalance
- ROC-AUC: **0.8818**
- Saved model with `joblib`

---

## API

### Run with Docker

```bash
docker build -t fraud-api .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your_key fraud-api
```

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Predict fraud from structured transaction data |
| `/agent` | POST | Analyze transaction described in natural language |
| `/chat` | GET | Web UI for the Claude Agent |
| `/docs` | GET | Auto-generated API documentation |
| `/health` | GET | Health check |

### Example — `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"income": 0.9, "customer_age": 60, "credit_risk_score": 131, ...}'
```

Response:
```json
{
  "fraud_probability": 0.7919,
  "is_fraud": true,
  "risk_level": "HIGH"
}
```

### Example — `/agent`

```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"message": "Customer age 20, very low income, no valid phones, device has 2 prior fraud cases..."}'
```

Response:
```json
{
  "analysis": "HIGH RISK (79%). Red flags: device fraud history, 17 distinct DOB-linked emails...",
  "model": "XGBoost + Claude"
}
```

---

## Claude Agent

The project includes a Claude-powered AI agent that:
- Accepts transaction descriptions in **natural language**
- Extracts parameters automatically
- Calls the XGBoost model via the API
- Returns a detailed explanation of the risk factors

Open `http://localhost:8000/chat` for a web chat interface.

---

## Key Takeaways

- **Isolation Forest alone is not enough** — anomaly signal too weak (AUC barely above random)
- **XGBoost with balanced class weights** outperforms Logistic Regression (0.8818 vs 0.872)
- **Feature engineering matters**: `balance_to_income_ratio` and velocity features are most discriminative
- **LLM + ML** combination enables natural language fraud analysis

---

## Requirements

```bash
pip install -r requirements.txt
```

---

## License

Educational and evaluation purposes.
