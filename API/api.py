from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import anthropic
import json
import os
import sys
from pathlib import Path
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from fastapi.responses import StreamingResponse

# Make the shared src/ package importable regardless of cwd (local run vs Docker).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.engineering import create_fraud_features

# Load model
artifact = joblib.load('fraud_model.joblib')
pipeline    = artifact['pipeline']
binary_cols = artifact['binary_cols']
num_cols    = artifact['num_cols']
cat_cols    = artifact['cat_cols']
sentinel_cols = artifact['sentinel_cols']

preprocessor = pipeline.named_steps['preprocessor']
model        = pipeline.named_steps['model']

# SHAP needs a version-clean Booster (avoids xgboost cross-version base_score parsing bugs).
# xgb_booster.json is a re-exported, version-normalized copy of the same trained model.
import xgboost as xgb
_shap_booster = xgb.Booster()
_shap_booster.load_model('xgb_booster.json')
shap_explainer = shap.TreeExplainer(_shap_booster)

# Claude client
claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE"))

app = FastAPI(title="Credit Card Fraud Detection API")

# ─── Models ───────────────────────────────────────────────────────────────────

class Transaction(BaseModel):
    income: float
    name_email_similarity: float
    prev_address_months_count: int = -1
    current_address_months_count: int = -1
    customer_age: int
    days_since_request: float
    intended_balcon_amount: float
    payment_type: str
    zip_count_4w: int
    velocity_6h: float
    velocity_24h: float
    velocity_4w: float
    bank_branch_count_8w: int
    date_of_birth_distinct_emails_4w: int
    employment_status: str
    credit_risk_score: int
    email_is_free: int
    housing_status: str
    phone_home_valid: int
    phone_mobile_valid: int
    bank_months_count: int = -1
    has_other_cards: int
    proposed_credit_limit: float
    foreign_request: int
    source: str
    session_length_in_minutes: float = -1
    device_os: str
    keep_alive_session: int
    device_distinct_emails_8w: int = -1
    device_fraud_count: int
    month: int

class AgentRequest(BaseModel):
    message: str

# ─── Tools for Claude ─────────────────────────────────────────────────────────

tools = [
    {
        "name": "check_fraud",
        "description": "Check if a credit card transaction is fraudulent.",
        "input_schema": {
            "type": "object",
            "properties": {
                "income": {"type": "number"},
                "name_email_similarity": {"type": "number"},
                "current_address_months_count": {"type": "integer"},
                "customer_age": {"type": "integer"},
                "days_since_request": {"type": "number"},
                "intended_balcon_amount": {"type": "number"},
                "payment_type": {"type": "string"},
                "zip_count_4w": {"type": "integer"},
                "velocity_6h": {"type": "number"},
                "velocity_24h": {"type": "number"},
                "velocity_4w": {"type": "number"},
                "bank_branch_count_8w": {"type": "integer"},
                "date_of_birth_distinct_emails_4w": {"type": "integer"},
                "employment_status": {"type": "string"},
                "credit_risk_score": {"type": "integer"},
                "email_is_free": {"type": "integer"},
                "housing_status": {"type": "string"},
                "phone_home_valid": {"type": "integer"},
                "phone_mobile_valid": {"type": "integer"},
                "has_other_cards": {"type": "integer"},
                "proposed_credit_limit": {"type": "number"},
                "foreign_request": {"type": "integer"},
                "source": {"type": "string"},
                "device_os": {"type": "string"},
                "keep_alive_session": {"type": "integer"},
                "device_fraud_count": {"type": "integer"},
                "month": {"type": "integer"}
            },
            "required": [
                "income", "name_email_similarity", "customer_age",
                "days_since_request", "intended_balcon_amount", "payment_type",
                "zip_count_4w", "velocity_6h", "velocity_24h", "velocity_4w",
                "bank_branch_count_8w", "date_of_birth_distinct_emails_4w",
                "employment_status", "credit_risk_score", "email_is_free",
                "housing_status", "phone_home_valid", "phone_mobile_valid",
                "has_other_cards", "proposed_credit_limit", "foreign_request",
                "source", "device_os", "keep_alive_session",
                "device_fraud_count", "month"
            ]
        }
    }
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _prepare_features(transaction: Transaction) -> pd.DataFrame:
    data = pd.DataFrame([transaction.dict()])
    data = create_fraud_features(data, sentinel_cols=sentinel_cols)
    return data[binary_cols + num_cols + cat_cols]

def _clean_feature_name(name: str) -> str:
    """Strip sklearn ColumnTransformer prefixes like 'num__' / 'cat__' / 'bin__'."""
    return name.split("__", 1)[-1]

def _top_shap_contributions(X: pd.DataFrame, top_n: int = 5):
    X_transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    shap_values = shap_explainer.shap_values(X_transformed)
    row = shap_values[0]

    contributions = [
        {"feature": _clean_feature_name(name), "impact": round(float(val), 4)}
        for name, val in zip(feature_names, row)
    ]
    contributions.sort(key=lambda c: abs(c["impact"]), reverse=True)
    return contributions[:top_n]

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(transaction: Transaction):
    X = _prepare_features(transaction)
    fraud_proba = pipeline.predict_proba(X)[0][1]
    return {
        "fraud_probability": round(float(fraud_proba), 4),
        "is_fraud": bool(fraud_proba >= 0.5),
        "risk_level": "HIGH" if fraud_proba >= 0.5 else "MEDIUM" if fraud_proba >= 0.2 else "LOW"
    }

@app.post("/explain")
def explain(transaction: Transaction):
    """Predict fraud probability AND explain which features drove the decision (SHAP)."""
    X = _prepare_features(transaction)
    fraud_proba = pipeline.predict_proba(X)[0][1]
    top_features = _top_shap_contributions(X, top_n=5)

    return {
        "fraud_probability": round(float(fraud_proba), 4),
        "is_fraud": bool(fraud_proba >= 0.5),
        "risk_level": "HIGH" if fraud_proba >= 0.5 else "MEDIUM" if fraud_proba >= 0.2 else "LOW",
        "top_contributing_features": [
            {
                "feature": f["feature"],
                "impact": f["impact"],
                "direction": "increases risk" if f["impact"] > 0 else "decreases risk"
            }
            for f in top_features
        ]
    }

@app.post("/explain/plot")
def explain_plot(transaction: Transaction):
    """Return a PNG bar chart of the top SHAP feature contributions for this transaction."""
    X = _prepare_features(transaction)
    png_bytes = _make_shap_plot(X, top_n=10)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")

@app.post("/agent")
def agent(request: AgentRequest):
    """Analyze a transaction described in natural language, with SHAP-grounded explanation."""
    messages = [{"role": "user", "content": request.message}]
    system = """You are a fraud detection assistant. When given transaction details,
    use the check_fraud tool to analyze it. You will receive the fraud probability
    AND a list of the top features that actually drove the model's decision (from SHAP analysis).
    Base your explanation ONLY on those listed features - do not invent other reasons.
    Reply with: 1) verdict + probability, 2) the top 2-3 real contributing features and their direction,
    3) a recommendation (approve/review/reject). Be concise."""

    last_transaction = None  # ← track the most recent successfully-parsed transaction

    while True:
        response = claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "tool_use":
            tool_use = next(b for b in response.content if b.type == "tool_use")

            defaults = {
                "prev_address_months_count": -1,
                "current_address_months_count": -1,
                "bank_months_count": -1,
                "session_length_in_minutes": -1,
                "device_distinct_emails_8w": -1
            }
            full_data = {**defaults, **tool_use.input}

            transaction = Transaction(**full_data)
            last_transaction = transaction  # ← remember it for the plot step below
            api_result = explain(transaction)  # ← now uses SHAP-backed explain()

            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(api_result)
                }]
            })

        elif response.stop_reason == "end_turn":
            final = next(b.text for b in response.content if hasattr(b, "text"))

            plot_base64 = None
            if last_transaction is not None:
                # Always generate the SHAP plot for whatever transaction was actually analyzed,
                # regardless of whether Claude's text explanation happens to mention it.
                X = _prepare_features(last_transaction)
                png_bytes = _make_shap_plot(X, top_n=10)
                plot_base64 = base64.b64encode(png_bytes).decode("utf-8")

            return {
                "analysis": final,
                "model": "XGBoost + SHAP + Claude",
                "plot_base64": plot_base64  # ← None if no transaction was ever analyzed (e.g. "hello")
            }

@app.get("/chat", response_class=HTMLResponse)
def chat_ui():
    """Simple web UI for the agent"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Fraud Detection Agent</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        #chat { background: white; border-radius: 10px; padding: 20px; min-height: 400px; margin-bottom: 20px; overflow-y: auto; max-height: 500px; }
        .user-msg { background: #007bff; color: white; padding: 10px 15px; border-radius: 10px; margin: 10px 0; text-align: right; }
        .agent-msg { background: #e9ecef; color: #333; padding: 10px 15px; border-radius: 10px; margin: 10px 0; white-space: pre-wrap; }
        #input-area { display: flex; gap: 10px; }
        #message { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .loading { color: #999; font-style: italic; }
    </style>
</head>
<body>
    <h1>🔍 Fraud Detection Agent</h1>
    <p>Describe a transaction in natural language and the AI will analyze it, grounded in real SHAP feature contributions.</p>
    <p style="font-size:13px;color:#666;">Tip: use <code>/explain/plot</code> (POST via /docs) to get a visual SHAP chart for a specific transaction.</p>
    <div id="chat"></div>
    <div id="input-area">
        <textarea id="message" rows="3" placeholder="E.g: Customer age 35, income 0.6, credit score 150, requesting 2000 credit limit, valid phones..."></textarea>
        <button onclick="sendMessage()">Analyze</button>
    </div>

    <script>
        async function sendMessage() {
            const msg = document.getElementById('message').value.trim();
            if (!msg) return;

            const chat = document.getElementById('chat');

            chat.innerHTML += `<div class="user-msg">${msg}</div>`;
            chat.innerHTML += `<div class="agent-msg loading" id="loading">🤔 Analyzing...</div>`;
            chat.scrollTop = chat.scrollHeight;
            document.getElementById('message').value = '';

            try {
                const response = await fetch('/agent', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: msg})
                });
                const data = await response.json();

                document.getElementById('loading').remove();
                chat.innerHTML += `<div class="agent-msg">🤖 ${data.analysis}</div>`;
                if (data.plot_base64) {
                    chat.innerHTML += `<div class="agent-msg"><img src="data:image/png;base64,${data.plot_base64}" style="max-width:100%; border-radius:8px;" /></div>`;
                }
                chat.scrollTop = chat.scrollHeight;
            } catch (e) {
                document.getElementById('loading').remove();
                chat.innerHTML += `<div class="agent-msg">❌ Error: ${e.message}</div>`;
            }
        }

        document.getElementById('message').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
"""
def _make_shap_plot(X: pd.DataFrame, top_n: int = 10) -> bytes:
    """Generate a horizontal bar chart of the top SHAP contributions and return PNG bytes."""
    X_transformed = preprocessor.transform(X)
    feature_names = [_clean_feature_name(n) for n in preprocessor.get_feature_names_out()]
    shap_values = shap_explainer.shap_values(X_transformed)
    row = shap_values[0]

    order = np.argsort(np.abs(row))[::-1][:top_n]
    labels = [feature_names[i] for i in order]
    values = [row[i] for i in order]
    colors = ['#ff4d6d' if v > 0 else '#4d94ff' for v in values]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = np.arange(len(labels))[::-1]
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color='#333', linewidth=0.8)
    ax.set_xlabel("Impact on fraud probability (SHAP value)")
    ax.set_title("Top factors driving this prediction", fontsize=12, fontweight='bold')
    for i, v in zip(y_pos, values):
        ax.text(v + (0.02 if v > 0 else -0.02), i, f"{v:+.2f}",
                va='center', ha='left' if v > 0 else 'right', fontsize=9)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


