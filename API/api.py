from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import anthropic
import requests
import json
import os

# Load model
artifact = joblib.load('fraud_model.joblib')
pipeline    = artifact['pipeline']
binary_cols = artifact['binary_cols']
num_cols    = artifact['num_cols']
cat_cols    = artifact['cat_cols']
sentinel_cols = artifact['sentinel_cols']

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

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(transaction: Transaction):
    data = pd.DataFrame([transaction.dict()])
    data['balance_to_income_ratio'] = data['intended_balcon_amount'] / data['income']
    data['balance_to_income_ratio'] = data['balance_to_income_ratio'].replace([np.inf, -np.inf], np.nan)
    data[sentinel_cols] = data[sentinel_cols].replace(-1, np.nan)
    X = data[binary_cols + num_cols + cat_cols]
    fraud_proba = pipeline.predict_proba(X)[0][1]
    return {
        "fraud_probability": round(float(fraud_proba), 4),
        "is_fraud": bool(fraud_proba >= 0.5),
        "risk_level": "HIGH" if fraud_proba >= 0.5 else "MEDIUM" if fraud_proba >= 0.2 else "LOW"
    }

@app.post("/agent")
def agent(request: AgentRequest):
    """Analyze a transaction described in natural language"""
    messages = [{"role": "user", "content": request.message}]
    system = """You are a fraud detection assistant. When given transaction details,
    use the check_fraud tool to analyze it, then explain:
    1. Whether it's fraudulent and the probability
    2. Key risk factors
    3. Recommendation (approve/review/reject)
    Be concise and clear."""

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

            # Fill defaults
            defaults = {
                "prev_address_months_count": -1,
                "current_address_months_count": -1,
                "bank_months_count": -1,
                "session_length_in_minutes": -1,
                "device_distinct_emails_8w": -1
            }
            full_data = {**defaults, **tool_use.input}

            # Call predict internally
            transaction = Transaction(**full_data)
            api_result = predict(transaction)

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
            return {
                "analysis": final,
                "model": "XGBoost + Claude"
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
    <p>Describe a transaction in natural language and the AI will analyze it.</p>
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

            // Show user message
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
                chat.scrollTop = chat.scrollHeight;
            } catch (e) {
                document.getElementById('loading').remove();
                chat.innerHTML += `<div class="agent-msg">❌ Error: ${e.message}</div>`;
            }
        }

        // Send on Ctrl+Enter
        document.getElementById('message').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
"""