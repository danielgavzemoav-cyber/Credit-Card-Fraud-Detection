import anthropic
import requests
import json

# ← שים את ה-API key שלך כאן
client = anthropic.Anthropic(api_key="YOUR_API_KEY_HERE")

# Tool definition — Claude ידע איך להשתמש ב-API שלנו
tools = [
    {
        "name": "check_fraud",
        "description": "Check if a credit card transaction is fraudulent. Call this tool when you have transaction details to analyze.",
        "input_schema": {
            "type": "object",
            "properties": {
                "income": {"type": "number", "description": "Customer income (0-1 normalized)"},
                "name_email_similarity": {"type": "number", "description": "Similarity between name and email (0-1)"},
                "current_address_months_count": {"type": "integer", "description": "Months at current address (-1 if unknown)"},
                "customer_age": {"type": "integer", "description": "Customer age"},
                "days_since_request": {"type": "number", "description": "Days since credit request"},
                "intended_balcon_amount": {"type": "number", "description": "Intended balance amount"},
                "payment_type": {"type": "string", "description": "Payment type (AA, AB, AC, AD, AE)"},
                "zip_count_4w": {"type": "integer", "description": "Zip code requests in last 4 weeks"},
                "velocity_6h": {"type": "number", "description": "Transaction velocity in last 6 hours"},
                "velocity_24h": {"type": "number", "description": "Transaction velocity in last 24 hours"},
                "velocity_4w": {"type": "number", "description": "Transaction velocity in last 4 weeks"},
                "bank_branch_count_8w": {"type": "integer", "description": "Bank branch visits in last 8 weeks"},
                "date_of_birth_distinct_emails_4w": {"type": "integer", "description": "Distinct emails with same DOB in 4 weeks"},
                "employment_status": {"type": "string", "description": "Employment status (CA, CB, CC, CD, CE, CF, CG)"},
                "credit_risk_score": {"type": "integer", "description": "Credit risk score"},
                "email_is_free": {"type": "integer", "description": "1 if free email provider, 0 otherwise"},
                "housing_status": {"type": "string", "description": "Housing status (BA, BB, BC, BD, BE, BF, BG)"},
                "phone_home_valid": {"type": "integer", "description": "1 if home phone is valid"},
                "phone_mobile_valid": {"type": "integer", "description": "1 if mobile phone is valid"},
                "bank_months_count": {"type": "integer", "description": "Months with bank (-1 if unknown)"},
                "has_other_cards": {"type": "integer", "description": "1 if has other cards"},
                "proposed_credit_limit": {"type": "number", "description": "Proposed credit limit"},
                "foreign_request": {"type": "integer", "description": "1 if request from foreign country"},
                "source": {"type": "string", "description": "Source (INTERNET, TELEAPP)"},
                "session_length_in_minutes": {"type": "number", "description": "Session length in minutes (-1 if unknown)"},
                "device_os": {"type": "string", "description": "Device OS (windows, linux, macintosh, other, x11)"},
                "keep_alive_session": {"type": "integer", "description": "1 if session kept alive"},
                "device_distinct_emails_8w": {"type": "integer", "description": "Distinct emails from device in 8 weeks (-1 if unknown)"},
                "device_fraud_count": {"type": "integer", "description": "Fraud count from this device"},
                "month": {"type": "integer", "description": "Month of request (0-7)"},
                "prev_address_months_count": {"type": "integer", "description": "Months at previous address (-1 if unknown)"}
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


def call_fraud_api(transaction_data: dict) -> dict:
    """Call our FastAPI fraud detection endpoint"""
    # Fill in defaults for optional fields
    defaults = {
        "prev_address_months_count": -1,
        "current_address_months_count": -1,
        "bank_months_count": -1,
        "session_length_in_minutes": -1,
        "device_distinct_emails_8w": -1
    }
    full_data = {**defaults, **transaction_data}

    response = requests.post(
        "http://localhost:8000/predict",
        json=full_data
    )
    return response.json()


def run_agent(user_message: str):
    """Run the fraud detection agent"""
    print(f"\n{'='*50}")
    print(f"User: {user_message}")
    print(f"{'='*50}")

    messages = [{"role": "user", "content": user_message}]

    system = """You are a fraud detection assistant. When a user describes a transaction or provides transaction details, 
    use the check_fraud tool to analyze it. 
    After getting the result, explain:
    1. Whether the transaction is fraudulent
    2. The risk level and probability
    3. Which details seem suspicious (if any)
    Be concise and clear."""

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            tools=tools,
            messages=messages
        )

        # If Claude wants to use a tool
        if response.stop_reason == "tool_use":
            tool_use = next(b for b in response.content if b.type == "tool_use")
            print(f"\n🔧 Calling fraud API with extracted parameters...")

            # Call our API
            api_result = call_fraud_api(tool_use.input)
            print(f"📊 API Result: {api_result}")

            # Add Claude's response and tool result to messages
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(api_result)
                }]
            })

        # If Claude is done
        elif response.stop_reason == "end_turn":
            final_response = next(b.text for b in response.content if hasattr(b, "text"))
            print(f"\n🤖 Agent: {final_response}")
            break

    return final_response


if __name__ == "__main__":
    # Test 1 — normal transaction
    run_agent("""
    I have a transaction to analyze:
    - Customer age: 35, income: 0.6
    - Requesting credit limit of 2000
    - Has valid home and mobile phone
    - Credit risk score: 150
    - Low velocity, been at address 24 months
    - Payment type AA, source INTERNET, Windows device
    - No foreign request, has other cards
    - zip_count_4w: 800, bank_branch_count_8w: 3
    - date_of_birth_distinct_emails_4w: 2
    - email_is_free: 0, housing_status: BC, employment_status: CA
    - device_fraud_count: 0, month: 3
    - velocity_6h: 5000, velocity_24h: 4000, velocity_4w: 3500
    - days_since_request: 0.005, intended_balcon_amount: 500
    - name_email_similarity: 0.85, keep_alive_session: 1
    """)

    # Test 2 — suspicious transaction
    run_agent("""
    Analyze this suspicious transaction:
    - Young customer age 20, very low income 0.1
    - Requesting very high credit limit of 9000
    - No valid phones at all
    - Very low credit risk score: 10
    - High velocity last 6h: 8139, 24h: 4312, 4w: 6341
    - Foreign request, free email, short session
    - Device has fraud count of 2
    - date_of_birth_distinct_emails_4w: 17
    - payment_type: AA, source: INTERNET, device_os: windows
    - employment_status: CA, housing_status: BA
    - email_is_free: 1, keep_alive_session: 1
    - zip_count_4w: 4079, bank_branch_count_8w: 2
    - days_since_request: 0.006, intended_balcon_amount: 35
    - name_email_similarity: 0.57, month: 0, has_other_cards: 0
    """)
