# tests/test_engineering.py
import pandas as pd
import numpy as np
from src.engineering import create_fraud_features


def test_balance_to_income_ratio_handles_zero_income():
    # Arrange: Explicitly build a single-row DataFrame with valid list values
    test_df = pd.DataFrame({
        'intended_balcon_amount': [5000.0],
        'income': [0.0]
    })

    # Act: Run your pipeline logic
    result_df = create_fraud_features(test_df)

    # Assert: Ensure it safely evaluates without hitting infinity or crashing
    calculated_ratio = result_df['balance_to_income_ratio'].iloc[0]

    assert not np.isinf(calculated_ratio), "Error: Division by zero caused an infinity value!"
    assert calculated_ratio > 0, "Error: Ratio should be greater than zero!"

def test_sentinel_values_are_converted_to_nan():
    # 1. ARRANGE: Create a fake applicant with missing data represented as -1
    test_df = pd.DataFrame({
        'intended_balcon_amount': [-1.0],
        'income': [4000.0]
    })

    # 2. ACT: Run your pipeline function
    result_df = create_fraud_features(test_df)

    # 3. ASSERT: Verify that the -1 was successfully changed to a NaN value
    assert pd.isna(result_df['intended_balcon_amount'].iloc[0]), "Error: -1 was not converted to NaN!"
