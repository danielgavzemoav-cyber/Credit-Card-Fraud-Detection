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

    # Assert: division by zero must not leak inf into the model; it should become NaN,
    # matching training-time behavior where the ratio is later median-imputed.
    calculated_ratio = result_df['balance_to_income_ratio'].iloc[0]

    assert not np.isinf(calculated_ratio), "Error: Division by zero caused an infinity value!"
    assert pd.isna(calculated_ratio), "Error: Zero income should yield NaN, not a fabricated finite ratio!"


def test_sentinel_values_are_converted_to_nan():
    # 1. ARRANGE: Create a fake applicant with missing data represented as -1
    #    in an actual sentinel column (per API/XGboost_train.py's sentinel_cols).
    test_df = pd.DataFrame({
        'intended_balcon_amount': [500.0],
        'income': [4000.0],
        'current_address_months_count': [-1]
    })

    # 2. ACT: Run your pipeline function
    result_df = create_fraud_features(test_df)

    # 3. ASSERT: Verify that the -1 was successfully changed to a NaN value
    assert pd.isna(result_df['current_address_months_count'].iloc[0]), "Error: -1 was not converted to NaN!"


def test_non_sentinel_negative_one_is_preserved():
    # intended_balcon_amount is NOT a sentinel column - a -1 there is a real value
    # (e.g. a negative intended balance transfer), not a missing-data marker.
    test_df = pd.DataFrame({
        'intended_balcon_amount': [-1.0],
        'income': [4000.0]
    })

    result_df = create_fraud_features(test_df)

    assert result_df['intended_balcon_amount'].iloc[0] == -1.0, \
        "Error: -1 outside sentinel_cols should be preserved, not blanked out!"
