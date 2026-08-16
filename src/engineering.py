import pandas as pd
import numpy as np

# Must match the sentinel_cols used at training time (see API/XGboost_train.py).
DEFAULT_SENTINEL_COLS = [
    'current_address_months_count',
    'bank_months_count',
    'session_length_in_minutes',
    'device_distinct_emails_8w',
]


def create_fraud_features(df: pd.DataFrame, sentinel_cols=None) -> pd.DataFrame:
    """
    Cleans raw application data and engineers custom fraud features.
    Converts -1 sentinel values (missing-data markers) to NaN, but only in
    `sentinel_cols` - a -1 outside those columns is a real value, not missing data.
    """
    df = df.copy()
    if sentinel_cols is None:
        sentinel_cols = DEFAULT_SENTINEL_COLS

    df['balance_to_income_ratio'] = df['intended_balcon_amount'] / df['income']
    df['balance_to_income_ratio'] = df['balance_to_income_ratio'].replace([np.inf, -np.inf], np.nan)

    present = [c for c in sentinel_cols if c in df.columns]
    df[present] = df[present].replace(-1, np.nan)

    return df
