import pandas as pd
import numpy as np


def create_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw application data and engineers custom fraud features.
    Handles -1 sentinel values and creates financial ratios.
    """
    df = df.copy()

    # Handle sentinel values safely
    df = df.replace(-1, np.nan)

    # Avoid division by zero bugs in production
    df['balance_to_income_ratio'] = df['intended_balcon_amount'] / (df['income'] + 1e-5)

    return df
