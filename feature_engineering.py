"""
feature_engineering.py
-------------------------
Contains functions for creating new features to improve model performance.
"""

import numpy as np
import pandas as pd

def engineer_features(df):
    """
    Create additional features for the fraud detection model.
    Example feature engineering steps:
      - Log transform the 'amount' feature (if exists)
      - Extract hour from a timestamp column (if exists)
    """
    df_fe = df.copy()

    # Example: Log transform on the 'amount' feature
    if "amount" in df_fe.columns:
        df_fe["amount_log"] = np.log1p(df_fe["amount"])

    # Example: If there is a 'transaction_time' column, extract hour
    if "transaction_time" in df_fe.columns:
        df_fe["transaction_time"] = pd.to_datetime(df_fe["transaction_time"], errors="coerce")
        df_fe["transaction_hour"] = df_fe["transaction_time"].dt.hour
        df_fe.drop(columns=["transaction_time"], inplace=True)

    # Additional feature engineering can be added here
    # For example: Aggregation, interaction features, etc.

    return df_fe
