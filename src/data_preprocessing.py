"""
data_preprocessing.py
----------------------
Contains functions for loading and preprocessing the raw data.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """
    Load data from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def preprocess_data(df):
    """
    Clean and preprocess the data.
    - Handle missing values.
    - Standardize numerical features.
    """
    # Drop duplicates
    df = df.drop_duplicates()

    # Fill missing values (simple example: fill with median for numerical columns)
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Scale numerical features (except target if present)
    if "is_fraud" in df.columns:
        features = df.drop(columns=["is_fraud"])
    else:
        features = df.copy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(scaled_features, columns=features.columns)

    # If target exists, add it back
    if "is_fraud" in df.columns:
        df_scaled["is_fraud"] = df["is_fraud"].values

    return df_scaled
