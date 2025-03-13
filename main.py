"""
main.py
---------
Main script to run the fraud detection pipeline.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src import data_preprocessing, feature_engineering, utils
from src.models import autoencoder, supervised_model

# Set random seed for reproducibility
np.random.seed(42)

def main():
    # Paths
    raw_data_path = os.path.join("data", "raw", "transactions.csv")
    processed_data_path = os.path.join("data", "processed", "transactions_processed.csv")

    # 1. Data Preprocessing
    print("Loading and preprocessing data...")
    df = data_preprocessing.load_data(raw_data_path)
    df = data_preprocessing.preprocess_data(df)

    # Optionally save processed data
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df.to_csv(processed_data_path, index=False)

    # 2. Feature Engineering
    print("Performing feature engineering...")
    df = feature_engineering.engineer_features(df)

    # Define features and target
    # Assuming 'is_fraud' is the target column in your dataset
    features = df.drop(columns=["is_fraud"])
    target = df["is_fraud"]

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42, stratify=target
    )

    # 4. Unsupervised Anomaly Detection with Autoencoder
    print("Training autoencoder for anomaly detection...")
    ae_model, threshold = autoencoder.train_autoencoder(X_train, contamination=0.01)
    # Evaluate autoencoder on test set
    autoencoder_predictions = autoencoder.evaluate_autoencoder(ae_model, X_test, threshold)
    print("Autoencoder-based anomaly detection complete.")

    # 5. Supervised Fraud Detection Model (Random Forest)
    print("Training supervised model (Random Forest)...")
    rf_model = supervised_model.train_supervised_model(X_train, y_train)
    y_pred, y_prob = supervised_model.evaluate_supervised_model(rf_model, X_test, y_test)

    # 6. Evaluation & Plotting
    print("Evaluating model performance...")
    utils.print_classification_report(y_test, y_pred)
    utils.plot_roc_curve(y_test, y_prob)

if __name__ == "__main__":
    main()
