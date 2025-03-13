"""
autoencoder.py
-------------------
Implements an autoencoder model for unsupervised anomaly detection.
"""

import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

def build_autoencoder(input_dim):
    """
    Build a simple autoencoder model.
    """
    input_layer = Input(shape=(input_dim,))

    # Encoder
    encoded = Dense(int(input_dim / 2), activation="relu")(input_layer)
    encoded = Dense(int(input_dim / 4), activation="relu")(encoded)

    # Bottleneck
    bottleneck = Dense(int(input_dim / 8), activation="relu")(encoded)

    # Decoder
    decoded = Dense(int(input_dim / 4), activation="relu")(bottleneck)
    decoded = Dense(int(input_dim / 2), activation="relu")(decoded)
    decoded = Dense(input_dim, activation="linear")(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer="adam", loss="mse")

    return autoencoder

def train_autoencoder(X_train, contamination=0.01, epochs=50, batch_size=32):
    """
    Train the autoencoder on training data and determine an anomaly threshold.
    """
    input_dim = X_train.shape[1]
    model = build_autoencoder(input_dim)

    # Early stopping for better training performance
    es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

    # Train only on non-fraud examples assuming fraud is a small fraction
    model.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        callbacks=[es]
    )

    # Compute reconstruction errors on training data
    X_train_pred = model.predict(X_train)
    mse = np.mean(np.power(X_train - X_train_pred, 2), axis=1)

    # Set threshold based on contamination level (e.g., 99th percentile)
    threshold = np.percentile(mse, 100 * (1 - contamination))
    print(f"Autoencoder training complete. Threshold for anomalies set at: {threshold:.4f}")

    return model, threshold

def evaluate_autoencoder(model, X, threshold):
    """
    Evaluate the autoencoder on data X and flag anomalies.
    Returns a binary prediction: 1 for anomaly, 0 for normal.
    """
    X_pred = model.predict(X)
    mse = np.mean(np.power(X - X_pred, 2), axis=1)
    predictions = (mse > threshold).astype(int)
    return predictions
