"""
supervised_model.py
----------------------
Trains and evaluates a supervised fraud detection model using Random Forest.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def train_supervised_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest classifier.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print("Supervised model training complete.")
    return clf

def evaluate_supervised_model(model, X_test, y_test):
    """
    Evaluate the supervised model on the test set.
    Returns predictions and prediction probabilities.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # probability for the positive class
    auc = roc_auc_score(y_test, y_prob)
    print(f"Supervised Model AUC: {auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return y_pred, y_prob
