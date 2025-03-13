"""
utils.py
-----------
Utility functions for model evaluation and plotting.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc

def print_classification_report(y_true, y_pred):
    """
    Print the classification report.
    """
    report = classification_report(y_true, y_pred)
    print("Classification Report:\n", report)

def plot_roc_curve(y_true, y_prob):
    """
    Plot the ROC curve for binary classification.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
