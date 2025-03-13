"""
test_models.py
----------------
Basic tests for model training functions.
"""

import unittest
import numpy as np
import pandas as pd
from src.models import autoencoder, supervised_model

class TestModels(unittest.TestCase):

    def setUp(self):
        # Generate synthetic data
        self.X = np.random.rand(100, 10)
        self.y = np.random.randint(0, 2, 100)

    def test_autoencoder_training(self):
        model, threshold = autoencoder.train_autoencoder(self.X, epochs=5)
        self.assertIsNotNone(model)
        self.assertTrue(threshold > 0)

    def test_supervised_model_training(self):
        clf = supervised_model.train_supervised_model(self.X, self.y)
        self.assertIsNotNone(clf)
        y_pred, _ = supervised_model.evaluate_supervised_model(clf, self.X, self.y)
        self.assertEqual(len(y_pred), len(self.y))

if __name__ == '__main__':
    unittest.main()
