"""
test_data_preprocessing.py
----------------------------
Basic tests for the data preprocessing functions.
"""

import unittest
import pandas as pd
from src import data_preprocessing

class TestDataPreprocessing(unittest.TestCase):

    def test_load_data(self):
        # Create a temporary CSV file
        df_expected = pd.DataFrame({"A": [1, 2, 3]})
        test_filepath = "temp_test.csv"
        df_expected.to_csv(test_filepath, index=False)

        df_loaded = data_preprocessing.load_data(test_filepath)
        self.assertTrue("A" in df_loaded.columns)

    def test_preprocess_data(self):
        df = pd.DataFrame({
            "num": [1, 2, None, 4],
            "is_fraud": [0, 1, 0, 0]
        })
        df_processed = data_preprocessing.preprocess_data(df)
        self.assertFalse(df_processed["num"].isnull().any())

if __name__ == '__main__':
    unittest.main()
