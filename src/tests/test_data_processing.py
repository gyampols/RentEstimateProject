import unittest 
import pandas as pd

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src import preprocessing

data_folder = os.path.join(project_root, "data")  # absolute path to data directory


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Use the provided training dataset for testing
        training_csv = os.path.join(data_folder, "TrainingSet.csv")
        if not os.path.exists(training_csv):
            raise FileNotFoundError(f"Required test data not found: {training_csv}")
        # sample up to 1000 rows for faster tests
        df = pd.read_csv(training_csv)
        n = min(1000, len(df))
        self.df = df.sample(n=n, random_state=42)
        self.processed_df = preprocessing.preprocess_data(self.df)

    def test_merge_zillow_data(self):
        self.assertIn('ZHI', self.processed_df.columns)
        self.assertTrue(self.processed_df['ZHI'].notna().all())

    def test_calc_distance_to_transit(self):
        self.assertIn('Distance_to_Transit', self.processed_df.columns)
        self.assertTrue(self.processed_df['Distance_to_Transit'].notna().all())
        self.assertTrue((self.processed_df['Distance_to_Transit'] >= 0).all())
        self.assertTrue((self.processed_df['Distance_to_Transit'] <= self.processed_df['Distance_to_Transit'].max()).all()) 
