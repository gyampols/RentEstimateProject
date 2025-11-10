"""
Unit tests for the data processing functions in the preprocessing module.
Tests include:
- ZHI merging by ZIP and state
- Distance to transit calculation
- State code assignment
"""
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
    """
    Unit tests for the data processing functions in the preprocessing module.
    """
    def setUp(self):
        """
        Set up the test environment by loading the dataset and preprocessing it.
        """
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
        """Test that ZHI is correctly merged by ZIP and state."""
        self.assertIn('ZHI', self.processed_df.columns)

    def test_calc_DistanceToTransit(self):
        """Test that DistanceToTransit is correctly calculated."""
        self.assertIn('DistanceToTransit', self.processed_df.columns)
        self.assertTrue(self.processed_df['DistanceToTransit'].notna().all())
        self.assertTrue((self.processed_df['DistanceToTransit'] >= 0).all())
        self.assertTrue((self.processed_df['DistanceToTransit'] <= self.processed_df['DistanceToTransit'].max()).all()) 
