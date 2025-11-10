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
        # lets read just 1000 random rows for testing
        self.df = pd.read_csv(os.path.join(data_folder, "properties.csv")).sample(n=1000, random_state=42)
        self.processed_df = preprocessing.preprocess_data(self.df)

    def test_add_state_codes(self):
        self.assertIn('State', self.processed_df.columns)
        self.assertTrue(self.processed_df['State'].notna().all())

    def test_merge_zillow_data_by_zip(self):
        self.assertIn('Zillow_Zip_Monthly', self.processed_df.columns)
        self.assertTrue(self.processed_df['Zillow_Zip_Monthly'].notna().all())

    def test_merge_zillow_data_by_state(self):
        self.assertIn('Zillow_State_Monthly', self.processed_df.columns)
        self.assertTrue(self.processed_df['Zillow_State_Monthly'].notna().all())

    def test_calc_distance_to_transit(self):
        self.assertIn('Distance_to_Transit', self.processed_df.columns)
        self.assertTrue(self.processed_df['Distance_to_Transit'].notna().all())
        self.assertTrue((self.processed_df['Distance_to_Transit'] >= 0).all())
        self.assertTrue((self.processed_df['Distance_to_Transit'] <= self.processed_df['Distance_to_Transit'].max()).all()) 
