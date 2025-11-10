"""
Unit tests for the Modeler class and model evaluation process.
Tests include:
- Modeler class initialization and data splitting
- Model training and evaluation
- Best model selection based on R2 score
"""

import unittest 
import pandas as pd

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src import preprocessing
from src.modeling import Modeler

data_folder = os.path.join(project_root, "data")  # absolute path to data directory

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


models = {
    # linear regression
    'LinearRegression': {
        'model': LinearRegression(),
        'params': {}
    },
    # decision tree 
    'decision_tree': {
        'model': DecisionTreeRegressor(random_state=42),
        'params': {
            'model__max_depth': [10, 20],
            'model__min_samples_split': [2, 5]
        }
    },
    # hist gradient boosting
    'HistGradientBoostingRegressor': {
        'model': HistGradientBoostingRegressor(random_state=42),
        'params': {
            'model__max_depth': [10, 20],
            'model__min_samples_leaf': [2, 5]
        }
    },
    # xgboost (slim grid for speed)
    'XGBoost': {
        'model': XGBRegressor(
            tree_method='hist',
            random_state=42,
            n_estimators=600,           
            learning_rate=0.06,        
            subsample=0.8,
            colsample_bytree=0.8,
            max_bin=128,               
            n_jobs=-1
        ),
        'params': {
            'model__objective': ['reg:squarederror'],     # single objective to reduce combos
            'model__max_depth': [6],                   # fewer, shallower trees
            'model__min_child_weight': [3],            # regularization via min child weight
            'model__gamma': [0, 1],                       # split loss threshold
            'model__reg_alpha': [0.01],                # L1
            'model__reg_lambda': [1]                  # L2
        }
    },
    # ANN 
    'ANN': {
        'model': MLPRegressor(
            random_state=42,
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=5,
            tol=1e-3
        ),
        'params': {
            'model__hidden_layer_sizes': [(64,)],
            'model__activation': ['relu'],
            'model__learning_rate_init': [1e-3],
            'model__alpha': [1e-4],
            'model__batch_size': [128]
        }
    }
}


class TestModelEvaluation(unittest.TestCase):
    """
    Unit tests for the Modeler class and model evaluation process.
    """
    def setUp(self):
        """
        Set up the test environment by loading the dataset, preprocessing it, and initializing the Modeler.
        """
        # Use the provided training dataset for testing
        training_csv = os.path.join(data_folder, "TrainingSet.csv")
        if not os.path.exists(training_csv):
            raise FileNotFoundError(f"Required test data not found: {training_csv}")
        # sample up to 1000 rows for faster tests
        self.df = pd.read_csv(training_csv)
        self.processed_df = preprocessing.preprocess_data(self.df)
        target_col = "Close Price"
        self.modeler = Modeler(self.processed_df, target_col)
        self.modeler.train_test_split(0.3)
        self.results = self.modeler.model_evals(models=models)
    def test_modeler_class(self):
        """Test that the Modeler class initializes correctly and splits data."""
        #make sure x train isnt empty
        self.assertFalse(self.modeler.X_train.empty)
        self.assertFalse(self.modeler.y_train.empty)
    def test_modeler_training(self):
        """Test that the model evaluation process runs and produces results."""
        #make sure trained models are not empty
        self.assertFalse(self.results.empty)
    def test_modeler_best_model_threshold(self):
        """Test that the best model's R2 score is above a certain threshold."""
        #make sure test R2 of top model is above 0.7
        threshold = 0.7
        top_model = self.results.loc[self.results['TestR2'].idxmax()]
        self.assertGreater(top_model['TestR2'], threshold)