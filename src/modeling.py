#import libraries
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import pandas as pd
from tqdm import tqdm
import os

class Modeler():
    def __init__(self,df,target_col='Close Price'):
        self.df=df
        self.X=self.df.drop(columns=[target_col])
        self.y=self.df[target_col]
        self.results_df=pd.DataFrame(columns=['Model Name','Best Parameters','Training_Latency','Train_MSE','Test_MSE','TrainR2','TestR2'])
        # Feature names after feature engineering (feature engineering must run first)
        # Apply global numeric pipeline to all but Bedrooms, which will get a dedicated
        # PowerTransformer branch to stabilize its distribution.
        numerical_features = [
            'House Age', 'Age at Sale', 'Close Month', 'Latitude', 'Longitude', 'ZHI', 'Close Year'
        ]
        skewed_features = ['Bedrooms', 'Square Feet', 'Bathrooms', 'DistanceToTransit']

        # Fix: proper Pipeline syntax and add imputation for NaNs in ZHI
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Dedicated transformer for Bedrooms only: impute -> PowerTransformer (Yeo-Johnson)
        # Yeo-Johnson supports zero/positive integers and standardize=True returns ~N(0,1)
        skewed_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('power', PowerTransformer(method='yeo-johnson', standardize=True))
        ])

        # Combine the transformers into a preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('bedrooms', skewed_transformer, skewed_features),
            ],
            remainder='drop'
        )
        self.best_model = None
        self.best_score = 0
        self.trained_models = {}

    def train_test_split(self, test_size=0.2, random_state=42):
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state) 

    def model_evals(self, models):
        self.trained_models = {}
        for name, model_info in tqdm(models.items()):
            print(f"Training {name}...")
            model = model_info['model']
            params = model_info['params']
            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('model', model)
            ])
            grid_search = GridSearchCV(pipeline,
                                params,
                                scoring="neg_mean_absolute_error",
                                cv=3, 
                                n_jobs=-1
                            )
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            self.trained_models[name] = best_model
            # Evaluate on training set
            y_train_pred = best_model.predict(self.X_train)
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            # Evaluate on test set
            y_test_pred = best_model.predict(self.X_test)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            # Store results
            new_row = pd.DataFrame([{
                'Model Name': name,
                'Best Parameters': grid_search.best_params_,
                'Training_Latency': grid_search.refit_time_,
                'Train_MSE': train_mse,
                'Test_MSE': test_mse,
                'TrainR2': train_r2,
                'TestR2': test_r2
            }])
            self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
            if test_r2 > self.best_score:
                self.best_model = best_model
                self.best_score = test_r2
            print('_' * 40)
        self.results_df.sort_values(by='Test_MSE', inplace=True)
        return self.results_df
    def save_models(self, directory='models/'):
        import os
        if not os.path.exists(directory):
                os.makedirs(directory)
        for name, model in self.trained_models.items():
            joblib.dump(model, f"{directory}{name}.joblib")
        print(f"Models saved to {directory}")
        #save the results dataframe as well
        self.results_df.to_csv(f"{directory}model_results.csv", index=False)
        print(f"Model results saved to {directory}model_results.csv")

    def train_final_model(self):
        # Train best model on entire dataset and save predictions
        if self.best_model is None:
            print("No best model found. Run model_evals() first.")
            return
        
        # Retrain best model on full dataset
        self.best_model.fit(self.X, self.y)
        
        # Save the final trained model
        if not os.path.exists('models/'):
            os.makedirs('models/')
        joblib.dump(self.best_model, 'models/final_model.joblib')
        print("Final model saved to models/final_model.joblib")
        


