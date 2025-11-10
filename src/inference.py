import joblib
from pathlib import Path
from datetime import datetime
from src import preprocessing



def predict_market_rent(df):
    """
    Predict market rent for the given DataFrame using the best trained model. 
    Exports results to a timestamped CSV file.
    Args:
        df (pd.DataFrame): Input DataFrame with property data.
    Returns:
        pd.DataFrame: DataFrame with original data and a new column 'calculated Market Rent'
    """
    # Preprocess test data (same function used for training)
    output_df = df.copy()
    processed_test_df = preprocessing.preprocess_data(df)
    # Load best model
    best_pipeline = joblib.load('./models/final_model.joblib')
    # Predict
    preds = best_pipeline.predict(processed_test_df)
    # Assemble output
    output_df['calculated Market Rent'] = preds
    export_dir = Path('.data/exports')
    export_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    export_file = export_dir / f"final_predictions_{stamp}.csv"
    output_df.to_csv(export_file, index=False)
    print(f"Predictions written to {export_file} (rows={len(output_df)})")
    return output_df