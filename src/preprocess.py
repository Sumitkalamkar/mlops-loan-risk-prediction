"""Preprocessing utilities for model training."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os


def preprocess_data(df: pd.DataFrame, models_dir: str = "models", test_data_path: str = "data/processed/test_data.csv"):
    """
    Preprocess data for model training.
    
    Args:
        df: DataFrame with features and 'risk' target column
        models_dir: Directory to save preprocessing artifacts
        test_data_path: Path to save test data for evaluation
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Separate features and target
    if "risk" not in df.columns:
        raise ValueError("'risk' column not found in data!")
    
    # Split BEFORE any preprocessing (keep original data for test file)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save test data (unprocessed) for evaluate.py
    os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
    test_df.to_csv(test_data_path, index=False)
    print(f"   Test data saved to {test_data_path} ({len(test_df)} samples)")
    
    # Now process training data
    y_train = train_df["risk"]
    X_train = train_df.drop(columns=["risk"])
    
    y_test = test_df["risk"]
    X_test = test_df.drop(columns=["risk"])
    
    # Keep only numeric columns
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64', 'uint8']).columns
    X_train = X_train[numeric_cols]
    X_test = X_test[numeric_cols]
    
    print(f"   Features used: {list(X_train.columns)}")
    
    # Encode target labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    joblib.dump(le, os.path.join(models_dir, "label_encoder.pkl"))
    print(f"   Label classes: {list(le.classes_)}")
    
    # Scale features (fit on train only!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    
    # Save feature names for prediction
    joblib.dump(list(X_train.columns), os.path.join(models_dir, "feature_names.pkl"))
    
    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded