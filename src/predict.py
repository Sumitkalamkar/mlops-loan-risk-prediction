"""Prediction module for loan risk API."""
import joblib
import numpy as np
import pandas as pd
import os

# Load models lazily (on first prediction)
_model = None
_scaler = None
_label_encoder = None
_feature_names = None

MODELS_DIR = "models"


def _load_models():
    """Load model artifacts from disk."""
    global _model, _scaler, _label_encoder, _feature_names
    
    if _model is None:
        print("Loading model artifacts...")
        _model = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
        _scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        _label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
        
        # Load feature names if available
        feature_path = os.path.join(MODELS_DIR, "feature_names.pkl")
        if os.path.exists(feature_path):
            _feature_names = joblib.load(feature_path)
        print(f"Models loaded. Features: {_feature_names}")


def predict(data: list) -> tuple:
    """
    Make prediction for loan risk.
    
    Args:
        data: [income, loan_amount, credit_score]
        
    Returns:
        (risk_label, confidence_score)
    """
    _load_models()
    
    # Create DataFrame with base features
    income, loan_amount, credit_score = data
    
    # Create features matching training data
    features = {
        "income": income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "dti": loan_amount / income,
    }
    
    # Add credit bucket one-hot features
    if credit_score < 580:
        credit_bucket = "Poor"
    elif credit_score < 670:
        credit_bucket = "Fair"
    elif credit_score < 740:
        credit_bucket = "Good"
    else:
        credit_bucket = "Excellent"
    
    features["credit_bucket_Fair"] = 1 if credit_bucket == "Fair" else 0
    features["credit_bucket_Good"] = 1 if credit_bucket == "Good" else 0
    features["credit_bucket_Excellent"] = 1 if credit_bucket == "Excellent" else 0
    
    # Add income bucket one-hot features
    if income < 30000:
        income_bucket = "Low"
    elif income < 50000:
        income_bucket = "Medium"
    elif income < 75000:
        income_bucket = "High"
    else:
        income_bucket = "VeryHigh"
    
    features["income_bucket_Medium"] = 1 if income_bucket == "Medium" else 0
    features["income_bucket_High"] = 1 if income_bucket == "High" else 0
    features["income_bucket_VeryHigh"] = 1 if income_bucket == "VeryHigh" else 0
    
    # Create DataFrame and select features in correct order
    df = pd.DataFrame([features])
    
    if _feature_names:
        # Ensure we have all required features
        for f in _feature_names:
            if f not in df.columns:
                df[f] = 0
        df = df[_feature_names]
    
    # Scale and predict
    X_scaled = _scaler.transform(df)
    pred = _model.predict(X_scaled)
    prob = float(_model.predict_proba(X_scaled).max())
    
    # Decode label
    risk_label = _label_encoder.inverse_transform(pred)[0]
    
    return risk_label, prob