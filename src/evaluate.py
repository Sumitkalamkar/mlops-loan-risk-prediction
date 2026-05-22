"""Model Evaluation Stage - Quality gate for model deployment."""
import json
import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)


def evaluate_model(
    data_path: str = "data/processed/test_data.csv",
    models_dir: str = "models",
    metrics_path: str = "metrics.json",
    min_accuracy: float = 0.70
) -> bool:
    """
    Evaluate model and act as quality gate.
    
    Args:
        data_path: Path to test data (test_data.csv from training stage)
        models_dir: Directory containing model artifacts
        metrics_path: Path to save metrics JSON
        min_accuracy: Minimum accuracy threshold
        
    Returns:
        True if model passes quality gate
    """
    
    print("Model Evaluation Stage")
    print("=" * 50)
    
    # Load artifacts
    model_path = os.path.join(models_dir, "model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    encoder_path = os.path.join(models_dir, "label_encoder.pkl")
    features_path = os.path.join(models_dir, "feature_names.pkl")
    
    for path in [model_path, scaler_path, encoder_path, data_path]:
        if not os.path.exists(path):
            print(f"Required file not found: {path}")
            sys.exit(1)
    
    # Load model and preprocessors
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(encoder_path)
    feature_names = joblib.load(features_path) if os.path.exists(features_path) else None
    
    # Load evaluation data
    df = pd.read_csv(data_path)
    
    # Prepare features (same preprocessing as training)
    y_true = le.transform(df["risk"])
    X = df.drop(columns=["risk"])
    
    # Keep only numeric columns used in training
    if feature_names:
        X = X[feature_names]
    else:
        X = X.select_dtypes(include=['int64', 'float64', 'uint8'])
    
    X_scaled = scaler.transform(X)
    
    # Predict
    y_pred = model.predict(X_scaled)
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average='weighted')),
        "recall": float(recall_score(y_true, y_pred, average='weighted')),
        "f1": float(f1_score(y_true, y_pred, average='weighted')),
        "samples_evaluated": len(y_true),
    }
    
    # Print results
    print(f"\nEvaluation Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    # Save metrics
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Quality Gate
    print(f"\nQuality Gate (min accuracy: {min_accuracy})")
    if metrics["accuracy"] >= min_accuracy:
        print(f"PASSED - Model accuracy {metrics['accuracy']:.4f} >= {min_accuracy}")
        return True
    else:
        print(f"FAILED - Model accuracy {metrics['accuracy']:.4f} < {min_accuracy}")
        sys.exit(1)


if __name__ == "__main__":
    # Can override threshold via environment variable
    threshold = float(os.environ.get("MIN_MODEL_ACCURACY", "0.70"))
    evaluate_model(min_accuracy=threshold)
