"""Model Training Stage - Train and save the loan risk model."""
import pandas as pd
import joblib
import os
import sys

import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_data


def train(input_path: str = "data/processed/featured_data.csv",
          models_dir: str = "models",
          mlflow_uri: str = None):
    """
    Train the loan risk prediction model.
    
    Args:
        input_path: Path to featured data
        models_dir: Directory to save model artifacts
        mlflow_uri: MLflow tracking URI (optional)
    """
    
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        sys.exit(1)
    
    os.makedirs(models_dir, exist_ok=True)
    
    # Configure MLflow (optional - works without server)
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("loan-risk-prediction")
    
    # Load data
    df = pd.read_csv(input_path)
    print(f"Training data shape: {df.shape}")
    
    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(df, models_dir)
    
    # Start MLflow run
    with mlflow.start_run():
        # Log build/version info for reproducibility
        build_number = os.environ.get("BUILD_NUMBER", "local")
        model_tag = f"model-v{build_number}" if build_number != "local" else "local-dev"
        
        mlflow.set_tag("jenkins_build", build_number)
        mlflow.set_tag("model_version_tag", model_tag)
        mlflow.set_tag("reproducibility", f"git checkout {model_tag} && dvc pull")
        
        # Log hyperparameters
        params = {
            "model_type": "XGBClassifier",
            "eval_metric": "mlogloss",
            "test_size": 0.3,
            "random_state": 42,
            "n_estimators": 100,
            "max_depth": 8,
        }
        mlflow.log_params(params)
        
        # Train model
        model = XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            eval_metric=params["eval_metric"],
            random_state=params["random_state"],
            use_label_encoder=False
        )
        model.fit(X_train, y_train)
        
        # Predictions
        preds = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, average='weighted'),
            "recall": recall_score(y_test, preds, average='weighted'),
            "f1": f1_score(y_test, preds, average='weighted'),
        }
        
        print(f"Model Metrics:")
        for k, v in metrics.items():
            print(f"   {k}: {v:.4f}")
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        
        # Save model locally (DVC will track this)
        model_path = os.path.join(models_dir, "model.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
        # Log model to MLflow
        mlflow.xgboost.log_model(model, "model")
        
        print("Training complete!")
        
        return metrics


if __name__ == "__main__":
    # Use environment variable for MLflow URI if set
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
    train(mlflow_uri=mlflow_uri)