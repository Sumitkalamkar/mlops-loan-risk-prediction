"""Data Validation Stage - Validate raw data quality before processing."""
import pandas as pd
import sys
import os


def validate_data(input_path: str) -> bool:
    """Validate raw data quality. Returns True if valid, exits with error if not."""
    
    if not os.path.exists(input_path):
        print(f"Data file not found: {input_path}")
        sys.exit(1)
    
    df = pd.read_csv(input_path)
    errors = []
    
    print(f"Validating data: {input_path}")
    print(f"Rows: {len(df)}, Columns: {list(df.columns)}")
    
    # Required columns check
    required_cols = ["income", "loan_amount", "credit_score", "risk"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Data type checks
    if df.empty:
        errors.append("Dataset is empty!")
    
    # Check for minimum rows
    if len(df) < 100:
        errors.append(f"Insufficient data: {len(df)} rows (minimum 100 required)")
    
    # Check target distribution (need all classes for training)
    if "risk" in df.columns:
        risk_values = df["risk"].unique()
        expected_risks = {"Low", "Medium", "High"}
        if not expected_risks.issubset(set(risk_values)):
            errors.append(f"Missing risk classes. Found: {list(risk_values)}, Expected: {expected_risks}")
    
    # Report validation results
    if errors:
        print("\nDATA VALIDATION FAILED:")
        for e in errors:
            print(f"   - {e}")
        sys.exit(1)
    
    # Warnings (non-fatal)
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        print(f"Warning: {null_count} null values found (will be handled in cleaning)")
    
    print("Data validation passed!")
    return True


if __name__ == "__main__":
    validate_data("data/loan_data.csv")
