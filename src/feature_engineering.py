"""Feature Engineering Stage - Create features for model training."""
import pandas as pd
import numpy as np
import os
import sys


def engineer_features(input_path: str, output_path: str) -> None:
    """Create features from cleaned data."""
    
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        sys.exit(1)
    
    df = pd.read_csv(input_path)
    print(f"Input shape: {df.shape}")
    
    # ------------------------
    # Feature Engineering
    # ------------------------
    
    # 1. Debt-to-Income Ratio (key financial metric)
    df["dti"] = df["loan_amount"] / df["income"]
    
    # 2. Credit Score Buckets (categorical feature)
    df["credit_bucket"] = pd.cut(
        df["credit_score"],
        bins=[300, 580, 670, 740, 850],
        labels=["Poor", "Fair", "Good", "Excellent"]
    ).astype(str)
    
    # 3. Income Bucket (categorical feature)
    df["income_bucket"] = pd.cut(
        df["income"],
        bins=[0, 30000, 50000, 75000, float('inf')],
        labels=["Low", "Medium", "High", "VeryHigh"]
    ).astype(str)
    
    # 4. Encode categorical features
    df = pd.get_dummies(df, columns=["credit_bucket", "income_bucket"], drop_first=True)
    
    print(f"Output shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Featured data saved to {output_path}")


if __name__ == "__main__":
    engineer_features(
        input_path="data/processed/cleaned_data.csv",
        output_path="data/processed/featured_data.csv"
    )
