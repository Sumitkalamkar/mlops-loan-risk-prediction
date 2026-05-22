"""Data Cleaning Stage - Clean raw loan data."""
import pandas as pd
import numpy as np
import os
import sys

def clean_data(input_path: str, output_path: str) -> None:
    """Clean raw data and save to processed folder."""
    
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        sys.exit(1)
    
    df = pd.read_csv(input_path)
    print(f"🔹 Initial shape: {df.shape}")
    
    # ------------------------
    # 1. Remove duplicates
    # ------------------------
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"   Removed {initial_rows - len(df)} duplicates")
    
    # ------------------------
    # 2. Handle missing/invalid values
    # ------------------------
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # ------------------------
    # 3. Remove invalid values
    # ------------------------
    df = df[
        (df["income"] > 0) &
        (df["loan_amount"] > 0) &
        (df["credit_score"].between(300, 850))
    ]
    
    # ------------------------
    # 4. Remove unrealistic outliers
    # ------------------------
    df = df[df["loan_amount"] < df["income"] * 5]
    
    print(f"🔹 Final cleaned shape: {df.shape}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")


if __name__ == "__main__":
    clean_data(
        input_path="data/loan_data.csv",
        output_path="data/processed/cleaned_data.csv"
    )
