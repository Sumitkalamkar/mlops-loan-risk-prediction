import pandas as pd
import numpy as np

np.random.seed(42)

# Generate base features
data = pd.DataFrame({
    "income": np.random.randint(20000, 100000, 500),
    "credit_score": np.random.randint(300, 850, 500),
})

# Generate loan amount based on income (realistic borrowing behavior)
data["loan_amount"] = (
    data["income"] * np.random.uniform(0.1, 0.8, 500)
).astype(int)

# Add Debt-to-Income ratio
data["dti"] = data["loan_amount"] / data["income"]


def assign_risk(row):
    dti = row["dti"]
    score = row["credit_score"]

    if dti > 0.6:
        if score < 650:
            return "High"
        else:
            return "Medium"

    elif dti > 0.3:
        if score < 600:
            return "High"
        elif score < 750:
            return "Medium"
        else:
            return "Low"

    else:
        if score < 600:
            return "Medium"
        else:
            return "Low"

# Apply labeling
data["risk"] = data.apply(assign_risk, axis=1)

# Optional: Drop dti if you want model to learn it implicitly
# data = data.drop(columns=["dti"])

# Save dataset
data.to_csv("data/loan_data.csv", index=False)

# Check distribution
print("\nRisk distribution:")
print(data["risk"].value_counts())
