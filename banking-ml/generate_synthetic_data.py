#!/usr/bin/env python
"""
Generate synthetic Bank Customer Churn dataset matching Kaggle schema.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_churn_dataset(n_samples=10000, random_state=42):
    """Generate synthetic bank churn dataset."""
    np.random.seed(random_state)

    df = pd.DataFrame({
        'RowNumber': range(1, n_samples + 1),
        'CustomerId': np.random.randint(100000, 100000 + n_samples, n_samples),
        'Surname': np.random.choice(['Smith', 'Brown', 'Johnson', 'Williams', 'Jones', 'Garcia', 'Miller', 'Davis'], n_samples),
        'CreditScore': np.random.normal(650, 100, n_samples).astype(int).clip(300, 850),
        'Geography': np.random.choice(['France', 'Germany', 'Spain'], n_samples, p=[0.5, 0.25, 0.25]),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.normal(39, 15, n_samples).astype(int).clip(18, 92),
        'Tenure': np.random.randint(0, 11, n_samples),
        'Balance': np.random.exponential(scale=100000, size=n_samples).astype(float),
        'NumOfProducts': np.random.randint(1, 5, n_samples),
        'HasCrCard': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'IsActiveMember': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'EstimatedSalary': np.random.uniform(11581, 199992, n_samples),
    })

    # Create target with some correlation to features
    churn_prob = (
        0.01 * (df['Age'] - 40) +  # older customers churn more
        0.02 * (df['Geography'] == 'Germany').astype(int) +  # Germany has higher churn
        -0.05 * df['IsActiveMember'] +  # active members churn less
        -0.02 * df['NumOfProducts'] +  # more products = less churn
        0.003 * (df['CreditScore'] < 500).astype(int) +
        0.01 * (df['Balance'] == 0).astype(int) +  # zero balance = higher churn
        np.random.normal(0, 0.1, n_samples)
    )
    churn_prob = 1 / (1 + np.exp(-churn_prob))  # sigmoid
    df['Exited'] = (churn_prob > 0.5).astype(int)

    return df

if __name__ == '__main__':
    output_path = Path('data/bank_churn.csv')

    print("Generating synthetic Bank Customer Churn dataset...")
    df = generate_churn_dataset(n_samples=10000)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✓ Dataset saved to {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Churn rate: {df['Exited'].mean():.1%}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
