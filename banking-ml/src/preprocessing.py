"""
Preprocessing pipeline for Bank Customer Churn dataset.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib


def preprocess(df: pd.DataFrame,
               test_size: float = 0.2,
               random_state: int = 42,
               apply_smote: bool = True) -> tuple:
    """
    Preprocess the dataset: split, scale, and handle imbalance.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with engineered features (from feature_engineering.py output)
    test_size : float, optional
        Proportion of data for test set (default 0.2)
    random_state : int, optional
        Random seed for reproducibility (default 42)
    apply_smote : bool, optional
        Whether to apply SMOTE to training data (default True)

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, scaler)
        - X_train, X_test: numpy arrays, scaled features
        - y_train, y_test: numpy arrays, target variable
        - scaler: fitted StandardScaler object for later transformation
    """
    df = df.copy()

    # Separate features and target
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    # Train-test split (stratified on target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrames for SMOTE (which expects column names)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Handle class imbalance with SMOTE (only on training data!)
    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(f"SMOTE applied - new training set size: {X_train_scaled.shape[0]}")
        print(f"Churn distribution after SMOTE: {y_train.value_counts().to_dict()}")

    # Convert back to numpy arrays
    X_train_scaled = X_train_scaled.values
    X_test_scaled = X_test_scaled.values

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler


def save_scaler(scaler, path: str = 'models/scaler.pkl'):
    """Save the fitted scaler for later use."""
    joblib.dump(scaler, path)
    print(f"Scaler saved to {path}")


def load_scaler(path: str = 'models/scaler.pkl'):
    """Load the fitted scaler."""
    return joblib.load(path)
