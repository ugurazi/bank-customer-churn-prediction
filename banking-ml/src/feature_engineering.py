"""
Feature Engineering for Bank Customer Churn dataset.
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from raw bank churn data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset with original features

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features, ready for preprocessing
    """
    df = df.copy()

    # 1. Drop unnecessary columns
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    # 2. Create new features
    # Balance-to-Salary ratio
    df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)

    # Is balance zero
    df['IsBalanceZero'] = (df['Balance'] == 0).astype(int)

    # Age groups
    df['AgeGroup'] = pd.cut(df['Age'],
                            bins=[0, 30, 40, 50, 100],
                            labels=['18-30', '31-40', '41-50', '51+'])

    # Credit score groups
    df['CreditScoreGroup'] = pd.cut(df['CreditScore'],
                                    bins=[0, 580, 670, 740, 850],
                                    labels=['Poor', 'Fair', 'Good', 'Excellent'])

    # Tenure to age ratio
    df['TenureAgeRatio'] = df['Tenure'] / (df['Age'] + 1)

    # Products per tenure
    df['ProductsPerTenure'] = df['NumOfProducts'] / (df['Tenure'] + 1)

    # 3. One-hot encode categorical variables
    df = pd.get_dummies(df,
                        columns=['Geography', 'Gender', 'AgeGroup', 'CreditScoreGroup'],
                        drop_first=False)

    return df


def get_feature_names(df_engineered: pd.DataFrame) -> list:
    """Get list of all feature names after engineering."""
    return [col for col in df_engineered.columns if col != 'Exited']
