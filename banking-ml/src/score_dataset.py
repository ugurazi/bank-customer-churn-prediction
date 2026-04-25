"""
Score the full dataset with the trained model for Power BI dashboard.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from feature_engineering import engineer_features
from preprocessing import load_scaler


def score_dataset(model_path='models/best_model.pkl',
                  scaler_path='models/scaler.pkl',
                  data_path='data/bank_churn.csv',
                  output_path='data/bank_churn_scored.csv'):
    """
    Score the full dataset with churn probabilities and risk levels.

    Parameters
    ----------
    model_path : str
        Path to saved model
    scaler_path : str
        Path to saved scaler
    data_path : str
        Path to raw data
    output_path : str
        Path to save scored data
    """
    print("Loading data and model...")

    # Load data
    df = pd.read_csv(data_path)
    original_df = df.copy()

    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Engineer features
    df_engineered = engineer_features(df)

    # Scale features
    feature_cols = [col for col in df_engineered.columns if col != 'Exited']
    X = df_engineered[feature_cols].values
    X_scaled = scaler.transform(X)

    # Get predictions and probabilities
    print("Scoring dataset...")
    churn_proba = model.predict_proba(X_scaled)[:, 1]
    churn_pred = model.predict(X_scaled)

    # Create risk levels
    risk_levels = pd.cut(churn_proba,
                         bins=[0, 0.3, 0.6, 1.0],
                         labels=['Low Risk', 'Medium Risk', 'High Risk'])

    # Add predictions to original dataframe
    output_df = original_df.copy()
    output_df['ChurnProbability'] = churn_proba
    output_df['ChurnPrediction'] = churn_pred
    output_df['RiskLevel'] = risk_levels

    # Save scored dataset
    output_df.to_csv(output_path, index=False)
    print(f"Scored dataset saved to {output_path}")

    # Print summary
    print(f"\nDataset Summary:")
    print(f"  Total customers: {len(output_df)}")
    print(f"  Predicted churners: {churn_pred.sum()} ({churn_pred.mean():.1%})")
    print(f"\nRisk Distribution:")
    print(output_df['RiskLevel'].value_counts())

    return output_df


def export_feature_importance(model_path='models/best_model.pkl',
                              feature_names=None,
                              output_path='data/feature_importance.csv'):
    """
    Export feature importance for Power BI.

    Parameters
    ----------
    model_path : str
        Path to saved model
    feature_names : list, optional
        List of feature names (if None, will be inferred)
    output_path : str
        Path to save feature importance
    """
    model = joblib.load(model_path)

    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return

    importances = model.feature_importances_

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(importances))]

    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    feat_imp_df.to_csv(output_path, index=False)
    print(f"Feature importance exported to {output_path}")

    return feat_imp_df


def export_customer_explanations(model_path='models/best_model.pkl',
                                 scaler_path='models/scaler.pkl',
                                 data_path='data/bank_churn.csv',
                                 output_path='data/customer_explanations.csv',
                                 use_shap=False):
    """
    Export per-customer SHAP values (optional, for detailed explanations).

    Parameters
    ----------
    model_path, scaler_path, data_path : str
        Paths to model, scaler, and data
    output_path : str
        Path to save explanations
    use_shap : bool
        Whether to use SHAP (slower but more interpretable)
    """
    print("Exporting customer explanations...")

    df = pd.read_csv(data_path)
    from feature_engineering import engineer_features
    df_engineered = engineer_features(df)

    # This is optional - just save summary stats
    summary = pd.DataFrame({
        'CustomerId': df['CustomerId'],
        'Age': df['Age'],
        'Geography': df['Geography'],
        'NumOfProducts': df['NumOfProducts'],
        'Balance': df['Balance'],
        'IsActiveMember': df['IsActiveMember']
    })

    summary.to_csv(output_path, index=False)
    print(f"Customer explanations exported to {output_path}")

    return summary


if __name__ == '__main__':
    # Main execution
    print("="*60)
    print("SCORING DATASET FOR POWER BI")
    print("="*60)

    # Score dataset
    scored_df = score_dataset()

    print("\n" + "="*60)
    print("All exports complete!")
    print("="*60)
