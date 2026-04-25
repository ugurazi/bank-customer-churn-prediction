#!/usr/bin/env python
"""
Main pipeline: Load data → Feature Engineering → Preprocessing → Train → Evaluate → Score
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
from feature_engineering import engineer_features, get_feature_names
from preprocessing import preprocess, save_scaler
from train import train_models, get_best_model, save_model, save_comparison
from evaluate import full_evaluation
from score_dataset import score_dataset
import joblib


def main():
    print("\n" + "="*70)
    print("BANK CUSTOMER CHURN PREDICTION - FULL PIPELINE")
    print("="*70)

    # 1. Load data
    print("\n[1/5] Loading data...")
    df = pd.read_csv('data/bank_churn.csv')
    print(f"✓ Loaded {df.shape[0]} customers with {df.shape[1]} features")
    print(f"  Churn rate: {df['Exited'].mean():.1%}")

    # 2. Feature Engineering
    print("\n[2/5] Engineering features...")
    df_engineered = engineer_features(df)
    feature_names = get_feature_names(df_engineered)
    print(f"✓ Created {len(feature_names)} features from {df.shape[1]} original columns")

    # 3. Preprocessing
    print("\n[3/5] Preprocessing...")
    X_train, X_test, y_train, y_test, scaler = preprocess(
        df_engineered,
        test_size=0.2,
        apply_smote=True
    )
    print(f"✓ Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"  Train churn rate: {y_train.mean():.1%}")
    print(f"  Test churn rate: {y_test.mean():.1%}")

    # Save scaler
    save_scaler(scaler)

    # 4. Training
    print("\n[4/5] Training models...")
    trained_models, all_results = train_models(
        X_train, X_test, y_train, y_test,
        use_grid_search=True
    )

    best_model_name, best_model, best_metrics = get_best_model(
        trained_models, all_results, metric='AUC_ROC'
    )

    # Save best model
    save_model(best_model)

    # Save comparison
    save_comparison(all_results)

    # 5. Evaluation
    print("\n[5/5] Model evaluation & visualization...")
    full_evaluation(trained_models, X_test, y_test, best_model_name, feature_names)

    # 6. Score dataset for Power BI
    print("\n[6/6] Scoring dataset for Power BI...")
    try:
        scored_df = score_dataset()
        print("✓ Dataset scored and ready for Power BI")
    except Exception as e:
        print(f"⚠ Scoring failed: {e}")

    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Open Power BI Web (app.powerbi.com)")
    print("2. Upload 'data/bank_churn_scored.csv' as data source")
    print("3. Create dashboard with 3 pages (see PLAN.md for specifications)")
    print("4. Publish and share the link")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
