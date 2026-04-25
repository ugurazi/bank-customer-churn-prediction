"""
Model training with hyperparameter tuning for Bank Customer Churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
import joblib
import warnings
warnings.filterwarnings('ignore')


def train_models(X_train, X_test, y_train, y_test, use_grid_search=True):
    """
    Train 4 models: Logistic Regression, Random Forest, XGBoost, LightGBM
    with optional hyperparameter tuning.

    Parameters
    ----------
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test targets
    use_grid_search : bool
        Whether to use GridSearchCV for tuning (slower but better results)

    Returns
    -------
    dict
        Dictionary with trained models and their metrics
    """
    models_config = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, use_label_encoder=False,
                                  eval_metric='logloss', n_jobs=-1),
            'params': {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        },
        'LightGBM': {
            'model': LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1),
            'params': {
                'max_depth': [5, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200],
                'num_leaves': [20, 31, 50]
            }
        }
    }

    trained_models = {}
    all_results = {}

    for model_name, config in models_config.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        print(f"{'='*50}")

        if use_grid_search and model_name != 'Logistic Regression':
            # Use reduced param grid for speed
            reduced_params = {
                'Random Forest': {'n_estimators': [100], 'max_depth': [10, 15]},
                'XGBoost': {'max_depth': [5, 7], 'learning_rate': [0.05, 0.1], 'n_estimators': [100, 200]},
                'LightGBM': {'max_depth': [10], 'learning_rate': [0.05, 0.1], 'n_estimators': [100]}
            }
            params = reduced_params.get(model_name, config['params'])

            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=params,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score (AUC): {grid_search.best_score_:.4f}")
        else:
            best_model = config['model']
            best_model.fit(X_train, y_train)
            print("Model trained without hyperparameter tuning")

        # Make predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'AUC_ROC': roc_auc_score(y_test, y_pred_proba)
        }

        trained_models[model_name] = best_model
        all_results[model_name] = metrics

        print(f"\nMetrics on Test Set:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    return trained_models, all_results


def get_best_model(trained_models, all_results, metric='AUC_ROC'):
    """
    Get the best performing model based on specified metric.

    Parameters
    ----------
    trained_models : dict
        Dictionary of trained models
    all_results : dict
        Dictionary of model metrics
    metric : str
        Metric to use for selection (default 'AUC_ROC')

    Returns
    -------
    tuple
        (best_model_name, best_model, best_metrics)
    """
    best_model_name = max(all_results.keys(),
                          key=lambda x: all_results[x][metric])
    best_model = trained_models[best_model_name]
    best_metrics = all_results[best_model_name]

    print(f"\n{'='*50}")
    print(f"Best Model: {best_model_name}")
    print(f"Metrics: {best_metrics}")
    print(f"{'='*50}\n")

    return best_model_name, best_model, best_metrics


def save_model(model, path='models/best_model.pkl'):
    """Save trained model."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path='models/best_model.pkl'):
    """Load trained model."""
    return joblib.load(path)


def save_comparison(all_results, path='outputs/model_comparison.csv'):
    """Save model comparison to CSV."""
    results_df = pd.DataFrame(all_results).T
    results_df.to_csv(path)
    print(f"\nModel comparison saved to {path}")
    print(results_df.round(4))
    return results_df
