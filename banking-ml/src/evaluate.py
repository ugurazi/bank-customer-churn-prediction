"""
Model evaluation and visualization for Bank Customer Churn prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             classification_report, roc_auc_score)
import shap
import warnings
warnings.filterwarnings('ignore')


def plot_confusion_matrix(y_true, y_pred, model_name='Model', save_path='outputs/confusion_matrix.png'):
    """
    Plot and save confusion matrix.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model for title
    save_path : str
        Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curves(trained_models, X_test, y_test, save_path='outputs/roc_curve.png'):
    """
    Plot ROC curves for all models.

    Parameters
    ----------
    trained_models : dict
        Dictionary of trained models
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    save_path : str
        Path to save the figure
    """
    plt.figure(figsize=(10, 8))

    colors = ['blue', 'red', 'green', 'orange']

    for (model_name, model), color in zip(trained_models.items(), colors):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=color, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

    # Diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"ROC curves saved to {save_path}")


def print_classification_report(y_true, y_pred, model_name='Model'):
    """Print detailed classification report."""
    print(f"\n{'='*60}")
    print(f"Classification Report - {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred,
                                target_names=['No Churn (0)', 'Churn (1)']))


def shap_analysis(best_model, X_test, feature_names, save_path='outputs/shap_summary.png'):
    """
    Perform SHAP analysis and save summary plot.

    Parameters
    ----------
    best_model : sklearn estimator
        Trained model (tree-based preferred)
    X_test : array-like or DataFrame
        Test features
    feature_names : list
        List of feature names
    save_path : str
        Path to save SHAP plot
    """
    print("\nRunning SHAP analysis...")

    try:
        # Use TreeExplainer for tree-based models
        if hasattr(best_model, 'booster'):  # XGBoost/LightGBM
            explainer = shap.TreeExplainer(best_model)
        else:  # Random Forest
            explainer = shap.TreeExplainer(best_model)

        shap_values = explainer.shap_values(X_test)

        # For binary classification, take the positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Create SHAP summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                          show=False, plot_type='bar')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"SHAP summary plot saved to {save_path}")

    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        print("Skipping SHAP analysis - model may not be compatible")


def plot_feature_importance(best_model, feature_names, save_path='outputs/feature_importance.png'):
    """
    Plot feature importance from tree-based model.

    Parameters
    ----------
    best_model : sklearn estimator
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    save_path : str
        Path to save the figure
    """
    if not hasattr(best_model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return

    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20 features

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.title('Top 20 Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Feature importance plot saved to {save_path}")

    # Also save as CSV
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    feat_imp_df.to_csv('outputs/feature_importance.csv', index=False)
    print("Feature importance saved to outputs/feature_importance.csv")

    return feat_imp_df


def full_evaluation(trained_models, X_test, y_test, best_model_name, feature_names):
    """
    Run full evaluation: confusion matrix, ROC curves, SHAP, feature importance.

    Parameters
    ----------
    trained_models : dict
        Dictionary of trained models
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    best_model_name : str
        Name of the best model
    feature_names : list
        List of feature names
    """
    best_model = trained_models[best_model_name]

    print("\n" + "="*60)
    print("STARTING FULL EVALUATION")
    print("="*60)

    # Confusion matrix
    y_pred = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, best_model_name)

    # Classification report
    print_classification_report(y_test, y_pred, best_model_name)

    # ROC curves
    plot_roc_curves(trained_models, X_test, y_test)

    # SHAP analysis
    shap_analysis(best_model, X_test, feature_names)

    # Feature importance
    plot_feature_importance(best_model, feature_names)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
