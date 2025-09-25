#!/usr/bin/env python3
"""
Evaluate sklearn models on test set with interpretability analysis.
Uses the same test split as the deep learning pipeline.
"""

import numpy as np
import pandas as pd
import pickle
import json
import h5py
import argparse
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    # Regression metrics
    mean_squared_error, r2_score, mean_absolute_error,
    explained_variance_score, max_error,
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import shap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_task_type(y_test: np.ndarray, model_data: dict = None) -> bool:
    """
    Detect if it's a classification or regression task.
    
    Args:
        y_test: Test labels
        model_data: Optional model data dictionary
        
    Returns:
        True if classification, False if regression
    """
    # First check if model data explicitly says it's classification
    if model_data and 'is_classification' in model_data:
        return model_data['is_classification']
    
    # Otherwise detect from the target values
    unique_vals = np.unique(y_test[~np.isnan(y_test)])
    is_classification = len(unique_vals) <= 2  # Binary classification
    
    return is_classification


def load_test_data(genotype_file: str, phenotype_file: str, splits_file: str):
    """
    Load test data using the same splits as DL pipeline.
    
    Args:
        genotype_file: HDF5 or NPZ file with genotypes
        phenotype_file: CSV file with phenotypes
        splits_file: NPZ file with train/val/test splits
        
    Returns:
        X_test, y_test, test_indices
    """
    # Load genotypes
    if genotype_file.endswith('.h5') or genotype_file.endswith('.hdf5'):
        with h5py.File(genotype_file, 'r') as f:
            X = f['genotypes'][:]
    elif genotype_file.endswith('.npz'):
        data = np.load(genotype_file)
        if 'data' in data:
            X = data['data']
        else:
            from scipy import sparse
            X = sparse.load_npz(genotype_file).toarray()
    else:
        X = np.load(genotype_file)
    
    # Load phenotypes
    pheno_df = pd.read_csv(phenotype_file)
    y = pheno_df.iloc[:, 0].values  # First column is the phenotype
    
    # Load test indices from DL pipeline splits
    splits_data = np.load(splits_file)
    test_indices = splits_data['test_indices']
    
    # Get test data
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    logger.info(f"Loaded test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    return X_test, y_test, test_indices


def load_models(model_files: list):
    """
    Load trained sklearn models.
    
    Args:
        model_files: List of pickle files containing models
        
    Returns:
        List of loaded models with metadata
    """
    models = []
    
    for model_file in model_files:
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
            models.append(model_data)
            logger.info(f"Loaded model from {model_file}")
    
    return models


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_pred_proba: np.ndarray = None,
                     is_classification: bool = False) -> dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_pred_proba: Predicted probabilities (for classification)
        is_classification: Whether it's a classification task
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    if is_classification:
        # Classification metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        
        # Handle binary classification
        if len(np.unique(y_true)) == 2:
            metrics['precision'] = float(precision_score(y_true, y_pred))
            metrics['recall'] = float(recall_score(y_true, y_pred))
            metrics['f1'] = float(f1_score(y_true, y_pred))
            
            # If probability predictions available
            if y_pred_proba is not None:
                if len(y_pred_proba.shape) > 1:
                    # Multi-class probabilities, get positive class
                    pos_proba = y_pred_proba[:, 1]
                else:
                    pos_proba = y_pred_proba
                    
                metrics['roc_auc'] = float(roc_auc_score(y_true, pos_proba))
                metrics['pr_auc'] = float(average_precision_score(y_true, pos_proba))
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    else:
        # Regression metrics
        metrics['mse'] = float(mean_squared_error(y_true, y_pred))
        metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        metrics['r2'] = float(r2_score(y_true, y_pred))
        metrics['explained_variance'] = float(explained_variance_score(y_true, y_pred))
        metrics['max_error'] = float(max_error(y_true, y_pred))
        metrics['mean_error'] = float(np.mean(y_pred - y_true))
        metrics['std_error'] = float(np.std(y_pred - y_true))
        
        # Correlation metrics
        pearson_r, pearson_p = stats.pearsonr(y_true.flatten(), y_pred.flatten())
        spearman_r, spearman_p = stats.spearmanr(y_true.flatten(), y_pred.flatten())
        
        metrics['pearson_r'] = float(pearson_r)
        metrics['pearson_p'] = float(pearson_p)
        metrics['spearman_r'] = float(spearman_r)
        metrics['spearman_p'] = float(spearman_p)
        
        # Percentile errors
        errors = np.abs(y_true - y_pred)
        metrics['mae_p50'] = float(np.percentile(errors, 50))
        metrics['mae_p90'] = float(np.percentile(errors, 90))
        metrics['mae_p95'] = float(np.percentile(errors, 95))
    
    return metrics


def evaluate_single_model(model_data: dict, X_test: np.ndarray, y_test: np.ndarray) -> tuple:
    """
    Evaluate a single model.
    
    Args:
        model_data: Dictionary containing model and scaler
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Predictions, metrics, task type, and probabilities
    """
    model = model_data['model']
    scaler = model_data.get('scaler', None)
    is_classification = detect_task_type(y_test, model_data)
    
    # Apply scaling if scaler is provided
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Get probabilities for classification
    y_pred_proba = None
    if is_classification and hasattr(model, 'predict_proba'):
        try:
            y_pred_proba = model.predict_proba(X_test_scaled)
        except:
            pass
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, is_classification)
    
    return y_pred, metrics, is_classification, y_pred_proba


def evaluate_ensemble(models: list, X_test: np.ndarray, y_test: np.ndarray) -> tuple:
    """
    Evaluate ensemble of models (e.g., from CV folds).
    
    Args:
        models: List of model dictionaries
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Ensemble predictions, metrics, all predictions, task type, and probabilities
    """
    all_predictions = []
    all_probabilities = []
    is_classification = detect_task_type(y_test, models[0] if models else None)
    
    for model_data in models:
        y_pred, _, _, y_pred_proba = evaluate_single_model(model_data, X_test, y_test)
        all_predictions.append(y_pred)
        if y_pred_proba is not None:
            all_probabilities.append(y_pred_proba)
    
    # Handle ensemble predictions differently for classification vs regression
    if is_classification:
        if all_probabilities:
            # If we have probabilities, average them and threshold
            y_pred_proba_ensemble = np.mean(all_probabilities, axis=0)
            # For binary classification, threshold at 0.5
            if len(y_pred_proba_ensemble.shape) > 1 and y_pred_proba_ensemble.shape[1] == 2:
                y_pred_ensemble = (y_pred_proba_ensemble[:, 1] > 0.5).astype(int)
            else:
                # Multi-class: take argmax
                y_pred_ensemble = np.argmax(y_pred_proba_ensemble, axis=1)
        else:
            # No probabilities available, use majority voting
            from scipy import stats
            # Convert to numpy array for easier manipulation
            all_preds_array = np.array(all_predictions)
            # Use mode (majority vote) for each sample
            y_pred_ensemble = stats.mode(all_preds_array, axis=0, keepdims=False)[0].astype(int)
            y_pred_proba_ensemble = None
    else:
        # For regression, average the predictions
        y_pred_ensemble = np.mean(all_predictions, axis=0)
        y_pred_proba_ensemble = None
    
    # Calculate ensemble metrics
    metrics_ensemble = calculate_metrics(y_test, y_pred_ensemble, 
                                        y_pred_proba_ensemble, is_classification)
    
    # Add ensemble-specific metrics
    if is_classification:
        # For classification, calculate agreement between models
        all_preds_array = np.array(all_predictions)
        agreement = np.mean([np.all(all_preds_array[:, i] == all_preds_array[0, i]) 
                            for i in range(all_preds_array.shape[1])])
        metrics_ensemble['model_agreement'] = float(agreement)
    else:
        # For regression, calculate standard deviation of predictions
        metrics_ensemble['prediction_std'] = float(np.std(all_predictions, axis=0).mean())
    
    metrics_ensemble['n_models'] = len(models)
    
    return y_pred_ensemble, metrics_ensemble, all_predictions, is_classification, y_pred_proba_ensemble


def calculate_shap_values(model, X_test: np.ndarray, X_train: np.ndarray = None,
                          n_samples: int = 100) -> np.ndarray:
    """
    Calculate SHAP values for model interpretability.
    
    Args:
        model: Trained model
        X_test: Test features
        X_train: Training features for background (optional)
        n_samples: Number of samples for SHAP analysis
        
    Returns:
        SHAP values array
    """
    logger.info(f"Calculating SHAP values for {n_samples} samples...")
    
    # Limit samples for computational efficiency
    if X_test.shape[0] > n_samples:
        sample_idx = np.random.choice(X_test.shape[0], n_samples, replace=False)
        X_sample = X_test[sample_idx]
    else:
        X_sample = X_test
    
    try:
        # Try different SHAP explainers based on model type
        model_type = model.__class__.__name__
        
        if 'Tree' in model_type or 'Forest' in model_type or 'XGB' in model_type:
            # Tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        elif 'Linear' in model_type or 'Lasso' in model_type or 'Ridge' in model_type:
            # Linear models
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
        else:
            # Generic explainer (slower)
            if X_train is not None and X_train.shape[0] > 100:
                background = shap.kmeans(X_train, 10)
            else:
                background = X_sample
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_sample)
        
        logger.info(f"SHAP values calculated: shape {shap_values.shape}")
        return shap_values, X_sample
        
    except Exception as e:
        logger.warning(f"Could not calculate SHAP values: {e}")
        return None, None


def create_classification_plots(y_true: np.ndarray, y_pred: np.ndarray,
                               output_file: str, model_type: str, 
                               y_pred_proba: np.ndarray = None):
    """Create classification evaluation plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Confusion Matrix
    ax = axes[0, 0]
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    # 2. ROC Curve
    ax = axes[0, 1]
    if y_pred_proba is not None:
        if len(y_pred_proba.shape) > 1:
            pos_proba = y_pred_proba[:, 1]
        else:
            pos_proba = y_pred_proba
        fpr, tpr, _ = roc_curve(y_true, pos_proba)
        roc_auc = roc_auc_score(y_true, pos_proba)
        ax.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No probabilities available', ha='center', va='center')
        ax.set_title('ROC Curve (Not Available)')
    
    # 3. Precision-Recall Curve
    ax = axes[0, 2]
    if y_pred_proba is not None:
        precision, recall, _ = precision_recall_curve(y_true, pos_proba)
        pr_auc = average_precision_score(y_true, pos_proba)
        ax.plot(recall, precision, label=f'PR (AUC = {pr_auc:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No probabilities available', ha='center', va='center')
        ax.set_title('PR Curve (Not Available)')
    
    # 4. Class Distribution
    ax = axes[1, 0]
    unique, counts = np.unique(y_true, return_counts=True)
    ax.bar(['Class 0', 'Class 1'], counts)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('True Class Distribution')
    
    # 5. Prediction Distribution
    ax = axes[1, 1]
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    ax.bar(['Class 0', 'Class 1'][:len(unique_pred)], counts_pred)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Predicted Class Distribution')
    
    # 6. Performance metrics text
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    metrics_text = f"""
    Performance Metrics:
    
    Accuracy: {acc:.4f}
    Precision: {prec:.4f}
    Recall: {rec:.4f}
    F1 Score: {f1:.4f}
    """
    
    if y_pred_proba is not None:
        if len(y_pred_proba.shape) > 1:
            pos_proba = y_pred_proba[:, 1]
        else:
            pos_proba = y_pred_proba
        roc_auc = roc_auc_score(y_true, pos_proba)
        pr_auc = average_precision_score(y_true, pos_proba)
        metrics_text += f"""ROC-AUC: {roc_auc:.4f}
    PR-AUC: {pr_auc:.4f}"""
    
    ax.text(0.1, 0.5, metrics_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='center')
    
    plt.suptitle(f'{model_type} Classification Evaluation Results', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Classification evaluation plots saved to {output_file}")


def create_regression_plots(y_true: np.ndarray, y_pred: np.ndarray,
                           output_file: str, model_type: str):
    """Create regression evaluation plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Scatter plot
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'{model_type}: Predictions vs True')
    
    # Add R² to plot
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
            verticalalignment='top')
    
    # 2. Residual plot
    ax = axes[0, 1]
    residuals = y_pred - y_true
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot')
    
    # 3. Histogram of residuals
    ax = axes[0, 2]
    ax.hist(residuals, bins=30, edgecolor='black')
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    ax.set_title('Residual Distribution')
    
    # 4. Q-Q plot
    ax = axes[1, 0]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot')
    
    # 5. Error distribution
    ax = axes[1, 1]
    errors = np.abs(residuals)
    ax.hist(errors, bins=30, edgecolor='black', cumulative=True, density=True)
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Error Distribution')
    
    # 6. Performance metrics text
    ax = axes[1, 2]
    ax.axis('off')
    metrics_text = f"""
    Performance Metrics:
    
    MSE: {mean_squared_error(y_true, y_pred):.4f}
    RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}
    MAE: {mean_absolute_error(y_true, y_pred):.4f}
    R²: {r2_score(y_true, y_pred):.4f}
    Pearson r: {stats.pearsonr(y_true.flatten(), y_pred.flatten())[0]:.4f}
    """
    ax.text(0.1, 0.5, metrics_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='center')
    
    plt.suptitle(f'{model_type} Regression Evaluation Results', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Regression evaluation plots saved to {output_file}")


def create_evaluation_plots(y_true: np.ndarray, y_pred: np.ndarray,
                           output_file: str, model_type: str = "Model",
                           is_classification: bool = False, 
                           y_pred_proba: np.ndarray = None):
    """
    Create evaluation plots for both classification and regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        output_file: Output PDF file
        model_type: Model type for title
        is_classification: Whether it's a classification task
        y_pred_proba: Predicted probabilities (for classification)
    """
    if is_classification:
        create_classification_plots(y_true, y_pred, output_file, model_type, y_pred_proba)
    else:
        create_regression_plots(y_true, y_pred, output_file, model_type)


def create_shap_analysis(shap_values: np.ndarray, X_test: np.ndarray,
                        feature_names: list, output_file: str):
    """
    Create SHAP analysis visualizations.
    
    Args:
        shap_values: SHAP values array
        X_test: Test features
        feature_names: Feature names
        output_file: Output HTML file
    """
    if shap_values is None:
        logger.warning("No SHAP values to visualize")
        return
    
    # Create HTML report
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SHAP Analysis Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            .plot-container { margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>SHAP Feature Importance Analysis</h1>
    """
    
    # Calculate mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    # Get top features
    n_features = min(20, len(mean_shap))
    top_indices = np.argsort(mean_shap)[-n_features:][::-1]
    
    # Feature importance plot data
    if feature_names is not None and len(feature_names) > 0:
        feature_labels = [str(feature_names[i]) for i in top_indices]
    else:
        feature_labels = [f'Feature_{i}' for i in top_indices]
    
    importance_values = mean_shap[top_indices]
    
    # Add feature importance bar plot
    html_content += f"""
        <h2>Top {n_features} Most Important Features</h2>
        <div id="importance-plot" class="plot-container"></div>
        
        <script>
            var data = [{{
                x: {list(importance_values)},
                y: {list(feature_labels)},
                type: 'bar',
                orientation: 'h'
            }}];
            
            var layout = {{
                title: 'Mean |SHAP| Values',
                xaxis: {{title: 'Mean |SHAP| Value'}},
                yaxis: {{title: 'Feature'}},
                margin: {{l: 150}}
            }};
            
            Plotly.newPlot('importance-plot', data, layout);
        </script>
    """
    
    # Add summary statistics
    html_content += f"""
        <h2>SHAP Analysis Summary</h2>
        <ul>
            <li>Total features analyzed: {shap_values.shape[1]}</li>
            <li>Samples analyzed: {shap_values.shape[0]}</li>
            <li>Most important feature: {feature_labels[0]}</li>
            <li>Max mean |SHAP| value: {importance_values[0]:.6f}</li>
            <li>Features with non-zero impact: {(mean_shap > 0).sum()}</li>
        </ul>
    """
    
    # Add SHAP value distribution for top features
    html_content += """
        <h2>SHAP Value Distribution for Top Features</h2>
        <p>Distribution of SHAP values shows how each feature impacts predictions across samples.</p>
        <div id="distribution-plot" class="plot-container"></div>
        
        <script>
            var traces = [];
    """
    
    for i, idx in enumerate(top_indices[:10]):
        label = feature_labels[i]
        values = shap_values[:, idx].tolist()
        html_content += f"""
            traces.push({{
                y: {values},
                name: '{label}',
                type: 'box'
            }});
        """
    
    html_content += """
            var layout = {
                title: 'SHAP Value Distributions',
                yaxis: {title: 'SHAP Value'},
                showlegend: true
            };
            
            Plotly.newPlot('distribution-plot', traces, layout);
        </script>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    logger.info(f"SHAP analysis saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate sklearn models')
    parser.add_argument('--models', nargs='+', required=True,
                       help='Model pickle files to evaluate')
    parser.add_argument('--genotype_file', type=str, required=True,
                       help='HDF5 or NPZ file with genotypes')
    parser.add_argument('--phenotype_file', type=str, required=True,
                       help='CSV file with phenotypes')
    parser.add_argument('--splits_file', type=str, required=True,
                       help='NPZ file with train/val/test splits')
    parser.add_argument('--model_type', type=str, required=True,
                       help='Model type for labeling')
    parser.add_argument('--output_metrics', type=str, required=True,
                       help='Output JSON file for metrics')
    parser.add_argument('--output_predictions', type=str, required=True,
                       help='Output CSV file for predictions')
    parser.add_argument('--output_plots', type=str, required=True,
                       help='Output PDF file for plots')
    parser.add_argument('--output_shap', type=str, required=True,
                       help='Output HTML file for SHAP analysis')
    parser.add_argument('--calculate_shap', type=str, default='true',
                       help='Whether to calculate SHAP values')
    parser.add_argument('--feature_names', type=str, default=None,
                       help='Text file with feature names')
    
    args = parser.parse_args()
    
    # Load test data
    X_test, y_test, test_indices = load_test_data(
        args.genotype_file, args.phenotype_file, args.splits_file
    )
    
    # Load feature names if provided
    feature_names = None
    if args.feature_names and Path(args.feature_names).exists():
        with open(args.feature_names, 'r') as f:
            feature_names = [line.strip() for line in f]
    
    # Load models
    models = load_models(args.models)
    logger.info(f"Evaluating {len(models)} model(s)")
    
    # Detect task type
    is_classification = detect_task_type(y_test, models[0] if models else None)
    task_type = "Classification" if is_classification else "Regression"
    logger.info(f"Task type detected: {task_type}")
    
    # Evaluate
    if len(models) == 1:
        # Single model evaluation
        y_pred, metrics, is_classification, y_pred_proba = evaluate_single_model(
            models[0], X_test, y_test
        )
        
        # SHAP analysis
        if args.calculate_shap.lower() == 'true':
            shap_values, X_shap = calculate_shap_values(
                models[0]['model'], X_test, n_samples=100
            )
        else:
            shap_values, X_shap = None, None
            
    else:
        # Ensemble evaluation (e.g., CV models)
        y_pred, metrics, all_predictions, is_classification, y_pred_proba = evaluate_ensemble(
            models, X_test, y_test
        )
        
        # SHAP for first model only (representative)
        if args.calculate_shap.lower() == 'true':
            shap_values, X_shap = calculate_shap_values(
                models[0]['model'], X_test, n_samples=100
            )
        else:
            shap_values, X_shap = None, None
    
    # Save metrics
    metrics['model_type'] = args.model_type
    metrics['task_type'] = task_type
    metrics['is_classification'] = is_classification
    metrics['n_test_samples'] = len(y_test)
    
    with open(args.output_metrics, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {args.output_metrics}")
    
    # Save predictions
    pred_df = pd.DataFrame({
        'sample_idx': test_indices,
        'true_value': y_test,
        'predicted_value': y_pred,
        'error': y_pred - y_test,
        'abs_error': np.abs(y_pred - y_test)
    })
    pred_df.to_csv(args.output_predictions, index=False)
    logger.info(f"Predictions saved to {args.output_predictions}")
    
    # Create plots
    create_evaluation_plots(y_test, y_pred, args.output_plots, 
                          args.model_type, is_classification, y_pred_proba)
    
    # Create SHAP analysis
    if shap_values is not None:
        create_shap_analysis(shap_values, X_shap, feature_names, args.output_shap)
    
    # Print summary
    print("\n" + "="*50)
    print(f"Evaluation Results for {args.model_type} ({task_type})")
    print("="*50)
    print(f"Test samples: {len(y_test)}")
    
    if is_classification:
        print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"F1 Score: {metrics.get('f1', 0):.4f}")
        print(f"Precision: {metrics.get('precision', 0):.4f}")
        print(f"Recall: {metrics.get('recall', 0):.4f}")
        print(f"ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
        print(f"PR-AUC: {metrics.get('pr_auc', 0):.4f}")
    else:
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"Pearson r: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.2e})")
    
    print("="*50)


if __name__ == '__main__':
    main()