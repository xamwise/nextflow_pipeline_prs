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
    mean_squared_error, r2_score, mean_absolute_error,
    explained_variance_score, max_error
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import shap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    y = pheno_df.iloc[:, -1].values
    
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


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'mse': float(mean_squared_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'explained_variance': float(explained_variance_score(y_true, y_pred)),
        'max_error': float(max_error(y_true, y_pred)),
        'mean_error': float(np.mean(y_pred - y_true)),
        'std_error': float(np.std(y_pred - y_true))
    }
    
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
        Predictions and metrics
    """
    model = model_data['model']
    scaler = model_data.get('scaler', None)
    
    # Apply scaling if scaler is provided
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    return y_pred, metrics


def evaluate_ensemble(models: list, X_test: np.ndarray, y_test: np.ndarray) -> tuple:
    """
    Evaluate ensemble of models (e.g., from CV folds).
    
    Args:
        models: List of model dictionaries
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Ensemble predictions and metrics
    """
    all_predictions = []
    
    for model_data in models:
        y_pred, _ = evaluate_single_model(model_data, X_test, y_test)
        all_predictions.append(y_pred)
    
    # Average predictions
    y_pred_ensemble = np.mean(all_predictions, axis=0)
    
    # Calculate ensemble metrics
    metrics_ensemble = calculate_metrics(y_test, y_pred_ensemble)
    
    # Add ensemble-specific metrics
    metrics_ensemble['prediction_std'] = float(np.std(all_predictions, axis=0).mean())
    metrics_ensemble['n_models'] = len(models)
    
    return y_pred_ensemble, metrics_ensemble, all_predictions


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


def create_evaluation_plots(y_true: np.ndarray, y_pred: np.ndarray,
                           output_file: str, model_type: str = "Model"):
    """
    Create evaluation plots.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        output_file: Output PDF file
        model_type: Model type for title
    """
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
    
    plt.suptitle(f'{model_type} Evaluation Results', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Evaluation plots saved to {output_file}")


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
    
    # Evaluate
    if len(models) == 1:
        # Single model evaluation
        y_pred, metrics = evaluate_single_model(models[0], X_test, y_test)
        
        # SHAP analysis
        if args.calculate_shap.lower() == 'true':
            shap_values, X_shap = calculate_shap_values(
                models[0]['model'], X_test, n_samples=100
            )
        else:
            shap_values, X_shap = None, None
            
    else:
        # Ensemble evaluation (e.g., CV models)
        y_pred, metrics, all_predictions = evaluate_ensemble(models, X_test, y_test)
        
        # SHAP for first model only (representative)
        if args.calculate_shap.lower() == 'true':
            shap_values, X_shap = calculate_shap_values(
                models[0]['model'], X_test, n_samples=100
            )
        else:
            shap_values, X_shap = None, None
    
    # Save metrics
    metrics['model_type'] = args.model_type
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
    create_evaluation_plots(y_test, y_pred, args.output_plots, args.model_type)
    
    # Create SHAP analysis
    if shap_values is not None:
        create_shap_analysis(shap_values, X_shap, feature_names, args.output_shap)
    
    # Print summary
    print("\n" + "="*50)
    print(f"Evaluation Results for {args.model_type}")
    print("="*50)
    print(f"Test samples: {len(y_test)}")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"Pearson r: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.2e})")
    print("="*50)


if __name__ == '__main__':
    main()