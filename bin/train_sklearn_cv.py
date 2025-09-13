#!/usr/bin/env python3
"""
Train sklearn models with cross-validation
"""

import argparse
import json
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    explained_variance_score, max_error
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, ARDRegression
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
)
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')


def load_data(genotype_file, phenotype_file):
    """Load genotype and phenotype data."""
    # Load genotype data (sparse matrix)
    if genotype_file.endswith('.npz'):
        data = np.load(genotype_file)
        if 'data' in data:
            X = data['data']
        else:
            X = sparse.load_npz(genotype_file)
    else:
        X = np.load(genotype_file)
    
    # Convert to dense if sparse
    if sparse.issparse(X):
        X = X.toarray()
    
    # Load phenotypes
    pheno_df = pd.read_csv(phenotype_file)
    y = pheno_df.iloc[:, -1].values  # Assume last column is the target
    
    return X, y


def load_splits(splits_file, cv_folds_file):
    """Load data splits and CV fold indices."""
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    with open(cv_folds_file, 'r') as f:
        cv_folds = json.load(f)
    
    return splits, cv_folds


def get_model(model_type, params):
    """Get sklearn model based on type and parameters."""
    
    models = {
        'linear': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'elasticnet': ElasticNet,
        'bayesian_ridge': BayesianRidge,
        'ard': ARDRegression,
        'rf': RandomForestRegressor,
        'gbm': GradientBoostingRegressor,
        'xgboost': xgb.XGBRegressor,
        'lightgbm': lgb.LGBMRegressor,
        'catboost': CatBoostRegressor,
        'extratrees': ExtraTreesRegressor,
        'adaboost': AdaBoostRegressor,
        'histgbm': HistGradientBoostingRegressor,
        'svm': SVR,
        'linear_svm': LinearSVR,
        'knn': KNeighborsRegressor,
        'mlp': MLPRegressor
    }
    
    model_class = models.get(model_type)
    if model_class is None:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Handle special cases
    if model_type == 'catboost':
        params['verbose'] = False
    elif model_type == 'xgboost':
        params['verbosity'] = 0
    elif model_type == 'lightgbm':
        params['verbose'] = -1
    
    return model_class(**params)


def custom_scorer(y_true, y_pred):
    """Calculate comprehensive metrics."""
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'explained_variance': explained_variance_score(y_true, y_pred),
        'max_error': max_error(y_true, y_pred),
        'pearson_r': np.corrcoef(y_true, y_pred)[0, 1],
        'spearman_r': pd.Series(y_true).corr(pd.Series(y_pred), method='spearman')
    }
    return metrics


def train_with_cv(X, y, model, cv_indices, scaler=None):
    """Train model with cross-validation."""
    cv_results = {
        'train_scores': [],
        'val_scores': [],
        'models': [],
        'scalers': [],
        'predictions': []
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_indices):
        print(f"  Training fold {fold_idx + 1}/{len(cv_indices)}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale features
        if scaler is None:
            scaler = StandardScaler()
        fold_scaler = StandardScaler()
        X_train_scaled = fold_scaler.fit_transform(X_train)
        X_val_scaled = fold_scaler.transform(X_val)
        
        # Clone model for this fold
        fold_model = get_model(model.__class__.__name__.lower(), model.get_params())
        
        # Train model
        fold_model.fit(X_train_scaled, y_train)
        
        # Predictions
        train_pred = fold_model.predict(X_train_scaled)
        val_pred = fold_model.predict(X_val_scaled)
        
        # Calculate metrics
        train_metrics = custom_scorer(y_train, train_pred)
        val_metrics = custom_scorer(y_val, val_pred)
        
        # Store results
        cv_results['train_scores'].append(train_metrics)
        cv_results['val_scores'].append(val_metrics)
        cv_results['models'].append(fold_model)
        cv_results['scalers'].append(fold_scaler)
        cv_results['predictions'].append({
            'val_indices': val_idx,
            'val_true': y_val,
            'val_pred': val_pred
        })
    
    return cv_results


def train_final_model(X_train, y_train, X_val, y_val, model_type, params):
    """Train final model on full training data."""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Get model
    model = get_model(model_type, params)
    
    # Train model
    print(f"Training final {model_type} model...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on validation set
    val_pred = model.predict(X_val_scaled)
    val_metrics = custom_scorer(y_val, val_pred)
    
    return model, scaler, val_metrics


def save_cv_predictions(predictions, output_file):
    """Save cross-validation predictions."""
    all_predictions = []
    
    for fold_idx, fold_pred in enumerate(predictions):
        for i, idx in enumerate(fold_pred['val_indices']):
            all_predictions.append({
                'sample_idx': idx,
                'fold': fold_idx,
                'true_value': fold_pred['val_true'][i],
                'predicted_value': fold_pred['val_pred'][i]
            })
    
    df = pd.DataFrame(all_predictions)
    df.to_csv(output_file, index=False)


def aggregate_cv_metrics(cv_scores):
    """Aggregate CV metrics across folds."""
    aggregated = {}
    
    for metric in cv_scores[0].keys():
        values = [fold[metric] for fold in cv_scores]
        aggregated[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }
    
    return aggregated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--genotype_file', required=True)
    parser.add_argument('--phenotype_file', required=True)
    parser.add_argument('--splits_file', required=True)
    parser.add_argument('--cv_folds', required=True)
    parser.add_argument('--best_params', required=True)
    parser.add_argument('--model_type', required=True)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_models', required=True)
    parser.add_argument('--output_results', required=True)
    parser.add_argument('--output_predictions', required=True)
    parser.add_argument('--output_log', required=True)
    args = parser.parse_args()
    
    print(f"\nTraining {args.model_type} model with cross-validation...")
    
    # Load data
    X, y = load_data(args.genotype_file, args.phenotype_file)
    splits, cv_folds = load_splits(args.splits_file, args.cv_folds)
    
    # Load best parameters
    with open(args.best_params, 'r') as f:
        best_params = json.load(f)
    
    # Update n_jobs if applicable
    if 'n_jobs' in best_params:
        best_params['n_jobs'] = args.n_jobs
    
    print(f"Data shape: {X.shape}")
    print(f"Best parameters: {best_params}")
    
    # Get train/test indices
    train_idx = splits['train_idx']
    test_idx = splits['test_idx']
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Create CV indices
    cv_indices = [(fold['train'], fold['val']) for fold in cv_folds['folds']]
    
    # Get model
    model = get_model(args.model_type, best_params)
    
    # Train with cross-validation
    print("\nPerforming cross-validation...")
    cv_results = train_with_cv(X_train, y_train, model, cv_indices)
    
    # Save CV models
    for fold_idx, (fold_model, fold_scaler) in enumerate(zip(cv_results['models'], cv_results['scalers'])):
        model_file = f"{args.output_models}_{fold_idx}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump({
                'model': fold_model,
                'scaler': fold_scaler,
                'fold': fold_idx,
                'params': best_params
            }, f)
        print(f"Saved fold {fold_idx} model to {model_file}")
    
    # Train final model on all training data
    if 'val_idx' in splits:
        val_idx = splits['val_idx']
        X_val, y_val = X[val_idx], y[val_idx]
    else:
        # Use last 10% of training as validation
        n_val = int(0.1 * len(X_train))
        X_val, y_val = X_train[-n_val:], y_train[-n_val:]
        X_train, y_train = X_train[:-n_val], y_train[:-n_val]
    
    final_model, final_scaler, val_metrics = train_final_model(
        X_train, y_train, X_val, y_val, args.model_type, best_params
    )
    
    # Save final model
    final_model_file = f"{args.output_models}_final.pkl"
    with open(final_model_file, 'wb') as f:
        pickle.dump({
            'model': final_model,
            'scaler': final_scaler,
            'params': best_params,
            'val_metrics': val_metrics
        }, f)
    print(f"Saved final model to {final_model_file}")
    
    # Aggregate CV results
    train_metrics_agg = aggregate_cv_metrics(cv_results['train_scores'])
    val_metrics_agg = aggregate_cv_metrics(cv_results['val_scores'])
    
    # Save results
    results = {
        'model_type': args.model_type,
        'parameters': best_params,
        'cv_train_metrics': train_metrics_agg,
        'cv_val_metrics': val_metrics_agg,
        'final_val_metrics': val_metrics,
        'n_features': X.shape[1],
        'n_samples': {
            'total': X.shape[0],
            'train': len(train_idx),
            'val': len(X_val),
            'test': len(test_idx)
        },
        'n_folds': len(cv_indices)
    }
    
    with open(args.output_results, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    save_cv_predictions(cv_results['predictions'], args.output_predictions)
    
    # Create training log
    log_data = []
    for fold_idx, (train_scores, val_scores) in enumerate(zip(
        cv_results['train_scores'], cv_results['val_scores']
    )):
        log_entry = {
            'fold': fold_idx,
            'train_mse': train_scores['mse'],
            'train_r2': train_scores['r2'],
            'val_mse': val_scores['mse'],
            'val_r2': val_scores['r2'],
            'val_mae': val_scores['mae'],
            'val_pearson': val_scores['pearson_r']
        }
        log_data.append(log_entry)
    
    # Add final model metrics
    log_data.append({
        'fold': 'final',
        'train_mse': np.nan,
        'train_r2': np.nan,
        'val_mse': val_metrics['mse'],
        'val_r2': val_metrics['r2'],
        'val_mae': val_metrics['mae'],
        'val_pearson': val_metrics['pearson_r']
    })
    
    pd.DataFrame(log_data).to_csv(args.output_log, index=False)
    
    print("\n" + "="*50)
    print(f"Cross-validation Results for {args.model_type}:")
    print(f"  Mean Val R²: {val_metrics_agg['r2']['mean']:.4f} ± {val_metrics_agg['r2']['std']:.4f}")
    print(f"  Mean Val MSE: {val_metrics_agg['mse']['mean']:.4f} ± {val_metrics_agg['mse']['std']:.4f}")
    print(f"  Final Val R²: {val_metrics['r2']:.4f}")
    print(f"  Final Val MSE: {val_metrics['mse']:.4f}")
    print("="*50)


if __name__ == '__main__':
    main()