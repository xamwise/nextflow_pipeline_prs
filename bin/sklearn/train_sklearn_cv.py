#!/usr/bin/env python3
"""
Train sklearn models with cross-validation.
Supports both classification and regression based on phenotype.
Uses HDF5 format from DL pipeline.
"""

import argparse
import json
import pickle
import numpy as np
import pandas as pd
import h5py
from scipy import sparse
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    # Regression metrics
    mean_squared_error, r2_score, mean_absolute_error,
    explained_variance_score, max_error,
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression, RidgeClassifier
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
)
from sklearn.svm import SVR, SVC, LinearSVR, LinearSVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


def load_data(genotype_file, phenotype_file):
    """Load genotype and phenotype data from HDF5 format."""
    # Load genotype data from HDF5 (DL pipeline format)
    if genotype_file.endswith('.h5') or genotype_file.endswith('.hdf5'):
        with h5py.File(genotype_file, 'r') as f:
            X = f['genotypes'][:]
    elif genotype_file.endswith('.npz'):
        data = np.load(genotype_file)
        if 'data' in data:
            X = data['data']
        else:
            X = sparse.load_npz(genotype_file)
            if sparse.issparse(X):
                X = X.toarray()
    else:
        X = np.load(genotype_file)
    
    # Load phenotypes
    pheno_df = pd.read_csv(phenotype_file)
    y = pheno_df.iloc[:, 0].values  # First column is the phenotype
    
    # Detect task type
    unique_vals = np.unique(y[~np.isnan(y)])
    is_classification = len(unique_vals) <= 2  # Binary classification
    
    return X, y, is_classification


def load_splits(splits_file):
    """Load data splits from DL pipeline format."""
    if splits_file.endswith('.npz'):
        # NPZ format from DL pipeline
        indices_data = np.load(splits_file)
        
        splits = {
            'train_idx': indices_data['train_indices'].tolist(),
            'val_idx': indices_data.get('val_indices', np.array([])).tolist(),
            'test_idx': indices_data['test_indices'].tolist()
        }
        
        # Extract CV folds
        cv_folds = {'folds': []}
        if 'n_folds' in indices_data:
            n_folds = int(indices_data['n_folds'])
            for i in range(n_folds):
                if f'fold_{i}_train' in indices_data and f'fold_{i}_val' in indices_data:
                    cv_folds['folds'].append({
                        'train': indices_data[f'fold_{i}_train'].tolist(),
                        'val': indices_data[f'fold_{i}_val'].tolist()
                    })
    else:
        # JSON format
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        cv_folds = splits.get('cv_folds', {'folds': []})
    
    return splits, cv_folds

def load_cv_folds(cv_folds_file):
    """Load CV folds from NPZ or JSON file."""
    if not cv_folds_file:
        return None
        
    if cv_folds_file.endswith('.npz'):
        # Load from NPZ file (same as splits file)
        indices_data = np.load(cv_folds_file)
        
        cv_folds = {'folds': []}
        
        # Get the training indices to create a mapping
        train_indices = indices_data['train_indices']
        
        # Create mapping from absolute to relative indices
        abs_to_rel = {abs_idx: rel_idx for rel_idx, abs_idx in enumerate(train_indices)}
        
        # Determine number of folds
        n_folds = 0
        for key in indices_data.files:
            if key.startswith('fold_') and '_train' in key:
                n_folds += 1
        
        # Extract fold indices and convert to relative
        for i in range(n_folds):
            if f'fold_{i}_train' in indices_data and f'fold_{i}_val' in indices_data:
                # Get absolute indices
                abs_train_idx = indices_data[f'fold_{i}_train']
                abs_val_idx = indices_data[f'fold_{i}_val']
                
                # Convert to relative indices within training set
                rel_train_idx = [abs_to_rel[idx] for idx in abs_train_idx if idx in abs_to_rel]
                rel_val_idx = [abs_to_rel[idx] for idx in abs_val_idx if idx in abs_to_rel]
                
                cv_folds['folds'].append({
                    'train': rel_train_idx,
                    'val': rel_val_idx
                })
        
        return cv_folds
        
    elif cv_folds_file.endswith('.json'):
        # Load from JSON file (legacy support)
        with open(cv_folds_file, 'r') as f:
            return json.load(f)
    
    return None


def get_model(model_type, params, is_classification=False):
    """Get sklearn model based on type and parameters."""
    
    # Model mapping for both classification and regression
    if is_classification:
        models = {
            'linear': LogisticRegression,
            'logistic': LogisticRegression,
            'ridge': RidgeClassifier,
            'lasso': lambda p: LogisticRegression(**{**p, 'penalty': 'l1', 'solver': 'liblinear'}),
            'elasticnet': lambda p: LogisticRegression(**{**p, 'penalty': 'elasticnet', 'solver': 'saga'}),
            'rf': RandomForestClassifier,
            'gbm': GradientBoostingClassifier,
            'xgboost': lambda p: xgb.XGBClassifier(**{**p, 'objective': 'binary:logistic'}),
            'lightgbm': lambda p: lgb.LGBMClassifier(**{**p, 'objective': 'binary'}),
            'catboost': CatBoostClassifier,
            'extratrees': ExtraTreesClassifier,
            'adaboost': AdaBoostClassifier,
            'histgbm': HistGradientBoostingClassifier,
            'svm': SVC,
            'linear_svm': LinearSVC,
            'knn': KNeighborsClassifier,
            'mlp': MLPClassifier
        }
    else:
        models = {
            'linear': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'elasticnet': ElasticNet,
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
    
    # Handle callable model classes
    if callable(model_class) and not isinstance(model_class, type):
        return model_class(params)
    else:
        return model_class(**params)


def custom_scorer(y_true, y_pred, y_pred_proba=None, is_classification=False):
    """Calculate comprehensive metrics for both classification and regression."""
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
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
                metrics['pr_auc'] = float(average_precision_score(y_true, y_pred_proba[:, 1]))
            
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
        
        # Correlation metrics
        if len(y_true) > 1:
            metrics['pearson_r'] = float(np.corrcoef(y_true, y_pred)[0, 1])
            metrics['spearman_r'] = float(pd.Series(y_true).corr(pd.Series(y_pred), method='spearman'))
    
    return metrics


def train_with_cv(X, y, model, cv_indices, is_classification=False):
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
        fold_scaler = StandardScaler()
        X_train_scaled = fold_scaler.fit_transform(X_train)
        X_val_scaled = fold_scaler.transform(X_val)
        
        # Determine model type from class name
        class_name = model.__class__.__name__
        if 'XGB' in class_name:
            model_type = 'xgboost'
        elif 'LGBM' in class_name:
            model_type = 'lightgbm'
        elif 'CatBoost' in class_name:
            model_type = 'catboost'
        elif 'RandomForest' in class_name:
            model_type = 'rf'
        elif 'GradientBoosting' in class_name:
            model_type = 'gbm'
        elif 'SV' in class_name:
            model_type = 'svm'
        elif 'Ridge' in class_name:
            model_type = 'ridge'
        elif 'Lasso' in class_name:
            model_type = 'lasso'
        elif 'ElasticNet' in class_name:
            model_type = 'elasticnet'
        elif 'Linear' in class_name or 'Logistic' in class_name:
            model_type = 'linear'
        else:
            # Fallback to original method
            model_type = class_name.lower().replace('classifier', '').replace('regressor', '')
        
        
        # Clone model for this fold
        fold_model = get_model(
            model_type,
            model.get_params(),
            is_classification
        )
        
        # Train model
        fold_model.fit(X_train_scaled, y_train)
        
        # Predictions
        if is_classification:
            train_pred = fold_model.predict(X_train_scaled)
            val_pred = fold_model.predict(X_val_scaled)
            
            # Get probabilities if available
            train_proba = None
            val_proba = None
            if hasattr(fold_model, 'predict_proba'):
                try:
                    train_proba = fold_model.predict_proba(X_train_scaled)
                    val_proba = fold_model.predict_proba(X_val_scaled)
                except:
                    pass
            
            # Calculate metrics
            train_metrics = custom_scorer(y_train, train_pred, train_proba, is_classification)
            val_metrics = custom_scorer(y_val, val_pred, val_proba, is_classification)
        else:
            train_pred = fold_model.predict(X_train_scaled)
            val_pred = fold_model.predict(X_val_scaled)
            
            # Calculate metrics
            train_metrics = custom_scorer(y_train, train_pred, None, is_classification)
            val_metrics = custom_scorer(y_val, val_pred, None, is_classification)
        
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


def aggregate_cv_metrics(cv_scores):
    """Aggregate CV metrics across folds."""
    aggregated = {}
    
    for metric in cv_scores[0].keys():
        values = [fold[metric] for fold in cv_scores]
        aggregated[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
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
    X, y, is_classification = load_data(args.genotype_file, args.phenotype_file)
    
    task_type = "Classification" if is_classification else "Regression"
    print(f"Task type detected: {task_type}")
    
    # Load splits
    splits, cv_folds_from_splits = load_splits(args.splits_file)
    
    # Load CV folds (prefer separate file if provided)
    cv_folds_loaded = load_cv_folds(args.cv_folds)
    if cv_folds_loaded:
        cv_folds = cv_folds_loaded
    else:
        cv_folds = cv_folds_from_splits
    
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
    model = get_model(args.model_type, best_params, is_classification)
    
    # Train with cross-validation
    print("\nPerforming cross-validation...")
    cv_results = train_with_cv(X_train, y_train, model, cv_indices, is_classification)
    
    # Save CV models
    for fold_idx, (fold_model, fold_scaler) in enumerate(zip(cv_results['models'], cv_results['scalers'])):
        model_file = f"{args.output_models}_{fold_idx}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump({
                'model': fold_model,
                'scaler': fold_scaler,
                'fold': fold_idx,
                'params': best_params,
                'is_classification': is_classification
            }, f)
        print(f"Saved fold {fold_idx} model to {model_file}")
    
    # Aggregate CV results
    train_metrics_agg = aggregate_cv_metrics(cv_results['train_scores'])
    val_metrics_agg = aggregate_cv_metrics(cv_results['val_scores'])
    
    # Save results
    results = {
        'model_type': args.model_type,
        'task_type': task_type,
        'is_classification': is_classification,
        'parameters': best_params,
        'cv_train_metrics': train_metrics_agg,
        'cv_val_metrics': val_metrics_agg,
        'n_features': X.shape[1],
        'n_samples': {
            'total': X.shape[0],
            'train': len(train_idx),
            'test': len(test_idx)
        },
        'n_folds': len(cv_indices)
    }
    
    with open(args.output_results, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    all_predictions = []
    for fold_idx, fold_pred in enumerate(cv_results['predictions']):
        for i, idx in enumerate(fold_pred['val_indices']):
            all_predictions.append({
                'sample_idx': idx,
                'fold': fold_idx,
                'true_value': fold_pred['val_true'][i],
                'predicted_value': fold_pred['val_pred'][i]
            })
    
    pd.DataFrame(all_predictions).to_csv(args.output_predictions, index=False)
    
    # Create training log
    log_data = []
    for fold_idx, (train_scores, val_scores) in enumerate(zip(
        cv_results['train_scores'], cv_results['val_scores']
    )):
        if is_classification:
            log_entry = {
                'fold': fold_idx,
                'train_accuracy': train_scores.get('accuracy', 0),
                'train_f1': train_scores.get('f1', 0),
                'train_roc_auc': train_scores.get('roc_auc', 0),  # Added
                'val_accuracy': val_scores.get('accuracy', 0),
                'val_f1': val_scores.get('f1', 0),
                'val_roc_auc': val_scores.get('roc_auc', 0),  # Added
                'val_precision': val_scores.get('precision', 0),
                'val_recall': val_scores.get('recall', 0)
            }
        else:
            log_entry = {
                'fold': fold_idx,
                'train_mse': train_scores['mse'],
                'train_r2': train_scores['r2'],
                'val_mse': val_scores['mse'],
                'val_r2': val_scores['r2'],
                'val_mae': val_scores['mae'],
                'val_pearson': val_scores.get('pearson_r', 0)
            }
        log_data.append(log_entry)
    
    pd.DataFrame(log_data).to_csv(args.output_log, index=False)
    
    print("\n" + "="*50)
    print(f"Cross-validation Results for {args.model_type} ({task_type}):")

    if is_classification:
        print(f"  Mean Val Accuracy: {val_metrics_agg.get('accuracy', {}).get('mean', 0):.4f} ± "
            f"{val_metrics_agg.get('accuracy', {}).get('std', 0):.4f}")
        print(f"  Mean Val F1: {val_metrics_agg.get('f1', {}).get('mean', 0):.4f} ± "
            f"{val_metrics_agg.get('f1', {}).get('std', 0):.4f}")
        print(f"  Mean Val ROC-AUC: {val_metrics_agg.get('roc_auc', {}).get('mean', 0):.4f} ± "
            f"{val_metrics_agg.get('roc_auc', {}).get('std', 0):.4f}")  # Added
        print(f"  Mean Val PR-AUC: {val_metrics_agg.get('pr_auc', {}).get('mean', 0):.4f} ± "
            f"{val_metrics_agg.get('pr_auc', {}).get('std', 0):.4f}")  # Added
    else:
        print(f"  Mean Val R²: {val_metrics_agg['r2']['mean']:.4f} ± {val_metrics_agg['r2']['std']:.4f}")
        print(f"  Mean Val MSE: {val_metrics_agg['mse']['mean']:.4f} ± {val_metrics_agg['mse']['std']:.4f}")

    print("="*50)

if __name__ == '__main__':
    main()