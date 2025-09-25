#!/usr/bin/env python3
"""
sklearn_hyperopt.py - Hyperparameter optimization for sklearn models using Optuna
Supports both classification and regression tasks
"""

import argparse
import json
import pickle
import numpy as np
import pandas as pd
import h5py
from scipy import sparse
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression, RidgeClassifier
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.svm import SVR, SVC
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


def load_data(genotype_file, phenotype_file):
    """Load genotype and phenotype data from HDF5 format."""
    # Load genotype data from HDF5 (matching DL pipeline format)
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
    y = pheno_df.iloc[:, 0].values  # First column is the phenotype (matching train_sklearn_cv.py)
    
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
    else:
        # JSON format
        with open(splits_file, 'r') as f:
            splits = json.load(f)
    
    return splits

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


def get_search_space(model_type, trial, is_classification=False):
    """Define hyperparameter search space for each model type."""
    
    if model_type == 'linear':
        if is_classification:
            return {
                'penalty': trial.suggest_categorical('penalty', ['l2', 'none']),
                'C': trial.suggest_float('C', 1e-4, 100, log=True),
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                'max_iter': trial.suggest_int('max_iter', 100, 1000)
            }
        else:
            return {'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])}
    
    elif model_type == 'ridge':
        if is_classification:
            return {
                'alpha': trial.suggest_float('alpha', 1e-4, 100, log=True),
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
            }
        else:
            return {
                'alpha': trial.suggest_float('alpha', 1e-4, 100, log=True),
                'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr']),
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
            }
    
    elif model_type == 'lasso':
        if is_classification:
            # Lasso classification via LogisticRegression with L1
            return {
                'C': trial.suggest_float('C', 1e-4, 10, log=True),
                'max_iter': trial.suggest_int('max_iter', 100, 5000),
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
            }
        else:
            return {
                'alpha': trial.suggest_float('alpha', 1e-4, 10, log=True),
                'max_iter': trial.suggest_int('max_iter', 100, 5000),
                'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
            }
    
    elif model_type == 'elasticnet':
        if is_classification:
            # ElasticNet classification via LogisticRegression
            return {
                'C': trial.suggest_float('C', 1e-4, 10, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9),
                'max_iter': trial.suggest_int('max_iter', 100, 5000),
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
            }
        else:
            return {
                'alpha': trial.suggest_float('alpha', 1e-4, 10, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9),
                'max_iter': trial.suggest_int('max_iter', 100, 5000),
                'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
            }
    
    elif model_type == 'rf':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
        if is_classification:
            params['criterion'] = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        else:
            params['criterion'] = trial.suggest_categorical('criterion', ['squared_error', 'absolute_error'])
        return params
    
    elif model_type == 'gbm':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8])
        }
        if is_classification:
            params['loss'] = trial.suggest_categorical('loss', ['log_loss', 'exponential'])
        else:
            params['loss'] = trial.suggest_categorical('loss', ['squared_error', 'absolute_error', 'huber'])
        return params
    
    elif model_type == 'xgboost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear'])
        }
        if is_classification:
            params['objective'] = 'binary:logistic'
        else:
            params['objective'] = 'reg:squarederror'
        return params
    
    elif model_type == 'lightgbm':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])
        }
        if is_classification:
            params['objective'] = 'binary'
            params['metric'] = 'binary_logloss'
        else:
            params['objective'] = 'regression'
            params['metric'] = 'rmse'
        return params
    
    elif model_type == 'catboost':
        params = {
            'iterations': trial.suggest_int('iterations', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 10, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 1),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'grow_policy': trial.suggest_categorical('grow_policy', 
                ['SymmetricTree', 'Depthwise', 'Lossguide'])
        }
        if is_classification:
            params['loss_function'] = 'Logloss'
        else:
            params['loss_function'] = 'RMSE'
        return params
    
    elif model_type == 'svm':
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        params = {
            'kernel': kernel,
            'C': trial.suggest_float('C', 1e-3, 100, log=True)
        }
        
        if not is_classification:
            params['epsilon'] = trial.suggest_float('epsilon', 1e-4, 1, log=True)
        
        params['shrinking'] = trial.suggest_categorical('shrinking', [True, False])
        
        if kernel == 'rbf':
            params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
        elif kernel == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 5)
            params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
        
        if is_classification:
            params['probability'] = True  # Enable probability predictions for AUC
        
        return params
    
    else:
        # Default
        if is_classification:
            return {'C': trial.suggest_float('C', 1e-4, 100, log=True)}
        else:
            return {'fit_intercept': True}


def get_model(model_type, params, is_classification=False):
    """Get model instance based on type and parameters."""
    
    if is_classification:
        models = {
            'linear': lambda p: LogisticRegression(**p),
            'ridge': lambda p: RidgeClassifier(**p),
            'lasso': lambda p: LogisticRegression(**{**p, 'penalty': 'l1', 'solver': 'liblinear'}),
            'elasticnet': lambda p: LogisticRegression(**{**p, 'penalty': 'elasticnet', 'solver': 'saga'}),
            'rf': lambda p: RandomForestClassifier(**p, n_jobs=-1, random_state=42),
            'gbm': lambda p: GradientBoostingClassifier(**p, random_state=42),
            'xgboost': lambda p: xgb.XGBClassifier(**p, n_jobs=-1, random_state=42, verbosity=0),
            'lightgbm': lambda p: lgb.LGBMClassifier(**p, n_jobs=-1, random_state=42, verbose=-1),
            'catboost': lambda p: CatBoostClassifier(**p, random_seed=42, verbose=False),
            'svm': lambda p: SVC(**p)
        }
    else:
        models = {
            'linear': lambda p: LinearRegression(**p),
            'ridge': lambda p: Ridge(**p),
            'lasso': lambda p: Lasso(**p),
            'elasticnet': lambda p: ElasticNet(**p),
            'rf': lambda p: RandomForestRegressor(**p, n_jobs=-1, random_state=42),
            'gbm': lambda p: GradientBoostingRegressor(**p, random_state=42),
            'xgboost': lambda p: xgb.XGBRegressor(**p, n_jobs=-1, random_state=42, verbosity=0),
            'lightgbm': lambda p: lgb.LGBMRegressor(**p, n_jobs=-1, random_state=42, verbose=-1),
            'catboost': lambda p: CatBoostRegressor(**p, random_seed=42, verbose=False),
            'svm': lambda p: SVR(**p)
        }
    
    if model_type not in models:
        if is_classification:
            return LogisticRegression()
        else:
            return LinearRegression()
    
    return models[model_type](params)


def objective(trial, X, y, model_type, cv_folds, n_jobs, is_classification):
    """Objective function for Optuna optimization."""
    # Get hyperparameters
    params = get_search_space(model_type, trial, is_classification)
    
    # Create model
    model = get_model(model_type, params, is_classification)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform cross-validation
    try:
        if is_classification:
            # For classification, use negative accuracy or negative ROC-AUC
            scores = cross_val_score(
                model, X_scaled, y,
                cv=cv_folds,
                scoring='roc_auc' if hasattr(model, 'predict_proba') else 'accuracy',
                n_jobs=1
            )
            # We want to maximize, so return negative
            return -scores.mean()
        else:
            # For regression, use negative MSE
            scores = cross_val_score(
                model, X_scaled, y,
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=1
            )
            # Return negative MSE (we want to minimize MSE)
            return -scores.mean()
    
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('inf')


# def optimize_hyperparameters(X, y, model_type, cv_folds, n_trials, n_jobs, seed, is_classification):
#     """Run hyperparameter optimization."""
#     # Create study
#     sampler = TPESampler(seed=seed)
#     pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
#     study = optuna.create_study(
#         direction='minimize',  # We're minimizing negative score
#         sampler=sampler,
#         pruner=pruner,
#         study_name=f'{model_type}_optimization'
#     )
    
#     # Optimize
#     study.optimize(
#         lambda trial: objective(trial, X, y, model_type, cv_folds, n_jobs, is_classification),
#         n_trials=n_trials,
#         n_jobs=n_jobs,
#         show_progress_bar=True
#     )
    
#     return study

def optimize_hyperparameters(X, y, model_type, cv_folds, n_trials, n_jobs, seed, is_classification):
    """Run hyperparameter optimization."""
    # Create study
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=10)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        study_name=f'{model_type}_optimization'
    )
    
    # Verbose callback
    def verbose_callback(study, trial):
        print(f"\n{'='*60}")
        print(f"Trial {trial.number} completed")
        print(f"Value: {trial.value:.6f}")
        print(f"Params: {trial.params}")
        print(f"Best value so far: {study.best_value:.6f}")
        print('='*60)
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, X, y, model_type, cv_folds, n_jobs, is_classification),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=False,  # Disable progress bar
        callbacks=[verbose_callback]
    )
    
    return study


def create_optimization_report(study, model_type, output_file, is_classification):
    """Create HTML report of optimization results."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    metric_name = 'ROC-AUC' if is_classification else 'MSE'
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Optimization History',
            'Parallel Coordinate Plot',
            'Parameter Importance',
            'Slice Plot'
        )
    )
    
    # Optimization history
    trials = study.trials
    values = [t.value for t in trials if t.value is not None]
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(values))),
            y=values,
            mode='lines+markers',
            name=metric_name
        ),
        row=1, col=1
    )
    
    # Parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        fig.add_trace(
            go.Bar(
                x=list(importance.keys()),
                y=list(importance.values()),
                name='Importance'
            ),
            row=2, col=1
        )
    except:
        pass
    
    # Update layout
    fig.update_layout(
        title=f'Hyperparameter Optimization Report - {model_type}',
        height=800,
        showlegend=True
    )
    
    task_type = "Classification" if is_classification else "Regression"
    
    # Save HTML
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Optimization Report - {model_type}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>Hyperparameter Optimization Report</h1>
        <h2>Model: {model_type} ({task_type})</h2>
        
        <h3>Best Parameters:</h3>
        <pre>{json.dumps(study.best_params, indent=2)}</pre>
        
        <h3>Best Score ({metric_name}): {abs(study.best_value):.6f}</h3>
        
        <h3>Optimization Visualizations:</h3>
        <div id="plot"></div>
        
        <script>
            var data = {fig.to_json()};
            Plotly.newPlot('plot', data.data, data.layout);
        </script>
        
        <h3>Trial Statistics:</h3>
        <ul>
            <li>Total trials: {len(study.trials)}</li>
            <li>Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}</li>
            <li>Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}</li>
            <li>Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}</li>
        </ul>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_template)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--genotype_file', required=True)
    parser.add_argument('--phenotype_file', required=True)
    parser.add_argument('--splits_file', required=True)
    parser.add_argument('--cv_folds', required=True)
    parser.add_argument('--model_type', required=True)
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_params', required=True)
    parser.add_argument('--output_study', required=True)
    parser.add_argument('--output_history', required=True)
    parser.add_argument('--output_report', required=True)
    args = parser.parse_args()
    
    print(f"\nOptimizing hyperparameters for {args.model_type}...")
    
    # Load data
    X, y, is_classification = load_data(args.genotype_file, args.phenotype_file)
    
    task_type = "Classification" if is_classification else "Regression"
    print(f"Task type detected: {task_type}")
    
    # Load splits
    splits = load_splits(args.splits_file)
    
    # Load CV folds
    cv_folds_data = load_cv_folds(args.cv_folds)

        
    
    # Get training data
    train_idx = splits['train_idx']
    X_train, y_train = X[train_idx], y[train_idx]
    
    print(f"Training data shape: {X_train.shape}")
    
    # Create CV folds
    if is_classification:
        cv_folds = StratifiedKFold(n_splits=len(cv_folds_data['folds']), shuffle=False)
    else:
        cv_folds = KFold(n_splits=len(cv_folds_data['folds']), shuffle=False)
    
    # Run optimization
    study = optimize_hyperparameters(
        X_train, y_train,
        args.model_type,
        cv_folds,
        args.n_trials,
        args.n_jobs,
        args.seed,
        is_classification
    )
    
    # Save best parameters
    best_params = study.best_params
    with open(args.output_params, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Save study
    with open(args.output_study, 'wb') as f:
        pickle.dump(study, f)
    
    # Save optimization history
    history_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            history_data.append({
                'trial': trial.number,
                'value': trial.value,
                'params': trial.params,
                'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                'duration': trial.duration.total_seconds() if trial.duration else None
            })
    
    pd.DataFrame(history_data).to_csv(args.output_history, index=False)
    
    # Create report
    create_optimization_report(study, args.model_type, args.output_report, is_classification)
    
    print("\n" + "="*50)
    print(f"Optimization complete for {args.model_type} ({task_type})")
    print(f"Best parameters: {best_params}")
    
    if is_classification:
        print(f"Best Score: {abs(study.best_value):.6f}")
    else:
        print(f"Best MSE: {abs(study.best_value):.6f}")
    
    print(f"Number of trials: {len(study.trials)}")
    print("="*50)


if __name__ == '__main__':
    main()