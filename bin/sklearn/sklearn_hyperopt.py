#!/usr/bin/env python3
"""
sklearn_hyperopt.py - Hyperparameter optimization for sklearn models using Optuna
Separate from DL hyperparameter_optimizer.py to avoid conflicts
"""

import argparse
import json
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')


def load_data(genotype_file, phenotype_file):
    """Load genotype and phenotype data."""
    if genotype_file.endswith('.npz'):
        data = np.load(genotype_file)
        if 'data' in data:
            X = data['data']
        else:
            X = sparse.load_npz(genotype_file)
    else:
        X = np.load(genotype_file)
    
    if sparse.issparse(X):
        X = X.toarray()
    
    pheno_df = pd.read_csv(phenotype_file)
    y = pheno_df.iloc[:, -1].values
    
    return X, y


def get_search_space(model_type, trial):
    """Define hyperparameter search space for each model type."""
    
    if model_type == 'ridge':
        return {
            'alpha': trial.suggest_float('alpha', 1e-4, 100, log=True),
            'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr']),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
        }
    
    elif model_type == 'lasso':
        return {
            'alpha': trial.suggest_float('alpha', 1e-4, 10, log=True),
            'max_iter': trial.suggest_int('max_iter', 100, 5000),
            'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
        }
    
    elif model_type == 'elasticnet':
        return {
            'alpha': trial.suggest_float('alpha', 1e-4, 10, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9),
            'max_iter': trial.suggest_int('max_iter', 100, 5000),
            'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
        }
    
    elif model_type == 'rf':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'criterion': trial.suggest_categorical('criterion', ['squared_error', 'absolute_error'])
        }
    
    elif model_type == 'gbm':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
            'loss': trial.suggest_categorical('loss', ['squared_error', 'absolute_error', 'huber'])
        }
    
    elif model_type == 'xgboost':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear']),
            'objective': 'reg:squarederror'
        }
    
    elif model_type == 'lightgbm':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
            'objective': 'regression',
            'metric': 'rmse'
        }
    
    elif model_type == 'catboost':
        return {
            'iterations': trial.suggest_int('iterations', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 10, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 1),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'grow_policy': trial.suggest_categorical('grow_policy', 
                ['SymmetricTree', 'Depthwise', 'Lossguide']),
            'loss_function': 'RMSE'
        }
    
    elif model_type == 'svm':
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        params = {
            'kernel': kernel,
            'C': trial.suggest_float('C', 1e-3, 100, log=True),
            'epsilon': trial.suggest_float('epsilon', 1e-4, 1, log=True),
            'shrinking': trial.suggest_categorical('shrinking', [True, False])
        }
        
        if kernel == 'rbf':
            params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
        elif kernel == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 5)
            params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
        
        return params
    
    else:
        # Default to linear regression (no hyperparameters)
        return {'fit_intercept': True}


def get_model(model_type, params):
    """Get model instance based on type and parameters."""
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
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    
    return models[model_type](params)


def objective(trial, X, y, model_type, cv_folds, n_jobs):
    """Objective function for Optuna optimization."""
    # Get hyperparameters
    params = get_search_space(model_type, trial)
    
    # Create model
    model = get_model(model_type, params)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform cross-validation
    try:
        scores = cross_val_score(
            model, X_scaled, y,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=1  # Use single job within trial
        )
        
        # Return negative MSE (we want to minimize MSE)
        return -scores.mean()
    
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('inf')


def optimize_hyperparameters(X, y, model_type, cv_folds, n_trials, n_jobs, seed):
    """Run hyperparameter optimization."""
    # Create study
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        study_name=f'{model_type}_optimization'
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, X, y, model_type, cv_folds, n_jobs),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True
    )
    
    return study


def create_optimization_report(study, model_type, output_file):
    """Create HTML report of optimization results."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
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
            name='MSE'
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
        <h2>Model: {model_type}</h2>
        
        <h3>Best Parameters:</h3>
        <pre>{json.dumps(study.best_params, indent=2)}</pre>
        
        <h3>Best Score (MSE): {study.best_value:.6f}</h3>
        
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
    X, y = load_data(args.genotype_file, args.phenotype_file)
    
    # Load splits
    with open(args.splits_file, 'r') as f:
        splits = json.load(f)
    
    with open(args.cv_folds, 'r') as f:
        cv_folds_data = json.load(f)
    
    # Get training data
    train_idx = splits['train_idx']
    X_train, y_train = X[train_idx], y[train_idx]
    
    print(f"Training data shape: {X_train.shape}")
    
    # Create CV folds
    cv_folds = KFold(n_splits=len(cv_folds_data['folds']), shuffle=False)
    
    # Run optimization
    study = optimize_hyperparameters(
        X_train, y_train,
        args.model_type,
        cv_folds,
        args.n_trials,
        args.n_jobs,
        args.seed
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
    create_optimization_report(study, args.model_type, args.output_report)
    
    print("\n" + "="*50)
    print(f"Optimization complete for {args.model_type}")
    print(f"Best parameters: {best_params}")
    print(f"Best MSE: {study.best_value:.6f}")
    print(f"Number of trials: {len(study.trials)}")
    print("="*50)


if __name__ == '__main__':
    main()