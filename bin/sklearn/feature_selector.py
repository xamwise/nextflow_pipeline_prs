#!/usr/bin/env python3
"""
Feature selection for high-dimensional genotype data.
Compatible with the existing deep learning pipeline data splits.
"""

import numpy as np
import pandas as pd
import h5py
import json
import argparse
from pathlib import Path
import logging
from scipy import sparse
from sklearn.feature_selection import (
    SelectKBest, f_regression, f_classif,
    mutual_info_regression, mutual_info_classif,
    SelectFromModel, VarianceThreshold
)
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(genotype_file: str, phenotype_file: str, indices_file: str = None):
    """
    Load genotype and phenotype data using the same format as DL pipeline.
    
    Args:
        genotype_file: HDF5 or NPZ file with genotypes
        phenotype_file: CSV file with phenotypes
        indices_file: Optional NPZ file with train/val/test splits
        
    Returns:
        X, y, indices_dict
    """
    # Load genotypes
    if genotype_file.endswith('.h5') or genotype_file.endswith('.hdf5'):
        with h5py.File(genotype_file, 'r') as f:
            X = f['genotypes'][:]
            if 'snp_ids' in f:
                feature_names = f['snp_ids'][:]
            else:
                feature_names = None
    elif genotype_file.endswith('.npz'):
        data = np.load(genotype_file)
        if 'data' in data:
            X = data['data']
        else:
            # Try loading as sparse
            X = sparse.load_npz(genotype_file).toarray()
        feature_names = data.get('feature_names', None)
    else:
        X = np.load(genotype_file)
        feature_names = None
    
    # Load phenotypes
    pheno_df = pd.read_csv(phenotype_file)
    y = pheno_df.iloc[:, -1].values  # Use last column as target
    
    # Load indices if provided (use same splits as DL pipeline)
    indices_dict = None
    if indices_file and Path(indices_file).exists():
        indices_data = np.load(indices_file)
        indices_dict = {
            'train': indices_data['train_indices'],
            'val': indices_data['val_indices'],
            'test': indices_data['test_indices']
        }
        logger.info(f"Using existing splits: {len(indices_dict['train'])} train, "
                   f"{len(indices_dict['val'])} val, {len(indices_dict['test'])} test")
    
    return X, y, feature_names, indices_dict


def is_classification(y: np.ndarray) -> bool:
    """Check if the task is classification based on target values."""
    unique_values = np.unique(y[~np.isnan(y)])
    return len(unique_values) <= 10  # Assume classification if <=10 unique values


def univariate_selection(X: np.ndarray, y: np.ndarray, n_features: int, 
                         is_classification: bool = False) -> np.ndarray:
    """
    Select features using univariate statistical tests.
    
    Args:
        X: Feature matrix
        y: Target values
        n_features: Number of features to select
        is_classification: Whether it's a classification task
        
    Returns:
        Selected feature indices
    """
    logger.info(f"Univariate feature selection: selecting {n_features} features")
    
    if is_classification:
        selector = SelectKBest(f_classif, k=min(n_features, X.shape[1]))
    else:
        selector = SelectKBest(f_regression, k=min(n_features, X.shape[1]))
    
    selector.fit(X, y)
    scores = selector.scores_
    
    # Get indices of top features
    indices = np.argsort(scores)[-n_features:][::-1]
    
    logger.info(f"Best score: {scores[indices[0]]:.4f}, "
               f"Worst selected: {scores[indices[-1]]:.4f}")
    
    return indices, scores


def lasso_selection(X: np.ndarray, y: np.ndarray, n_features: int,
                   alpha: float = None) -> np.ndarray:
    """
    Select features using LASSO regularization.
    """
    logger.info(f"LASSO feature selection: selecting {n_features} features")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if alpha is None:
        # Use cross-validation with a wider range of alphas
        alphas = np.logspace(-6, 1, 100)  # Wider range including smaller alphas
        lasso = LassoCV(alphas=alphas, cv=5, max_iter=5000, n_jobs=-1)
        lasso.fit(X_scaled, y)
        alpha = lasso.alpha_
        logger.info(f"Optimal alpha from CV: {alpha:.6f}")
    else:
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=alpha, max_iter=5000)
        lasso.fit(X_scaled, y)
    
    # Get feature importance (absolute coefficients)
    importance = np.abs(lasso.coef_)
    
    # Check if LASSO selected any features
    n_nonzero = (importance > 1e-10).sum()
    logger.info(f"Non-zero coefficients: {n_nonzero}")
    
    if n_nonzero == 0:
        logger.warning("LASSO selected no features, using variance as fallback")
        # Fallback: use variance to select features
        variances = np.var(X_scaled, axis=0)
        indices = np.argsort(variances)[-n_features:][::-1]
        importance = variances
    elif n_nonzero < n_features:
        logger.warning(f"LASSO selected only {n_nonzero} features, less than requested {n_features}")
        # Use all non-zero features plus highest variance for the rest
        nonzero_indices = np.where(importance > 1e-10)[0]
        variances = np.var(X_scaled, axis=0)
        variances[nonzero_indices] = -1  # Exclude already selected
        additional_indices = np.argsort(variances)[-(n_features - n_nonzero):]
        indices = np.concatenate([nonzero_indices, additional_indices])
    else:
        # Select top features as normal
        indices = np.argsort(importance)[-n_features:][::-1]
    
    logger.info(f"Max coefficient: {importance[indices[0]]:.6f}")
    
    return indices, importance


def elastic_net_selection(X: np.ndarray, y: np.ndarray, n_features: int,
                         l1_ratio: float = 0.5) -> np.ndarray:
    """
    Select features using Elastic Net regularization.
    
    Args:
        X: Feature matrix
        y: Target values
        n_features: Number of features to select
        l1_ratio: Balance between L1 and L2 regularization
        
    Returns:
        Selected feature indices
    """
    logger.info(f"Elastic Net feature selection: selecting {n_features} features")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use cross-validation
    elastic_net = ElasticNetCV(l1_ratio=l1_ratio, cv=5, max_iter=5000, n_jobs=-1)
    elastic_net.fit(X_scaled, y)
    
    # Get feature importance
    importance = np.abs(elastic_net.coef_)
    
    # Select top features
    indices = np.argsort(importance)[-n_features:][::-1]
    
    logger.info(f"Optimal alpha: {elastic_net.alpha_:.6f}")
    logger.info(f"Non-zero coefficients: {(importance > 0).sum()}")
    
    return indices, importance


def random_forest_selection(X: np.ndarray, y: np.ndarray, n_features: int,
                           is_classification: bool = False) -> np.ndarray:
    """
    Select features using Random Forest feature importance.
    
    Args:
        X: Feature matrix
        y: Target values
        n_features: Number of features to select
        is_classification: Whether it's a classification task
        
    Returns:
        Selected feature indices
    """
    logger.info(f"Random Forest feature selection: selecting {n_features} features")
    
    if is_classification:
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                  n_jobs=-1, random_state=42)
    else:
        rf = RandomForestRegressor(n_estimators=100, max_depth=10,
                                 n_jobs=-1, random_state=42)
    
    rf.fit(X, y)
    
    # Get feature importance
    importance = rf.feature_importances_
    
    # Select top features
    indices = np.argsort(importance)[-n_features:][::-1]
    
    logger.info(f"Max importance: {importance[indices[0]]:.4f}, "
               f"Min selected: {importance[indices[-1]]:.4f}")
    
    return indices, importance


def mutual_info_selection(X: np.ndarray, y: np.ndarray, n_features: int,
                         is_classification: bool = False) -> np.ndarray:
    """
    Select features using mutual information.
    
    Args:
        X: Feature matrix
        y: Target values
        n_features: Number of features to select
        is_classification: Whether it's a classification task
        
    Returns:
        Selected feature indices
    """
    logger.info(f"Mutual information feature selection: selecting {n_features} features")
    
    if is_classification:
        mi_scores = mutual_info_classif(X, y, random_state=42)
    else:
        mi_scores = mutual_info_regression(X, y, random_state=42)
    
    # Select top features
    indices = np.argsort(mi_scores)[-n_features:][::-1]
    
    logger.info(f"Max MI: {mi_scores[indices[0]]:.4f}, "
               f"Min selected: {mi_scores[indices[-1]]:.4f}")
    
    return indices, mi_scores


def variance_threshold_selection(X: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    Remove features with low variance.
    
    Args:
        X: Feature matrix
        threshold: Variance threshold
        
    Returns:
        Selected feature indices
    """
    logger.info(f"Variance threshold selection: threshold={threshold}")
    
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    
    indices = selector.get_support(indices=True)
    logger.info(f"Selected {len(indices)} out of {X.shape[1]} features")
    
    return indices, selector.variances_


def combine_selections(selections_dict: dict, n_features: int, 
                      method: str = 'union') -> np.ndarray:
    """
    Combine feature selections from different methods.
    
    Args:
        selections_dict: Dictionary of method -> indices
        n_features: Number of features to select
        method: Combination method ('union', 'intersection', 'weighted')
        
    Returns:
        Combined feature indices
    """
    if method == 'union':
        # Union of all selections
        all_indices = set()
        for indices in selections_dict.values():
            all_indices.update(indices[:n_features])
        selected = np.array(list(all_indices))
        
    elif method == 'intersection':
        # Intersection of all selections
        common = None
        for indices in selections_dict.values():
            indices_set = set(indices[:n_features])
            if common is None:
                common = indices_set
            else:
                common = common.intersection(indices_set)
        selected = np.array(list(common))
        
    elif method == 'weighted':
        # Weighted by frequency of selection
        feature_counts = {}
        for method_name, indices in selections_dict.items():
            for rank, idx in enumerate(indices[:n_features]):
                if idx not in feature_counts:
                    feature_counts[idx] = 0
                # Weight by inverse rank (top features get more weight)
                feature_counts[idx] += 1.0 / (rank + 1)
        
        # Sort by weighted count
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        selected = np.array([f[0] for f in sorted_features[:n_features]])
    
    else:
        raise ValueError(f"Unknown combination method: {method}")
    
    logger.info(f"Combined selection: {len(selected)} features using {method}")
    return selected


def save_selected_data(X: np.ndarray, selected_indices: np.ndarray, 
                       output_file: str, format: str = None):
    """Save selected features."""
    try:
        # Ensure indices are valid
        selected_indices = np.array(selected_indices, dtype=int)
        selected_indices = selected_indices[selected_indices < X.shape[1]]  # Remove invalid indices
        
        if len(selected_indices) == 0:
            raise ValueError("No valid features selected")
        
        X_selected = X[:, selected_indices]
        logger.info(f"Selected data shape: {X_selected.shape}")
        
        # Auto-detect format from filename if not specified
        if format is None:
            if output_file.endswith('.h5') or output_file.endswith('.hdf5'):
                format = 'h5'
            else:
                format = 'npz'
        
        if format == 'npz':
            np.savez_compressed(output_file, data=X_selected, indices=selected_indices)
        elif format == 'h5':
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('genotypes', data=X_selected, compression='gzip')
                f.create_dataset('selected_indices', data=selected_indices)
        
        logger.info(f"Saved selected features to {output_file} (format: {format})")
        
    except Exception as e:
        logger.error(f"Failed to save selected data: {e}")
        raise
    

def create_report(importance_dict: dict, selected_indices: np.ndarray,
                 feature_names: np.ndarray, output_file: str):
    """Create HTML report of feature selection results."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create subplots
    n_methods = len(importance_dict)
    fig = make_subplots(
        rows=(n_methods + 1) // 2, cols=2,
        subplot_titles=list(importance_dict.keys())
    )
    
    # Plot importance scores for each method
    for idx, (method, scores) in enumerate(importance_dict.items()):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        # Get top features for this method
        top_indices = np.argsort(scores)[-20:][::-1]
        
        fig.add_trace(
            go.Bar(
                x=list(range(len(top_indices))),
                y=scores[top_indices],
                name=method
            ),
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        title='Feature Selection Results',
        height=400 * ((n_methods + 1) // 2),
        showlegend=False
    )
    
    # Save HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Feature Selection Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>Feature Selection Report</h1>
        
        <h2>Summary</h2>
        <ul>
            <li>Total features analyzed: {len(scores)}</li>
            <li>Features selected: {len(selected_indices)}</li>
            <li>Methods used: {', '.join(importance_dict.keys())}</li>
        </ul>
        
        <h2>Feature Importance by Method</h2>
        <div id="importance-plot"></div>
        
        <script>
            var data = {fig.to_json()};
            Plotly.newPlot('importance-plot', data.data, data.layout);
        </script>
        
        <h2>Top Selected Features</h2>
        <table border="1">
            <tr><th>Rank</th><th>Feature Index</th><th>Feature Name</th></tr>
            {"".join([f"<tr><td>{i+1}</td><td>{idx}</td><td>{feature_names[idx] if feature_names is not None else 'N/A'}</td></tr>" 
                     for i, idx in enumerate(selected_indices[:20])])}
        </table>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Report saved to {output_file}")


def main():
        
    parser = argparse.ArgumentParser(description='Feature selection for genotype data')
    parser.add_argument('--genotype_file', type=str, required=True,
                    help='HDF5 or NPZ file with genotypes')
    parser.add_argument('--phenotype_file', type=str, required=True,
                    help='CSV file with phenotypes')
    parser.add_argument('--feature_names', type=str, default=None,
                    help='Text file with feature names')
    parser.add_argument('--indices_file', type=str, default=None,
                    help='NPZ file with train/val/test splits from DL pipeline')
    parser.add_argument('--n_features', type=int, default=1000,
                    help='Number of features to select')
    parser.add_argument('--methods', type=str, default='univariate,lasso,rf',
                    help='Comma-separated list of methods')
    parser.add_argument('--combine_method', type=str, default='weighted',
                    choices=['union', 'intersection', 'weighted'],
                    help='How to combine selections')
    parser.add_argument('--output_data', type=str, required=True,
                    help='Output file for selected features')
    parser.add_argument('--output_importance', type=str, required=True,
                    help='Output CSV for feature importance')
    parser.add_argument('--output_indices', type=str, required=True,
                    help='Output file for selected indices')
    parser.add_argument('--output_report', type=str, required=True,
                    help='Output HTML report')
    parser.add_argument('--n_jobs', type=int, default=-1,
                    help='Number of parallel jobs')
    
    args = parser.parse_args()
    
    # Load data
    X, y, feature_names_array, indices_dict = load_data(
        args.genotype_file, args.phenotype_file, args.indices_file
    )
    
    # Load feature names if provided separately
    if args.feature_names and Path(args.feature_names).exists():
        with open(args.feature_names, 'r') as f:
            feature_names_array = np.array([line.strip() for line in f])
    
    # Use training data only for feature selection
    if indices_dict is not None:
        X_train = X[indices_dict['train']]
        y_train = y[indices_dict['train']]
        logger.info(f"Using training set for feature selection: {X_train.shape}")
    else:
        X_train = X
        y_train = y
    
    # Check if classification or regression
    is_class = is_classification(y_train)
    logger.info(f"Task type: {'Classification' if is_class else 'Regression'}")
    
    # Apply methods
    methods = args.methods.split(',')
    selections = {}
    importance_scores = {}
    
    for method in methods:
        method = method.strip().lower()
        
        if method == 'univariate':
            indices, scores = univariate_selection(X_train, y_train, args.n_features, is_class)
            selections[method] = indices
            importance_scores[method] = scores
            
        elif method == 'lasso':
            indices, scores = lasso_selection(X_train, y_train, args.n_features)
            selections[method] = indices
            importance_scores[method] = scores
            
        elif method == 'elasticnet':
            indices, scores = elastic_net_selection(X_train, y_train, args.n_features)
            selections[method] = indices
            importance_scores[method] = scores
            
        elif method in ['rf', 'random_forest']:
            indices, scores = random_forest_selection(X_train, y_train, args.n_features, is_class)
            selections[method] = indices
            importance_scores[method] = scores
            
        elif method in ['mutual_info', 'mi']:
            indices, scores = mutual_info_selection(X_train, y_train, args.n_features, is_class)
            selections[method] = indices
            importance_scores[method] = scores
    
    # Combine selections
    if len(selections) > 1:
        selected_indices = combine_selections(selections, args.n_features, args.combine_method)
    else:
        selected_indices = list(selections.values())[0][:args.n_features]
        
    selected_indices = np.array(selected_indices, dtype=int)
    if len(selected_indices) == 0:
        logger.error("No features were selected!")
        # Fallback to selecting top variance features
        variances = np.var(X_train, axis=0)
        selected_indices = np.argsort(variances)[-args.n_features:][::-1]
        logger.warning(f"Using top {args.n_features} high-variance features as fallback")

        
    if not Path(args.output_data).exists():
        raise FileNotFoundError(f"Failed to create output file: {args.output_data}")
    
    logger.info(f"Final selection: {len(selected_indices)} features")
    
    # Save selected data
    logger.info(f"Saving {len(selected_indices)} features to {args.output_data}")
    
    output_format = 'h5' if args.output_data.endswith('.h5') else 'npz'
    save_selected_data(X, selected_indices, args.output_data, format=output_format)    
    
    # Save indices
    np.save(args.output_indices, selected_indices)
    
    # Save importance scores
    importance_df = pd.DataFrame()
    for method, scores in importance_scores.items():
        importance_df[method] = scores
    importance_df.to_csv(args.output_importance, index=False)
    
    # Create report
    create_report(importance_scores, selected_indices, feature_names_array, args.output_report)
    
    print("\n" + "="*50)
    print("Feature Selection Complete")
    print("="*50)
    print(f"Features selected: {len(selected_indices)}")
    print(f"Methods used: {', '.join(methods)}")
    print(f"Combination method: {args.combine_method}")
    print("="*50)
    



if __name__ == '__main__':
    main()