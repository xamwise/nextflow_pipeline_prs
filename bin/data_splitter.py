"""
Split genotype data into train/validation/test sets with k-fold cross-validation.
"""

import numpy as np
import pandas as pd
import h5py
import json
import argparse
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_data(
    genotype_file: str,
    phenotype_file: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    n_folds: int = 5,
    stratified: bool = False,
    seed: int = 42
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Split data into train/validation/test sets with k-fold cross-validation.
    
    Args:
        genotype_file: Path to HDF5 genotype file
        phenotype_file: Path to phenotype CSV file
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation
        n_folds: Number of cross-validation folds
        stratified: Whether to use stratified splitting for binary outcomes
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (split information dict, train indices, val indices, test indices, fold indices)
    """
    np.random.seed(seed)
    logger.info(f"Splitting data with seed {seed}")
    
    # Get number of samples from genotype file
    with h5py.File(genotype_file, 'r') as f:
        n_samples_genotype = f['genotypes'].shape[0]
    
    # Load phenotypes
    phenotypes = pd.read_csv(phenotype_file)
    n_samples_phenotype = len(phenotypes)
    
    # Check for mismatch
    if n_samples_genotype != n_samples_phenotype:
        logger.warning(f"Sample size mismatch: {n_samples_genotype} genotypes, {n_samples_phenotype} phenotypes")
        # Use the minimum to ensure we have both genotype and phenotype for each sample
        n_samples = min(n_samples_genotype, n_samples_phenotype)
        logger.info(f"Using {n_samples} samples (minimum of both)")
    else:
        n_samples = n_samples_genotype
    
    y = phenotypes.iloc[:n_samples, 0].values
    
    # Check if stratification is appropriate
    unique_values = np.unique(y[~np.isnan(y)])
    is_binary = len(unique_values) == 2
    
    if stratified and not is_binary:
        logger.warning("Stratified splitting requested but phenotype is not binary. Using regular splitting.")
        stratified = False
    
    # Create indices - only use samples that have both genotype and phenotype
    indices = np.arange(n_samples)
    
    # Remove samples with missing phenotypes
    valid_indices = indices[~np.isnan(y)]
    if len(valid_indices) < len(indices):
        logger.info(f"Removed {len(indices) - len(valid_indices)} samples with missing phenotypes")
        indices = valid_indices
        y_valid = y[valid_indices]
    else:
        y_valid = y
    
    # Split into train+val and test
    if stratified and is_binary:
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=y_valid
        )
    else:
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed
        )
    
    logger.info(f"Test set: {len(test_indices)} samples ({test_size*100:.1f}%)")
    
    # Split train+val into train and val
    if stratified and is_binary:
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_size,
            random_state=seed,
            stratify=y_valid[np.isin(indices, train_val_indices)]
        )
    else:
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_size,
            random_state=seed
        )
    
    logger.info(f"Training set: {len(train_indices)} samples")
    logger.info(f"Validation set: {len(val_indices)} samples ({val_size*100:.1f}% of train+val)")
    
    # Create k-fold splits
    if stratified and is_binary:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = kf.split(train_val_indices, y_valid[np.isin(indices, train_val_indices)])
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = kf.split(train_val_indices)
    
    fold_indices = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        fold_indices.append({
            'fold': fold,
            'train': train_val_indices[train_idx],
            'val': train_val_indices[val_idx]
        })
        logger.info(f"Fold {fold}: {len(train_idx)} train, {len(val_idx)} val")
    
    # Calculate phenotype statistics for each split
    def get_phenotype_stats(indices):
        # Ensure indices are within bounds
        valid_mask = indices < len(y)
        valid_indices = indices[valid_mask]
        if len(valid_indices) < len(indices):
            logger.warning(f"Skipped {len(indices) - len(valid_indices)} indices out of bounds for phenotypes")
        
        subset_y = y[valid_indices]
        valid_y = subset_y[~np.isnan(subset_y)]
        if is_binary:
            return {
                'n_samples': len(valid_indices),
                'n_positive': int(np.sum(valid_y == 1)),
                'n_negative': int(np.sum(valid_y == 0)),
                'positive_rate': float(np.mean(valid_y)) if len(valid_y) > 0 else 0.0
            }
        else:
            return {
                'n_samples': len(valid_indices),
                'mean': float(np.mean(valid_y)) if len(valid_y) > 0 else 0.0,
                'std': float(np.std(valid_y)) if len(valid_y) > 0 else 0.0,
                'min': float(np.min(valid_y)) if len(valid_y) > 0 else 0.0,
                'max': float(np.max(valid_y)) if len(valid_y) > 0 else 0.0
            }
    
    # Create split information
    splits_info = {
        'n_samples': len(indices),
        'n_samples_genotype': n_samples_genotype,
        'n_samples_phenotype': n_samples_phenotype,
        'n_train': len(train_indices),
        'n_val': len(val_indices),
        'n_test': len(test_indices),
        'n_folds': n_folds,
        'test_size': test_size,
        'val_size': val_size,
        'seed': seed,
        'stratified': stratified,
        'is_binary': is_binary,
        'train_stats': get_phenotype_stats(train_indices),
        'val_stats': get_phenotype_stats(val_indices),
        'test_stats': get_phenotype_stats(test_indices)
    }
    
    return splits_info, train_indices, val_indices, test_indices, fold_indices


def verify_splits(
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    fold_indices: list
) -> bool:
    """
    Verify that data splits are valid (no overlap between sets).
    
    Args:
        train_indices: Training set indices
        val_indices: Validation set indices
        test_indices: Test set indices
        fold_indices: K-fold indices
        
    Returns:
        True if splits are valid
    """
    # Check train/val/test overlap
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    
    if train_set & val_set:
        logger.error("Overlap found between train and validation sets")
        return False
    
    if train_set & test_set:
        logger.error("Overlap found between train and test sets")
        return False
    
    if val_set & test_set:
        logger.error("Overlap found between validation and test sets")
        return False
    
    # Check fold overlap
    for i, fold in enumerate(fold_indices):
        fold_train = set(fold['train'])
        fold_val = set(fold['val'])
        
        if fold_train & fold_val:
            logger.error(f"Overlap found in fold {i} between train and val")
            return False
        
        if fold_train & test_set:
            logger.error(f"Overlap found between fold {i} train and test set")
            return False
        
        if fold_val & test_set:
            logger.error(f"Overlap found between fold {i} val and test set")
            return False
    
    logger.info("All splits verified: no overlaps found")
    return True


def main():
    parser = argparse.ArgumentParser(description='Split data for PRS model training')
    parser.add_argument('--genotype_file', type=str, required=True,
                        help='Path to HDF5 genotype file')
    parser.add_argument('--phenotype_file', type=str, required=True,
                        help='Path to phenotype CSV file')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data for test set')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Proportion of training data for validation')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--stratified', action='store_true',
                        help='Use stratified splitting for binary outcomes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_splits', type=str, required=True,
                        help='Output JSON file for split information')
    parser.add_argument('--output_indices', type=str, required=True,
                        help='Output NPZ file for indices')
    
    args = parser.parse_args()
    
    # Split data
    splits_info, train_idx, val_idx, test_idx, fold_idx = split_data(
        genotype_file=args.genotype_file,
        phenotype_file=args.phenotype_file,
        test_size=args.test_size,
        val_size=args.val_size,
        n_folds=args.n_folds,
        stratified=args.stratified,
        seed=args.seed
    )
    
    # Verify splits
    if verify_splits(train_idx, val_idx, test_idx, fold_idx):
        logger.info("Data splits are valid")
    else:
        logger.error("Data splits validation failed")
        raise ValueError("Invalid data splits detected")
    
    # Save split information
    with open(args.output_splits, 'w') as f:
        json.dump(splits_info, f, indent=2)
    logger.info(f"Split information saved to {args.output_splits}")
    
    # Save indices
    # Save fold indices separately - they may have different sizes
    save_dict = {
        'train_indices': train_idx,
        'val_indices': val_idx,
        'test_indices': test_idx,
        'n_folds': args.n_folds
    }
    
    # Add each fold's indices with a unique key
    for i, fold in enumerate(fold_idx):
        save_dict[f'fold_{i}_train'] = fold['train']
        save_dict[f'fold_{i}_val'] = fold['val']
    
    np.savez(args.output_indices, **save_dict)
    logger.info(f"Indices saved to {args.output_indices}")
    
    # Print summary
    print("\n" + "="*50)
    print("Data Splitting Summary")
    print("="*50)
    print(f"Total samples: {splits_info['n_samples']}")
    print(f"Training set: {splits_info['n_train']} samples")
    print(f"Validation set: {splits_info['n_val']} samples")
    print(f"Test set: {splits_info['n_test']} samples")
    print(f"Number of folds: {splits_info['n_folds']}")
    print(f"Stratified: {splits_info['stratified']}")
    print("="*50)


if __name__ == '__main__':
    main()