#!/usr/bin/env python3
"""
Data splitter for sklearn pipeline.
Compatible with existing DL pipeline splits - can either use existing splits or create new ones.
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
from scipy import sparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_genotype_data(genotype_file: str) -> Tuple[np.ndarray, int]:
    """
    Load genotype data from various formats.
    
    Args:
        genotype_file: Path to genotype file (HDF5, NPZ, or numpy)
        
    Returns:
        Tuple of (genotype_array, n_samples)
    """
    if genotype_file.endswith('.h5') or genotype_file.endswith('.hdf5'):
        with h5py.File(genotype_file, 'r') as f:
            if 'genotypes' in f:
                n_samples = f['genotypes'].shape[0]
            else:
                raise KeyError("No 'genotypes' dataset found in HDF5 file")
    elif genotype_file.endswith('.npz'):
        data = np.load(genotype_file)
        if 'data' in data:
            n_samples = data['data'].shape[0]
        else:
            # Try loading as sparse
            X = sparse.load_npz(genotype_file)
            n_samples = X.shape[0]
    else:
        X = np.load(genotype_file)
        n_samples = X.shape[0]
    
    return None, n_samples  # We don't load full data, just get shape


def load_existing_splits(splits_file: str) -> Dict[str, Any]:
    """
    Load existing splits from DL pipeline.
    
    Args:
        splits_file: Path to NPZ file with existing splits
        
    Returns:
        Dictionary with split indices
    """
    logger.info(f"Loading existing splits from {splits_file}")
    
    splits_data = np.load(splits_file)
    
    splits = {
        'train_idx': splits_data['train_indices'],
        'val_idx': splits_data.get('val_indices', np.array([])),
        'test_idx': splits_data['test_indices']
    }
    
    # Load fold indices if available
    fold_indices = []
    if 'n_folds' in splits_data:
        n_folds = int(splits_data['n_folds'])
        for i in range(n_folds):
            if f'fold_{i}_train' in splits_data and f'fold_{i}_val' in splits_data:
                fold_indices.append({
                    'train': splits_data[f'fold_{i}_train'].tolist(),
                    'val': splits_data[f'fold_{i}_val'].tolist()
                })
        splits['folds'] = fold_indices
    
    logger.info(f"Loaded splits: {len(splits['train_idx'])} train, "
               f"{len(splits['val_idx'])} val, {len(splits['test_idx'])} test")
    
    return splits


def create_new_splits(n_samples: int, phenotype_file: str,
                     test_size: float = 0.2, val_size: float = 0.1,
                     n_folds: int = 5, stratify: str = 'auto',
                     seed: int = 42) -> Dict[str, Any]:
    """
    Create new data splits.
    
    Args:
        n_samples: Number of samples
        phenotype_file: Path to phenotype CSV
        test_size: Proportion for test set
        val_size: Proportion for validation set
        n_folds: Number of CV folds
        stratify: Stratification strategy ('auto', 'yes', 'no')
        seed: Random seed
        
    Returns:
        Dictionary with split indices
    """
    np.random.seed(seed)
    logger.info(f"Creating new splits with seed {seed}")
    
    # Load phenotypes for stratification check
    phenotypes = pd.read_csv(phenotype_file)
    y = phenotypes.iloc[:, -1].values[:n_samples]
    
    # Check if stratification should be used
    unique_values = np.unique(y[~np.isnan(y)])
    is_binary = len(unique_values) == 2
    
    if stratify == 'auto':
        use_stratify = is_binary
    elif stratify == 'yes':
        use_stratify = is_binary
    else:
        use_stratify = False
    
    if use_stratify and not is_binary:
        logger.warning("Stratification requested but target is not binary")
        use_stratify = False
    
    # Create indices
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
    if use_stratify:
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=seed, stratify=y_valid
        )
    else:
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=seed
        )
    
    # Split train+val into train and val
    if use_stratify:
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_size, random_state=seed,
            stratify=y_valid[np.isin(indices, train_val_indices)]
        )
    else:
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_size, random_state=seed
        )
    
    logger.info(f"Created splits: {len(train_indices)} train, "
               f"{len(val_indices)} val, {len(test_indices)} test")
    
    # Create k-fold splits
    if use_stratify:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits_gen = kf.split(train_val_indices, y_valid[np.isin(indices, train_val_indices)])
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits_gen = kf.split(train_val_indices)
    
    fold_indices = []
    for fold, (train_idx, val_idx) in enumerate(splits_gen):
        fold_indices.append({
            'fold': fold,
            'train': train_val_indices[train_idx].tolist(),
            'val': train_val_indices[val_idx].tolist()
        })
        logger.info(f"Fold {fold}: {len(train_idx)} train, {len(val_idx)} val")
    
    return {
        'train_idx': train_indices.tolist(),
        'val_idx': val_indices.tolist(),
        'test_idx': test_indices.tolist(),
        'folds': fold_indices,
        'stratified': use_stratify,
        'n_samples': n_samples,
        'n_folds': n_folds
    }


def save_splits(splits: Dict[str, Any], output_splits: str, output_indices: str):
    """
    Save splits in formats compatible with both sklearn and DL pipelines.
    
    Args:
        splits: Dictionary with split information
        output_splits: Path for JSON output
        output_indices: Path for NPZ output
    """
    # Save JSON format for sklearn pipeline
    with open(output_splits, 'w') as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Saved splits to {output_splits}")
    
    # Save NPZ format compatible with DL pipeline
    save_dict = {
        'train_indices': np.array(splits['train_idx']),
        'val_indices': np.array(splits['val_idx']),
        'test_indices': np.array(splits['test_idx'])
    }
    
    # Add fold information if available
    if 'folds' in splits:
        save_dict['n_folds'] = len(splits['folds'])
        for i, fold in enumerate(splits['folds']):
            save_dict[f'fold_{i}_train'] = np.array(fold['train'])
            save_dict[f'fold_{i}_val'] = np.array(fold['val'])
    
    np.savez(output_indices, **save_dict)
    logger.info(f"Saved indices to {output_indices}")


def save_cv_folds(splits: Dict[str, Any], output_cv: str):
    """
    Save CV fold information in separate file.
    
    Args:
        splits: Dictionary with split information
        output_cv: Path for CV folds JSON
    """
    cv_data = {
        'n_folds': len(splits.get('folds', [])),
        'folds': splits.get('folds', [])
    }
    
    with open(output_cv, 'w') as f:
        json.dump(cv_data, f, indent=2)
    logger.info(f"Saved CV folds to {output_cv}")


def main():
    parser = argparse.ArgumentParser(description='Split data for sklearn pipeline')
    parser.add_argument('--genotype_file', type=str, required=True,
                       help='Path to genotype file (HDF5, NPZ, or numpy)')
    parser.add_argument('--phenotype_file', type=str, required=True,
                       help='Path to phenotype CSV file')
    parser.add_argument('--existing_splits', type=str, default=None,
                       help='Path to existing splits NPZ from DL pipeline')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion for test set')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Proportion for validation set')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--stratify', type=str, default='auto',
                       choices=['auto', 'yes', 'no'],
                       help='Use stratified splitting')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_splits', type=str, required=True,
                       help='Output JSON file for splits')
    parser.add_argument('--output_indices', type=str, required=True,
                       help='Output NPZ file for indices')
    parser.add_argument('--output_cv', type=str, required=True,
                       help='Output JSON file for CV folds')
    
    args = parser.parse_args()
    
    # Check if we should use existing splits
    if args.existing_splits and Path(args.existing_splits).exists():
        logger.info("Using existing splits from DL pipeline")
        splits = load_existing_splits(args.existing_splits)
    else:
        logger.info("Creating new splits")
        # Get number of samples
        _, n_samples = load_genotype_data(args.genotype_file)
        
        # Create new splits
        splits = create_new_splits(
            n_samples=n_samples,
            phenotype_file=args.phenotype_file,
            test_size=args.test_size,
            val_size=args.val_size,
            n_folds=args.n_folds,
            stratify=args.stratify,
            seed=args.seed
        )
    
    # Save splits
    save_splits(splits, args.output_splits, args.output_indices)
    save_cv_folds(splits, args.output_cv)
    
    # Print summary
    print("\n" + "="*50)
    print("Data Splitting Summary")
    print("="*50)
    print(f"Training samples: {len(splits['train_idx'])}")
    print(f"Validation samples: {len(splits['val_idx'])}")
    print(f"Test samples: {len(splits['test_idx'])}")
    if 'folds' in splits:
        print(f"CV folds: {len(splits['folds'])}")
    print("="*50)


if __name__ == '__main__':
    main()