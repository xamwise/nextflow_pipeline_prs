import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
import logging
from pathlib import Path
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


class GenotypeDataset(Dataset):
    """
    PyTorch Dataset for large-scale genotype data stored in HDF5 format.
    Implements lazy loading to handle datasets that don't fit in memory.
    """
    
    def __init__(
        self,
        h5_file: str,
        phenotype_file: str,
        indices: Optional[np.ndarray] = None,
        augment: bool = False,
        augmentation_params: Optional[Dict] = None,
        cache_size: int = 1000,
        normalize: bool = False
    ):
        """
        Initialize the GenotypeDataset.
        
        Args:
            h5_file: Path to HDF5 file containing genotype data
            phenotype_file: Path to CSV file containing phenotypes
            indices: Optional array of sample indices to use (for train/val/test splits)
            augment: Whether to apply data augmentation
            augmentation_params: Parameters for augmentation
            cache_size: Number of samples to cache in memory
            normalize: Whether to normalize genotypes
        """
        self.h5_file = h5_file
        self.phenotypes = pd.read_csv(phenotype_file)
        self.augment = augment
        self.augmentation_params = augmentation_params or {}
        self.normalize = normalize
        
        # Don't open HDF5 file here - will open lazily
        self._h5_handle = None
        
        # Get dimensions by temporarily opening the file
        with h5py.File(h5_file, 'r') as f:
            self.n_samples, self.n_snps = f['genotypes'].shape
        
        # Handle indices for data splits
        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(self.n_samples)
        
        # Setup cache
        self.cache = {}
        self.cache_size = cache_size
        
        # Calculate statistics for normalization
        if self.normalize:
            self._calculate_stats()
        
        logger.info(f"Initialized dataset with {len(self.indices)} samples and {self.n_snps} SNPs")
    
    @property
    def h5_handle(self):
        """Lazy loading of HDF5 file handle."""
        if self._h5_handle is None:
            self._h5_handle = h5py.File(self.h5_file, 'r')
        return self._h5_handle
    
    @property
    def genotypes(self):
        """Access genotypes from HDF5 file."""
        return self.h5_handle['genotypes']
    
    def _calculate_stats(self):
        """Calculate mean and std for normalization."""
        # Sample a subset for statistics calculation
        sample_size = min(1000, len(self.indices))
        sample_indices = np.random.choice(self.indices, sample_size, replace=False)
        
        sample_data = []
        # Temporarily open file for stats calculation
        with h5py.File(self.h5_file, 'r') as f:
            genotypes = f['genotypes']
            for idx in sample_indices:
                sample_data.append(genotypes[idx])
        sample_data = np.array(sample_data)
        
        self.mean = np.mean(sample_data, axis=0)
        self.std = np.std(sample_data, axis=0) + 1e-8  # Add small epsilon
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (genotype tensor, phenotype tensor)
        """
        # Get actual index
        actual_idx = self.indices[idx]
        
        # Check cache
        if actual_idx in self.cache:
            genotype = self.cache[actual_idx].copy()
        else:
            # Load from HDF5
            genotype = self.genotypes[actual_idx][:]
            
            # Update cache
            if len(self.cache) >= self.cache_size:
                # Remove oldest item
                self.cache.pop(next(iter(self.cache)))
            self.cache[actual_idx] = genotype.copy()
        
        # Normalize
        if self.normalize:
            genotype = (genotype - self.mean) / self.std
        
        # Apply augmentation
        if self.augment:
            genotype = self._augment_genotype(genotype)
        
        # Get phenotype
        phenotype = self.phenotypes.iloc[actual_idx].values
        
        # Convert to tensors
        genotype_tensor = torch.FloatTensor(genotype)
        phenotype_tensor = torch.FloatTensor(phenotype)
        
        return genotype_tensor, phenotype_tensor
    
    def _augment_genotype(self, genotype: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to genotype data.
        
        Args:
            genotype: Input genotype array
            
        Returns:
            Augmented genotype array
        """
        augmented = genotype.copy()
        
        # Random SNP dropout
        if 'snp_dropout' in self.augmentation_params:
            dropout_rate = self.augmentation_params['snp_dropout']
            mask = np.random.random(len(augmented)) > dropout_rate
            augmented = augmented * mask
        
        # Add noise
        if 'noise_std' in self.augmentation_params:
            noise_std = self.augmentation_params['noise_std']
            noise = np.random.normal(0, noise_std, size=augmented.shape)
            augmented = augmented + noise
        
        # Random SNP shuffling (for specific regions)
        if 'shuffle_regions' in self.augmentation_params:
            n_regions = self.augmentation_params['shuffle_regions']
            region_size = len(augmented) // n_regions
            for i in range(n_regions):
                if np.random.random() < 0.1:  # 10% chance to shuffle each region
                    start = i * region_size
                    end = min((i + 1) * region_size, len(augmented))
                    np.random.shuffle(augmented[start:end])
        
        return augmented
    
    def close(self):
        """Close the HDF5 file handle."""
        if self._h5_handle is not None:
            self._h5_handle.close()
            self._h5_handle = None
    
    def __del__(self):
        """Ensure HDF5 file is closed on deletion."""
        self.close()


class GenotypeDataModule:
    """
    Data module for managing train/validation/test splits and data loaders.
    """
    
    def __init__(
        self,
        h5_file: str,
        phenotype_file: str,
        indices_file: str,
        batch_size: int = 32,
        num_workers: int = 4,
        augment_train: bool = True,
        augmentation_params: Optional[Dict] = None
    ):
        """
        Initialize the data module.
        
        Args:
            h5_file: Path to HDF5 file containing genotype data
            phenotype_file: Path to CSV file containing phenotypes
            indices_file: Path to NPZ file containing split indices
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            augment_train: Whether to augment training data
            augmentation_params: Parameters for augmentation
        """
        self.h5_file = h5_file
        self.phenotype_file = phenotype_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment_train = augment_train
        self.augmentation_params = augmentation_params
        
        # Load split indices
        indices_data = np.load(indices_file)
        self.train_indices = indices_data['train_indices']
        self.val_indices = indices_data['val_indices']
        self.test_indices = indices_data['test_indices']
        
        # Load fold indices if available
        self.fold_indices = []
        if 'n_folds' in indices_data:
            n_folds = int(indices_data['n_folds'])
            for i in range(n_folds):
                if f'fold_{i}_train' in indices_data and f'fold_{i}_val' in indices_data:
                    self.fold_indices.append({
                        'train': indices_data[f'fold_{i}_train'],
                        'val': indices_data[f'fold_{i}_val']
                    })
        
        # If no fold indices found, create them from train/val split
        if not self.fold_indices:
            logger.info("No fold indices found, creating from train/val split")
            from sklearn.model_selection import KFold
            n_folds = 5  # Default
            train_val_indices = np.concatenate([self.train_indices, self.val_indices])
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            for train_idx, val_idx in kf.split(train_val_indices):
                self.fold_indices.append({
                    'train': train_val_indices[train_idx],
                    'val': train_val_indices[val_idx]
                })
        
        logger.info(f"Data splits - Train: {len(self.train_indices)}, "
                   f"Val: {len(self.val_indices)}, Test: {len(self.test_indices)}")
    
    def train_dataloader(self, fold: Optional[int] = None) -> DataLoader:
        """Get training data loader."""
        if fold is not None and hasattr(self, 'fold_indices'):
            indices = self.fold_indices[fold]['train']
        else:
            indices = self.train_indices
        
        dataset = GenotypeDataset(
            self.h5_file,
            self.phenotype_file,
            indices=indices,
            augment=self.augment_train,
            augmentation_params=self.augmentation_params
        )
        
        # Set num_workers to 0 to avoid multiprocessing issues with HDF5
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Changed from self.num_workers to 0
            pin_memory=torch.cuda.is_available()
        )
    
    def val_dataloader(self, fold: Optional[int] = None) -> DataLoader:
        """Get validation data loader."""
        if fold is not None and hasattr(self, 'fold_indices'):
            indices = self.fold_indices[fold]['val']
        else:
            indices = self.val_indices
        
        dataset = GenotypeDataset(
            self.h5_file,
            self.phenotype_file,
            indices=indices,
            augment=False
        )
        
        # Set num_workers to 0 to avoid multiprocessing issues with HDF5
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Changed from self.num_workers to 0
            pin_memory=torch.cuda.is_available()
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test data loader."""
        dataset = GenotypeDataset(
            self.h5_file,
            self.phenotype_file,
            indices=self.test_indices,
            augment=False
        )
        
        # Set num_workers to 0 to avoid multiprocessing issues with HDF5
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Changed from self.num_workers to 0
            pin_memory=torch.cuda.is_available()
        )
    
    def get_input_dim(self) -> int:
        """Get input dimension (number of SNPs)."""
        with h5py.File(self.h5_file, 'r') as f:
            return f['genotypes'].shape[1]
    
    def get_output_dim(self) -> int:
        """Get output dimension (number of phenotypes)."""
        phenotypes = pd.read_csv(self.phenotype_file)
        return phenotypes.shape[1]


# Custom collate function for variable-length sequences if needed
def genotype_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for genotype batches.
    
    Args:
        batch: List of (genotype, phenotype) tuples
        
    Returns:
        Batched tensors
    """
    genotypes, phenotypes = zip(*batch)
    
    # Stack tensors
    genotypes = torch.stack(genotypes)
    phenotypes = torch.stack(phenotypes)
    
    return genotypes, phenotypes