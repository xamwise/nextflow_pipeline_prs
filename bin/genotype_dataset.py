import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
import logging
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

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
        target_scaler: Optional[Dict] = None,
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
        self.target_scaler = target_scaler
        
        # Don't open HDF5 file here - will open lazily
        self._h5_handle = None
        
        # Get dimensions by temporarily opening the file
        with h5py.File(h5_file, 'r') as f:
            shape = f['genotypes'].shape
            self.n_samples = shape[0]
            self.n_snps = shape[1]
            self.n_channels = shape[2] if len(shape) == 3 else 1
            # Encoding may not be present in older files
            self.encoding = f.attrs.get('encoding', 'raw')
            if isinstance(self.encoding, bytes):
                self.encoding = self.encoding.decode('utf-8')  
                
        # Handle indices for data splits
        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(self.n_samples)
        
        # Setup cache
        self.cache = {}
        self.cache_size = cache_size
        
        # Only normalize raw {0,1,2} counts; for one-hot / two-dim it's not meaningful.
        if self.normalize and self.encoding != 'raw':
            logger.warning(
                f"normalize=True ignored for encoding={self.encoding!r}; "
                "normalization is only applied to 'raw' encoding."
            )
            self.normalize = False

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
        if self.target_scaler is not None:
            phenotype = self.target_scaler.transform(phenotype.reshape(1, -1)).ravel()
        
        # Convert to tensors
        genotype_tensor = torch.FloatTensor(genotype)
        phenotype_tensor = torch.FloatTensor(phenotype)
        
        return genotype_tensor, phenotype_tensor
    
    # def _augment_genotype(self, genotype: np.ndarray) -> np.ndarray:
    #     """
    #     Apply data augmentation to genotype data.
        
    #     Args:
    #         genotype: Input genotype array
            
    #     Returns:
    #         Augmented genotype array
    #     """
    #     augmented = genotype.copy()
        
    #     # Random SNP dropout
    #     if 'snp_dropout' in self.augmentation_params:
    #         dropout_rate = self.augmentation_params['snp_dropout']
    #         mask = np.random.random(len(augmented)) > dropout_rate
    #         augmented = augmented * mask
        
    #     # Add noise
    #     if 'noise_std' in self.augmentation_params:
    #         noise_std = self.augmentation_params['noise_std']
    #         noise = np.random.normal(0, noise_std, size=augmented.shape)
    #         augmented = augmented + noise
        
    #     # Random SNP shuffling (for specific regions)
    #     if 'shuffle_regions' in self.augmentation_params:
    #         n_regions = self.augmentation_params['shuffle_regions']
    #         region_size = len(augmented) // n_regions
    #         for i in range(n_regions):
    #             if np.random.random() < 0.1:  # 10% chance to shuffle each region
    #                 start = i * region_size
    #                 end = min((i + 1) * region_size, len(augmented))
    #                 np.random.shuffle(augmented[start:end])
        
    #     return augmented
    
    def _augment_genotype(self, genotype: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to genotype data.

        For 'raw' encoding (1D input), augmentation acts per element, matching
        the original behavior. For 'one-hot' and 'two-dim' encodings (2D input
        of shape (n_snps, n_channels)), augmentation acts per SNP so that all
        channels of a SNP are modified together — this keeps every augmented
        sample inside the space of valid encodings.
        """
        augmented = genotype.copy()

        # Raw / 1D path — original behavior, unchanged.
        if augmented.ndim == 1:
            if 'snp_dropout' in self.augmentation_params:
                dropout_rate = self.augmentation_params['snp_dropout']
                mask = np.random.random(len(augmented)) > dropout_rate
                augmented = augmented * mask

            if 'noise_std' in self.augmentation_params:
                noise_std = self.augmentation_params['noise_std']
                noise = np.random.normal(0, noise_std, size=augmented.shape)
                augmented = augmented + noise

            if 'shuffle_regions' in self.augmentation_params:
                n_regions = self.augmentation_params['shuffle_regions']
                region_size = len(augmented) // n_regions
                for i in range(n_regions):
                    if np.random.random() < 0.1:
                        start = i * region_size
                        end = min((i + 1) * region_size, len(augmented))
                        np.random.shuffle(augmented[start:end])

            return augmented

        # Encoded / 2D path — shape (n_snps, n_channels), act per SNP.
        n_snps = augmented.shape[0]

        # SNP dropout: zero out whole SNPs (all channels together) so the result
        # matches how the encoder represents a missing genotype.
        if 'snp_dropout' in self.augmentation_params:
            dropout_rate = self.augmentation_params['snp_dropout']
            keep = (np.random.random(n_snps) > dropout_rate).astype(augmented.dtype)
            augmented = augmented * keep[:, None]

        # Gaussian noise: skipped for one-hot since it produces non-categorical
        # values the model would never see at inference. Allowed for two-dim.
        if 'noise_std' in self.augmentation_params and self.encoding != 'one-hot':
            noise_std = self.augmentation_params['noise_std']
            noise = np.random.normal(0, noise_std, size=augmented.shape)
            augmented = augmented + noise

        # Region shuffle: permute SNP positions within a region, carrying each
        # SNP's channels with it (rows move together).
        if 'shuffle_regions' in self.augmentation_params:
            n_regions = self.augmentation_params['shuffle_regions']
            region_size = max(1, n_snps // n_regions)
            for i in range(n_regions):
                if np.random.random() < 0.1:
                    start = i * region_size
                    end = min((i + 1) * region_size, n_snps)
                    perm = np.random.permutation(end - start)
                    augmented[start:end] = augmented[start:end][perm]

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
        scale_target: bool = False,
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
            scale_target: Whether to scale target phenotypes
            target_scaler: Optional scaler for target phenotypes (if scale_target is True)
        """
        self.h5_file = h5_file
        self.phenotype_file = phenotype_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment_train = augment_train
        self.augmentation_params = augmentation_params
        self.target_scalers: Dict[int, StandardScaler] = {}
        self.scale_target = scale_target
        
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
            augmentation_params=self.augmentation_params,
            target_scaler=self.get_target_scaler(fold) if self.scale_target else None
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
            augment=False,
            target_scaler=self.get_target_scaler(fold) if self.scale_target else None
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
            augment=False,
            target_scaler=self.get_target_scaler(0) if self.scale_target else None
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
        """Number of SNPs (length of the SNP axis)."""
        with h5py.File(self.h5_file, 'r') as f:
            return f['genotypes'].shape[1]

    def get_n_channels(self) -> int:
        """Per-SNP channel count: 1 for 'raw', 2 for 'two-dim', 3 for 'one-hot'."""
        with h5py.File(self.h5_file, 'r') as f:
            shape = f['genotypes'].shape
            return shape[2] if len(shape) == 3 else 1

    def get_encoding(self) -> str:
        with h5py.File(self.h5_file, 'r') as f:
            enc = f.attrs.get('encoding', 'raw')
            return enc.decode('utf-8') if isinstance(enc, bytes) else enc
    
    def get_output_dim(self) -> int:
        """Get output dimension (number of phenotypes)."""
        phenotypes = pd.read_csv(self.phenotype_file)
        return phenotypes.shape[1]
    
    def get_target_scaler(self, fold: int):
        if not self.scale_target:
            return None
        if fold not in self.target_scalers:
            train_idx = self.fold_indices[fold]['train']
            y_train = pd.read_csv(self.phenotype_file).iloc[train_idx].values
            scaler = StandardScaler().fit(y_train)
            self.target_scalers[fold] = scaler
        return self.target_scalers[fold]


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