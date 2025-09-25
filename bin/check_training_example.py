import torch
import numpy as np
import sys
import os
sys.path.append('bin')  # Add your scripts directory

from genotype_dataset import GenotypeDataModule

# Load the data module
data_module = GenotypeDataModule(
    h5_file='out/genotypes_test.h5',
    phenotype_file='out/phenotypes_test.csv',
    indices_file='out/indices_test.npz',
    batch_size=500,
    num_workers=0
)

# Get a training data loader for fold 0
train_loader = data_module.train_dataloader(fold=0)

# Get one batch
for batch_idx, (genotypes, phenotypes) in enumerate(train_loader):
    print(f"\n{'='*50}")
    print("Training Example Shapes:")
    print(f"{'='*50}")
    
    print(f"\nBatch {batch_idx + 1}:")
    print(f"  Genotypes shape: {genotypes.shape}")
    print(f"  Phenotypes shape: {phenotypes.shape}")
    
    print(f"\nData types:")
    print(f"  Genotypes dtype: {genotypes.dtype}")
    print(f"  Phenotypes dtype: {phenotypes.dtype}")
    
    print(f"\nSingle sample shapes:")
    print(f"  Single genotype: {genotypes[0].shape}")
    print(f"  Single phenotype: {phenotypes[0].shape}")
    
    print(f"\nValue ranges:")
    print(f"  Genotypes: min={genotypes.min():.4f}, max={genotypes.max():.4f}, mean={genotypes.mean():.4f}")
    print(f"  Phenotypes: min={phenotypes.min():.4f}, max={phenotypes.max():.4f}, mean={phenotypes.mean():.4f}")
    
    print(f"\nFirst sample preview:")
    print(f"  First 10 SNPs: {genotypes[0][:10].numpy()}")
    print(f"  Phenotype: {phenotypes[0].numpy()}")
    
    print(f"{'='*50}\n")
    
    # Only show first batch
    break