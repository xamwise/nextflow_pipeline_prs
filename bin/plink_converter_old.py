"""
Convert PLINK files to HDF5 format for efficient loading in PyTorch.
"""

import numpy as np
import pandas as pd
import h5py
import argparse
import json
from pathlib import Path
import logging
from typing import Dict, Any, Optional
from pandas_plink import read_plink

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_plink_to_h5(
    plink_prefix: str,
    output_h5: str,
    output_pheno: str,
    phenotype_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convert PLINK files to HDF5 format.
    
    Args:
        plink_prefix: Prefix for PLINK files (.bed, .bim, .fam)
        output_h5: Output HDF5 file path
        output_pheno: Output phenotype CSV file path
        phenotype_file: Optional separate phenotype file path
        
    Returns:
        Dictionary of data statistics
    """
    logger.info(f"Reading PLINK files with prefix: {plink_prefix}")
    
    # Read PLINK files using pandas-plink
    (bim, fam, bed) = read_plink(plink_prefix)
    genotypes = bed.compute()  # Convert dask array to numpy
    snp_info = bim
    sample_info = fam
    
    # Check dimensions and transpose if necessary
    logger.info(f"Raw genotype matrix shape: {genotypes.shape}")
    
    # pandas-plink should return (n_samples, n_snps)
    # If we have more rows than columns, it's likely transposed
    if genotypes.shape[0] > genotypes.shape[1]:
        logger.info(f"Detected possible transposition: {genotypes.shape[0]} rows > {genotypes.shape[1]} columns")
        # Check against fam and bim files to determine correct orientation
        n_samples_expected = len(sample_info)
        n_snps_expected = len(snp_info)
        
        logger.info(f"Expected from .fam file: {n_samples_expected} samples")
        logger.info(f"Expected from .bim file: {n_snps_expected} SNPs")
        
        if genotypes.shape[0] == n_snps_expected and genotypes.shape[1] == n_samples_expected:
            logger.info("Matrix is transposed. Transposing to (samples x SNPs)...")
            genotypes = genotypes.T
        elif genotypes.shape[0] == n_samples_expected and genotypes.shape[1] == n_snps_expected:
            logger.info("Matrix is already in correct orientation (samples x SNPs)")
        else:
            logger.warning("Matrix dimensions don't match expected values. Proceeding with caution.")
    
    logger.info(f"Final genotype matrix shape: {genotypes.shape} (samples x SNPs)")
    n_samples, n_snps = genotypes.shape
    
    # Handle missing genotypes: substitute NaN with 0
    n_missing = np.isnan(genotypes).sum()
    missing_rate_original = n_missing / genotypes.size
    
    if n_missing > 0:
        logger.info(f"Found {n_missing:,} missing genotypes ({missing_rate_original:.2%} of total)")
        logger.info("Substituting missing genotypes (NaN) with 0")
        genotypes = np.nan_to_num(genotypes, nan=0.0)
        logger.info("Missing genotype substitution completed")
    else:
        logger.info("No missing genotypes found")
    
    # Calculate basic statistics (after missing value substitution)
    stats = {
        'n_samples': genotypes.shape[0],
        'n_snps': genotypes.shape[1],
        'missing_rate_original': float(missing_rate_original),
        'n_missing_substituted': int(n_missing),
        'missing_rate_final': 0.0,  # After substitution, no missing values remain
        'mean_genotype': float(np.mean(genotypes)),
        'std_genotype': float(np.std(genotypes)),
        'min_genotype': float(np.min(genotypes)),
        'max_genotype': float(np.max(genotypes))
    }
    
    # Save to HDF5
    logger.info(f"Saving to HDF5: {output_h5}")
    with h5py.File(output_h5, 'w') as f:
        # Save genotypes
        f.create_dataset(
            'genotypes',
            data=genotypes.astype(np.float32),
            dtype='float32',
            compression='gzip',
            compression_opts=4
        )
        
        # Save SNP information
        for col in snp_info.columns:
            try:
                data = snp_info[col].values
                
                # Handle different data types
                if data.dtype == np.object_ or data.dtype.kind in ['O', 'U', 'S']:
                    # Convert object/string types to fixed-length strings
                    max_len = max(len(str(x)) for x in data)
                    dtype = f'S{max_len}'
                    data = np.array([str(x).encode('utf-8') for x in data], dtype=dtype)
                elif np.issubdtype(data.dtype, np.integer):
                    data = data.astype(np.int64)
                elif np.issubdtype(data.dtype, np.floating):
                    data = data.astype(np.float64)
                
                f.create_dataset(f'snp_info/{col}', data=data)
            except Exception as e:
                logger.warning(f"Could not save SNP info column '{col}': {e}")
        
        # Save sample information
        for col in sample_info.columns:
            try:
                data = sample_info[col].values
                
                # Handle different data types
                if data.dtype == np.object_ or data.dtype.kind in ['O', 'U', 'S']:
                    # Convert object/string types to fixed-length strings
                    max_len = max(len(str(x)) for x in data)
                    dtype = f'S{max_len}'
                    data = np.array([str(x).encode('utf-8') for x in data], dtype=dtype)
                elif np.issubdtype(data.dtype, np.integer):
                    data = data.astype(np.int64)
                elif np.issubdtype(data.dtype, np.floating):
                    data = data.astype(np.float64)
                
                f.create_dataset(f'sample_info/{col}', data=data)
            except Exception as e:
                logger.warning(f"Could not save sample info column '{col}': {e}")
        
        # Save metadata
        f.attrs['n_samples'] = stats['n_samples']
        f.attrs['n_snps'] = stats['n_snps']
        f.attrs['n_missing_substituted'] = stats['n_missing_substituted']
        f.attrs['missing_rate_original'] = stats['missing_rate_original']
    
    # Handle phenotypes
    logger.info(f"Saving phenotypes to: {output_pheno}")
    
    if phenotype_file:
        # Load phenotypes from separate file
        logger.info(f"Loading phenotypes from: {phenotype_file}")
        
        # Try to read the phenotype file
        # First, try to detect the format
        with open(phenotype_file, 'r') as f:
            first_line = f.readline().strip()
            
        # Check if it has a header or not
        has_header = not first_line.replace('.', '').replace('-', '').replace('e', '').replace('+', '').split()[0].isdigit()
        
        if has_header:
            # File has headers
            phenotypes_raw = pd.read_csv(phenotype_file, sep='\s+', engine='python')
        else:
            # No headers, assume FID, IID, phenotype format (standard PLINK phenotype format)
            phenotypes_raw = pd.read_csv(phenotype_file, sep='\s+', engine='python',
                                    names=['FID', 'IID', 'phenotype'])
        
        # If the file only has one column (just phenotype values)
        if phenotypes_raw.shape[1] == 1:
            # Assume phenotypes are in same order as samples
            if len(phenotypes_raw) != len(sample_info):
                logger.warning(f"Phenotype count ({len(phenotypes_raw)}) doesn't match sample count ({len(sample_info)})")
                # Pad with NaN if needed
                if len(phenotypes_raw) < len(sample_info):
                    padding = len(sample_info) - len(phenotypes_raw)
                    phenotypes = pd.DataFrame({
                        'phenotype': list(phenotypes_raw.iloc[:, 0]) + [np.nan] * padding
                    })
                    logger.info(f"Padded phenotypes with {padding} NaN values")
                else:
                    # Truncate if too many
                    phenotypes = phenotypes_raw.iloc[:len(sample_info), [0]]
                    phenotypes.columns = ['phenotype']
                    logger.info(f"Truncated phenotypes to match sample count")
            else:
                phenotypes = phenotypes_raw
                phenotypes.columns = ['phenotype']
        else:
            # If we have FID and IID columns, match with fam file order
            if 'IID' in phenotypes_raw.columns or phenotypes_raw.shape[1] >= 2:
                # Assume second column is IID if not named
                if 'IID' not in phenotypes_raw.columns:
                    phenotypes_raw.columns = ['FID', 'IID'] + list(phenotypes_raw.columns[2:])
                
                # Create a dataframe with all samples
                phenotypes = pd.DataFrame({
                    'IID': sample_info['iid'],
                    'order': range(len(sample_info))
                })
                
                # Get phenotype column (last non-ID column)
                pheno_cols = [c for c in phenotypes_raw.columns if c not in ['FID', 'IID']]
                if pheno_cols:
                    pheno_col = pheno_cols[-1]
                else:
                    pheno_col = 'phenotype'
                    phenotypes_raw['phenotype'] = phenotypes_raw.iloc[:, -1]
                
                # Merge phenotypes
                phenotypes = phenotypes.merge(
                    phenotypes_raw[['IID', pheno_col]], 
                    on='IID', 
                    how='left'
                )
                phenotypes = phenotypes.sort_values('order')
                phenotypes = phenotypes[[pheno_col]]
                phenotypes.columns = ['phenotype']
                
                n_matched = phenotypes['phenotype'].notna().sum()
                logger.info(f"Matched {n_matched} phenotypes out of {len(sample_info)} samples")
            else:
                phenotypes = phenotypes_raw.iloc[:, [0]]
                phenotypes.columns = ['phenotype']
    else:
        # Try to get phenotype from fam file
        if 'phenotype' in sample_info.columns:
            phenotypes = sample_info[['phenotype']].copy()
        else:
            # Use the last column of fam file (standard location for phenotype)
            phenotypes = sample_info.iloc[:, -1:].copy()
            phenotypes.columns = ['phenotype']
    
    # Handle PLINK phenotype coding if applicable
    phenotype_col = phenotypes.columns[0]
    if phenotypes[phenotype_col].dtype in [np.int32, np.int64, np.float32, np.float64]:
        unique_vals = phenotypes[phenotype_col].dropna().unique()
        
        # Check for PLINK binary coding (1=control, 2=case)
        if len(unique_vals) == 2 and set(unique_vals).issubset({1, 2}):
            # Binary trait: convert 1->0 (control), 2->1 (case)
            phenotypes[phenotype_col] = phenotypes[phenotype_col] - 1
            logger.info("Converted binary phenotypes from PLINK coding (1/2) to (0/1)")
        
        # Handle missing phenotypes (-9 in PLINK)
        phenotypes.loc[phenotypes[phenotype_col] < 0, phenotype_col] = np.nan
    
    # Rename column to standard name
    phenotypes.columns = ['phenotype_1']
    phenotypes.to_csv(output_pheno, index=False)
    
    logger.info(f"Saved {len(phenotypes)} phenotype values")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Convert PLINK files to HDF5 format')
    parser.add_argument('--plink_prefix', type=str, required=True,
                        help='Prefix for PLINK files (.bed, .bim, .fam)')
    parser.add_argument('--output_h5', type=str, required=True,
                        help='Output HDF5 file path')
    parser.add_argument('--output_pheno', type=str, required=True,
                        help='Output phenotype CSV file path')
    parser.add_argument('--stats_file', type=str, required=True,
                        help='Output statistics JSON file')
    parser.add_argument('--phenotype_file', type=str, default=None,
                        help='Optional separate phenotype file (if not in .fam file)')
    
    args = parser.parse_args()
    
    # Convert PLINK to HDF5
    stats = convert_plink_to_h5(
        plink_prefix=args.plink_prefix,
        output_h5=args.output_h5,
        output_pheno=args.output_pheno,
        phenotype_file=args.phenotype_file
    )
    
    # Save statistics
    with open(args.stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Conversion completed. Statistics saved to {args.stats_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("PLINK to HDF5 Conversion Summary")
    print("="*50)
    print(f"Samples: {stats['n_samples']:,}")
    print(f"SNPs: {stats['n_snps']:,}")
    print(f"Original missing rate: {stats['missing_rate_original']:.2%}")
    print(f"Missing genotypes substituted: {stats['n_missing_substituted']:,}")
    print(f"Final missing rate: {stats['missing_rate_final']:.2%}")
    print(f"Mean genotype: {stats['mean_genotype']:.4f}")
    print(f"Std genotype: {stats['std_genotype']:.4f}")
    print("="*50)


if __name__ == '__main__':
    main()