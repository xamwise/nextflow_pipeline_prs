#!/usr/bin/env python3
"""
Convert PLINK format genotype data to sklearn-compatible numpy arrays.
Uses the same HDF5 format as the deep learning pipeline for consistency.
"""

import numpy as np
import pandas as pd
import h5py
import json
import argparse
from pathlib import Path
import logging
from scipy import sparse
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plink_to_numpy(plink_prefix: str, handle_missing: str = 'impute') -> tuple:
    """
    Convert PLINK files to numpy arrays.
    
    Args:
        plink_prefix: Prefix for PLINK files (.bed, .bim, .fam)
        handle_missing: How to handle missing values ('impute', 'drop', 'zero')
        
    Returns:
        Tuple of (genotype_matrix, sample_ids, snp_ids)
    """
    # Check if files exist
    bed_file = f"{plink_prefix}.bed"
    bim_file = f"{plink_prefix}.bim"
    fam_file = f"{plink_prefix}.fam"
    
    for f in [bed_file, bim_file, fam_file]:
        if not Path(f).exists():
            raise FileNotFoundError(f"PLINK file not found: {f}")
    
    # Try to use PLINK to convert to raw format first
    try:
        logger.info("Converting PLINK to raw format...")
        cmd = f"plink --bfile {plink_prefix} --recodeA --out {plink_prefix}_raw"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Load the raw file
            raw_file = f"{plink_prefix}_raw.raw"
            if Path(raw_file).exists():
                logger.info("Loading raw genotype file...")
                raw_data = pd.read_csv(raw_file, sep=r'\s+')
                
                # Extract sample IDs
                sample_ids = raw_data[['FID', 'IID']].values
                
                # Extract genotype data (skip first 6 columns)
                genotype_cols = raw_data.columns[6:]
                genotypes = raw_data[genotype_cols].values
                
                # Extract SNP IDs from column names
                snp_ids = [col.replace('_A', '').replace('_T', '').replace('_G', '').replace('_C', '') 
                          for col in genotype_cols]
                
                # Clean up temporary files
                for ext in ['.raw', '.log', '.nosex']:
                    temp_file = f"{plink_prefix}_raw{ext}"
                    if Path(temp_file).exists():
                        Path(temp_file).unlink()
                
                logger.info(f"Loaded genotypes: {genotypes.shape}")
                return genotypes, sample_ids, snp_ids
    except Exception as e:
        logger.warning(f"Could not use PLINK to convert: {e}")
    
    # Fallback: Use pandas-plink if available
    try:
        import pandas_plink
        logger.info("Using pandas-plink to read PLINK files...")
        
        (bim, fam, bed) = pandas_plink.read_plink(plink_prefix)
        
        # Convert to numpy array
        genotypes = bed.compute()  # Convert dask array to numpy
        sample_ids = fam[['fid', 'iid']].values
        snp_ids = bim['snp'].values
        
        return genotypes, sample_ids, snp_ids
        
    except ImportError:
        logger.error("pandas-plink not installed. Install with: pip install pandas-plink")
        sys.exit(1)


def handle_missing_values(genotypes: np.ndarray, method: str = 'impute') -> np.ndarray:
    """
    Handle missing values in genotype data.
    
    Args:
        genotypes: Genotype matrix
        method: Method to handle missing values
        
    Returns:
        Processed genotype matrix
    """
    n_missing = np.isnan(genotypes).sum()
    logger.info(f"Found {n_missing} missing values ({n_missing/genotypes.size*100:.2f}%)")
    
    if method == 'impute':
        # Impute with mean
        logger.info("Imputing missing values with mean...")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        genotypes = imputer.fit_transform(genotypes)
        
    elif method == 'drop':
        # Drop SNPs with any missing values
        logger.info("Dropping SNPs with missing values...")
        mask = ~np.isnan(genotypes).any(axis=0)
        genotypes = genotypes[:, mask]
        logger.info(f"Kept {mask.sum()} out of {len(mask)} SNPs")
        
    elif method == 'zero':
        # Replace missing with zero
        logger.info("Replacing missing values with zero...")
        genotypes = np.nan_to_num(genotypes, nan=0)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return genotypes


def save_as_hdf5(genotypes: np.ndarray, output_file: str, 
                 sample_ids: np.ndarray = None, snp_ids: np.ndarray = None):
    """
    Save genotype data in HDF5 format (compatible with DL pipeline).
    
    Args:
        genotypes: Genotype matrix
        output_file: Output HDF5 file path
        sample_ids: Sample identifiers
        snp_ids: SNP identifiers
    """
    logger.info(f"Saving to HDF5: {output_file}")
    
    with h5py.File(output_file, 'w') as f:
        # Save genotypes
        f.create_dataset('genotypes', data=genotypes, compression='gzip', compression_opts=4)
        
        # Save sample IDs if provided
        if sample_ids is not None:
            if sample_ids.dtype == 'O':  # String type
                dt = h5py.special_dtype(vlen=str)
                f.create_dataset('sample_ids', data=sample_ids, dtype=dt)
            else:
                f.create_dataset('sample_ids', data=sample_ids)
        
        # Save SNP IDs if provided
        if snp_ids is not None:
            if isinstance(snp_ids[0], str):
                dt = h5py.special_dtype(vlen=str)
                f.create_dataset('snp_ids', data=snp_ids, dtype=dt)
            else:
                f.create_dataset('snp_ids', data=snp_ids)
        
        # Save metadata
        f.attrs['n_samples'] = genotypes.shape[0]
        f.attrs['n_snps'] = genotypes.shape[1]
        f.attrs['format'] = 'sklearn_compatible'
    
    logger.info(f"Saved {genotypes.shape[0]} samples x {genotypes.shape[1]} SNPs")


def save_as_sparse(genotypes: np.ndarray, output_file: str):
    """
    Save genotype data as sparse matrix for memory efficiency.
    
    Args:
        genotypes: Genotype matrix
        output_file: Output NPZ file path
    """
    logger.info(f"Saving as sparse matrix: {output_file}")
    
    # Convert to sparse matrix if beneficial
    sparsity = (genotypes == 0).sum() / genotypes.size
    logger.info(f"Matrix sparsity: {sparsity*100:.2f}%")
    
    if sparsity > 0.5:
        # Use sparse format
        sparse_matrix = sparse.csr_matrix(genotypes)
        sparse.save_npz(output_file, sparse_matrix)
        logger.info(f"Saved as sparse CSR matrix")
    else:
        # Save as dense
        np.savez_compressed(output_file, data=genotypes)
        logger.info(f"Saved as compressed dense matrix")


def extract_phenotypes(fam_file: str, pheno_file: str = None) -> pd.DataFrame:
    """
    Extract phenotypes from FAM file or external phenotype file.
    
    Args:
        fam_file: Path to FAM file
        pheno_file: Optional external phenotype file
        
    Returns:
        DataFrame with phenotypes
    """
    # Load FAM file
    fam_df = pd.read_csv(fam_file, sep=r'\s+', header=None,
                         names=['FID', 'IID', 'PID', 'MID', 'SEX', 'PHENO'])
    
    if pheno_file and Path(pheno_file).exists():
        logger.info(f"Loading phenotypes from: {pheno_file}")
        # Load external phenotype file
        pheno_df = pd.read_csv(pheno_file, sep=r'\s+')
        
        # Merge with FAM file
        if 'FID' in pheno_df.columns and 'IID' in pheno_df.columns:
            merged = fam_df[['FID', 'IID']].merge(pheno_df, on=['FID', 'IID'])
            phenotypes = merged.drop(['FID', 'IID'], axis=1)
        else:
            # Assume same order as FAM file
            phenotypes = pheno_df
    else:
        # Use phenotype from FAM file
        phenotypes = fam_df[['PHENO']]
    
    # Replace -9 with NaN (PLINK missing phenotype code)
    phenotypes = phenotypes.replace(-9, np.nan)
    
    return phenotypes


def calculate_stats(genotypes: np.ndarray) -> dict:
    """
    Calculate statistics about the genotype data.
    
    Args:
        genotypes: Genotype matrix
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'n_samples': int(genotypes.shape[0]),
        'n_snps': int(genotypes.shape[1]),
        'missing_rate': float(np.isnan(genotypes).sum() / genotypes.size),
        'mean': float(np.nanmean(genotypes)),
        'std': float(np.nanstd(genotypes)),
        'min': float(np.nanmin(genotypes)),
        'max': float(np.nanmax(genotypes)),
        'sparsity': float((genotypes == 0).sum() / genotypes.size)
    }
    
    # Calculate allele frequencies
    allele_freqs = np.nanmean(genotypes, axis=0) / 2  # Assuming 0/1/2 coding
    stats['maf_mean'] = float(np.mean(np.minimum(allele_freqs, 1 - allele_freqs)))
    stats['maf_min'] = float(np.min(np.minimum(allele_freqs, 1 - allele_freqs)))
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Convert PLINK to sklearn format')
    parser.add_argument('--plink_prefix', type=str, required=True,
                       help='Prefix for PLINK files')
    parser.add_argument('--output_matrix', type=str, required=True,
                       help='Output file for genotype matrix (.h5 or .npz)')
    parser.add_argument('--output_pheno', type=str, required=True,
                       help='Output CSV file for phenotypes')
    parser.add_argument('--output_features', type=str, required=True,
                       help='Output file for feature names')
    parser.add_argument('--stats_file', type=str, required=True,
                       help='Output JSON file for data statistics')
    parser.add_argument('--handle_missing', type=str, default='impute',
                       choices=['impute', 'drop', 'zero'],
                       help='How to handle missing values')
    parser.add_argument('--external_pheno', type=str, default=None,
                       help='External phenotype file')
    parser.add_argument('--format', type=str, default='auto',
                       choices=['hdf5', 'npz', 'auto'],
                       help='Output format')
    
    args = parser.parse_args()
    
    # Convert PLINK to numpy
    logger.info(f"Loading PLINK files: {args.plink_prefix}")
    genotypes, sample_ids, snp_ids = plink_to_numpy(args.plink_prefix, args.handle_missing)
    
    # Handle missing values
    genotypes = handle_missing_values(genotypes, args.handle_missing)
    
    # Calculate statistics
    stats = calculate_stats(genotypes)
    logger.info(f"Data shape: {genotypes.shape}")
    logger.info(f"Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    
    # Save genotype matrix
    output_ext = Path(args.output_matrix).suffix.lower()
    
    if args.format == 'auto':
        if output_ext == '.h5' or output_ext == '.hdf5':
            format_type = 'hdf5'
        else:
            format_type = 'npz'
    else:
        format_type = args.format
    
    if format_type == 'hdf5':
        # Ensure output file has .h5 extension
        output_file = str(Path(args.output_matrix).with_suffix('.h5'))
        save_as_hdf5(genotypes, output_file, sample_ids, snp_ids)
    else:
        # Save as NPZ (sparse or dense)
        save_as_sparse(genotypes, args.output_matrix)
    
    # Extract and save phenotypes
    fam_file = f"{args.plink_prefix}.fam"
    phenotypes = extract_phenotypes(fam_file, args.external_pheno)
    phenotypes.to_csv(args.output_pheno, index=False)
    logger.info(f"Saved phenotypes: {args.output_pheno}")
    
    # Save feature names
    with open(args.output_features, 'w') as f:
        for snp_id in snp_ids:
            f.write(f"{snp_id}\n")
    logger.info(f"Saved {len(snp_ids)} feature names")
    
    # Save statistics
    with open(args.stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics: {args.stats_file}")
    
    print("\n" + "="*50)
    print("Conversion Summary")
    print("="*50)
    print(f"Samples: {stats['n_samples']}")
    print(f"SNPs: {stats['n_snps']}")
    print(f"Missing rate: {stats['missing_rate']*100:.2f}%")
    print(f"Matrix sparsity: {stats['sparsity']*100:.2f}%")
    print(f"Mean MAF: {stats['maf_mean']:.4f}")
    print("="*50)


if __name__ == '__main__':
    main()