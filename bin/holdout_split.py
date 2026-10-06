import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse


def holdout_split(pheno_file, fam_file, test_size, output_dir, random_state=42):
    """
    Split the samples into a hold-out test set and a development set (used for k-fold CV).

    Only samples that are in the genotype .fam file and have a non-missing phenotype are used.
    Samples are matched on IID, like the fold files written by create_folds.py.

    Writes to output_dir:
        test_ids.txt  FID IID of the hold-out test samples
        dev_ids.txt   FID IID of the development samples
        dev.pheno     .pheno file restricted to the development samples (input for create_folds.py)
    """
    pheno_df = pd.read_csv(pheno_file, sep=r'\s+', dtype={'FID': str, 'IID': str})

    if 'phenotype' not in pheno_df.columns:
        raise ValueError("The .pheno file must contain a 'phenotype' column for stratification.")

    fam_iids = set(pd.read_csv(fam_file, sep=r'\s+', header=None, usecols=[1], dtype=str)[1])
    n_pheno = len(pheno_df)
    pheno_df = pheno_df[pheno_df['IID'].isin(fam_iids) & pheno_df['phenotype'].notna()]
    print(f"{len(pheno_df)} of {n_pheno} samples in {pheno_file} are genotyped and have a phenotype.")

    # Check if 'phenotype' column is binary for stratification
    if pheno_df['phenotype'].nunique() == 2:
        print("Using a stratified split for binary phenotype.")
        stratify = pheno_df['phenotype']
    else:
        print("Phenotype is not binary. Using a random split without stratification.")
        stratify = None

    dev_df, test_df = train_test_split(
        pheno_df, test_size=test_size, shuffle=True, stratify=stratify, random_state=random_state
    )
    # Keep the original sample order in the output files
    dev_df = dev_df.sort_index()
    test_df = test_df.sort_index()

    os.makedirs(output_dir, exist_ok=True)

    test_df[['FID', 'IID']].to_csv(f"{output_dir}/test_ids.txt", sep=' ', index=False)
    dev_df[['FID', 'IID']].to_csv(f"{output_dir}/dev_ids.txt", sep=' ', index=False)
    dev_df.to_csv(f"{output_dir}/dev.pheno", sep='\t', index=False)

    print(f"Hold-out test set created with {len(test_df)} samples.")
    print(f"Development set created with {len(dev_df)} samples.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Split a .pheno file into a hold-out test set and a development set")
    parser.add_argument("--pheno_file", type=str, required=True, help="Path to .pheno file")
    parser.add_argument("--fam_file", type=str, required=True, help="Path to the .fam file of the QC'd genotypes")
    parser.add_argument("--test_size", type=float, default=0.1, help="Fraction of samples in the hold-out test set")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the split files")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    args = parser.parse_args()

    holdout_split(args.pheno_file, args.fam_file, args.test_size, args.output_dir, random_state=args.random_state)
