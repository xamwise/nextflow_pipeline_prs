import pandas as pd
import os
import argparse
import sys


def align_pheno(pheno_file, fam_file, out_file):
    """
    Write the .pheno file with exactly one row per sample of a PLINK .fam file, in .fam order.

    SCT.R uses the phenotype vector positionally and lassosum.R labels its PRS with the IDs of
    the .pheno file, so both silently go wrong if the .pheno is not in .fam order.

    Samples are matched on IID (like create_folds.py and the notebooks); FID and IID are taken
    from the .fam. Samples of the .fam without a phenotype get NA, samples that are not in the
    .fam are dropped. All other columns are copied unchanged.
    """
    pheno_df = pd.read_csv(pheno_file, sep=r'\s+', dtype=str, keep_default_na=False)

    if not {'FID', 'IID'}.issubset(pheno_df.columns):
        raise ValueError("The .pheno file must have a header with 'FID' and 'IID' columns.")

    fam_df = pd.read_csv(fam_file, sep=r'\s+', header=None, usecols=[0, 1], dtype=str)
    fam_df.columns = ['FID', 'IID']

    n_dup = pheno_df['IID'].duplicated().sum()
    if n_dup:
        print(f"WARNING: {n_dup} duplicated IIDs in {pheno_file}, keeping the first row of each.", file=sys.stderr)
        pheno_df = pheno_df.drop_duplicates('IID')

    aligned = fam_df.merge(pheno_df.drop(columns='FID'), on='IID', how='left')
    aligned = aligned[pheno_df.columns]

    n_matched = fam_df['IID'].isin(pheno_df['IID']).sum()
    n_missing = len(fam_df) - n_matched
    n_dropped = (~pheno_df['IID'].isin(fam_df['IID'])).sum()

    if n_matched == 0:
        raise ValueError(f"No IIDs of {fam_file} found in {pheno_file}.")

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    aligned.to_csv(out_file, sep='\t', index=False, na_rep='NA')

    print(f"{out_file}: {len(aligned)} samples in .fam order, {n_matched} with a phenotype.")
    if n_missing:
        print(f"WARNING: {n_missing} samples of {fam_file} are not in {pheno_file} and get NA.", file=sys.stderr)
    if n_dropped:
        print(f"WARNING: {n_dropped} samples of {pheno_file} are not in {fam_file} and were dropped.", file=sys.stderr)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Align a .pheno file to the samples and sample order of a .fam file")
    parser.add_argument("--pheno_file", type=str, required=True, help="Path to .pheno file")
    parser.add_argument("--fam_file", type=str, required=True, help="Path to the .fam file of the QC'd genotypes")
    parser.add_argument("--out", type=str, required=True, help="Output .pheno file")
    args = parser.parse_args()

    align_pheno(args.pheno_file, args.fam_file, args.out)
