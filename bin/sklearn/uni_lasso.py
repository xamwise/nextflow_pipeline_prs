import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
import sys 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from pandas_plink import read_plink1_bin
import glob
import argparse
from unilasso import *
from sklearn.model_selection import train_test_split



def main(data_prefix, pheno_file, cov_file, pcs_file, n_folds, out_dir):
    
    
    G = read_plink1_bin(os.path.join(data_prefix, '.bed'), os.path.join(data_prefix, '.bim'), os.path.join(data_prefix, '.fam'))
    
    
    # Load phenotype and covariate data
    pheno_df = pd.read_csv(pheno_file, sep='\t')
    cov_df = pd.read_csv(cov_file, sep='\t')
    
    y = pheno_df['phenotype'].values
    
    ids = pheno_df['IID'].values
    
    X_cov = cov_df.drop(columns=['FID', 'IID'])

    train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=42, stratify=y)

    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run univariate Lasso regression for PRS modeling.')
    parser.add_argument('--data_prefix', type=str, required=True, help='Directory containing the data files.')
    parser.add_argument('--pheno_file', type=str, required=True, help='Path to the phenotype file (tab-delimited).')
    parser.add_argument('--cov_file', type=str, required=False, help='Path to the covariate file (tab-delimited).')
    parser.add_argument('--pcs_file', type=str, required=False, help='Path to the principal components file (tab-delimited).')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation.')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save the results.')

    args = parser.parse_args()
    
    main(args.data_prefix, args.pheno_file, args.cov_file, args.pcs_file, args.n_folds, args.out_dir)