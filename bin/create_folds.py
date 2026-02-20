import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import os
import argparse


def create_folds(pheno_file, n_folds, output_dir, random_state=42):    
    # Read the .pheno file
    pheno_df = pd.read_csv(pheno_file, sep='\t')
    
    print(pheno_df.head())

    # Create KFold object
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)   

    # Create output directory if it doesn't exist
    os.makedirs(f"{output_dir}/folds", exist_ok=True)
    
    # remove any existing fold files
    for file in os.listdir(f"{output_dir}/folds"):
        if file.startswith("fold_") and file.endswith(".txt"):
            os.remove(os.path.join(f"{output_dir}/folds", file))   
    
    # Generate stratified folds and save to separate files
    for fold, (train_index, test_index) in enumerate(kf.split(pheno_df['IID'], pheno_df['phenotype'])):
        fold_df = pheno_df.iloc[test_index]
        
        fold_df = fold_df[['FID', 'IID']]
        
        fold_df.to_csv(f"{output_dir}/folds/fold_{fold + 1}.txt", sep=' ', index=False)
        print(f"Fold {fold + 1} created with {len(fold_df)} samples.")
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Create K folds from .pheno file")
    parser.add_argument("--pheno_file", type=str, required=True, help="Path to .pheno file")
    parser.add_argument("--n_folds", type=int, required=True, help="Number of folds to create")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fold files")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    args = parser.parse_args()      
    
  
    create_folds(args.pheno_file, args.n_folds, args.output_dir, random_state=args.random_state)