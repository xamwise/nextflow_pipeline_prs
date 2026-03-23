import numpy as np
import pandas as pd
from sklearn.base import accuracy_score
from sklearn.model_selection import StratifiedKFold
import os
import sys 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc, average_precision_score, balanced_accuracy_score, mean_absolute_percentage_error
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
    
    predictor_type = ''
    
    # Check if y is binary or continuous
    if set(np.unique(y)) <= {0, 1}:
        print("Phenotype is binary. Using logistic regression.")
        predictor_type = 'binary'
    elif set(np.unique(y)) <= {1, 2}:
        print("Phenotype is binary (coded as 1/2). Using logistic regression.")
        y = y - 1  # Convert to 0/1
        predictor_type = 'binary'
    else:
        print("Phenotype is continuous. Using linear regression.")  
        predictor_type = 'continuous'
    
    ids = pheno_df['IID'].values
    
    X_cov = cov_df.drop(columns=['FID', 'IID']).values
    
    X = np.hstack((X_cov, G.values))

    G_test, G_train, y_test, y_train = train_test_split(X.values, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {G_train.shape[0]} samples, {G_train.shape[1]} features")
    print("Starting univariate Lasso regression with cross-validation...")
    
    if predictor_type == 'binary':
        cv_fit = cv_unilasso(G_train, y_train, n_folds=n_folds, family='binomial')
        
    else:
    
        cv_fit = cv_unilasso(G_train, y_train, n_folds=n_folds, family='gaussian')
        
        
    extracted_fit = extract_cv(cv_fit)
    
    if predictor_type == 'binary':
        
        linear_preds = predict(extracted_fit, G_test)
        probabilities = 1 / (1 + np.exp(-linear_preds))
        classes = (probabilities >= 0.5).astype(int)
        
        print(f"ROC AUC: {roc_auc_score(y_test, probabilities):.4f}")
        print(f"Accuracy: {accuracy_score(y_test, classes):.4f}")
        print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, classes):.4f}")
        print(f"Average Precision: {average_precision_score(y_test, probabilities):.4f}")
        
    if predictor_type == 'continuous':
        
        predictions = predict(extracted_fit, G_test)
        print("Predictions:", predictions[:10])
        print("True values:", y_test[:10])
        r2 = np.corrcoef(predictions, y_test)[0, 1] ** 2
        print(f"R^2: {r2:.4f}")
        mape = mean_absolute_percentage_error(y_test, predictions)
        print(f"MAPE: {mape:.4f}")
    
    
    
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