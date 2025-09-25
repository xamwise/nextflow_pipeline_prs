MLBF-PRS:  MACHINE LEARNING MODEL DEVELOPMENT AND BENCHMARKING FRAMEWORK FOR POLYGENIC RISK SCORES

Here we present a state-of-the-art pipeline for predicting Polygenic Risk Scores (PRS) from genotype data using established PRS methods,
PyTorch, sklearn and Nextflow.
Features

Deep Learning Models: Multiple architectures (MLP, CNN, Transformer, Attention-based)
Machine Learning Models: SVM, XGBoost, Random Forests, Elasticnet etc.
PRS-Methods
Data Processing: Efficient handling of large-scale PLINK format genotype data
Memory Efficient: Lazy loading with HDF5 for datasets that don't fit in memory
Hyperparameter Optimization: Automated tuning using Optuna
K-fold Cross-validation: Robust model evaluation
Data Augmentation: SNP dropout, noise injection, and region shuffling
Comprehensive Logging: Integration with Weights & Biases
Visualization: Interactive reports and performance metrics

Directory Structure
nextflow_pipeline_prs/
├── LICENSE
├── README.md
├── bin
├── config
│   ├── params.yaml
│   └── training_config.yaml
├── data
│   ├── qc
│   ├── raw
│   ├── results
│   └── supplement_data
├── models
│   ├── attention_model.py
│   ├── bayesian_model.py
│   ├── cnn_model.py
│   ├── ensemble_model.py
│   ├── mlp_model.py
│   ├── models.py
│   └── transformer_model.py
├── modules
│   └── local
├── nextflow.config
├── out
├── requirements.txt
├── run_pipelines.sh
├── wandb
└── workflows
    ├── baselinePRS.nf
    ├── config
    ├── dl_prs.nf
    ├── main_integrated.nf
    ├── prs_models_pipeline.nf
    ├── qc_pipeline.nf
    ├── qc_test.nf
    ├── sklearn_pipeline.nf


Installation
Prerequisites

Nextflow (version 21.04+)
Python 3.8+
CUDA 11.0+ (for GPU support, optional)

R packages:
install.packages(c("optparse", "bigsnpr", "tidyr", "ggplot", "Matrix",
                   "data", "fmsb", "devtools", "tidyverse", "magrittr"))


Python Dependencies
Install required Python packages:
pip install -r requirements.txt
For GPU support with PyTorch:
bashpip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
For CPU only:
bashpip install torch torchvision torchaudio
Weights & Biases Setup
To use W&B logging (recommended):
bashwandb login

PRS methods:
Follow the intructions on the respective Git:
LDpred-2:   https://choishingwan.github.io/PRS-Tutorial/ldpred/
PRSice:     https://choishingwan.github.io/PRS-Tutorial/prsice/
lassosum:   https://choishingwan.github.io/PRS-Tutorial/lassosum/
lassosum2:  https://privefl.github.io/bigsnpr/reference/snp_lassosum2.html
PRSet:      https://choishingwan.github.io/PRSice/quick_start_prset/
PRScs:      https://github.com/getian107/PRScs
PRScsx:     https://github.com/getian107/PRScsx
SBayesRC:   https://github.com/zhilizheng/SBayesRC
Plink1.9:   https://www.cog-genomics.org/plink/


Usage
1. Prepare your data
Place your PLINK files in the data/ directory:

genotypes.bed - Binary genotype file
genotypes.bim - SNP information
genotypes.fam - Sample information

2. Configure the pipeline
Edit config/training_config.yaml to customize:

Model architectures
Training parameters
Hyperparameter search space
Data augmentation settings

3. Run the pipelines
Overall run

bash run_pipelines.h

Basic run:

nextflow run workflows/qc_pipeline.nf -params-file workflows/config/params_qc.yaml
nextflow run workflows/prs_models_pipeline.nf -params-file workflows/config/params_prs.yaml
nextflow run workflows/dl_prs.nf --params-file workflows/config/dl_config.yaml
nextflow run workflows/sklearn_pipeline.nf -params-file workflows/config/sklearn_config.yaml

Resume a previous run:
nextflow run main.nf -resume ...

4. Monitor training
Track experiments in Weights & Biases dashboard at https://wandb.ai

Output Files
Trained Models

results/models/fold_*/best_model_fold_*.pt - Best model checkpoints
results/models/fold_*/training_history_fold_*.json - Training metrics

Evaluation Reports

results/evaluation/final_evaluation_report.html - Interactive HTML report
results/evaluation/ensemble_predictions.csv - PRS predictions with risk categories
results/evaluation/model_comparison.json - Performance metrics

Visualizations

results/visualizations/data_overview.png - Data quality plots
results/visualizations/interactive_viz.html - Interactive visualizations

Performance Metrics
The pipeline evaluates models using PRS-specific metrics:

R² Score: Variance explained
AUC-ROC: Classification evaluation
Pearson/Spearman Correlation: Linear/rank correlation
Risk Stratification: Odds ratios, risk ratios
Decile Analysis: Performance across risk deciles

Running Individual Scripts
You can also run pipeline components individually:
Convert PLINK to HDF5
python bin/plink_converter.py \
    --plink_prefix data/genotypes \
    --output_h5 results/genotypes.h5 \
    --output_pheno results/phenotypes.csv \
    --stats_file results/stats.json
Split data
python bin/data_splitter.py \
    --genotype_file results/genotypes.h5 \
    --phenotype_file results/phenotypes.csv \
    --output_splits results/splits.json \
    --output_indices results/indices.npz
Train a model
python bin/train_model.py \
    --genotype_file results/genotypes.h5 \
    --phenotype_file results/phenotypes.csv \
    --indices_file results/indices.npz \
    --params_file config/training_config.yaml \
    --fold 0 \
    --output_model results/model.pt \
    --output_metrics results/metrics.json \
    --output_log results/log.csv


Troubleshooting
Memory Issues

Reduce batch_size in configuration
Increase cache_size in dataset settings
Use data subset for testing

GPU Issues

Check CUDA: python -c "import torch; print(torch.cuda.is_available())"
Run on CPU by setting device in config

Nextflow Issues

Clear cache: nextflow clean -f
Check logs: cat .nextflow.log
Update Nextflow: nextflow self-update

