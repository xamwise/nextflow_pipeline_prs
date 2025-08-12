Polygenic Risk Score Prediction Pipeline
A state-of-the-art deep learning pipeline for predicting Polygenic Risk Scores (PRS) from genotype data using PyTorch and Nextflow.
Features

Deep Learning Models: Multiple architectures (MLP, CNN, Transformer, Attention-based)
Data Processing: Efficient handling of large-scale PLINK format genotype data
Memory Efficient: Lazy loading with HDF5 for datasets that don't fit in memory
Hyperparameter Optimization: Automated tuning using Optuna
K-fold Cross-validation: Robust model evaluation
Data Augmentation: SNP dropout, noise injection, and region shuffling
Comprehensive Logging: Integration with Weights & Biases
Visualization: Interactive reports and performance metrics

Directory Structure
prs-pipeline/
├── main.nf                    # Main Nextflow pipeline
├── nextflow.config           # Nextflow configuration
├── requirements.txt          # Python dependencies
├── config/
│   └── training_config.yaml # Training configuration
├── bin/
│   ├── plink_converter.py   # PLINK to HDF5 conversion
│   ├── data_splitter.py     # Data splitting utilities
│   ├── data_visualizer.py   # Visualization tools
│   ├── genotype_dataset.py  # PyTorch dataset implementation
│   ├── models.py            # DL model architectures
│   ├── train_model.py       # Training script
│   ├── hyperparameter_optimizer.py # Hyperparameter tuning
│   └── evaluate_models.py   # Model evaluation
├── data/
│   └── genotypes.*         # Input PLINK files (.bed, .bim, .fam)
└── results/
    ├── processed_data/      # Converted HDF5 files
    ├── data_splits/         # Train/val/test indices
    ├── visualizations/      # Data visualizations
    ├── hyperparameter_search/ # Optuna results
    ├── models/              # Trained model checkpoints
    └── evaluation/          # Final evaluation reports
Installation
Prerequisites

Nextflow (version 21.04+)
Python 3.8+
CUDA 11.0+ (for GPU support, optional)

Python Dependencies
Install required Python packages:
bashpip install -r requirements.txt
For GPU support with PyTorch:
bashpip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
For CPU only:
bashpip install torch torchvision torchaudio
Weights & Biases Setup
To use W&B logging (recommended):
bashwandb login
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

3. Run the pipeline
Basic run:
bashnextflow run main.nf
With custom parameters:
bashnextflow run main.nf \
    --input_plink data/my_genotypes \
    --outdir results \
    --max_epochs 100 \
    --n_folds 5 \
    --n_trials 30 \
    --wandb_project my-prs-project
Resume a previous run:
bashnextflow run main.nf -resume
Run with specific profile:
bashnextflow run main.nf -profile gpu    # For GPU execution
nextflow run main.nf -profile slurm  # For SLURM cluster
4. Monitor training
Track experiments in Weights & Biases dashboard at https://wandb.ai
Pipeline Parameters
ParameterDefaultDescription--input_plinkdata/genotypesPLINK file prefix--outdirresultsOutput directory--n_folds5Number of CV folds--test_size0.2Test set proportion--val_size0.1Validation set proportion--max_epochs100Maximum training epochs--batch_size32Batch size--n_trials20Optuna trials--wandb_projectprs-predictionW&B project name--wandb_entitynullW&B entity/team name
Model Architectures
1. MLP (Multi-Layer Perceptron)

Fully connected layers with batch normalization
Configurable hidden dimensions and dropout
Suitable for smaller SNP sets

2. CNN (1D Convolutional Neural Network)

Treats SNPs as sequential data
Multiple convolutional blocks with pooling
Effective for capturing local SNP patterns

3. Transformer

Self-attention mechanism for SNP interactions
Positional encoding for SNP positions
Best for capturing long-range dependencies

4. Attention Model

Multi-head self-attention
Learns SNP importance weights
Good interpretability

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
Pearson/Spearman Correlation: Linear/rank correlation
Calibration Metrics: Slope, intercept, ECE
Risk Stratification: Odds ratios, risk ratios
Decile Analysis: Performance across risk deciles

Running Individual Scripts
You can also run pipeline components individually:
Convert PLINK to HDF5
pythonpython scripts/plink_converter.py \
    --plink_prefix data/genotypes \
    --output_h5 results/genotypes.h5 \
    --output_pheno results/phenotypes.csv \
    --stats_file results/stats.json
Split data
pythonpython scripts/data_splitter.py \
    --genotype_file results/genotypes.h5 \
    --phenotype_file results/phenotypes.csv \
    --output_splits results/splits.json \
    --output_indices results/indices.npz
Train a model
pythonpython scripts/train_model.py \
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

Citation
If you use this pipeline in your research, please cite:
bibtex@software{prs_pipeline,
  title={Deep Learning Pipeline for Polygenic Risk Score Prediction},
  author={Max Schuran},
  year={2024},
  url={https://github.com/yourusername/prs-pipeline}
}
