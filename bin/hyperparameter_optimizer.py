import optuna
from optuna import Trial
from optuna.visualization import plot_optimization_history, plot_param_importances
import torch
import torch.nn as nn
import numpy as np
import yaml
import json
import argparse
from pathlib import Path
import logging
from typing import Dict, Any
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bin.genotype_dataset import GenotypeDataModule
from models.models import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna.
    """
    
    def __init__(
        self,
        data_module: GenotypeDataModule,
        base_config: Dict[str, Any],
        n_trials: int = 20,
        study_name: str = "genotype_model_optimization"
    ):
        """
        Initialize the optimizer.
        
        Args:
            data_module: Data module for loading data
            base_config: Base configuration
            n_trials: Number of optimization trials
            study_name: Name of the Optuna study
        """
        self.data_module = data_module
        self.base_config = base_config
        self.n_trials = n_trials
        self.study_name = study_name
        
        # Get data dimensions
        self.input_dim = data_module.get_input_dim()
        self.output_dim = data_module.get_output_dim()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def objective(self, trial: Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation loss to minimize
        """
        # Sample hyperparameters
        params = self.sample_hyperparameters(trial)
        
        # Create model
        model_config = params['model']
        model_config['input_dim'] = self.input_dim
        model_config['output_dim'] = self.output_dim
        model = create_model(model_config).to(self.device)
        
        # Create optimizer
        optimizer = self.create_optimizer(model, params['optimizer'])
        
        # Create loss function
        criterion = nn.MSELoss()
        
        # Training parameters
        n_epochs = params['training']['n_epochs']
        early_stopping_patience = params['training']['early_stopping_patience']
        
        # Get data loaders (use fold 0 for optimization)
        train_loader = self.data_module.train_dataloader(fold=0)
        val_loader = self.data_module.val_dataloader(fold=0)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_idx, (genotypes, phenotypes) in enumerate(train_loader):
                genotypes = genotypes.to(self.device)
                phenotypes = phenotypes.to(self.device)
                
                optimizer.zero_grad()
                predictions = model(genotypes)
                loss = criterion(predictions, phenotypes)
                loss.backward()
                
                # Gradient clipping
                if 'gradient_clip' in params['training']:
                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        params['training']['gradient_clip']
                    )
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for genotypes, phenotypes in val_loader:
                    genotypes = genotypes.to(self.device)
                    phenotypes = phenotypes.to(self.device)
                    
                    predictions = model(genotypes)
                    loss = criterion(predictions, phenotypes)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                break
            
            # Report intermediate value for pruning
            trial.report(avg_val_loss, epoch)
            
            # Prune trial if needed
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_val_loss
    
    def sample_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """
        Sample hyperparameters for the trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of sampled hyperparameters
        """
        params = {}
        
        # Model architecture parameters
        model_type = trial.suggest_categorical('model_type', ['mlp', 'cnn', 'transformer', 'attention'])
        
        if model_type == 'mlp':
            params['model'] = {
                'model_type': 'mlp',
                'hidden_dims': [
                    trial.suggest_int('mlp_hidden_1', 128, 2048, step=128),
                    trial.suggest_int('mlp_hidden_2', 64, 1024, step=64),
                    trial.suggest_int('mlp_hidden_3', 32, 512, step=32)
                ],
                'dropout_rate': trial.suggest_float('mlp_dropout', 0.1, 0.5),
                'activation': trial.suggest_categorical('mlp_activation', ['relu', 'elu', 'leaky_relu']),
                'batch_norm': trial.suggest_categorical('mlp_batch_norm', [True, False])
            }
        
        elif model_type == 'cnn':
            params['model'] = {
                'model_type': 'cnn',
                'channels': [
                    trial.suggest_int('cnn_channels_1', 16, 64, step=16),
                    trial.suggest_int('cnn_channels_2', 32, 128, step=32),
                    trial.suggest_int('cnn_channels_3', 64, 256, step=64)
                ],
                'kernel_sizes': [
                    trial.suggest_int('cnn_kernel_1', 3, 11, step=2),
                    trial.suggest_int('cnn_kernel_2', 3, 9, step=2),
                    trial.suggest_int('cnn_kernel_3', 3, 7, step=2)
                ],
                'dropout_rate': trial.suggest_float('cnn_dropout', 0.1, 0.5)
            }
        
        elif model_type == 'transformer':
            params['model'] = {
                'model_type': 'transformer',
                'd_model': trial.suggest_int('transformer_d_model', 64, 512, step=64),
                'n_heads': trial.suggest_int('transformer_n_heads', 2, 16, step=2),
                'n_layers': trial.suggest_int('transformer_n_layers', 2, 8),
                'd_ff': trial.suggest_int('transformer_d_ff', 256, 2048, step=256),
                'dropout_rate': trial.suggest_float('transformer_dropout', 0.0, 0.3)
            }
        
        else:  # attention
            params['model'] = {
                'model_type': 'attention',
                'hidden_dim': trial.suggest_int('attention_hidden', 64, 512, step=64),
                'attention_dim': trial.suggest_int('attention_dim', 32, 256, step=32),
                'n_attention_heads': trial.suggest_int('attention_heads', 2, 8),
                'dropout_rate': trial.suggest_float('attention_dropout', 0.1, 0.5)
            }
        
        # Optimizer parameters
        optimizer_type = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
        params['optimizer'] = {
            'type': optimizer_type,
            'lr': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        }
        
        if optimizer_type == 'sgd':
            params['optimizer']['momentum'] = trial.suggest_float('momentum', 0.5, 0.99)
        
        # Training parameters
        params['training'] = {
            'n_epochs': trial.suggest_int('n_epochs', 20, 100, step=10),
            'early_stopping_patience': trial.suggest_int('early_stopping_patience', 5, 20),
            'gradient_clip': trial.suggest_float('gradient_clip', 0.5, 5.0)
        }
        
        # Data augmentation parameters
        params['augmentation'] = {
            'snp_dropout': trial.suggest_float('snp_dropout', 0.0, 0.2),
            'noise_std': trial.suggest_float('noise_std', 0.0, 0.1)
        }
        
        return params
    
    def create_optimizer(self, model: nn.Module, opt_config: Dict[str, Any]) -> torch.optim.Optimizer:
        """Create optimizer from configuration."""
        opt_type = opt_config['type'].lower()
        
        if opt_type == 'adam':
            return torch.optim.Adam(
                model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_type == 'adamw':
            return torch.optim.AdamW(
                model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_type == 'sgd':
            return torch.optim.SGD(
                model.parameters(),
                lr=opt_config['lr'],
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Returns:
            Best parameters found
        """
        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            )
        )
        
        # Optimize
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best trial value: {best_value}")
        logger.info(f"Best parameters: {best_params}")
        
        # Convert to configuration format
        best_config = self.params_to_config(best_params)
        
        return {
            'best_params': best_params,
            'best_config': best_config,
            'best_value': best_value,
            'study': study
        }
    
    def params_to_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Optuna parameters to configuration format.
        
        Args:
            params: Optuna parameters
            
        Returns:
            Configuration dictionary
        """
        config = self.base_config.copy()
        
        # Model configuration
        model_type = params['model_type']
        config['model'] = {'model_type': model_type}
        
        if model_type == 'mlp':
            config['model'].update({
                'hidden_dims': [params['mlp_hidden_1'], params['mlp_hidden_2'], params['mlp_hidden_3']],
                'dropout_rate': params['mlp_dropout'],
                'activation': params['mlp_activation'],
                'batch_norm': params['mlp_batch_norm']
            })
        elif model_type == 'cnn':
            config['model'].update({
                'channels': [params['cnn_channels_1'], params['cnn_channels_2'], params['cnn_channels_3']],
                'kernel_sizes': [params['cnn_kernel_1'], params['cnn_kernel_2'], params['cnn_kernel_3']],
                'pool_sizes': [2, 2, 2],
                'dropout_rate': params['cnn_dropout']
            })
        elif model_type == 'transformer':
            config['model'].update({
                'd_model': params['transformer_d_model'],
                'n_heads': params['transformer_n_heads'],
                'n_layers': params['transformer_n_layers'],
                'd_ff': params['transformer_d_ff'],
                'dropout_rate': params['transformer_dropout']
            })
        else:  # attention
            config['model'].update({
                'hidden_dim': params['attention_hidden'],
                'attention_dim': params['attention_dim'],
                'n_attention_heads': params['attention_heads'],
                'dropout_rate': params['attention_dropout']
            })
        
        # Optimizer configuration
        config['optimizer'] = {
            'type': params['optimizer'],
            'lr': params['learning_rate'],
            'weight_decay': params['weight_decay']
        }
        if params['optimizer'] == 'sgd':
            config['optimizer']['momentum'] = params['momentum']
        
        # Training configuration
        config['max_epochs'] = params['n_epochs']
        config['early_stopping'] = {'patience': params['early_stopping_patience']}
        config['gradient_clip'] = params['gradient_clip']
        
        # Augmentation configuration
        config['augmentation_params'] = {
            'snp_dropout': params['snp_dropout'],
            'noise_std': params['noise_std']
        }
        
        return config
    
    def generate_report(self, study: optuna.Study, output_path: str):
        """
        Generate HTML report with optimization results.
        
        Args:
            study: Optuna study object
            output_path: Path to save HTML report
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Optimization History', 'Parameter Importances',
                          'Parallel Coordinate Plot', 'Slice Plot')
        )
        
        # Optimization history
        trials = study.trials
        x = [t.number for t in trials]
        y = [t.value for t in trials if t.value is not None]
        
        fig.add_trace(
            go.Scatter(x=x[:len(y)], y=y, mode='lines+markers', name='Objective Value'),
            row=1, col=1
        )
        
        # Parameter importances (simplified visualization)
        params = list(study.best_params.keys())
        importances = [1.0 / len(params)] * len(params)  # Simplified
        
        fig.add_trace(
            go.Bar(x=params[:5], y=importances[:5], name='Importance'),
            row=1, col=2
        )
        
        # Best parameters as table
        param_text = "<br>".join([f"{k}: {v}" for k, v in study.best_params.items()])
        
        fig.add_annotation(
            text=f"Best Parameters:<br>{param_text}",
            xref="paper", yref="paper",
            x=0.5, y=-0.1,
            showarrow=False,
            font=dict(size=10)
        )
        
        # Update layout
        fig.update_layout(
            title=f"Hyperparameter Optimization Report - {study.study_name}",
            height=800,
            showlegend=True
        )
        
        # Save to HTML
        fig.write_html(output_path)
        logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for genotype models')
    parser.add_argument('--genotype_file', type=str, required=True)
    parser.add_argument('--phenotype_file', type=str, required=True)
    parser.add_argument('--splits_file', type=str, required=True)
    parser.add_argument('--indices_file', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--output_params', type=str, required=True)
    parser.add_argument('--study_db', type=str, required=True)
    parser.add_argument('--report', type=str, required=True)
    
    args = parser.parse_args()
    
    # Load base configuration
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create data module
    data_module = GenotypeDataModule(
        h5_file=args.genotype_file,
        phenotype_file=args.phenotype_file,
        indices_file=args.indices_file,
        batch_size=base_config.get('batch_size', 32),
        num_workers=base_config.get('num_workers', 4)
    )
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        data_module=data_module,
        base_config=base_config,
        n_trials=args.n_trials,
        study_name="genotype_optimization"
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    # Save best parameters
    with open(args.output_params, 'w') as f:
        yaml.dump(results['best_config'], f, default_flow_style=False)
    
    # Save study
    import joblib
    joblib.dump(results['study'], args.study_db)
    
    # Generate report
    optimizer.generate_report(results['study'], args.report)
    
    logger.info("Hyperparameter optimization completed successfully")


if __name__ == '__main__':
    main()