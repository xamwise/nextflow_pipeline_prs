"""
Example training script modifications for Bayesian Neural Network.

This shows how to properly train a BNN with ELBO loss and uncertainty evaluation.
Add these modifications to your train_model.py when using model_type='bayesian'.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
import wandb
from time import datetime

def train_epoch_bayesian(
    model,
    train_loader,
    optimizer,
    device,
    kl_weight=0.01
) -> Tuple[float, Dict]:
    """
    Training epoch for Bayesian Neural Network.
    
    Args:
        model: BayesianNeuralNetwork instance
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to use
        kl_weight: Weight for KL divergence term
        
    Returns:
        Tuple of (average loss, metrics dict)
    """
    model.train()
    total_loss = 0
    total_nll = 0
    total_kl = 0
    n_batches = len(train_loader)
    
    all_predictions = []
    all_targets = []
    
    for batch_idx, (genotypes, phenotypes) in enumerate(train_loader):
        genotypes = genotypes.to(device)
        phenotypes = phenotypes.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(genotypes)
        
        # Calculate ELBO loss
        # Negative log likelihood (MSE for continuous phenotypes)
        nll = nn.functional.mse_loss(predictions, phenotypes)
        
        # KL divergence
        kl_div = model.get_kl_divergence() / n_batches
        
        # Total loss
        loss = nll + kl_weight * kl_div
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        total_nll += nll.item()
        total_kl += kl_div.item()
        
        all_predictions.append(predictions.detach().cpu())
        all_targets.append(phenotypes.detach().cpu())
    
    # Calculate metrics
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    metrics = {
        'loss': total_loss / n_batches,
        'nll': total_nll / n_batches,
        'kl_div': total_kl / n_batches,
        'mse': nn.functional.mse_loss(all_predictions, all_targets).item(),
        'mae': nn.functional.l1_loss(all_predictions, all_targets).item()
    }
    
    return total_loss / n_batches, metrics


def evaluate_bayesian_with_uncertainty(
    model,
    val_loader,
    device,
    n_samples=10
) -> Dict:
    """
    Evaluate Bayesian model with uncertainty quantification.
    
    Args:
        model: BayesianNeuralNetwork instance
        val_loader: Validation data loader
        device: Device to use
        n_samples: Number of forward passes for uncertainty
        
    Returns:
        Dictionary of metrics including uncertainty measures
    """
    model.eval()
    
    all_means = []
    all_stds = []
    all_targets = []
    
    with torch.no_grad():
        for genotypes, phenotypes in val_loader:
            genotypes = genotypes.to(device)
            phenotypes = phenotypes.to(device)
            
            # Get predictions with uncertainty
            mean_pred, std_pred = model.predict_with_uncertainty(
                genotypes, 
                n_samples=n_samples
            )
            
            all_means.append(mean_pred.cpu())
            all_stds.append(std_pred.cpu())
            all_targets.append(phenotypes.cpu())
    
    # Concatenate all batches
    all_means = torch.cat(all_means)
    all_stds = torch.cat(all_stds)
    all_targets = torch.cat(all_targets)
    
    # Calculate metrics
    mse = nn.functional.mse_loss(all_means, all_targets).item()
    mae = nn.functional.l1_loss(all_means, all_targets).item()
    
    # R-squared
    ss_res = torch.sum((all_targets - all_means) ** 2)
    ss_tot = torch.sum((all_targets - all_targets.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot).item()
    
    # Uncertainty metrics
    mean_uncertainty = all_stds.mean().item()
    
    # Calibration: Check if uncertainty correlates with error
    errors = (all_means - all_targets).abs()
    uncertainty_error_correlation = torch.corrcoef(
        torch.stack([errors.flatten(), all_stds.flatten()])
    )[0, 1].item()
    
    # Coverage: What percentage of true values fall within prediction intervals
    z_score = 1.96  # 95% confidence interval
    lower_bound = all_means - z_score * all_stds
    upper_bound = all_means + z_score * all_stds
    coverage = ((all_targets >= lower_bound) & 
                (all_targets <= upper_bound)).float().mean().item()
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'mean_uncertainty': mean_uncertainty,
        'uncertainty_error_corr': uncertainty_error_correlation,
        'coverage_95': coverage,
        'mean_std': all_stds.mean().item(),
        'std_std': all_stds.std().item()
    }
    
    return metrics


def visualize_uncertainty(
    model,
    test_loader,
    device,
    save_path='uncertainty_plot.png'
):
    """
    Visualize predictions with uncertainty bounds.
    
    Args:
        model: BayesianNeuralNetwork instance
        test_loader: Test data loader
        device: Device to use
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    model.eval()
    
    # Get a batch of predictions
    for genotypes, phenotypes in test_loader:
        genotypes = genotypes.to(device)
        phenotypes = phenotypes.to(device)
        
        # Get predictions with uncertainty
        mean_pred, std_pred = model.predict_with_uncertainty(genotypes, n_samples=50)
        
        # Move to CPU
        mean_pred = mean_pred.cpu().numpy()
        std_pred = std_pred.cpu().numpy()
        phenotypes = phenotypes.cpu().numpy()
        
        # Plot first 50 samples
        n_plot = min(50, len(phenotypes))
        indices = np.arange(n_plot)
        
        plt.figure(figsize=(15, 6))
        
        # Plot predictions with uncertainty
        plt.errorbar(
            indices, 
            mean_pred[:n_plot, 0], 
            yerr=1.96 * std_pred[:n_plot, 0],  # 95% CI
            fmt='o', 
            alpha=0.6, 
            label='Predictions ± 95% CI',
            capsize=3
        )
        
        # Plot true values
        plt.scatter(
            indices, 
            phenotypes[:n_plot, 0], 
            color='red', 
            s=20, 
            alpha=0.8, 
            label='True values'
        )
        
        plt.xlabel('Sample Index')
        plt.ylabel('Phenotype Value')
        plt.title('Bayesian Neural Network Predictions with Uncertainty')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        break  # Only visualize first batch


def log_bayesian_metrics_wandb(metrics: Dict, prefix: str = 'val'):
    """
    Log Bayesian-specific metrics to Weights & Biases.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for metric names (e.g., 'train', 'val', 'test')
    """
    wandb.log({
        f'{prefix}/mse': metrics['mse'],
        f'{prefix}/mae': metrics['mae'],
        f'{prefix}/r2': metrics['r2'],
        f'{prefix}/mean_uncertainty': metrics.get('mean_uncertainty', 0),
        f'{prefix}/uncertainty_error_correlation': metrics.get('uncertainty_error_corr', 0),
        f'{prefix}/coverage_95': metrics.get('coverage_95', 0),
        f'{prefix}/nll': metrics.get('nll', 0),
        f'{prefix}/kl_divergence': metrics.get('kl_div', 0)
    })


# Example modification for the Trainer class in train_model.py
class BayesianTrainerMixin:
    """
    Mixin class with Bayesian-specific training methods.
    Add these methods to your Trainer class when using Bayesian models.
    """
    
    def train_epoch_bayesian(self) -> Tuple[float, Dict[str, float]]:
        """Train epoch for Bayesian model."""
        self.model.train()
        train_loader = self.data_module.train_dataloader(self.fold)
        
        total_loss = 0
        total_nll = 0
        total_kl = 0
        n_batches = len(train_loader)
        
        all_predictions = []
        all_targets = []
        
        for batch_idx, (genotypes, phenotypes) in enumerate(train_loader):
            genotypes = genotypes.to(self.device)
            phenotypes = phenotypes.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(genotypes)
            
            # ELBO loss
            nll = self.criterion(predictions, phenotypes)
            kl_div = self.model.get_kl_divergence() / n_batches
            kl_weight = self.config.get('kl_weight', 0.01)
            
            loss = nll + kl_weight * kl_div
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if 'gradient_clip' in self.config:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_nll += nll.item()
            total_kl += kl_div.item()
            
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(phenotypes.detach().cpu())
        
        # Calculate metrics
        avg_loss = total_loss / n_batches
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        metrics = self.calculate_metrics(all_predictions, all_targets)
        metrics['nll'] = total_nll / n_batches
        metrics['kl_div'] = total_kl / n_batches
        
        return avg_loss, metrics
    
    def validate_bayesian(self) -> Tuple[float, Dict[str, float]]:
        """Validate Bayesian model with uncertainty."""
        self.model.eval()
        val_loader = self.data_module.val_dataloader(self.fold)
        
        return evaluate_bayesian_with_uncertainty(
            self.model,
            val_loader,
            self.device,
            n_samples=self.config.get('n_samples', 10)
        )


# Example usage in training script
def main_bayesian_training():
    """Complete main function for training Bayesian model."""
    import argparse
    import yaml
    import json
    from pathlib import Path
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from bin.genotype_dataset import GenotypeDataModule
    from models.models import create_model
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Bayesian Neural Network for PRS')
    parser.add_argument('--genotype_file', type=str, required=True,
                        help='Path to HDF5 genotype file')
    parser.add_argument('--phenotype_file', type=str, required=True,
                        help='Path to phenotype CSV file')
    parser.add_argument('--indices_file', type=str, required=True,
                        help='Path to indices NPZ file')
    parser.add_argument('--params_file', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold number for cross-validation')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output_model', type=str, required=True,
                        help='Path to save model')
    parser.add_argument('--output_metrics', type=str, required=True,
                        help='Path to save metrics JSON')
    parser.add_argument('--output_log', type=str, required=True,
                        help='Path to save training log CSV')
    parser.add_argument('--wandb_project', type=str, default='prs-bayesian',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity name')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.params_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    config['max_epochs'] = args.max_epochs
    config['batch_size'] = args.batch_size
    
    # Ensure model type is Bayesian
    if config['model']['model_type'] not in ['bayesian', 'bnn']:
        print("Warning: Config doesn't specify Bayesian model. Setting to 'bayesian'")
        config['model']['model_type'] = 'bayesian'
    
    # Set default Bayesian parameters if not present
    if 'kl_weight' not in config['model']:
        config['model']['kl_weight'] = 0.01
    if 'n_samples' not in config['model']:
        config['model']['n_samples'] = 10
    if 'prior_var' not in config['model']:
        config['model']['prior_var'] = 1.0
    
    # Initialize W&B
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"bayesian_fold_{args.fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=config,
        tags=['bayesian', f'fold_{args.fold}', 'prs']
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data module
    data_module = GenotypeDataModule(
        h5_file=args.genotype_file,
        phenotype_file=args.phenotype_file,
        indices_file=args.indices_file,
        batch_size=args.batch_size,
        num_workers=0,  # HDF5 doesn't work well with multiprocessing
        augment_train=config.get('augment_train', False),
        augmentation_params=config.get('augmentation_params', {})
    )
    
    # Get data dimensions
    input_dim = data_module.get_input_dim()
    output_dim = data_module.get_output_dim()
    
    # Update config with dimensions
    config['model']['input_dim'] = input_dim
    config['model']['output_dim'] = output_dim
    
    print(f"Data dimensions - Input: {input_dim}, Output: {output_dim}")
    
    # Create model
    model = create_model(config['model'])
    model.to(device)
    
    # Log model to W&B
    wandb.watch(model, log='all', log_freq=100)
    
    # Create optimizer
    optimizer_config = config.get('optimizer', {'type': 'adam', 'lr': 0.001})
    if optimizer_config['type'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config.get('lr', 0.001),
            weight_decay=optimizer_config.get('weight_decay', 0.0)
        )
    elif optimizer_config['type'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config.get('lr', 0.001),
            weight_decay=optimizer_config.get('weight_decay', 0.01)
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optimizer_config.get('lr', 0.001),
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=optimizer_config.get('weight_decay', 0.0)
        )
    
    # Create learning rate scheduler
    scheduler = None
    if 'scheduler' in config:
        if config['scheduler']['type'] == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=config['scheduler'].get('patience', 10),
                factor=config['scheduler'].get('factor', 0.1)
            )
        elif config['scheduler']['type'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.max_epochs
            )
    
    # Get data loaders
    train_loader = data_module.train_dataloader(fold=args.fold)
    val_loader = data_module.val_dataloader(fold=args.fold)
    test_loader = data_module.test_dataloader()
    
    # Training history
    training_history = {
        'train_loss': [],
        'train_metrics': [],
        'val_metrics': [],
        'uncertainties': []
    }
    
    # CSV logging setup
    csv_log_path = Path(args.output_log)
    csv_data = []
    
    # Best model tracking
    best_val_loss = float('inf')
    best_val_r2 = -float('inf')
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = config.get('early_stopping', {}).get('patience', 20)
    
    # KL weight (can be annealed during training)
    kl_weight = config['model'].get('kl_weight', 0.01)
    kl_annealing = config['model'].get('kl_annealing', False)
    
    print(f"\nStarting Bayesian Neural Network training for fold {args.fold}")
    print(f"KL weight: {kl_weight}, Annealing: {kl_annealing}")
    print("="*50)
    
    # Training loop
    for epoch in range(args.max_epochs):
        # Optionally anneal KL weight
        if kl_annealing:
            current_kl_weight = kl_weight * min(1.0, epoch / 10)  # Linear annealing over 10 epochs
        else:
            current_kl_weight = kl_weight
        
        # Training phase
        train_loss, train_metrics = train_epoch_bayesian(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            kl_weight=current_kl_weight
        )
        
        # Validation phase with uncertainty
        val_metrics = evaluate_bayesian_with_uncertainty(
            model=model,
            val_loader=val_loader,
            device=device,
            n_samples=config['model'].get('n_samples', 10)
        )
        
        # Calculate validation loss for scheduler and early stopping
        val_loss = val_metrics['mse']
        
        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Log to W&B
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'train/nll': train_metrics['nll'],
            'train/kl_div': train_metrics['kl_div'],
            'train/mse': train_metrics['mse'],
            'train/mae': train_metrics['mae'],
            'val/mse': val_metrics['mse'],
            'val/mae': val_metrics['mae'],
            'val/r2': val_metrics['r2'],
            'val/mean_uncertainty': val_metrics['mean_uncertainty'],
            'val/uncertainty_error_corr': val_metrics['uncertainty_error_corr'],
            'val/coverage_95': val_metrics['coverage_95'],
            'learning_rate': current_lr,
            'kl_weight': current_kl_weight
        })
        
        # Console output
        print(f"Epoch {epoch+1}/{args.max_epochs}:")
        print(f"  Train Loss: {train_loss:.4f} (NLL: {train_metrics['nll']:.4f}, "
              f"KL: {train_metrics['kl_div']:.4f})")
        print(f"  Val MSE: {val_loss:.4f}, R²: {val_metrics['r2']:.4f}")
        print(f"  Uncertainty: {val_metrics['mean_uncertainty']:.4f} "
              f"(Corr: {val_metrics['uncertainty_error_corr']:.3f})")
        print(f"  95% Coverage: {val_metrics['coverage_95']:.2%}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save to CSV log
        csv_data.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_nll': train_metrics['nll'],
            'train_kl': train_metrics['kl_div'],
            'val_mse': val_loss,
            'val_r2': val_metrics['r2'],
            'val_uncertainty': val_metrics['mean_uncertainty'],
            'val_coverage': val_metrics['coverage_95'],
            'lr': current_lr
        })
        
        # Update training history
        training_history['train_loss'].append(train_loss)
        training_history['train_metrics'].append(train_metrics)
        training_history['val_metrics'].append(val_metrics)
        training_history['uncertainties'].append({
            'mean': val_metrics['mean_uncertainty'],
            'std': val_metrics.get('std_std', 0),
            'coverage': val_metrics['coverage_95']
        })
        
        # Check for best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_r2 = val_metrics['r2']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  ✓ New best model (Val MSE: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        # Periodic visualization
        if (epoch + 1) % 10 == 0:
            save_path = f'uncertainty_plot_epoch_{epoch+1}.png'
            visualize_uncertainty(model, val_loader, device, save_path)
            wandb.log({"uncertainty_plot": wandb.Image(save_path)})
        
        print("-" * 50)
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best validation MSE: {best_val_loss:.4f}")
    print(f"Best validation R²: {best_val_r2:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final test evaluation with uncertainty
    print("\nEvaluating on test set...")
    test_metrics = evaluate_bayesian_with_uncertainty(
        model=model,
        val_loader=test_loader,
        device=device,
        n_samples=config['model'].get('n_samples', 10) * 2  # More samples for final evaluation
    )
    
    print(f"Test MSE: {test_metrics['mse']:.4f}")
    print(f"Test R²: {test_metrics['r2']:.4f}")
    print(f"Test Uncertainty: {test_metrics['mean_uncertainty']:.4f}")
    print(f"Test 95% Coverage: {test_metrics['coverage_95']:.2%}")
    
    # Log test results to W&B
    wandb.log({
        'test/mse': test_metrics['mse'],
        'test/mae': test_metrics['mae'],
        'test/r2': test_metrics['r2'],
        'test/mean_uncertainty': test_metrics['mean_uncertainty'],
        'test/coverage_95': test_metrics['coverage_95']
    })
    
    # Create final uncertainty visualization
    visualize_uncertainty(model, test_loader, device, 'final_uncertainty_plot.png')
    wandb.log({"final_uncertainty_plot": wandb.Image('final_uncertainty_plot.png')})
    
    # Save model
    output_model_path = Path(args.output_model)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'best_val_r2': best_val_r2,
        'config': config,
        'training_history': training_history,
        'test_metrics': test_metrics,
        'weight_statistics': model.get_weight_statistics()
    }
    
    torch.save(checkpoint, output_model_path)
    print(f"\nModel saved to {output_model_path}")
    
    # Save metrics
    metrics_summary = {
        'best_val_loss': float(best_val_loss),
        'best_val_r2': float(best_val_r2),
        'test_metrics': test_metrics,
        'final_epoch': epoch,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'weight_statistics': model.get_weight_statistics(),
        'uncertainty_calibration': {
            'mean_uncertainty': test_metrics['mean_uncertainty'],
            'uncertainty_error_correlation': test_metrics['uncertainty_error_corr'],
            'coverage_95': test_metrics['coverage_95']
        }
    }
    
    with open(args.output_metrics, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Metrics saved to {args.output_metrics}")
    
    # Save CSV log
    import pandas as pd
    df_log = pd.DataFrame(csv_data)
    df_log.to_csv(args.output_log, index=False)
    print(f"Training log saved to {args.output_log}")
    
    # Finish W&B run
    wandb.finish()
    
    return model, test_metrics


if __name__ == '__main__':
    # Run the Bayesian training
    model, metrics = main_bayesian_training()
    print("\nBayesian Neural Network training completed successfully!")