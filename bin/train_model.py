import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import yaml
import argparse
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import wandb
from tqdm import tqdm
from scipy import stats
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bin.genotype_dataset import GenotypeDataModule
from models.models import create_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for genotype models with comprehensive logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        data_module: GenotypeDataModule,
        config: Dict[str, Any],
        fold: int,
        output_dir: Path,
        use_wandb: bool = True
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            data_module: Data module with data loaders
            config: Training configuration
            fold: Current fold number
            output_dir: Directory for outputs
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model
        self.data_module = data_module
        self.config = config
        self.fold = fold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Loss function
        self.criterion = self._create_loss_function()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup logging with W&B
        self.setup_logging(use_wandb)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_r2 = -float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        opt_config = self.config['optimizer']
        opt_type = opt_config['type'].lower()
        
        params = self.model.parameters()
        
        if opt_type == 'adam':
            return optim.Adam(
                params,
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0)
            )
        elif opt_type == 'adamw':
            return optim.AdamW(
                params,
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        elif opt_type == 'sgd':
            return optim.SGD(
                params,
                lr=opt_config['lr'],
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration."""
        loss_type = self.config.get('loss_function', 'mse').lower()
        
        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'mae':
            return nn.L1Loss()
        elif loss_type == 'huber':
            return nn.HuberLoss()
        elif loss_type == 'crossentropy':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")
    
    def _create_scheduler(self) -> Optional[object]:
        """Create learning rate scheduler."""
        if 'scheduler' not in self.config:
            return None
        
        sched_config = self.config['scheduler']
        sched_type = sched_config['type'].lower()
        
        if sched_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config['gamma']
            )
        elif sched_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['max_epochs']
            )
        elif sched_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=sched_config.get('patience', 10),
                factor=sched_config.get('factor', 0.1)
            )
        else:
            return None
    
    def setup_logging(self, use_wandb: bool):
        """Setup logging with W&B only."""
        # Weights & Biases
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=self.config.get('wandb_project', 'prs-prediction'),
                entity=self.config.get('wandb_entity'),
                name=f"fold_{self.fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config,
                reinit=True,
                tags=["training", f"fold_{self.fold}", "prs"]
            )
            wandb.watch(self.model, log='all', log_freq=100)
        
        # CSV logger for backup
        self.csv_log_path = self.output_dir / f'training_log_fold_{self.fold}.csv'
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        self.model.train()
        train_loader = self.data_module.train_dataloader(self.fold)
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc=f'Training Epoch {self.current_epoch}')
        for batch_idx, (genotypes, phenotypes) in enumerate(pbar):
            genotypes = genotypes.to(self.device)
            phenotypes = phenotypes.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(genotypes)
            loss = self.criterion(predictions, phenotypes)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if 'gradient_clip' in self.config:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(phenotypes.detach().cpu())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to W&B
            if self.use_wandb and batch_idx % 10 == 0:
                global_step = self.current_epoch * len(train_loader) + batch_idx
                wandb.log({'train/batch_loss': loss.item(), 'global_step': global_step})
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        metrics = self.calculate_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        self.model.eval()
        val_loader = self.data_module.val_dataloader(self.fold)
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for genotypes, phenotypes in tqdm(val_loader, desc='Validation'):
                genotypes = genotypes.to(self.device)
                phenotypes = phenotypes.to(self.device)
                
                predictions = self.model(genotypes)
                loss = self.criterion(predictions, phenotypes)
                
                total_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_targets.append(phenotypes.cpu())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        metrics = self.calculate_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def calculate_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics for PRS.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            Dictionary of metrics
        """
        predictions = predictions.numpy()
        targets = targets.numpy()
        
        metrics = {}
        
        # Mean Squared Error
        mse = np.mean((predictions - targets) ** 2)
        metrics['mse'] = float(mse)
        metrics['rmse'] = float(np.sqrt(mse))
        
        # Mean Absolute Error
        mae = np.mean(np.abs(predictions - targets))
        metrics['mae'] = float(mae)
        
        # R-squared (important for PRS)
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        metrics['r2'] = float(r2)
        
        # Pearson correlation (critical for PRS)
        if predictions.shape[1] == 1:
            from scipy.stats import pearsonr, spearmanr
            corr, p_val = pearsonr(predictions.flatten(), targets.flatten())
            metrics['pearson_r'] = float(corr)
            metrics['pearson_p'] = float(p_val)
            
            # Spearman correlation
            rho, p_val = spearmanr(predictions.flatten(), targets.flatten())
            metrics['spearman_r'] = float(rho)
            metrics['spearman_p'] = float(p_val)
        
        # Variance explained
        metrics['variance_explained'] = float(
            1 - np.var(targets - predictions) / (np.var(targets) + 1e-8)
        )
        
        return metrics
    
    def train(self, max_epochs: int):
        """
        Train the model for multiple epochs.
        
        Args:
            max_epochs: Maximum number of epochs to train
        """
        logger.info(f"Starting training for fold {self.fold}")
        
        for epoch in range(max_epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss, train_metrics = self.train_epoch()
            
            # Validation
            val_loss, val_metrics = self.validate()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Logging
            self.log_epoch(train_loss, train_metrics, val_loss, val_metrics)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            
            # Also track best R2
            if val_metrics.get('r2', 0) > self.best_val_r2:
                self.best_val_r2 = val_metrics['r2']
                self.save_checkpoint(is_best=True, metric='r2')
                logger.info(f"New best model saved with validation R²: {self.best_val_r2:.4f}")
            
            # Early stopping
            if self.check_early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Save final model
        self.save_checkpoint(is_best=False, final=True)
        
        # Save training history
        self.save_training_history()
        
        # Finish W&B run
        if self.use_wandb:
            wandb.finish()
        
        logger.info(f"Training completed for fold {self.fold}")
    
    def log_epoch(
        self,
        train_loss: float,
        train_metrics: Dict[str, float],
        val_loss: float,
        val_metrics: Dict[str, float]
    ):
        """Log epoch results to W&B and CSV."""
        # Console logging
        logger.info(
            f"Epoch {self.current_epoch}: "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Val R²: {val_metrics.get('r2', 0):.4f}, "
            f"Val Pearson r: {val_metrics.get('pearson_r', 0):.4f}"
        )
        
        # Weights & Biases
        if self.use_wandb:
            wandb.log({
                'epoch': self.current_epoch,
                'train/loss': train_loss,
                'val/loss': val_loss,
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()},
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
        
        # CSV logging
        log_data = {
            'epoch': self.current_epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()}
        }
        
        df = pd.DataFrame([log_data])
        if self.csv_log_path.exists():
            df.to_csv(self.csv_log_path, mode='a', header=False, index=False)
        else:
            df.to_csv(self.csv_log_path, index=False)
        
        # Update training history
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['train_metrics'].append(train_metrics)
        self.training_history['val_metrics'].append(val_metrics)
    
    def check_early_stopping(self, val_loss: float, patience: int = 20) -> bool:
        """
        Check if early stopping should be triggered.
        
        Args:
            val_loss: Current validation loss
            patience: Number of epochs to wait
            
        Returns:
            True if training should stop
        """
        if 'early_stopping' not in self.config:
            return False
        
        if not hasattr(self, 'patience_counter'):
            self.patience_counter = 0
        
        if val_loss >= self.best_val_loss:
            self.patience_counter += 1
        else:
            self.patience_counter = 0
        
        return self.patience_counter >= patience
    
    def save_checkpoint(self, is_best: bool = False, final: bool = False, metric: str = 'loss'):
        """Save model checkpoint."""
        if is_best:
            if metric == 'r2':
                path = self.output_dir / f'best_model_r2_fold_{self.fold}.pt'
            else:
                path = self.output_dir / f'best_model_fold_{self.fold}.pt'
        elif final:
            path = self.output_dir / f'final_model_fold_{self.fold}.pt'
        else:
            path = self.output_dir / f'checkpoint_fold_{self.fold}_epoch_{self.current_epoch}.pt'
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_r2': self.best_val_r2,
            'config': self.config,
            'training_history': self.training_history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def save_training_history(self):
        """Save training history to JSON file."""
        history_path = self.output_dir / f'training_history_fold_{self.fold}.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Also save metrics summary
        metrics_summary = {
            'best_val_loss': float(self.best_val_loss),
            'best_val_r2': float(self.best_val_r2),
            'final_train_loss': float(self.training_history['train_loss'][-1]),
            'final_val_loss': float(self.training_history['val_loss'][-1]),
            'final_val_metrics': self.training_history['val_metrics'][-1],
            'total_epochs': len(self.training_history['train_loss'])
        }
        
        summary_path = self.output_dir / f'metrics_fold_{self.fold}.json'
        with open(summary_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train genotype deep learning model for PRS prediction')
    parser.add_argument('--genotype_file', type=str, required=True)
    parser.add_argument('--phenotype_file', type=str, required=True)
    parser.add_argument('--indices_file', type=str, required=True)
    parser.add_argument('--params_file', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_model', type=str, required=True)
    parser.add_argument('--output_metrics', type=str, required=True)
    parser.add_argument('--output_log', type=str, required=True)
    parser.add_argument('--wandb_project', type=str, default='prs-prediction')
    parser.add_argument('--wandb_entity', type=str, default=None)
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.params_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command line arguments
    config['max_epochs'] = args.max_epochs
    config['batch_size'] = args.batch_size
    config['use_wandb'] = True
    config['wandb_project'] = args.wandb_project
    config['wandb_entity'] = args.wandb_entity
    
    # Create data module
    data_module = GenotypeDataModule(
        h5_file=args.genotype_file,
        phenotype_file=args.phenotype_file,
        indices_file=args.indices_file,
        batch_size=args.batch_size,
        num_workers=config.get('num_workers', 4),
        augment_train=config.get('augment_train', True),
        augmentation_params=config.get('augmentation_params', {})
    )
    
    # Get dimensions
    input_dim = data_module.get_input_dim()
    output_dim = data_module.get_output_dim()
    
    # Create model
    model_config = config['model']
    model_config['input_dim'] = input_dim
    model_config['output_dim'] = output_dim
    model = create_model(model_config)
    
    # Create output directory
    output_dir = Path(args.output_model).parent
    
    # Create trainer
    trainer = Trainer(
        model=model,
        data_module=data_module,
        config=config,
        fold=args.fold,
        output_dir=output_dir,
        use_wandb=config.get('use_wandb', True)
    )
    
    # Train model
    trainer.train(args.max_epochs)
    
    # Copy final files to expected locations
    import shutil
    best_model_path = output_dir / f'best_model_fold_{args.fold}.pt'
    if best_model_path.exists():
        shutil.copy(best_model_path, args.output_model)
    
    metrics_path = output_dir / f'metrics_fold_{args.fold}.json'
    if metrics_path.exists():
        shutil.copy(metrics_path, args.output_metrics)
    
    log_path = output_dir / f'training_log_fold_{args.fold}.csv'
    if log_path.exists():
        shutil.copy(log_path, args.output_log)


if __name__ == '__main__':
    main()