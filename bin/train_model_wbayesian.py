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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report


from nagelkerke_r2 import calculate_nagelkerke_r2

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
    Supports standard and Bayesian neural networks.
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
        self.model = model
        self.data_module = data_module
        self.config = config
        self.fold = fold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect if model is Bayesian
        self.is_bayesian = self._check_bayesian()
        if self.is_bayesian:
            logger.info("Bayesian model detected — using ELBO training loop")
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Loss function (not used directly for Bayesian, but kept for validation reference)
        self.criterion = self._create_loss_function()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Bayesian-specific config
        if self.is_bayesian:
            model_cfg = self.config.get('model', {})
            self.kl_weight = model_cfg.get('kl_weight', 0.01)
            self.kl_warmup_epochs = model_cfg.get('kl_warmup_epochs', 10)
            self.n_val_samples = model_cfg.get('n_val_samples', 30)
            self.n_datapoints = self._count_training_samples()
            logger.info(
                f"Bayesian config: kl_weight={self.kl_weight}, "
                f"kl_warmup_epochs={self.kl_warmup_epochs}, "
                f"n_datapoints={self.n_datapoints}, "
                f"n_val_samples={self.n_val_samples}"
            )
        
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_bayesian(self) -> bool:
        """Check whether the model is a BayesianNeuralNetwork."""
        from models.bayesian_model import BayesianNeuralNetwork
        return isinstance(self.model, BayesianNeuralNetwork)

    def _count_training_samples(self) -> int:
        """Count total training samples for correct KL scaling."""
        loader = self.data_module.train_dataloader(self.fold)
        return len(loader.dataset)

    def _get_kl_weight_for_epoch(self) -> float:
        """Linear KL warm-up: ramp from 0 → kl_weight over kl_warmup_epochs."""
        if self.kl_warmup_epochs <= 0:
            return self.kl_weight
        progress = min(1.0, self.current_epoch / self.kl_warmup_epochs)
        return self.kl_weight * progress
        
    def _detect_task_type(self, phenotypes: torch.Tensor) -> str:
        unique_values = torch.unique(phenotypes)
        n_unique = len(unique_values)
        if torch.all(phenotypes == phenotypes.long()):
            if n_unique == 2:
                return 'binary'
            elif n_unique <= 20:
                return 'multiclass' 
        return 'regression'
        
    def _create_optimizer(self) -> optim.Optimizer:
        opt_config = self.config['optimizer']
        opt_type = opt_config['type'].lower()
        params = self.model.parameters()
        
        if opt_type == 'adam':
            return optim.Adam(
                params, lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0)
            )
        elif opt_type == 'adamw':
            return optim.AdamW(
                params, lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0.0)
            )
        elif opt_type == 'sgd':
            return optim.SGD(
                params, lr=opt_config['lr'],
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")
        
    def _create_loss_function(self) -> nn.Module:
        if not hasattr(self, 'task_type'):
            sample_loader = self.data_module.train_dataloader(self.fold)
            _, sample_phenotypes = next(iter(sample_loader))
            self.task_type = self._detect_task_type(sample_phenotypes)
            logger.info(f"Detected task type: {self.task_type}")
        
        loss_type = self.config.get('loss_function', 'auto').lower()
        
        if loss_type == 'auto':
            if self.task_type == 'binary':
                return nn.BCEWithLogitsLoss()
            elif self.task_type == 'multiclass':
                return nn.CrossEntropyLoss()
            else:
                return nn.MSELoss()
        
        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'mae':
            return nn.L1Loss()
        elif loss_type == 'bce':
            return nn.BCEWithLogitsLoss()
        elif loss_type == 'crossentropy':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")
    
    def _create_scheduler(self) -> Optional[object]:
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
                self.optimizer, T_max=self.config['max_epochs']
            )
        elif sched_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min',
                patience=sched_config.get('patience', 10),
                factor=sched_config.get('factor', 0.1)
            )
        else:
            return None
    
    def setup_logging(self, use_wandb: bool):
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=self.config.get('wandb_project', 'prs-prediction'),
                entity=self.config.get('wandb_entity'),
                name=f"fold_{self.fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config,
                reinit=True,
                tags=["training", f"fold_{self.fold}", "prs",
                      *(["bayesian"] if self.is_bayesian else [])]
            )
            wandb.watch(self.model, log='all', log_freq=100)
        
        self.csv_log_path = self.output_dir / f'training_log_fold_{self.fold}.csv'
        
    def calculate_nagelkerke_r2(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        return calculate_nagelkerke_r2(predictions, targets)

    # ------------------------------------------------------------------
    # Training loops
    # ------------------------------------------------------------------
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        if self.is_bayesian:
            self.model.set_sampling(True)

        train_loader = self.data_module.train_dataloader(self.fold)
        
        total_loss = 0.0
        total_nll = 0.0
        total_kl = 0.0
        all_predictions = []
        all_targets = []
        
        current_kl_weight = self._get_kl_weight_for_epoch() if self.is_bayesian else 0.0
        
        pbar = tqdm(train_loader, desc=f'Training Epoch {self.current_epoch}')
        for batch_idx, (genotypes, phenotypes) in enumerate(pbar):
            genotypes = genotypes.to(self.device)
            phenotypes = phenotypes.to(self.device)
            
            self.optimizer.zero_grad()

            if self.is_bayesian:
                loss, nll, kl, preds = self._bayesian_train_step(
                    genotypes, phenotypes, current_kl_weight
                )
                total_nll += nll
                total_kl += kl
            else:
                predictions = self.model(genotypes)
                loss = self.criterion(predictions, phenotypes)
                preds = predictions

            loss.backward()
            
            if 'gradient_clip' in self.config:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            all_predictions.append(preds.detach().cpu())
            all_targets.append(phenotypes.detach().cpu())
            
            postfix = {'loss': loss.item()}
            if self.is_bayesian:
                postfix['nll'] = nll
                postfix['kl'] = kl
            pbar.set_postfix(postfix)
            
            if self.use_wandb and batch_idx % 10 == 0:
                global_step = self.current_epoch * len(train_loader) + batch_idx
                log_dict = {'train/batch_loss': loss.item(), 'global_step': global_step}
                if self.is_bayesian:
                    log_dict['train/batch_nll'] = nll
                    log_dict['train/batch_kl'] = kl
                    log_dict['train/kl_weight'] = current_kl_weight
                wandb.log(log_dict)
        
        n_batches = len(train_loader)
        avg_loss = total_loss / n_batches
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        metrics = self.calculate_metrics(all_predictions, all_targets)

        if self.is_bayesian:
            metrics['avg_nll'] = total_nll / n_batches
            metrics['avg_kl'] = total_kl / n_batches
            metrics['kl_weight'] = current_kl_weight
        
        return avg_loss, metrics

    def _bayesian_train_step(
        self,
        genotypes: torch.Tensor,
        phenotypes: torch.Tensor,
        kl_weight: float,
    ) -> Tuple[torch.Tensor, float, float, torch.Tensor]:
        """Single training step for BNN. Returns (loss, nll_value, kl_value, predictions)."""
        output = self.model(genotypes)

        if self.model.task == 'regression':
            pred_mean, pred_log_var = output
            loss = self.model.elbo_loss(
                pred_mean, phenotypes,
                n_datapoints=self.n_datapoints,
                kl_weight=kl_weight,
                pred_log_var=pred_log_var,
            )
            preds = pred_mean
        else:
            # binary: output is logits
            logits = output
            loss = self.model.elbo_loss(
                logits, phenotypes,
                n_datapoints=self.n_datapoints,
                kl_weight=kl_weight,
            )
            preds = logits

        kl_val = self.model.get_kl_divergence().item()
        nll_val = (loss - (kl_weight / self.n_datapoints) * kl_val)
        # nll_val is approximate; compute cleanly
        nll_val = loss.item() - (kl_weight / self.n_datapoints) * kl_val

        return loss, nll_val, kl_val, preds
    
    # def validate(self) -> Tuple[float, Dict[str, float]]:
    #     self.model.eval()
    #     if self.is_bayesian:
    #         self.model.set_sampling(False)

    #     val_loader = self.data_module.val_dataloader(self.fold)
        
    #     total_loss = 0.0
    #     all_predictions = []
    #     all_targets = []
        
    #     with torch.no_grad():
    #         for genotypes, phenotypes in tqdm(val_loader, desc='Validation'):
    #             genotypes = genotypes.to(self.device)
    #             phenotypes = phenotypes.to(self.device)

    #             if self.is_bayesian:
    #                 # Deterministic forward (mean weights) for stable val loss
    #                 output = self.model(genotypes)
    #                 if self.model.task == 'regression':
    #                     pred_mean, pred_log_var = output
    #                     loss = self.criterion(pred_mean, phenotypes)
    #                     preds = pred_mean
    #                 else:
    #                     logits = output
    #                     loss = self.criterion(logits, phenotypes)
    #                     preds = logits
    #             else:
    #                 preds = self.model(genotypes)
    #                 loss = self.criterion(preds, phenotypes)
                
    #             total_loss += loss.item()
    #             all_predictions.append(preds.cpu())
    #             all_targets.append(phenotypes.cpu())
        
    #     avg_loss = total_loss / len(val_loader)
    #     all_predictions = torch.cat(all_predictions)
    #     all_targets = torch.cat(all_targets)
    #     metrics = self.calculate_metrics(all_predictions, all_targets)

    #     # Bayesian: run MC uncertainty estimation on val set periodically
    #     if self.is_bayesian and (self.current_epoch + 1) % 5 == 0:
    #         unc_metrics = self._evaluate_uncertainty(val_loader)
    #         metrics.update(unc_metrics)
        
    #     return avg_loss, metrics
    
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        self.model.eval()

        val_loader = self.data_module.val_dataloader(self.fold)

        if self.is_bayesian:
            return self._validate_bayesian_mc(val_loader)

        # Non-Bayesian path (unchanged)
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for genotypes, phenotypes in tqdm(val_loader, desc='Validation'):
                genotypes = genotypes.to(self.device)
                phenotypes = phenotypes.to(self.device)
                preds = self.model(genotypes)
                loss = self.criterion(preds, phenotypes)
                total_loss += loss.item()
                all_predictions.append(preds.cpu())
                all_targets.append(phenotypes.cpu())

        avg_loss = total_loss / len(val_loader)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        metrics = self.calculate_metrics(all_predictions, all_targets)
        return avg_loss, metrics
    
    def _validate_bayesian_mc(
                            self, val_loader: torch.utils.data.DataLoader
                            ) -> Tuple[float, Dict[str, float]]:
        """
        Bayesian validation:
        - val LOSS computed with deterministic forward (low variance, easy to monitor)
        - val METRICS (AUC, accuracy, etc.) computed on MC-averaged predictions
        """
        total_loss = 0.0
        det_predictions = []          # for stable val loss
        mc_mean_probs = []            # for AUC + accuracy
        mc_epistemic = []
        mc_aleatoric = []
        all_targets = []

        # --- Deterministic pass for val loss ---
        self.model.set_sampling(False)
        with torch.no_grad():
            for genotypes, phenotypes in tqdm(val_loader, desc='Validation (det)'):
                genotypes = genotypes.to(self.device)
                phenotypes = phenotypes.to(self.device)

                output = self.model(genotypes)
                if self.model.task == 'regression':
                    pred_mean, _ = output
                    loss = self.criterion(pred_mean, phenotypes)
                    preds = pred_mean
                else:
                    logits = output
                    loss = self.criterion(logits, phenotypes)
                    preds = logits

                total_loss += loss.item()
                det_predictions.append(preds.cpu())
                all_targets.append(phenotypes.cpu())

        avg_loss = total_loss / len(val_loader)

        # --- MC-averaged pass for metrics ---
        self.model.set_sampling(True)
        with torch.no_grad():
            for genotypes, _phenotypes in tqdm(val_loader, desc='Validation (MC)'):
                genotypes = genotypes.to(self.device)
                mean_pred, ep_std, al_std = self.model.predict_with_uncertainty(
                    genotypes, n_samples=self.n_val_samples
                )
                mc_mean_probs.append(mean_pred.cpu())
                mc_epistemic.append(ep_std.cpu())
                mc_aleatoric.append(al_std.cpu())

        self.model.set_sampling(False)

        mc_mean_probs = torch.cat(mc_mean_probs)
        all_targets = torch.cat(all_targets)

        # Compute primary metrics from MC-averaged predictions.
        # NOTE: predict_with_uncertainty returns sigmoid(logits) for binary
        # and pred_mean for regression. calculate_metrics expects logits for
        # binary (it applies sigmoid internally), so for binary we convert back.
        if self.model.task == 'binary':
            # Avoid log(0) — clamp before logit
            probs = mc_mean_probs.clamp(1e-7, 1 - 1e-7)
            logits_equivalent = torch.log(probs / (1 - probs))
            metrics = self.calculate_metrics(logits_equivalent, all_targets)
        else:
            metrics = self.calculate_metrics(mc_mean_probs, all_targets)

        # Add uncertainty diagnostics
        ep = torch.cat(mc_epistemic)
        al = torch.cat(mc_aleatoric)
        metrics['epistemic_std_mean'] = ep.mean().item()
        metrics['aleatoric_std_mean'] = al.mean().item()

        # Uncertainty-error correlation
        if self.model.task == 'binary':
            errors = (mc_mean_probs - all_targets.float()).abs()
        else:
            errors = (mc_mean_probs - all_targets).abs()
        errors_np = errors.numpy().flatten()
        ep_np = ep.numpy().flatten()
        if len(errors_np) > 2:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(ep_np, errors_np)
            metrics['uncertainty_error_spearman'] = float(corr)

        return avg_loss, metrics
        
    

    def _evaluate_uncertainty(
        self, val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Run MC sampling on validation set and report uncertainty diagnostics."""
        self.model.set_sampling(True)

        all_means, all_ep, all_al, all_targets = [], [], [], []
        with torch.no_grad():
            for genotypes, phenotypes in val_loader:
                genotypes = genotypes.to(self.device)
                mean, ep_std, al_std = self.model.predict_with_uncertainty(
                    genotypes, n_samples=self.n_val_samples
                )
                all_means.append(mean.cpu())
                all_ep.append(ep_std.cpu())
                all_al.append(al_std.cpu())
                all_targets.append(phenotypes)

        means = torch.cat(all_means)
        ep = torch.cat(all_ep)
        al = torch.cat(all_al)
        targets = torch.cat(all_targets)

        metrics = {
            'epistemic_std_mean': ep.mean().item(),
            'epistemic_std_median': ep.median().item(),
            'aleatoric_std_mean': al.mean().item(),
        }

        # Coverage calibration (regression only)
        if self.model.task == 'regression':
            total_std = torch.sqrt(ep ** 2 + al ** 2 + 1e-8)
            errors = (means - targets).abs()
            for level in [0.5, 0.9, 0.95]:
                from scipy.stats import norm as _norm
                z = _norm.ppf(0.5 + level / 2.0)
                covered = (errors < z * total_std).float().mean().item()
                metrics[f'coverage_{int(level*100)}'] = covered

        # Uncertainty-error correlation (should be positive if well-calibrated)
        errors_np = (means - targets).abs().numpy().flatten()
        ep_np = ep.numpy().flatten()
        if len(errors_np) > 2:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(ep_np, errors_np)
            metrics['uncertainty_error_spearman'] = float(corr)

        self.model.set_sampling(False)
        return metrics
    
    def calculate_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        predictions_np = predictions.numpy()
        targets_np = targets.numpy()
        
        metrics = {}
        
        if self.task_type == 'regression':
            mse = np.mean((predictions_np - targets_np) ** 2)
            metrics['mse'] = float(mse)
            metrics['rmse'] = float(np.sqrt(mse))
            metrics['mae'] = float(np.mean(np.abs(predictions_np - targets_np)))
            
            ss_res = np.sum((targets_np - predictions_np) ** 2)
            ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            metrics['r2'] = float(r2)
            
            from scipy.stats import pearsonr, spearmanr
            if predictions_np.size > 1:
                corr, p_val = pearsonr(predictions_np.flatten(), targets_np.flatten())
                metrics['pearson_r'] = float(corr)
                metrics['pearson_p'] = float(p_val)
        
        elif self.task_type == 'binary':
            probs = torch.sigmoid(torch.tensor(predictions_np)).numpy()
            pred_labels = (probs > 0.5).astype(int)
            
            metrics['accuracy'] = float(accuracy_score(targets_np, pred_labels))
            metrics['precision'] = float(precision_score(targets_np, pred_labels, average='binary', zero_division=0))
            metrics['recall'] = float(recall_score(targets_np, pred_labels, average='binary', zero_division=0))
            metrics['f1'] = float(f1_score(targets_np, pred_labels, average='binary', zero_division=0))
            
            try:
                metrics['auc_roc'] = float(roc_auc_score(targets_np, probs))
            except ValueError:
                metrics['auc_roc'] = 0.0
                
            metrics['nagelkerke_r2'] = self.calculate_nagelkerke_r2(predictions_np, targets_np)
        
        elif self.task_type == 'multiclass':
            pred_labels = np.argmax(predictions_np, axis=1)
            
            metrics['accuracy'] = float(accuracy_score(targets_np, pred_labels))
            metrics['precision'] = float(precision_score(targets_np, pred_labels, average='weighted', zero_division=0))
            metrics['recall'] = float(recall_score(targets_np, pred_labels, average='weighted', zero_division=0))
            metrics['f1'] = float(f1_score(targets_np, pred_labels, average='weighted', zero_division=0))
        
        return metrics
    
    def train(self, max_epochs: int):
        logger.info(f"Starting training for fold {self.fold} — Task type: {self.task_type}")
        if self.is_bayesian:
            logger.info(
                f"Bayesian training: n_datapoints={self.n_datapoints}, "
                f"kl_warmup={self.kl_warmup_epochs} epochs"
            )
        
        if self.task_type == 'regression':
            primary_metric = 'r2'
            best_metric_value = -float('inf')
            metric_improved = lambda new, best: new > best
        elif self.task_type == 'binary':
            primary_metric = 'auc_roc'
            best_metric_value = -float('inf') 
            metric_improved = lambda new, best: new > best
        else:
            primary_metric = 'accuracy'
            best_metric_value = -float('inf')
            metric_improved = lambda new, best: new > best
        
        for epoch in range(max_epochs):
            self.current_epoch = epoch
            
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate()
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            self.log_epoch(train_loss, train_metrics, val_loss, val_metrics)
            
            if self.current_epoch > self.kl_warmup_epochs and self.is_bayesian:
                logger.info(f"Epoch {epoch} is larger than {self.kl_warmup_epochs}: Starting early stopping checks based on validation loss for Bayesian model.")
                if self.check_early_stopping(val_loss):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
                    
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)
                    logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
                    
            elif not self.is_bayesian:
                if self.check_early_stopping(val_loss):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)
                    logger.info(f"New best model saved with validation loss: {val_loss:.4f}")

   
            
            current_metric = val_metrics.get(primary_metric, 0)
            if metric_improved(current_metric, best_metric_value):
                best_metric_value = current_metric
                self.save_checkpoint(is_best=True, metric=primary_metric)
                logger.info(f"New best model saved with {primary_metric}: {current_metric:.4f}")
            
       
        
        self.save_checkpoint(is_best=False, final=True)

        # Final uncertainty calibration for Bayesian models
        if self.is_bayesian:
            self._final_bayesian_report()

        self.save_training_history()
        
        if self.use_wandb:
            wandb.finish()
        
        logger.info(f"Training completed for fold {self.fold}")

    def _final_bayesian_report(self):
        """Run full calibration and weight diagnostics at end of training."""
        logger.info("Running final Bayesian diagnostics...")

        # Weight statistics
        weight_stats = self.model.get_weight_statistics()
        logger.info(f"Weight statistics: {json.dumps(weight_stats, indent=2)}")

        stats_path = self.output_dir / f'bayesian_weight_stats_fold_{self.fold}.json'
        with open(stats_path, 'w') as f:
            json.dump(weight_stats, f, indent=2)

        # Full calibration on validation set
        if self.model.task == 'regression':
            val_loader = self.data_module.val_dataloader(self.fold)
            cal = self.model.calibrate_uncertainty(
                val_loader, n_samples=50, device=self.device
            )
            logger.info(f"Calibration: {cal}")
            cal_path = self.output_dir / f'bayesian_calibration_fold_{self.fold}.json'
            with open(cal_path, 'w') as f:
                json.dump(cal, f, indent=2)

            if self.use_wandb:
                for level, cov in cal['coverage'].items():
                    wandb.log({f'calibration/coverage_{int(float(level)*100)}': cov})
                wandb.log({
                    'calibration/mean_epistemic_std': cal['mean_epistemic_std'],
                    'calibration/mean_aleatoric_std': cal['mean_aleatoric_std'],
                })

        if self.use_wandb:
            for layer_name, lstats in weight_stats.items():
                for k, v in lstats.items():
                    if isinstance(v, (int, float)):
                        wandb.log({f'weights/{layer_name}/{k}': v})
    
    def log_epoch(
        self,
        train_loss: float,
        train_metrics: Dict[str, float],
        val_loss: float,
        val_metrics: Dict[str, float]
    ):
        if self.task_type == 'regression':
            key_metrics = f"R²: {val_metrics.get('r2', 0):.4f}, Pearson r: {val_metrics.get('pearson_r', 0):.4f}"
        elif self.task_type == 'binary':
            key_metrics = f"Accuracy: {val_metrics.get('accuracy', 0):.4f}, AUC: {val_metrics.get('auc_roc', 0):.4f}"
        else:
            key_metrics = f"Accuracy: {val_metrics.get('accuracy', 0):.4f}, F1: {val_metrics.get('f1', 0):.4f}"
        
        extra = ""
        if self.is_bayesian:
            extra = (
                f", NLL: {train_metrics.get('avg_nll', 0):.4f}, "
                f"KL: {train_metrics.get('avg_kl', 0):.2f}, "
                f"KL_w: {train_metrics.get('kl_weight', 0):.4f}"
            )
            if 'epistemic_std_mean' in val_metrics:
                extra += f", Ep.Std: {val_metrics['epistemic_std_mean']:.4f}"

        logger.info(
            f"Epoch {self.current_epoch}: "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"{key_metrics}{extra}"
        )
        
        if self.use_wandb:
            log_dict = {
                'epoch': self.current_epoch,
                'train/loss': train_loss,
                'val/loss': val_loss,
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()},
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            wandb.log(log_dict)
        
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
        
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['train_metrics'].append(train_metrics)
        self.training_history['val_metrics'].append(val_metrics)
    
    def check_early_stopping(self, val_loss: float, patience: int = 20) -> bool:
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
            'training_history': self.training_history,
            'is_bayesian': self.is_bayesian,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def save_training_history(self):
        history_path = self.output_dir / f'training_history_fold_{self.fold}.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        metrics_summary = {
            'best_val_loss': float(self.best_val_loss),
            'best_val_r2': float(self.best_val_r2),
            'final_train_loss': float(self.training_history['train_loss'][-1]),
            'final_val_loss': float(self.training_history['val_loss'][-1]),
            'final_val_metrics': self.training_history['val_metrics'][-1],
            'total_epochs': len(self.training_history['train_loss']),
            'is_bayesian': self.is_bayesian,
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
    
    with open(args.params_file, 'r') as f:
        config = yaml.safe_load(f)
    
    config['max_epochs'] = args.max_epochs
    config['batch_size'] = args.batch_size
    config['use_wandb'] = True
    config['wandb_project'] = args.wandb_project
    config['wandb_entity'] = args.wandb_entity
    
    data_module = GenotypeDataModule(
        h5_file=args.genotype_file,
        phenotype_file=args.phenotype_file,
        indices_file=args.indices_file,
        batch_size=args.batch_size,
        num_workers=config.get('num_workers', 4),
        augment_train=config.get('augment_train', True),
        augmentation_params=config.get('augmentation_params', {})
    )
    
    input_dim = data_module.get_input_dim()
    output_dim = data_module.get_output_dim()
    
    model_config = config['model']
    model_config['input_dim'] = input_dim
    model_config['output_dim'] = output_dim
    
    logger.info(f"Model configuration: {model_config}")
    logger.info(f"Complete configuration: {config}")
    
    if 'task_type' not in model_config:
        sample_loader = data_module.train_dataloader(args.fold)
        _, sample_phenotypes = next(iter(sample_loader))
        model_config['task_type'] = Trainer._detect_task_type(None, sample_phenotypes)
        logger.info(f"Auto-detected task type: {model_config['task_type']}")
    else:
        logger.info(f"Using specified task type: {model_config['task_type']}")
    
    data_module.scale_target = (model_config['task_type'] == 'regression')
    
    model = create_model(model_config)
        
    output_dir = Path(args.output_model).parent
    
    trainer = Trainer(
        model=model,
        data_module=data_module,
        config=config,
        fold=args.fold,
        output_dir=output_dir,
        use_wandb=config.get('use_wandb', True)
    )
    
    trainer.train(args.max_epochs)
    
    import shutil
    final_model = output_dir / f'final_model_fold_{args.fold}.pt'
    best_model = output_dir / f'best_model_fold_{args.fold}.pt'
    source = best_model if best_model.exists() else final_model
    shutil.copy(str(source), args.output_model)


if __name__ == '__main__':
    main()