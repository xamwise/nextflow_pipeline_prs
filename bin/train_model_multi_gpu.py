"""
Training script for genotype deep learning models with multi-GPU support.

Supports single-GPU and multi-GPU training via DistributedDataParallel (DDP).
Launch multi-GPU training with:
    torchrun --nproc_per_node=NUM_GPUS train_model.py --genotype_file ... --distributed

Single-GPU training works as before:
    python train_model.py --genotype_file ...
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
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

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

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


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    """
    Initialize the distributed process group.

    Uses environment variables set by torchrun (RANK, WORLD_SIZE, LOCAL_RANK,
    MASTER_ADDR, MASTER_PORT). Returns (local_rank, global_rank, world_size).
    """
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size


def cleanup_distributed():
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if this is the main process (rank 0)."""
    return rank == 0


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    All-reduce a tensor across processes and return the mean.

    Args:
        tensor: Scalar tensor to reduce
        world_size: Number of processes

    Returns:
        Averaged tensor (identical on all ranks after all-reduce)
    """
    rt = tensor.clone().detach()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def gather_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    world_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gather predictions and targets from all ranks to rank 0.

    Args:
        predictions: Local predictions tensor
        targets: Local targets tensor
        world_size: Number of processes

    Returns:
        Concatenated (predictions, targets) on rank 0, unchanged on other ranks
    """
    pred_list = [torch.zeros_like(predictions) for _ in range(world_size)]
    tgt_list = [torch.zeros_like(targets) for _ in range(world_size)]

    dist.all_gather(pred_list, predictions)
    dist.all_gather(tgt_list, targets)

    return torch.cat(pred_list, dim=0), torch.cat(tgt_list, dim=0)


class Trainer:
    """
    Trainer class for genotype models with multi-GPU support.

    Handles single-GPU and DDP multi-GPU training transparently.
    Logging, checkpointing, and metric computation happen only on
    the main process (rank 0) in distributed mode.
    """

    def __init__(
        self,
        model: nn.Module,
        data_module: GenotypeDataModule,
        config: Dict[str, Any],
        fold: int,
        output_dir: Path,
        use_wandb: bool = True,
        distributed: bool = False,
        local_rank: int = 0,
        global_rank: int = 0,
        world_size: int = 1,
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
            distributed: Whether to use DDP
            local_rank: Local GPU rank
            global_rank: Global process rank
            world_size: Total number of processes
        """
        self.data_module = data_module
        self.config = config
        self.fold = fold
        self.output_dir = Path(output_dir)
        self.distributed = distributed
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.is_main = is_main_process(global_rank)

        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if distributed:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(self.device)

        # Wrap in DDP
        if distributed:
            self.model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
        else:
            self.model = model

        logger.info(f"Rank {global_rank}/{world_size} using device: {self.device}")

        # Optimizer, loss, scheduler
        self.optimizer = self._create_optimizer()
        self.criterion = self._create_loss_function()
        self.scheduler = self._create_scheduler()

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler()

        # Logging — only on main process
        if self.is_main:
            self.setup_logging(use_wandb)
        else:
            self.use_wandb = False

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_r2 = -float("inf")
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
        }

    # ------------------------------------------------------------------
    # Property to access the unwrapped model (useful for saving state)
    # ------------------------------------------------------------------

    @property
    def raw_model(self) -> nn.Module:
        """Return the underlying model without DDP wrapper."""
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _detect_task_type(self, phenotypes: torch.Tensor) -> str:
        """
        Detect task type based on phenotype data.

        Returns:
            'binary', 'multiclass', or 'regression'
        """
        unique_values = torch.unique(phenotypes)
        n_unique = len(unique_values)

        if torch.all(phenotypes == phenotypes.long()):
            if n_unique == 2:
                return "binary"
            elif n_unique <= 20:
                return "multiclass"

        return "regression"

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        opt_config = self.config["optimizer"]
        opt_type = opt_config["type"].lower()
        params = self.model.parameters()

        if opt_type == "adam":
            return optim.Adam(
                params,
                lr=opt_config["lr"],
                weight_decay=opt_config.get("weight_decay", 0),
            )
        elif opt_type == "adamw":
            return optim.AdamW(
                params,
                lr=opt_config["lr"],
                weight_decay=opt_config.get("weight_decay", 0.01),
            )
        elif opt_type == "sgd":
            return optim.SGD(
                params,
                lr=opt_config["lr"],
                momentum=opt_config.get("momentum", 0.9),
                weight_decay=opt_config.get("weight_decay", 0),
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on task type and configuration."""
        if not hasattr(self, "task_type"):
            sample_loader = self.data_module.train_dataloader(self.fold)
            _, sample_phenotypes = next(iter(sample_loader))
            self.task_type = self._detect_task_type(sample_phenotypes)
            if self.is_main:
                logger.info(f"Detected task type: {self.task_type}")

        loss_type = self.config.get("loss_function", "auto").lower()

        if loss_type == "auto":
            if self.task_type == "binary":
                return nn.BCEWithLogitsLoss()
            elif self.task_type == "multiclass":
                return nn.CrossEntropyLoss()
            else:
                return nn.MSELoss()

        mapping = {
            "mse": nn.MSELoss,
            "mae": nn.L1Loss,
            "bce": nn.BCEWithLogitsLoss,
            "crossentropy": nn.CrossEntropyLoss,
        }
        if loss_type in mapping:
            return mapping[loss_type]()
        raise ValueError(f"Unknown loss function: {loss_type}")

    def _create_scheduler(self) -> Optional[object]:
        """Create learning rate scheduler."""
        if "scheduler" not in self.config:
            return None

        sched_config = self.config["scheduler"]
        sched_type = sched_config["type"].lower()

        if sched_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config["step_size"],
                gamma=sched_config["gamma"],
            )
        elif sched_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config["max_epochs"]
            )
        elif sched_type == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=sched_config.get("patience", 10),
                factor=sched_config.get("factor", 0.1),
            )
        return None

    def setup_logging(self, use_wandb: bool):
        """Setup logging with W&B (main process only)."""
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=self.config.get("wandb_project", "prs-prediction"),
                entity=self.config.get("wandb_entity"),
                name=f"fold_{self.fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config,
                reinit=True,
                tags=["training", f"fold_{self.fold}", "prs"],
            )
            wandb.watch(self.model, log="all", log_freq=100)

        self.csv_log_path = self.output_dir / f"training_log_fold_{self.fold}.csv"

    # ------------------------------------------------------------------
    # Data loaders with distributed samplers
    # ------------------------------------------------------------------

    def _make_train_loader(self) -> Tuple[DataLoader, Optional[DistributedSampler]]:
        """
        Create training dataloader with optional distributed sampler.

        Returns:
            (DataLoader, sampler or None)
        """
        base_loader = self.data_module.train_dataloader(self.fold)
        dataset = base_loader.dataset

        if self.distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=True,
            )
            loader = DataLoader(
                dataset,
                batch_size=base_loader.batch_size,
                sampler=sampler,
                num_workers=base_loader.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            return loader, sampler
        return base_loader, None

    def _make_val_loader(self) -> DataLoader:
        """
        Create validation dataloader with optional distributed sampler.

        Returns:
            DataLoader
        """
        base_loader = self.data_module.val_dataloader(self.fold)
        dataset = base_loader.dataset

        if self.distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=False,
            )
            return DataLoader(
                dataset,
                batch_size=base_loader.batch_size,
                sampler=sampler,
                num_workers=base_loader.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        return base_loader

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch with gradient accumulation and mixed precision.

        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        self.model.train()
        train_loader, train_sampler = self._make_train_loader()
        accum_steps = self.config.get("gradient_accumulation_steps", 1)

        # Set epoch for distributed sampler (ensures proper shuffling)
        if train_sampler is not None:
            train_sampler.set_epoch(self.current_epoch)

        total_loss = 0.0
        all_predictions = []
        all_targets = []
        n_batches = 0

        pbar = (
            tqdm(train_loader, desc=f"Training Epoch {self.current_epoch}")
            if self.is_main
            else train_loader
        )

        for batch_idx, (genotypes, phenotypes) in enumerate(pbar):
            genotypes = genotypes.to(self.device, non_blocking=True)
            phenotypes = phenotypes.to(self.device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = self.model(genotypes)
                loss = self.criterion(predictions, phenotypes) / accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                if "gradient_clip" in self.config:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config["gradient_clip"]
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            batch_loss = loss.item() * accum_steps
            total_loss += batch_loss
            n_batches += 1

            all_predictions.append(predictions.detach().float().cpu())
            all_targets.append(phenotypes.detach().cpu())

            if self.is_main and isinstance(pbar, tqdm):
                pbar.set_postfix({"loss": batch_loss})

            if self.use_wandb and batch_idx % 10 == 0:
                global_step = self.current_epoch * len(train_loader) + batch_idx
                wandb.log(
                    {"train/batch_loss": batch_loss, "global_step": global_step}
                )

        # Handle leftover gradients if total batches not divisible by accum_steps
        if n_batches % accum_steps != 0:
            if "gradient_clip" in self.config:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["gradient_clip"]
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        # Aggregate loss across ranks
        avg_loss = total_loss / n_batches
        if self.distributed:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            avg_loss = reduce_tensor(loss_tensor, self.world_size).item()

        # Gather predictions for metrics (main process only computes metrics)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        if self.distributed:
            all_predictions = all_predictions.to(self.device)
            all_targets = all_targets.to(self.device)
            all_predictions, all_targets = gather_predictions(
                all_predictions, all_targets, self.world_size
            )
            all_predictions = all_predictions.cpu()
            all_targets = all_targets.cpu()

        metrics = {}
        if self.is_main:
            metrics = self.calculate_metrics(all_predictions, all_targets)

        return avg_loss, metrics

    def validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.

        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        self.model.eval()
        val_loader = self._make_val_loader()

        total_loss = 0.0
        all_predictions = []
        all_targets = []
        n_batches = 0

        with torch.no_grad():
            pbar = (
                tqdm(val_loader, desc="Validation")
                if self.is_main
                else val_loader
            )
            for genotypes, phenotypes in pbar:
                genotypes = genotypes.to(self.device, non_blocking=True)
                phenotypes = phenotypes.to(self.device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predictions = self.model(genotypes)
                    loss = self.criterion(predictions, phenotypes)

                total_loss += loss.item()
                n_batches += 1
                all_predictions.append(predictions.float().cpu())
                all_targets.append(phenotypes.cpu())

        # Aggregate loss
        avg_loss = total_loss / n_batches
        if self.distributed:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            avg_loss = reduce_tensor(loss_tensor, self.world_size).item()

        # Gather predictions
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        if self.distributed:
            all_predictions = all_predictions.to(self.device)
            all_targets = all_targets.to(self.device)
            all_predictions, all_targets = gather_predictions(
                all_predictions, all_targets, self.world_size
            )
            all_predictions = all_predictions.cpu()
            all_targets = all_targets.cpu()

        metrics = {}
        if self.is_main:
            metrics = self.calculate_metrics(all_predictions, all_targets)

        return avg_loss, metrics

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def calculate_nagelkerke_r2(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> float:
        """Calculate Nagelkerke R2 for binary classification."""
        return calculate_nagelkerke_r2(predictions, targets)

    def calculate_metrics(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate evaluation metrics based on task type."""
        predictions_np = predictions.numpy()
        targets_np = targets.numpy()
        metrics = {}

        if self.task_type == "regression":
            mse = np.mean((predictions_np - targets_np) ** 2)
            metrics["mse"] = float(mse)
            metrics["rmse"] = float(np.sqrt(mse))
            metrics["mae"] = float(np.mean(np.abs(predictions_np - targets_np)))

            ss_res = np.sum((targets_np - predictions_np) ** 2)
            ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
            metrics["r2"] = float(1 - (ss_res / (ss_tot + 1e-8)))

            from scipy.stats import pearsonr

            if predictions_np.size > 1:
                corr, p_val = pearsonr(
                    predictions_np.flatten(), targets_np.flatten()
                )
                metrics["pearson_r"] = float(corr)
                metrics["pearson_p"] = float(p_val)

        elif self.task_type == "binary":
            probs = torch.sigmoid(torch.tensor(predictions_np)).numpy()
            pred_labels = (probs > 0.5).astype(int)

            metrics["accuracy"] = float(accuracy_score(targets_np, pred_labels))
            metrics["precision"] = float(
                precision_score(targets_np, pred_labels, average="binary", zero_division=0)
            )
            metrics["recall"] = float(
                recall_score(targets_np, pred_labels, average="binary", zero_division=0)
            )
            metrics["f1"] = float(
                f1_score(targets_np, pred_labels, average="binary", zero_division=0)
            )

            try:
                metrics["auc_roc"] = float(roc_auc_score(targets_np, probs))
            except ValueError:
                metrics["auc_roc"] = 0.0

            metrics["nagelkerke_r2"] = self.calculate_nagelkerke_r2(
                predictions_np, targets_np
            )

        elif self.task_type == "multiclass":
            pred_labels = np.argmax(predictions_np, axis=1)
            metrics["accuracy"] = float(accuracy_score(targets_np, pred_labels))
            metrics["precision"] = float(
                precision_score(
                    targets_np, pred_labels, average="weighted", zero_division=0
                )
            )
            metrics["recall"] = float(
                recall_score(
                    targets_np, pred_labels, average="weighted", zero_division=0
                )
            )
            metrics["f1"] = float(
                f1_score(
                    targets_np, pred_labels, average="weighted", zero_division=0
                )
            )

        return metrics

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, max_epochs: int):
        """
        Train the model for multiple epochs.

        Args:
            max_epochs: Maximum number of epochs to train
        """
        if self.is_main:
            logger.info(
                f"Starting training for fold {self.fold} — "
                f"Task type: {self.task_type} — "
                f"World size: {self.world_size}"
            )

        # Primary metric for model selection
        if self.task_type == "regression":
            primary_metric = "r2"
        elif self.task_type == "binary":
            primary_metric = "auc_roc"
        else:
            primary_metric = "accuracy"

        best_metric_value = -float("inf")
        metric_improved = lambda new, best: new > best

        for epoch in range(max_epochs):
            self.current_epoch = epoch

            # Training
            train_loss, train_metrics = self.train_epoch()

            # Validation
            val_loss, val_metrics = self.validate()

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # --- Everything below is main-process only ---
            if not self.is_main:
                continue

            self.log_epoch(train_loss, train_metrics, val_loss, val_metrics)
                 # Early stopping
                 
            if self.check_early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                # Signal other ranks to stop
                if self.distributed:
                    stop_tensor = torch.tensor(1, device=self.device)
                    dist.broadcast(stop_tensor, src=0)
                break
            else:
                # Save best model by loss
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)
                    logger.info(
                        f"New best model saved with validation loss: {val_loss:.4f}"
                    )

                # Save best model by primary metric
                current_metric = val_metrics.get(primary_metric, 0)
                if metric_improved(current_metric, best_metric_value):
                    best_metric_value = current_metric
                    self.save_checkpoint(is_best=True, metric=primary_metric)
                    logger.info(
                        f"New best model saved with {primary_metric}: {current_metric:.4f}"
                    )
                    
                if self.distributed:
                    stop_tensor = torch.tensor(0, device=self.device)
                    dist.broadcast(stop_tensor, src=0)
            
        
            # # Early stopping
            # if self.check_early_stopping(val_loss):
            #     logger.info(f"Early stopping triggered at epoch {epoch}")
            #     # Signal other ranks to stop
            #     if self.distributed:
            #         stop_tensor = torch.tensor(1, device=self.device)
            #         dist.broadcast(stop_tensor, src=0)
            #     break
           

        # Non-main ranks: listen for early stopping signal
        # (handled inside the loop above via broadcast)

        if self.is_main:
            self.save_checkpoint(is_best=False, final=True)
            self.save_training_history()

            if self.use_wandb:
                wandb.finish()

            logger.info(f"Training completed for fold {self.fold}")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_epoch(
        self,
        train_loss: float,
        train_metrics: Dict[str, float],
        val_loss: float,
        val_metrics: Dict[str, float],
    ):
        """Log epoch results to W&B and CSV (main process only)."""
        if self.task_type == "regression":
            key_metrics = (
                f"R²: {val_metrics.get('r2', 0):.4f}, "
                f"Pearson r: {val_metrics.get('pearson_r', 0):.4f}"
            )
        elif self.task_type == "binary":
            key_metrics = (
                f"Accuracy: {val_metrics.get('accuracy', 0):.4f}, "
                f"AUC: {val_metrics.get('auc_roc', 0):.4f}"
            )
        else:
            key_metrics = (
                f"Accuracy: {val_metrics.get('accuracy', 0):.4f}, "
                f"F1: {val_metrics.get('f1', 0):.4f}"
            )

        logger.info(
            f"Epoch {self.current_epoch}: "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"{key_metrics}"
        )

        if self.use_wandb:
            wandb.log(
                {
                    "epoch": self.current_epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }
            )

        log_data = {
            "epoch": self.current_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

        df = pd.DataFrame([log_data])
        if self.csv_log_path.exists():
            df.to_csv(self.csv_log_path, mode="a", header=False, index=False)
        else:
            df.to_csv(self.csv_log_path, index=False)

        self.training_history["train_loss"].append(train_loss)
        self.training_history["val_loss"].append(val_loss)
        self.training_history["train_metrics"].append(train_metrics)
        self.training_history["val_metrics"].append(val_metrics)

    # ------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------

    def check_early_stopping(self, val_loss: float) -> bool:
        """
        Check if early stopping should be triggered.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if "early_stopping" not in self.config:
            return False

        patience = self.config["early_stopping"].get("patience", 20)

        if not hasattr(self, "patience_counter"):
            self.patience_counter = 0

        if val_loss >= self.best_val_loss:
            self.patience_counter += 1
        else:
            self.patience_counter = 0

        return self.patience_counter >= patience

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        is_best: bool = False,
        final: bool = False,
        metric: str = "loss",
    ):
        """Save model checkpoint (main process only)."""
        if is_best:
            if metric == "r2":
                path = self.output_dir / f"best_model_r2_fold_{self.fold}.pt"
            else:
                path = self.output_dir / f"best_model_fold_{self.fold}.pt"
        elif final:
            path = self.output_dir / f"final_model_fold_{self.fold}.pt"
        else:
            path = (
                self.output_dir
                / f"checkpoint_fold_{self.fold}_epoch_{self.current_epoch}.pt"
            )

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_r2": self.best_val_r2,
            "config": self.config,
            "training_history": self.training_history,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def save_training_history(self):
        """Save training history to JSON file (main process only)."""
        history_path = (
            self.output_dir / f"training_history_fold_{self.fold}.json"
        )
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        metrics_summary = {
            "best_val_loss": float(self.best_val_loss),
            "best_val_r2": float(self.best_val_r2),
            "final_train_loss": float(self.training_history["train_loss"][-1]),
            "final_val_loss": float(self.training_history["val_loss"][-1]),
            "final_val_metrics": self.training_history["val_metrics"][-1],
            "total_epochs": len(self.training_history["train_loss"]),
        }

        summary_path = self.output_dir / f"metrics_fold_{self.fold}.json"
        with open(summary_path, "w") as f:
            json.dump(metrics_summary, f, indent=2)


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train genotype deep learning model for PRS prediction"
    )
    parser.add_argument("--genotype_file", type=str, required=True)
    parser.add_argument("--phenotype_file", type=str, required=True)
    parser.add_argument("--indices_file", type=str, required=True)
    parser.add_argument("--params_file", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_model", type=str, required=True)
    parser.add_argument("--output_metrics", type=str, required=True)
    parser.add_argument("--output_log", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="prs-prediction")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable multi-GPU DDP training (launch with torchrun)",
    )

    args = parser.parse_args()

    # ---- Distributed setup ----
    distributed = args.distributed
    if distributed:
        local_rank, global_rank, world_size = setup_distributed()
    else:
        local_rank, global_rank, world_size = 0, 0, 1

    # ---- Configuration ----
    with open(args.params_file, "r") as f:
        config = yaml.safe_load(f)

    config["max_epochs"] = args.max_epochs
    config["batch_size"] = args.batch_size
    config["use_wandb"] = True
    config["wandb_project"] = args.wandb_project
    config["wandb_entity"] = args.wandb_entity

    # ---- Data ----
    data_module = GenotypeDataModule(
        h5_file=args.genotype_file,
        phenotype_file=args.phenotype_file,
        indices_file=args.indices_file,
        batch_size=args.batch_size,
        num_workers=config.get("num_workers", 4),
        augment_train=config.get("augment_train", True),
        augmentation_params=config.get("augmentation_params", {}),
    )

    input_dim = data_module.get_input_dim()
    output_dim = data_module.get_output_dim()

    # ---- Model ----
    model_config = config["model"]
    model_config["input_dim"] = input_dim
    model_config["output_dim"] = output_dim

    if is_main_process(global_rank):
        logger.info(f"Model configuration: {model_config}")

    # Auto-detect task type
    if "task_type" not in model_config:
        sample_loader = data_module.train_dataloader(args.fold)
        _, sample_phenotypes = next(iter(sample_loader))
        model_config["task_type"] = Trainer._detect_task_type(
            None, sample_phenotypes
        )
        if is_main_process(global_rank):
            logger.info(f"Auto-detected task type: {model_config['task_type']}")

    model = create_model(model_config)

    # ---- Trainer ----
    output_dir = Path(args.output_model).parent

    trainer = Trainer(
        model=model,
        data_module=data_module,
        config=config,
        fold=args.fold,
        output_dir=output_dir,
        use_wandb=config.get("use_wandb", True),
        distributed=distributed,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
    )

    # ---- Train ----
    trainer.train(args.max_epochs)

    # ---- Copy best model (main only) ----
    if is_main_process(global_rank):
        import shutil

        best_model = output_dir / f"best_model_fold_{args.fold}.pt"
        final_model = output_dir / f"final_model_fold_{args.fold}.pt"
        source = best_model if best_model.exists() else final_model
        shutil.copy(str(source), args.output_model)

    # ---- Cleanup ----
    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()