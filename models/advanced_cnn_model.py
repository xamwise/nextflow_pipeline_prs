"""
AdvancedCNN for classification PRS prediction.

The CNN counterpart to AdvancedMLP. Same three improvements over a vanilla
classification CNN:

1. Wide-and-deep architecture: a Linear(n_channels * input_dim -> logits)
   skip runs in parallel with the deep CNN. Guarantees the model can recover
   logistic regression on the raw genotype vector when the deep arm doesn't
   help. For genomics this is a non-trivial floor -- linear PRS methods
   (LDpred2, lassosum) are very competitive baselines.

2. Class-imbalance-aware losses: `make_loss(y_train=...)` builds
   BCEWithLogitsLoss(pos_weight) or CrossEntropyLoss(weight) from training
   class frequencies. Optional focal loss and label smoothing.

3. Post-hoc temperature scaling: `fit_temperature(val_logits, val_labels)`
   fits a single scalar T on a held-out validation fold; predict_proba
   returns calibrated probabilities afterwards.

Note on parameter cost: with input_dim ~ 500K SNPs the wide arm has ~500K
parameters per logit, which is comparable to or smaller than the conv stack.
For very large input_dim and tight memory budgets, set wide_skip=False.
"""

from typing import List, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse classification helpers from advanced_mlp to keep these in one place.
from .advanced_mlp_model import FocalLoss, _SmoothedBCEWithLogitsLoss


VALID_TASKS = ("binary", "multiclass")


def _pick_groups(num_channels: int, target: int = 8) -> int:
    """Pick a GroupNorm group count that divides num_channels and is <= target."""
    for g in range(min(target, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class AdvancedCNN(nn.Module):
    """
    Wide-and-deep 1D CNN classifier.

    Architecture:
                   +-- conv stack -> fc stack -> deep_head ---+
        x  -- ... -+                                          +--> logits  (-> /T at predict)
                   +-- wide_head (linear over flattened x) ---+

    Notes:
        - No norm immediately precedes deep_head (BN/GN running stats can
          drift between train/eval and degrade calibration).
        - Forward returns RAW logits; predict_proba / predict apply T.
    """

    def __init__(
        self,
        input_dim: int,
        task: str = "binary",
        num_classes: int = 2,
        channels: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [7, 5, 3],
        pool_sizes: List[int] = [2, 2, 2],
        fc_dims: List[int] = [256, 128],
        dropout_rate: float = 0.3,
        norm: str = "batch",            # "batch" | "group" | "none"
        pool_type: str = "max",         # "max" | "avg"
        n_channels: int = 1,
        wide_skip: bool = True,
        first_layer_l1: float = 0.0
    ):
        super().__init__()

        if task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}; got {task}")
        if task == "multiclass" and num_classes < 2:
            raise ValueError("multiclass requires num_classes >= 2")
        if not (len(channels) == len(kernel_sizes) == len(pool_sizes)):
            raise ValueError(
                "channels, kernel_sizes, and pool_sizes must have the same length"
            )
        if norm not in ("batch", "group", "none"):
            raise ValueError(f"norm must be 'batch'|'group'|'none'; got {norm}")
        if pool_type not in ("max", "avg"):
            raise ValueError(f"pool_type must be 'max' or 'avg'; got {pool_type}")
        if n_channels not in (1, 2, 3):
            raise ValueError(f"n_channels must be 1, 2, or 3; got {n_channels}")

        self.task = task
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.n_channels = n_channels
        self.wide_skip_enabled = wide_skip
        self.first_layer_l1 = float(first_layer_l1)
        self.norm_type = norm
        self.pool_type = pool_type
        self.channels = list(channels)
        self.kernel_sizes = list(kernel_sizes)
        self.pool_sizes = list(pool_sizes)

        self.logit_dim = 1 if task == "binary" else num_classes

        # ----- conv stack ---------------------------------------------- #
        conv_layers: List[nn.Module] = []
        c_in = n_channels
        current_len = input_dim
        for c_out, k, p in zip(channels, kernel_sizes, pool_sizes):
            conv_layers.append(nn.Conv1d(c_in, c_out, k, padding=k // 2))
            if norm == "batch":
                conv_layers.append(nn.BatchNorm1d(c_out))
            elif norm == "group":
                conv_layers.append(nn.GroupNorm(_pick_groups(c_out), c_out))
            conv_layers.append(nn.ReLU())
            conv_layers.append(
                nn.MaxPool1d(p) if pool_type == "max" else nn.AvgPool1d(p)
            )
            conv_layers.append(nn.Dropout(dropout_rate))
            c_in = c_out
            current_len = current_len // p

        if current_len <= 0:
            raise ValueError(
                f"After pooling, conv output length is {current_len}. "
                "Reduce pool_sizes or use a larger input_dim."
            )

        self.conv_layers = nn.Sequential(*conv_layers)
        self.flatten_dim = channels[-1] * current_len

        # ----- FC stack ------------------------------------------------ #
        fc_layers: List[nn.Module] = []
        prev_dim = self.flatten_dim
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(prev_dim, fc_dim))
            if norm == "batch":
                fc_layers.append(nn.BatchNorm1d(fc_dim))
            elif norm == "group":
                fc_layers.append(nn.LayerNorm(fc_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout_rate))
            prev_dim = fc_dim

        self.fc_layers = nn.Sequential(*fc_layers)
        self.deep_head = nn.Linear(prev_dim, self.logit_dim)

        # Wide arm: linear over flattened raw input.
        self.wide_head = (
            nn.Linear(n_channels * input_dim, self.logit_dim)
            if wide_skip else None
        )

        # Temperature stored as a buffer so it isn't picked up by
        # model.parameters() and trained accidentally.
        self.register_buffer("temperature", torch.ones(1))
        self._temperature_fitted = False

        self._initialize_weights()

    # ------------------------------------------------------------------ #
    # Init                                                               #
    # ------------------------------------------------------------------ #

    def _initialize_weights(self) -> None:
        for m in self.conv_layers.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        for m in self.fc_layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Heads: Xavier (no nonlinearity follows).
        nn.init.xavier_uniform_(self.deep_head.weight)
        nn.init.zeros_(self.deep_head.bias)
        if self.wide_head is not None:
            nn.init.xavier_uniform_(self.wide_head.weight)
            nn.init.zeros_(self.wide_head.bias)

    # ------------------------------------------------------------------ #
    # Forward / inference                                                #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns RAW (uncalibrated) logits.
            binary     -> (B, 1)
            multiclass -> (B, num_classes)

        Accepts (B, n_snps) for n_channels=1 or (B, n_snps, n_channels) for
        multi-channel input.
        """
        if x.dim() == 2:
            x_conv = x.unsqueeze(1)                    # (B, 1, M)
            x_flat = x
        elif x.dim() == 3:
            # (B, M, k) -> (B, k, M) for Conv1d.
            x_conv = x.transpose(1, 2).contiguous()
            x_flat = x.flatten(start_dim=1)            # (B, M*k)
        else:
            raise ValueError(f"Expected x.dim() in (2, 3); got {x.dim()}")

        h = self.conv_layers(x_conv)
        h = h.flatten(start_dim=1)
        h = self.fc_layers(h)
        deep_logits = self.deep_head(h)

        if self.wide_head is not None:
            return deep_logits + self.wide_head(x_flat)
        return deep_logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calibrated probabilities (T-scaled if fitted; T=1 otherwise).
            binary     -> (B,)
            multiclass -> (B, num_classes)
        """
        logits = self.forward(x)
        scaled = logits / self.temperature
        if self.task == "binary":
            return torch.sigmoid(scaled).squeeze(-1)
        return F.softmax(scaled, dim=-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Class predictions on calibrated probabilities.
            binary     -> (B,) long, 1 iff P(class=1) >= threshold
            multiclass -> (B,) long, argmax of softmax
        """
        proba = self.predict_proba(x)
        if self.task == "binary":
            return (proba >= threshold).long()
        return proba.argmax(dim=-1)

    def get_conv_features(self, x: torch.Tensor) -> torch.Tensor:
        """Conv features before FC layers, shape (B, C_last, L_last)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            x = x.transpose(1, 2).contiguous()
        
        return self.conv_layers(x)

    def get_receptive_field(self) -> int:
        """Receptive field of the last conv block, in input-SNP units."""
        rf = 1
        jump = 1
        for k, p in zip(self.kernel_sizes, self.pool_sizes):
            rf = rf + (k - 1) * jump
            rf = rf + (p - 1) * jump
            jump = jump * p
        return rf

    # ------------------------------------------------------------------ #
    # Regularization helper                                              #
    # ------------------------------------------------------------------ #

    def first_layer_l1_penalty(self) -> torch.Tensor:
        """
        Returns first_layer_l1 * ||W_conv1||_1, or zero if first_layer_l1 == 0.

        Note: for Conv1d, this gives kernel-pattern sparsity (whole filters can
        go to zero), not per-SNP feature sparsity like in an MLP. For per-SNP
        sparsity in a CNN you would need a learned input mask or group lasso
        over input positions; this simple L1 is the cheap version.
        """
        device = next(self.parameters()).device
        if self.first_layer_l1 <= 0.0:
            return torch.tensor(0.0, device=device)
        first_conv = self.conv_layers[0]      # always nn.Conv1d
        return self.first_layer_l1 * first_conv.weight.abs().sum()

    # ------------------------------------------------------------------ #
    # Loss helpers                                                       #
    # ------------------------------------------------------------------ #

    def make_loss(
        self,
        y_train: Optional[torch.Tensor] = None,
        use_class_weights: bool = True,
        focal_gamma: Optional[float] = None,
        label_smoothing: float = 0.0,
    ) -> nn.Module:
        """
        Build a task-appropriate loss; mirrors AdvancedMLP.make_loss.

        binary    -> pos_weight = n_neg / n_pos
        multiclass-> weight[c]  = N / (num_classes * n_c)
        """
        alpha = None
        pos_weight = None
        class_weight = None

        if y_train is not None and use_class_weights:
            y = y_train.detach().cpu()
            if self.task == "binary":
                y_flat = y.flatten().float()
                n_pos = (y_flat == 1).sum().clamp(min=1).float()
                n_neg = (y_flat == 0).sum().clamp(min=1).float()
                pw = (n_neg / n_pos).reshape(1)
                pos_weight = pw
                alpha = torch.tensor([1.0, pw.item()])
            else:
                y_flat = y.flatten().long()
                counts = torch.bincount(
                    y_flat, minlength=self.num_classes
                ).float().clamp(min=1)
                w = counts.sum() / (self.num_classes * counts)
                class_weight = w
                alpha = w

        if focal_gamma is not None:
            if label_smoothing > 0:
                warnings.warn("label_smoothing is ignored when focal_gamma is set")
            return FocalLoss(task=self.task, gamma=focal_gamma, alpha=alpha)

        if self.task == "binary":
            if label_smoothing > 0:
                return _SmoothedBCEWithLogitsLoss(
                    pos_weight=pos_weight, smoothing=label_smoothing
                )
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        return nn.CrossEntropyLoss(
            weight=class_weight, label_smoothing=label_smoothing
        )

    # ------------------------------------------------------------------ #
    # Temperature scaling                                                #
    # ------------------------------------------------------------------ #

    def fit_temperature(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> float:
        """
        Fit a single scalar T on a held-out validation set by minimizing NLL
        of T-scaled logits (Guo et al., 2017). Updates self.temperature.

        For BNN variants of this model, prefer to pass the MC-averaged
        predictive logits (i.e., logit of mean(p)) rather than a single
        deterministic forward pass, otherwise calibration throws away the
        epistemic uncertainty you computed.
        """
        device = val_logits.device
        T = nn.Parameter(torch.ones(1, device=device))
        optimizer = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter)

        if self.task == "binary":
            logits = val_logits.detach().reshape(-1)
            targets = val_labels.detach().reshape(-1).float().to(device)

            def nll() -> torch.Tensor:
                return F.binary_cross_entropy_with_logits(logits / T, targets)
        else:
            logits = val_logits.detach()
            targets = val_labels.detach().long().to(device)

            def nll() -> torch.Tensor:
                return F.cross_entropy(logits / T, targets)

        def closure():
            optimizer.zero_grad()
            loss = nll()
            loss.backward()
            return loss

        optimizer.step(closure)

        fitted = T.detach().clamp(min=0.05, max=20.0)
        self.temperature.copy_(fitted.to(self.temperature.device))
        self._temperature_fitted = True
        return float(fitted.item())

    @property
    def is_calibrated(self) -> bool:
        return self._temperature_fitted

    def reset_temperature(self) -> None:
        """Reset T to 1 (uncalibrated)."""
        self.temperature.fill_(1.0)
        self._temperature_fitted = False