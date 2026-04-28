"""
AdvancedMLP for classification PRS prediction.

Three substantive improvements over a vanilla classification MLP for case/control
or multiclass phenotype prediction:

1. Wide-and-deep architecture: an input -> logit linear skip runs in parallel
   with the deep MLP. This guarantees the model can recover logistic regression
   when the deep arm doesn't help -- which is the most reliable PRS baseline
   and the floor you don't want to fall below.

2. Class-imbalance-aware losses: `make_loss(y_train=...)` builds a
   BCEWithLogitsLoss(pos_weight) or CrossEntropyLoss(weight) with weights
   derived from training class frequencies. Optional focal loss and label
   smoothing for further imbalance / calibration handling.

3. Post-hoc temperature scaling: `fit_temperature(val_logits, val_labels)`
   fits a single scalar T on a held-out validation fold; predict_proba then
   returns calibrated probabilities. Almost free to add and consistently
   improves ECE / Brier on tabular classifiers.

Typical usage:
    model = AdvancedMLP(input_dim=n_snps, task="binary")
    criterion = model.make_loss(y_train=y_train)            # uses pos_weight
    optim = torch.optim.AdamW(model.parameters(),
                              lr=1e-3, weight_decay=1e-2)

    # ... train loop; optionally add model.first_layer_l1_penalty() to loss ...

    # Calibrate after training, on the inner-validation fold:
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
    model.fit_temperature(val_logits, y_val)

    # Calibrated outputs:
    probs = model.predict_proba(X_test)
"""

from typing import List, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


VALID_TASKS = ("binary", "multiclass")


# --------------------------------------------------------------------------- #
# Auxiliary loss modules                                                      #
# --------------------------------------------------------------------------- #


class FocalLoss(nn.Module):
    """
    Focal loss for binary or multiclass logits.

    binary:     gamma=0 reduces to BCE; larger gamma down-weights easy examples.
    multiclass: same idea applied to the predicted-class probability.

    Consumes raw logits (BCEWithLogitsLoss-style for binary, CrossEntropyLoss-
    style for multiclass). Calibration tends to suffer with focal loss, so pair
    with temperature scaling when probabilities matter.
    """

    def __init__(
        self,
        task: str,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        if task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}; got {task}")
        self.task = task
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.task == "binary":
            logits = logits.reshape(-1)
            targets = targets.reshape(-1).float()

            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            p = torch.sigmoid(logits)
            p_t = p * targets + (1 - p) * (1 - targets)
            focal = (1 - p_t).pow(self.gamma) * bce

            if self.alpha is not None:
                # alpha is a 2-vector [neg_weight, pos_weight]
                alpha_t = self.alpha[1] * targets + self.alpha[0] * (1 - targets)
                focal = alpha_t * focal
        else:  # multiclass
            log_p = F.log_softmax(logits, dim=-1)
            log_p_t = log_p.gather(1, targets.long().unsqueeze(1)).squeeze(1)
            p_t = log_p_t.exp()
            focal = -(1 - p_t).pow(self.gamma) * log_p_t

            if self.alpha is not None:
                alpha_t = self.alpha.gather(0, targets.long())
                focal = alpha_t * focal

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class _SmoothedBCEWithLogitsLoss(nn.Module):
    """BCE-with-logits + label smoothing + optional pos_weight."""

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        smoothing: float = 0.0,
    ):
        super().__init__()
        self.smoothing = smoothing
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.reshape(-1)
        targets = targets.reshape(-1).float()
        # Symmetric smoothing toward 0.5
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(
            logits, targets_smooth, pos_weight=self.pos_weight
        )


# --------------------------------------------------------------------------- #
# Main model                                                                  #
# --------------------------------------------------------------------------- #


class AdvancedMLP(nn.Module):
    """
    Wide-and-deep classification MLP for genotype -> phenotype prediction.

    Architecture:
        x --+-- feature_extractor --> deep_head -----+
            |                                        +--> logits  (-> /T at predict)
            +-- wide_head (linear) ------------------+

    Notes:
        - No norm layer immediately precedes deep_head; BN/LN running stats can
          drift between train/eval and degrade calibration.
        - With wide_skip=True (default), the model strictly contains logistic
          regression as a special case, so it can't underperform the linear PRS
          baseline due to optimization issues alone.
        - Forward returns RAW logits. predict_proba / predict apply temperature.
    """

    def __init__(
        self,
        input_dim: int,
        task: str = "binary",
        num_classes: int = 2,
        hidden_dims: List[int] = [1024, 512, 256],
        dropout_rate: float = 0.3,
        activation: str = "relu",
        norm: str = "batch",        # "batch" | "layer" | "none"
        wide_skip: bool = True,
        first_layer_l1: float = 0.0,
    ):
        super().__init__()

        if task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}; got {task}")
        if task == "multiclass" and num_classes < 2:
            raise ValueError("multiclass requires num_classes >= 2")
        if norm not in ("batch", "layer", "none"):
            raise ValueError(f"norm must be 'batch'|'layer'|'none'; got {norm}")

        self.task = task
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.wide_skip_enabled = wide_skip
        self.first_layer_l1 = float(first_layer_l1)
        self.activation_name = activation
        self.norm_type = norm

        # Single logit for binary; one per class for multiclass.
        self.logit_dim = 1 if task == "binary" else num_classes

        # Match init nonlinearity to chosen activation.
        init_nl = {"relu": "relu", "leaky_relu": "leaky_relu", "elu": "relu"}.get(
            activation
        )
        if init_nl is None:
            raise ValueError(f"Unknown activation: {activation}")
        self._init_nonlinearity = init_nl

        # Build deep arm.
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if norm == "batch":
                layers.append(nn.BatchNorm1d(h))
            elif norm == "layer":
                layers.append(nn.LayerNorm(h))
            layers.append(self._make_activation(activation))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h

        self.feature_extractor = nn.Sequential(*layers)
        self.deep_head = nn.Linear(prev_dim, self.logit_dim)

        # Wide arm: input -> logits.
        self.wide_head = (
            nn.Linear(input_dim, self.logit_dim) if wide_skip else None
        )

        # Temperature for post-hoc scaling. Stored as a buffer so it is NOT
        # picked up by model.parameters() and trained accidentally during
        # the main optimization loop.
        self.register_buffer("temperature", torch.ones(1))
        self._temperature_fitted = False

        self._initialize_weights()

    # ------------------------------------------------------------------ #
    # Construction helpers                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        if name == "elu":
            return nn.ELU()
        if name == "leaky_relu":
            return nn.LeakyReLU()
        raise ValueError(f"Unknown activation: {name}")

    def _initialize_weights(self) -> None:
        # Hidden Linears: Kaiming matched to activation.
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity=self._init_nonlinearity
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
        """
        deep_logits = self.deep_head(self.feature_extractor(x))
        if self.wide_head is not None:
            return deep_logits + self.wide_head(x)
        return deep_logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calibrated probabilities (temperature-scaled if fitted; T=1 otherwise).
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

    # ------------------------------------------------------------------ #
    # Regularization helper                                              #
    # ------------------------------------------------------------------ #

    def first_layer_l1_penalty(self) -> torch.Tensor:
        """
        Returns first_layer_l1 * ||W_1||_1, or zero if first_layer_l1 == 0.

        Add to the training loss when first_layer_l1 > 0; encourages SNP-level
        sparsity at the input layer (most SNPs are not causal -> sparse first
        layer is biologically reasonable and an effective regularizer in
        p >> n regimes).
        """
        device = next(self.parameters()).device
        if self.first_layer_l1 <= 0.0:
            return torch.tensor(0.0, device=device)
        first_linear = self.feature_extractor[0]
        return self.first_layer_l1 * first_linear.weight.abs().sum()

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
        Build an appropriate loss for this task, optionally weighted by training
        class frequencies.

        Class weight derivation (when y_train is given and use_class_weights):
            binary    -> pos_weight = n_neg / n_pos
            multiclass-> weight[c] = N / (num_classes * n_c)   (inverse-frequency)

        Args:
            y_train: training labels (binary: float 0/1; multiclass: long).
            use_class_weights: disable to keep an unweighted loss even when
                               y_train is provided.
            focal_gamma: if not None, return a FocalLoss instead of BCE/CE.
            label_smoothing: nonzero -> smooth labels (BCE/CE only; ignored
                             with focal loss).
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
                # Focal expects 2-vector [neg, pos].
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

        # multiclass
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
        Fit a single scalar temperature T on a held-out validation set by
        minimizing NLL of T-scaled logits (Guo et al., 2017). Updates
        self.temperature in place.

        Args:
            val_logits: raw logits from forward() on validation data.
                        binary -> (N,) or (N, 1); multiclass -> (N, C).
            val_labels: matching labels.
                        binary -> (N,) or (N, 1) float (0/1).
                        multiclass -> (N,) long.

        Returns:
            The fitted temperature (clamped to [0.05, 20]).
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