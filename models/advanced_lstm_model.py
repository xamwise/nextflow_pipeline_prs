"""
AdvancedLSTM for classification PRS prediction.

The BiLSTM counterpart to AdvancedMLP / AdvancedCNN. Same three improvements
over a vanilla classification BiLSTM:

1. Wide-and-deep architecture: a Linear(input_dim -> logits) skip operates on
   the float dosage in parallel with the embed-LSTM-pool deep arm. Guarantees
   the model can recover linear PRS / logistic regression on raw genotype
   when the deep arm doesn't help -- which for clinical PRS is the floor you
   genuinely don't want to fall below.

2. Class-imbalance-aware losses: `make_loss(y_train=...)` builds
   BCEWithLogitsLoss(pos_weight) or CrossEntropyLoss(weight) from training
   class frequencies. Optional focal loss and label smoothing.

3. Post-hoc temperature scaling: `fit_temperature(val_logits, val_labels)`
   fits a single scalar T on a held-out validation fold; predict_proba then
   returns calibrated probabilities.

Structural notes:
    - The deep arm uses an integer embedding over dosage tokens; the wide arm
      uses raw float dosage. This lets the same input tensor feed both arms
      with a single forward pass.
    - first_layer_l1 here penalizes the wide_head's columns rather than the
      embedding (which has no per-SNP weights -- 3 tokens are shared across
      every position). This gives interpretable per-SNP sparsity in the
      linear arm; the deep arm remains unregularized by this penalty.
"""

from typing import List, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse classification helpers from advanced_mlp.
from .advanced_mlp_model import FocalLoss, _SmoothedBCEWithLogitsLoss


VALID_TASKS = ("binary", "multiclass")
VALID_POOLS = ("mean", "max", "mean_max", "last")


class AdvancedLSTM(nn.Module):
    """
    Wide-and-deep BiLSTM classifier.

    Architecture:
                     +-- embed -> BiLSTM -> pool -> fc -> deep_head --+
        x  -- ... ---+                                                +--> logits  (-> /T at predict)
                     +-- wide_head (linear over float dosage) --------+
    """

    def __init__(
        self,
        input_dim: int,
        task: str = "binary",
        num_classes: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout_rate: float = 0.3,
        pool: str = "mean",
        num_genotypes: int = 3,
        wide_skip: bool = True,
        first_layer_l1: float = 0.0,
        n_channels: int = 1
    ):
        super().__init__()

        if task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}; got {task}")
        if task == "multiclass" and num_classes < 2:
            raise ValueError("multiclass requires num_classes >= 2")
        if pool not in VALID_POOLS:
            raise ValueError(f"pool must be one of {VALID_POOLS}; got {pool}")
        if num_genotypes < 3:
            raise ValueError("num_genotypes must be >= 3")
        if n_channels not in (1, 2, 3):
            raise ValueError(f"n_channels must be 1, 2, or 3; got {n_channels}")
        self.n_channels = n_channels

        self.task = task
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.pool_type = pool
        self.num_genotypes = num_genotypes
        self.wide_skip_enabled = wide_skip
        self.first_layer_l1 = float(first_layer_l1)

        if first_layer_l1 > 0 and not wide_skip:
            warnings.warn(
                "first_layer_l1 > 0 but wide_skip=False; the L1 penalty has "
                "nothing to act on (the deep arm's embedding has no per-SNP "
                "weights). Penalty will return zero."
            )

        self.logit_dim = 1 if task == "binary" else num_classes

        # ----- deep arm ------------------------------------------------ #
        # Position encoder: Embedding for raw, Linear for encoded.
        if n_channels == 1:
            self.position_encoder = nn.Embedding(
                num_embeddings=num_genotypes, embedding_dim=hidden_dim
            )
            self._encoder_kind = "embedding"
        else:
            self.position_encoder = nn.Linear(n_channels, hidden_dim)
            self._encoder_kind = "linear"
        
        self.embedding = self.position_encoder if n_channels == 1 else None        
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if n_layers > 1 else 0.0,
        )

        bi_hidden = hidden_dim * 2
        pooled_dim = bi_hidden * 2 if pool == "mean_max" else bi_hidden

        # FC stack between pooled representation and deep_head. Ends in a
        # Dropout (no norm immediately before deep_head, for calibration).
        self.fc_layers = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.deep_head = nn.Linear(hidden_dim, self.logit_dim)

        # ----- wide arm ------------------------------------------------ #
        # Wide arm sized by effective input dim.
        self.wide_head = (
            nn.Linear(input_dim * n_channels, self.logit_dim) if wide_skip else None
        )

        # Temperature buffer (excluded from model.parameters()).
        self.register_buffer("temperature", torch.ones(1))
        self._temperature_fitted = False

        self._initialize_weights()

    # ------------------------------------------------------------------ #
    # Init                                                               #
    # ------------------------------------------------------------------ #

    def _initialize_weights(self) -> None:
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                h = self.hidden_dim
                p.data[h : 2 * h].fill_(1.0)        # forget-gate bias = 1

        for m in self.fc_layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Heads: Xavier (no nonlinearity follows).
        nn.init.xavier_uniform_(self.deep_head.weight)
        nn.init.zeros_(self.deep_head.bias)
        if self.wide_head is not None:
            nn.init.xavier_uniform_(self.wide_head.weight)
            nn.init.zeros_(self.wide_head.bias)
            
        if self._encoder_kind == "linear":
            nn.init.xavier_uniform_(self.position_encoder.weight)
            nn.init.zeros_(self.position_encoder.bias)

    # ------------------------------------------------------------------ #
    # Encode / pool                                                      #
    # ------------------------------------------------------------------ #

    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point() and x.dtype == torch.long:
            return x.clamp(0, self.num_genotypes - 1)
        return x.round().long().clamp(0, self.num_genotypes - 1)

    def _pool(self, output: torch.Tensor) -> torch.Tensor:
        if self.pool_type == "mean":
            return output.mean(dim=1)
        if self.pool_type == "max":
            return output.max(dim=1).values
        if self.pool_type == "last":
            return output[:, -1, :]
        return torch.cat([output.mean(dim=1), output.max(dim=1).values], dim=-1)

    # ------------------------------------------------------------------ #
    # Forward / inference                                                #
    # ------------------------------------------------------------------ #

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Returns RAW (uncalibrated) logits.
    #         binary     -> (B, 1)
    #         multiclass -> (B, num_classes)

    #     x is expected as (B, n_snps) and used in two forms:
    #         - integer tokens (clamped/rounded) for the deep arm,
    #         - float dosage for the wide arm.
    #     """
    #     tokens = self._to_tokens(x)
    #     embedded = self.embedding(tokens)
    #     lstm_out, _ = self.lstm(embedded)
    #     pooled = self._pool(lstm_out)
    #     h = self.fc_layers(pooled)
    #     deep_logits = self.deep_head(h)

    #     if self.wide_head is not None:
    #         x_float = x if x.is_floating_point() else x.float()
    #         return deep_logits + self.wide_head(x_float)
    #     return deep_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts:
            (B, M)     — raw float dosage or integer tokens; n_channels must be 1.
            (B, M, k)  — encoded, channels-last; n_channels must equal k.
        """
        if self._encoder_kind == "embedding":
            tokens = self._to_tokens(x)
            embedded = self.position_encoder(tokens)             # (B, M, H)
            x_flat = x if x.is_floating_point() else x.float()   # (B, M)
        else:
            # Encoded (B, M, k) -> per-position Linear -> (B, M, H).
            if x.dim() != 3 or x.shape[-1] != self.n_channels:
                raise ValueError(
                    f"Expected (B, M, {self.n_channels}); got shape {tuple(x.shape)}"
                )
            x_float = x if x.is_floating_point() else x.float()
            embedded = self.position_encoder(x_float)            # (B, M, H)
            x_flat = x_float.flatten(start_dim=1)                # (B, M*k)

        lstm_out, _ = self.lstm(embedded)
        pooled = self._pool(lstm_out)
        h = self.fc_layers(pooled)
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
        proba = self.predict_proba(x)
        if self.task == "binary":
            return (proba >= threshold).long()
        return proba.argmax(dim=-1)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled BiLSTM representation, before the head."""
        # tokens = self._to_tokens(x)
        # embedded = self.embedding(tokens)
        
        if self._encoder_kind == "embedding":
            embedded = self.position_encoder(self._to_tokens(x))
        else:
            embedded = self.position_encoder(x.float() if not x.is_floating_point() else x)
            
        lstm_out, _ = self.lstm(embedded)
        return self._pool(lstm_out)

    def get_snp_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-SNP saliency via |d(sum logits)/d(LSTM output_t)|, averaged over
        batch and hidden. Uses torch.autograd.grad (does not pollute
        parameter .grad).
        """
        # tokens = self._to_tokens(x)
        # embedded = self.embedding(tokens)
        
        if self._encoder_kind == "embedding":
            embedded = self.position_encoder(self._to_tokens(x))
        else:
            embedded = self.position_encoder(x.float() if not x.is_floating_point() else x)
            
        lstm_out, _ = self.lstm(embedded)
        pooled = self._pool(lstm_out)
        h = self.fc_layers(pooled)
        deep_logits = self.deep_head(h)

        if self.wide_head is not None:
            x_float = x if x.is_floating_point() else x.float()
            logits = deep_logits + self.wide_head(x_float)
        else:
            logits = deep_logits

        grad = torch.autograd.grad(
            outputs=logits.sum(),
            inputs=lstm_out,
            create_graph=False,
            retain_graph=False,
        )[0]
        return grad.abs().mean(dim=(0, 2))

    # ------------------------------------------------------------------ #
    # Regularization helper                                              #
    # ------------------------------------------------------------------ #

    def first_layer_l1_penalty(self) -> torch.Tensor:
        """
        L1 penalty on the wide_head's weights (per-SNP sparsity in the linear
        arm). Returns zero if first_layer_l1 == 0 or wide_skip is disabled.

        This intentionally does NOT penalize the embedding, because every SNP
        position shares the same 3-token embedding -- there is no per-SNP
        weight in the deep arm to drive toward zero.
        """
        device = next(self.parameters()).device
        if self.first_layer_l1 <= 0.0 or self.wide_head is None:
            return torch.tensor(0.0, device=device)
        
        W = self.wide_head.weight                                  # (logit, M*k)
        if self.n_channels == 1:
            return self.first_layer_l1 * W.abs().sum()
        W_grouped = W.view(W.shape[0], self.input_dim, self.n_channels)
        return self.first_layer_l1 * W_grouped.norm(p=2, dim=2).sum()
         
        # return self.first_layer_l1 * self.wide_head.weight.abs().sum()

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
        """Identical API to AdvancedMLP.make_loss / AdvancedCNN.make_loss."""
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
        """LBFGS fit of a single scalar T on val NLL (Guo et al., 2017)."""
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
        self.temperature.fill_(1.0)
        self._temperature_fitted = False