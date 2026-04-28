"""
1D Convolutional Neural Network for PRS regression.

Treats SNPs as a 1D sequence (single-channel dosage by default; pass
in_channels=3 with one-hot-encoded input to capture genotype semantics
explicitly). Useful when SNPs are ordered along the genome and you want the
model to capture local LD / haplotype structure -- not useful if the input is
a clumped or pruned set with no spatial ordering, in which case an MLP is
more appropriate.

Forward returns raw continuous predictions of shape (batch_size, output_dim).
Pair with nn.MSELoss, nn.HuberLoss, or nn.GaussianNLLLoss externally.
"""

from typing import List

import torch
import torch.nn as nn


class SimpleCNNModel_R(nn.Module):
    """1D CNN for continuous phenotype prediction from genotype data."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        channels: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [7, 5, 3],
        pool_sizes: List[int] = [2, 2, 2],
        fc_dims: List[int] = [256, 128],
        dropout_rate: float = 0.3,
        norm: str = "batch",        # "batch" | "group" | "none"
        pool_type: str = "max",     # "max" | "avg"
        in_channels: int = 1,
    ):
        """
        Args:
            input_dim: number of SNPs (sequence length).
            output_dim: number of continuous targets (typically 1).
            channels / kernel_sizes / pool_sizes: per-block conv config; must
                be the same length.
            fc_dims: hidden FC widths after the conv stack.
            dropout_rate: dropout after each conv and FC block.
            norm: 'batch' (BN1d on conv, BN1d on FC),
                  'group' (GroupNorm on conv, LayerNorm on FC),
                  'none'.
            pool_type: 'max' (default) or 'avg'. AvgPool is often more
                appropriate for additive polygenic signals where most SNP
                contributions are small; MaxPool emphasizes peak activations.
            in_channels: 1 for dosage (default), 3 for one-hot genotypes.
        """
        super().__init__()

        if not (len(channels) == len(kernel_sizes) == len(pool_sizes)):
            raise ValueError(
                "channels, kernel_sizes, and pool_sizes must have the same length"
            )
        if output_dim < 1:
            raise ValueError("output_dim must be >= 1 for regression.")
        if norm not in ("batch", "group", "none"):
            raise ValueError(f"norm must be 'batch'|'group'|'none'; got {norm}")
        if pool_type not in ("max", "avg"):
            raise ValueError(f"pool_type must be 'max' or 'avg'; got {pool_type}")
        if in_channels < 1:
            raise ValueError("in_channels must be >= 1")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_channels = in_channels
        # Stored for get_receptive_field() and for advanced subclass use.
        self.channels = list(channels)
        self.kernel_sizes = list(kernel_sizes)
        self.pool_sizes = list(pool_sizes)
        self.norm_type = norm
        self.pool_type = pool_type

        # ----- conv stack ---------------------------------------------- #
        conv_layers: List[nn.Module] = []
        c_in = in_channels
        current_len = input_dim
        for c_out, k, p in zip(channels, kernel_sizes, pool_sizes):
            # Padding k//2 keeps length exactly only for odd k; even k drifts
            # by 1 per layer, which is fine and what the original did.
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
                "Reduce pool_sizes / number of pooling layers, or use a larger "
                "input_dim."
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
        self.output_layer = nn.Linear(prev_dim, output_dim)

        self._initialize_weights()

    # ------------------------------------------------------------------ #
    # Init                                                               #
    # ------------------------------------------------------------------ #

    def _initialize_weights(self) -> None:
        """
        Conv1d + hidden Linear: Kaiming-ReLU.
        Output Linear: Xavier (no nonlinearity follows the head; ReLU-tuned
                       Kaiming would over-scale weights by sqrt(2)).
        """
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

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    # ------------------------------------------------------------------ #
    # Forward / utilities                                                #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_snps) for in_channels=1, OR
               (B, in_channels, n_snps) for pre-formatted multi-channel input.
        Returns:
            (B, output_dim) raw continuous predictions.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)        # (B, 1, n_snps)
        x = self.conv_layers(x)
        x = x.flatten(start_dim=1)
        x = self.fc_layers(x)
        return self.output_layer(x)

    def get_conv_features(self, x: torch.Tensor) -> torch.Tensor:
        """Conv features before FC layers, shape (B, C_last, L_last)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.conv_layers(x)

    def get_receptive_field(self) -> int:
        """
        Receptive field of the last conv block, in input-SNP units.

        Standard formula, applied separately per layer:
            r_new    = r_old + (k - 1) * jump_old
            jump_new = jump_old * stride
        Conv layers use stride 1; pool layers use stride = pool_size. The
        original implementation's formula (multiply by pool at each step)
        under-counted later layers because it didn't accumulate `jump`.
        """
        rf = 1
        jump = 1
        for k, p in zip(self.kernel_sizes, self.pool_sizes):
            # conv (stride 1)
            rf = rf + (k - 1) * jump
            # pool (kernel=p, stride=p)
            rf = rf + (p - 1) * jump
            jump = jump * p
        return rf


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _pick_groups(num_channels: int, target: int = 8) -> int:
    """Pick a GroupNorm group count that divides num_channels and is <= target."""
    for g in range(min(target, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1