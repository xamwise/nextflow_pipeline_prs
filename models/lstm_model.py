"""
Bidirectional LSTM model for Polygenic Risk Score prediction.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LSTMModel(nn.Module):
    """
    Bidirectional LSTM model for genotype data.

    This model treats SNPs as a sequence and processes them with a
    bidirectional LSTM, capturing dependencies in both directions
    along the genome. The forward and backward hidden states are
    concatenated and pooled to produce a fixed-size representation
    for phenotype prediction.

    Memory scales as O(n × d) rather than O(n²), enabling training
    on large SNP panels without the quadratic attention bottleneck.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout_rate: float = 0.3,
    ):
        """
        Initialize the Bidirectional LSTM model.

        Args:
            input_dim: Number of input features (SNPs)
            output_dim: Number of output features (phenotypes)
            hidden_dim: Hidden dimension for each LSTM direction
                        (total hidden = hidden_dim × 2 due to bidirectional)
            n_layers: Number of stacked LSTM layers
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Embed discrete genotype (0, 1, 2) into hidden dimension
        self.embedding = nn.Embedding(num_embeddings=3, embedding_dim=hidden_dim)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if n_layers > 1 else 0.0,
        )

        # Output projection (hidden_dim × 2 because bidirectional)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                # Set forget gate bias to 1 for better gradient flow
                hidden = self.hidden_dim
                p.data[hidden : 2 * hidden].fill_(1.0)

        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode genotype sequence through embedding and BiLSTM.

        Embeds each SNP, processes the full sequence bidirectionally,
        and mean-pools the outputs across the sequence.

        Args:
            x: Input tensor of shape (batch_size, n_snps) with values in {0, 1, 2}

        Returns:
            Pooled representation of shape (batch_size, hidden_dim × 2)
        """
        x = x.long()
        x = self.embedding(x)  # (batch, seq, hidden_dim)

        # LSTM output: (batch, seq, hidden_dim × 2)
        output, _ = self.lstm(x)

        # Mean pool over sequence
        pooled = output.mean(dim=1)

        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: (batch_size, n_snps) integer tensor with values in {0, 1, 2}

        Returns:
            (batch_size, output_dim)
        """
        pooled = self._encode(x)
        return self.head(pooled)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the BiLSTM embeddings before the output layers.

        Args:
            x: Input tensor of shape (batch_size, n_snps)

        Returns:
            Embeddings tensor of shape (batch_size, hidden_dim × 2)
        """
        return self._encode(x)

    def get_snp_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-SNP importance via gradient-based attribution.

        Calculates the absolute gradient of the summed output with
        respect to the LSTM outputs at each sequence position,
        averaged across batch samples and hidden dimensions.

        Args:
            x: Input tensor of shape (batch_size, n_snps)

        Returns:
            SNP importance scores of shape (n_snps,)
        """
        x = x.long()
        embedded = self.embedding(x)

        # Need gradients for the LSTM output
        output, _ = self.lstm(embedded)
        output.retain_grad()

        pooled = output.mean(dim=1)
        prediction = self.head(pooled)
        prediction.sum().backward()

        # Importance = mean absolute gradient per position
        # output.grad: (batch, seq, hidden_dim × 2)
        return output.grad.abs().mean(dim=(0, 2))