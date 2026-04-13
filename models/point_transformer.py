"""
Transformer-based model for Polygenic Risk Score prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Store as (1, max_len, d_model) for batch_first usage
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PointTransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout_rate: float = 0.1,
        max_seq_length: int = 10000,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model

        # Embed discrete genotype (0, 1, 2) rather than projecting a scalar
        self.embedding = nn.Embedding(num_embeddings=3, embedding_dim=d_model)

        self.positional_encoding = PositionalEncoding(
            d_model, dropout_rate, max_seq_length
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable training)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, output_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "LayerNorm" not in name and "layernorm" not in name:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_snps) integer tensor with values in {0, 1, 2}
        Returns:
            (batch_size, output_dim)
        """
        x = self.embedding(x)  # (batch, seq, d_model)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        # Mean pooling
        x = x.mean(dim=1)
        return self.head(x)

    def get_attention_weights(self, x: torch.Tensor):
        """Extract per-layer attention weights for interpretability."""
        x = self.embedding(x)
        x = self.positional_encoding(x)

        attn_weights = []
        for layer in self.transformer_encoder.layers:
            # Pre-norm path: normalise before self-attn (matches norm_first=True)
            x_norm = layer.norm1(x)
            _, w = layer.self_attn(
                x_norm, x_norm, x_norm, need_weights=True, average_attn_weights=False
            )
            attn_weights.append(w)
            # Run the full layer to keep hidden states consistent
            x = layer(x)

        # list of (batch, n_heads, seq, seq)
        return attn_weights

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        return x.mean(dim=1)