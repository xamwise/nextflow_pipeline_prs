"""
Attention-based model for Polygenic Risk Score prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ChunkedLocalAttention(nn.Module):
    """
    Memory-efficient chunked local attention with global landmarks.

    Splits the SNP sequence into non-overlapping chunks and computes
    self-attention independently within each chunk. Global landmark
    tokens are prepended to every chunk to relay information across
    distant genomic regions.

    Reduces peak memory from O(n²) to O(n × (chunk_size + n_landmarks)).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        chunk_size: int = 512,
        n_landmarks: int = 32,
        dropout: float = 0.1,
    ):
        """
        Initialize chunked local attention.

        Args:
            d_model: Dimension of model embeddings
            n_heads: Number of attention heads
            chunk_size: Number of SNP tokens per local attention chunk
            n_landmarks: Number of global landmark tokens (0 to disable)
            dropout: Attention dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.chunk_size = chunk_size
        self.n_landmarks = n_landmarks
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _compute_landmarks(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute global landmark tokens by pooling evenly-spaced segments.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Landmark tokens of shape (batch_size, n_landmarks, d_model)
        """
        batch, seq_len, d = x.shape
        seg_len = seq_len // self.n_landmarks
        trimmed = x[:, : seg_len * self.n_landmarks, :]
        return trimmed.reshape(batch, self.n_landmarks, seg_len, d).mean(dim=2)

    def _pad_to_chunks(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Pad sequence length to be divisible by chunk_size.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Padded tensor and original sequence length
        """
        seq_len = x.size(1)
        remainder = seq_len % self.chunk_size
        if remainder == 0:
            return x, seq_len
        pad_len = self.chunk_size - remainder
        return F.pad(x, (0, 0, 0, pad_len)), seq_len

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with chunked local attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            return_weights: Whether to return per-chunk attention weights

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
            Optionally attention weights of shape
            (batch_size, n_chunks, n_heads, chunk_size, ctx_len)
        """
        batch, seq_len, _ = x.shape

        if self.n_landmarks > 0:
            landmarks = self._compute_landmarks(x)

        x_padded, orig_len = self._pad_to_chunks(x)
        padded_len = x_padded.size(1)
        n_chunks = padded_len // self.chunk_size

        q_local = self.q_proj(x_padded)
        q_chunks = q_local.reshape(batch, n_chunks, self.chunk_size, self.d_model)

        x_chunks = x_padded.reshape(batch, n_chunks, self.chunk_size, self.d_model)

        if self.n_landmarks > 0:
            lm_expanded = landmarks.unsqueeze(1).expand(
                batch, n_chunks, self.n_landmarks, self.d_model
            )
            kv_context = torch.cat([lm_expanded, x_chunks], dim=2)
        else:
            kv_context = x_chunks

        k_chunks = self.k_proj(kv_context)
        v_chunks = self.v_proj(kv_context)
        ctx_len = kv_context.size(2)

        q = (
            q_chunks.reshape(batch * n_chunks, self.chunk_size, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            k_chunks.reshape(batch * n_chunks, ctx_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            v_chunks.reshape(batch * n_chunks, ctx_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(torch.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(
            batch, n_chunks, self.chunk_size, self.d_model
        )
        out = out.reshape(batch, padded_len, self.d_model)[:, :orig_len, :]
        out = self.out_proj(out)

        if return_weights:
            weights = attn.reshape(batch, n_chunks, self.n_heads, self.chunk_size, ctx_len)
            return out, weights
        return out


class AttentionModel(nn.Module):
    """
    Attention-based model with chunked self-attention for SNP importance.

    This model uses chunked local attention with global landmarks to
    efficiently learn which SNPs are most important for phenotype prediction.
    Each SNP is treated as a token, with local windows capturing linkage
    disequilibrium and landmarks relaying long-range genomic signals.

    Scales to large SNP panels (50k–100k+) with O(n × chunk_size) memory.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_attention_heads: int = 4,
        dropout_rate: float = 0.3,
        chunk_size: int = 512,
        n_landmarks: int = 32,
    ):
        """
        Initialize the Attention model.

        Args:
            input_dim: Number of input features (SNPs)
            output_dim: Number of output features (phenotypes)
            hidden_dim: Hidden dimension for projections
            n_attention_heads: Number of attention heads
            dropout_rate: Dropout probability for regularization
            chunk_size: Local attention chunk size (maps to genomic neighbourhood)
            n_landmarks: Number of global landmark tokens for cross-region communication
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_attention_heads

        # Embed each discrete genotype value (0, 1, 2) into hidden dimension
        self.snp_embedding = nn.Embedding(num_embeddings=3, embedding_dim=hidden_dim)

        # Learned positional embedding for each SNP slot
        self.pos_embedding = nn.Embedding(
            num_embeddings=input_dim, embedding_dim=hidden_dim
        )

        # Chunked local attention replaces full self-attention
        self.attention = ChunkedLocalAttention(
            d_model=hidden_dim,
            n_heads=n_attention_heads,
            chunk_size=chunk_size,
            n_landmarks=n_landmarks,
            dropout=dropout_rate,
        )

        # Feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Layer normalization for stable training
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Output projection layers
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Shared encoder producing pooled representations and optional attention weights.

        Embeds each SNP as a token, adds positional information, applies
        chunked self-attention with a residual feed-forward block, and
        mean-pools across the sequence.

        Args:
            x: Input tensor of shape (batch_size, n_snps) with values in {0, 1, 2}
            return_weights: Whether to compute and return attention weights

        Returns:
            pooled: Pooled representation of shape (batch_size, hidden_dim)
            attn_weights: Per-chunk attention weights or None
        """
        x = x.long()
        positions = torch.arange(x.size(1), device=x.device)
        x = self.snp_embedding(x) + self.pos_embedding(positions)

        # Apply chunked self-attention with residual connection
        if return_weights:
            attn_output, attn_weights = self.attention(
                self.norm1(x), return_weights=True
            )
        else:
            attn_output = self.attention(self.norm1(x))
            attn_weights = None

        x = x + attn_output

        # Apply feed-forward network with residual connection
        x = x + self.ffn(self.norm2(x))

        # Mean pool over SNP tokens
        pooled = x.mean(dim=1)

        return pooled, attn_weights

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, n_snps) with integer
               genotype values in {0, 1, 2}
            return_attention: Whether to return attention weights

        Returns:
            Output tensor of shape (batch_size, output_dim)
            Optionally returns per-chunk attention weights if return_attention=True
        """
        pooled, attn_weights = self._encode(x, return_weights=return_attention)
        output = self.head(pooled)

        if return_attention:
            return output, attn_weights
        return output

    def get_snp_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-SNP importance scores based on local attention received.

        Within each chunk, importance is the column-wise mean of the attention
        matrix (excluding landmark columns), measuring how much each SNP is
        attended to by its neighbours. Scores are averaged across heads,
        batch samples, and reassembled into the full sequence.

        Args:
            x: Input tensor of shape (batch_size, n_snps)

        Returns:
            SNP importance scores of shape (n_snps,)
        """
        _, attn_weights = self._encode(x, return_weights=True)

        # attn_weights: (batch, n_chunks, n_heads, chunk_size, ctx_len)
        # Strip landmark columns to get local-only attention
        n_lm = self.attention.n_landmarks
        local_weights = attn_weights[:, :, :, :, n_lm:]

        # Column mean within each chunk = per-token importance
        # (batch, n_chunks, n_heads, chunk_size) -> (n_chunks, chunk_size)
        chunk_importance = local_weights.mean(dim=(0, 2, 3))

        # Flatten chunks back to sequence and trim padding
        return chunk_importance.reshape(-1)[: self.input_dim]

    def get_hidden_representations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get hidden representations after the attention block.

        Args:
            x: Input tensor of shape (batch_size, n_snps)

        Returns:
            Hidden representations of shape (batch_size, hidden_dim)
        """
        pooled, _ = self._encode(x)
        return pooled