"""
Transformer-based model for Polygenic Risk Score prediction.

Uses chunked local attention to reduce memory from O(n²) to O(n × chunk_size),
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.

    Adds positional information to the input embeddings to help the model
    understand the sequential nature of SNP positions along the genome.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100000):
        """
        Initialize positional encoding.

        Args:
            d_model: Dimension of the model embeddings
            dropout: Dropout probability
            max_len: Maximum sequence length to support
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Input with positional encoding added
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ChunkedLocalAttention(nn.Module):
    """
    Memory-efficient chunked local attention.

    Splits the SNP sequence into non-overlapping chunks and computes
    self-attention independently within each chunk. This reduces peak
    memory from O(n²) to O(n × chunk_size).

    An optional set of global landmark tokens are prepended to every
    chunk to relay information across distant genomic regions. Landmarks
    are a small learned summary of the full sequence, computed via
    average pooling over evenly-spaced segments.
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
        # Split sequence into n_landmarks segments and average each
        seg_len = seq_len // self.n_landmarks
        # Trim to exact multiple
        trimmed = x[:, : seg_len * self.n_landmarks, :]
        trimmed = trimmed.reshape(batch, self.n_landmarks, seg_len, d)
        return trimmed.mean(dim=2)

    def _pad_to_chunks(self, x: torch.Tensor) -> tuple:
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
        x = nn.functional.pad(x, (0, 0, 0, pad_len))
        return x, seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with chunked local attention.

        Each chunk independently attends over its local tokens plus shared
        global landmark tokens. Only the local token outputs are kept.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Compute global landmarks from full sequence
        if self.n_landmarks > 0:
            landmarks = self._compute_landmarks(x)  # (B, n_landmarks, d)
        
        # Pad sequence to chunk-aligned length
        x_padded, orig_len = self._pad_to_chunks(x)
        padded_len = x_padded.size(1)
        n_chunks = padded_len // self.chunk_size

        # Project queries from local tokens only
        q_local = self.q_proj(x_padded)

        # Reshape into chunks: (batch, n_chunks, chunk_size, d_model)
        q_chunks = q_local.reshape(batch, n_chunks, self.chunk_size, self.d_model)

        # Build key/value context: local chunk tokens + landmarks
        x_chunks = x_padded.reshape(batch, n_chunks, self.chunk_size, self.d_model)

        if self.n_landmarks > 0:
            # Expand landmarks to prepend to every chunk
            # (batch, n_landmarks, d) -> (batch, n_chunks, n_landmarks, d)
            lm_expanded = landmarks.unsqueeze(1).expand(
                batch, n_chunks, self.n_landmarks, self.d_model
            )
            # Context = landmarks + local tokens
            kv_context = torch.cat([lm_expanded, x_chunks], dim=2)
        else:
            kv_context = x_chunks

        k_chunks = self.k_proj(kv_context)
        v_chunks = self.v_proj(kv_context)

        ctx_len = kv_context.size(2)  # chunk_size (+ n_landmarks)

        # Reshape for multi-head attention
        # (batch, n_chunks, seq, d) -> (batch * n_chunks, n_heads, seq, head_dim)
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

        # Compute attention within each chunk
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(torch.softmax(attn, dim=-1))

        # Apply attention to values — keep only local token outputs
        out = (attn @ v).transpose(1, 2).reshape(
            batch, n_chunks, self.chunk_size, self.d_model
        )

        # Flatten chunks back to sequence
        out = out.reshape(batch, padded_len, self.d_model)

        # Remove padding and project
        out = out[:, :orig_len, :]
        return self.out_proj(out)


class ChunkedTransformerBlock(nn.Module):
    """
    Single transformer block using chunked local attention.

    Uses pre-norm architecture (LayerNorm before attention and FFN) for
    more stable training, with GELU activation in the feed-forward network.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        chunk_size: int,
        n_landmarks: int,
        dropout: float,
    ):
        """
        Initialize chunked transformer block.

        Args:
            d_model: Dimension of model embeddings
            n_heads: Number of attention heads
            d_ff: Dimension of the feed-forward network
            chunk_size: Local attention chunk size
            n_landmarks: Number of global landmark tokens
            dropout: Dropout probability
        """
        super().__init__()
        self.attn = ChunkedLocalAttention(
            d_model, n_heads, chunk_size, n_landmarks, dropout
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the chunked transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TransformerModel(nn.Module):
    """
    Transformer-based model for genotype data.

    This model treats SNPs as tokens in a sequence and uses chunked local
    attention with global landmarks to efficiently capture both local linkage
    disequilibrium structure and long-range genomic interactions.

    Memory usage scales as O(n × (chunk_size + n_landmarks)) rather than O(n²),
    enabling training on full SNP panels (50k–100k+).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout_rate: float = 0.1,
        max_seq_length: int = 100000,
        chunk_size: int = 512,
        n_landmarks: int = 32,
    ):
        """
        Initialize the Transformer model.

        Args:
            input_dim: Number of input features (SNPs)
            output_dim: Number of output features (phenotypes)
            d_model: Dimension of the transformer embeddings
            n_heads: Number of attention heads
            n_layers: Number of transformer encoder layers
            d_ff: Dimension of the feed-forward network
            dropout_rate: Dropout probability for regularization
            max_seq_length: Maximum sequence length for positional encoding
            chunk_size: Local attention chunk size (maps to genomic neighbourhood)
            n_landmarks: Number of global landmark tokens for cross-region communication
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model

        # Embed discrete genotype (0, 1, 2) rather than projecting a scalar
        self.embedding = nn.Embedding(num_embeddings=3, embedding_dim=d_model)

        self.positional_encoding = PositionalEncoding(
            d_model, dropout_rate, max_seq_length
        )

        # Chunked transformer blocks replace the standard encoder
        self.layers = nn.ModuleList([
            ChunkedTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                chunk_size=chunk_size,
                n_landmarks=n_landmarks,
                dropout=dropout_rate,
            )
            for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, output_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for name, p in self.named_parameters():
            if p.dim() > 1 and "LayerNorm" not in name and "layernorm" not in name:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: (batch_size, n_snps) integer tensor with values in {0, 1, 2}

        Returns:
            (batch_size, output_dim)
        """
        x = x.long()
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x)

        # Mean pooling
        x = x.mean(dim=1)
        return self.head(x)

    def get_attention_weights(self, x: torch.Tensor):
        """
        Extract per-layer chunked attention weights for interpretability.

        Each returned tensor contains per-chunk attention matrices. The
        context dimension includes landmark tokens (first n_landmarks
        columns) followed by local chunk tokens.

        Args:
            x: Input tensor of shape (batch_size, n_snps)

        Returns:
            List of attention weight tensors, one per layer, each of shape
            (batch_size, n_chunks, n_heads, chunk_size, chunk_size + n_landmarks)
        """
        x = x.long()
        x = self.embedding(x)
        x = self.positional_encoding(x)

        all_weights = []
        for layer in self.layers:
            # Extract attention from the chunked attention module
            norm_x = layer.norm1(x)
            sa = layer.attn
            batch, seq_len, _ = norm_x.shape

            # Reproduce the chunked attention forward to capture weights
            if sa.n_landmarks > 0:
                landmarks = sa._compute_landmarks(norm_x)

            x_padded, orig_len = sa._pad_to_chunks(norm_x)
            padded_len = x_padded.size(1)
            n_chunks = padded_len // sa.chunk_size

            q_local = sa.q_proj(x_padded)
            q_chunks = q_local.reshape(batch, n_chunks, sa.chunk_size, sa.d_model)

            x_chunks = x_padded.reshape(batch, n_chunks, sa.chunk_size, sa.d_model)

            if sa.n_landmarks > 0:
                lm_expanded = landmarks.unsqueeze(1).expand(
                    batch, n_chunks, sa.n_landmarks, sa.d_model
                )
                kv_context = torch.cat([lm_expanded, x_chunks], dim=2)
            else:
                kv_context = x_chunks

            k_chunks = sa.k_proj(kv_context)
            ctx_len = kv_context.size(2)

            q = (
                q_chunks.reshape(batch * n_chunks, sa.chunk_size, sa.n_heads, sa.head_dim)
                .transpose(1, 2)
            )
            k = (
                k_chunks.reshape(batch * n_chunks, ctx_len, sa.n_heads, sa.head_dim)
                .transpose(1, 2)
            )

            scores = (q @ k.transpose(-2, -1)) * sa.scale
            weights = torch.softmax(scores, dim=-1)

            # Reshape: (batch, n_chunks, n_heads, chunk_size, ctx_len)
            weights = weights.reshape(batch, n_chunks, sa.n_heads, sa.chunk_size, ctx_len)
            all_weights.append(weights)

            # Run the full block to keep hidden states consistent
            x = layer(x)

        return all_weights

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the transformer embeddings before the output layers.

        Args:
            x: Input tensor of shape (batch_size, n_snps)

        Returns:
            Embeddings tensor of shape (batch_size, d_model)
        """
        x = x.long()
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x)

        return x.mean(dim=1)