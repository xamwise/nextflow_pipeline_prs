"""
Bidirectional LSTM model for PRS regression.

Treats SNPs as a sequence and embeds discrete dosages {0, 1, 2} before passing
through a stacked BiLSTM. Useful when SNPs are ordered along the genome and you
want long-range dependencies without the O(n^2) cost of attention. Not useful
if the input is a clumped/pruned set with no spatial ordering.

Forward returns raw continuous predictions of shape (batch_size, output_dim).
Pair with nn.MSELoss, nn.HuberLoss, or nn.GaussianNLLLoss externally.

Memory note: with hidden=256, n_snps=100K, batch=128, the LSTM output tensor
alone is ~25 GB in fp32. For large inputs reduce hidden_dim, batch size, or
chunk the sequence.
"""

from typing import Tuple

import torch
import torch.nn as nn


VALID_POOLS = ("mean", "max", "mean_max", "last")


class LSTMModel_R(nn.Module):
    """BiLSTM for continuous phenotype prediction from genotype dosage."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout_rate: float = 0.3,
        pool: str = "mean",
        num_genotypes: int = 3,    # 3 = {0,1,2}; pass 4 if you reserve a missing token
    ):
        """
        Args:
            input_dim: number of SNPs (sequence length).
            output_dim: number of continuous targets (typically 1).
            hidden_dim: hidden size per LSTM direction (concat -> 2*hidden_dim).
            n_layers: stacked LSTM layers.
            dropout_rate: PyTorch LSTM dropout (between stacked layers, only
                          active when n_layers > 1) + dropout in the head.
            pool: 'mean' | 'max' | 'mean_max' | 'last'. mean_max concatenates
                  mean and max pools and doubles the head input dim.
            num_genotypes: vocabulary size for the embedding table. 3 is
                  standard for hard-called dosage; use 4 if you encode a
                  missing-genotype token.
        """
        super().__init__()

        if pool not in VALID_POOLS:
            raise ValueError(f"pool must be one of {VALID_POOLS}; got {pool}")
        if output_dim < 1:
            raise ValueError("output_dim must be >= 1 for regression.")
        if num_genotypes < 3:
            raise ValueError("num_genotypes must be >= 3")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.pool_type = pool
        self.num_genotypes = num_genotypes

        self.embedding = nn.Embedding(
            num_embeddings=num_genotypes, embedding_dim=hidden_dim
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if n_layers > 1 else 0.0,
        )

        # Pooled dimension after BiLSTM.
        bi_hidden = hidden_dim * 2
        pooled_dim = bi_hidden * 2 if pool == "mean_max" else bi_hidden

        # Head: LayerNorm -> hidden FC + GELU + Dropout -> output Linear.
        # Xavier init throughout (correct for GELU and for the linear head).
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

        self._initialize_weights()

    # ------------------------------------------------------------------ #
    # Init                                                               #
    # ------------------------------------------------------------------ #

    def _initialize_weights(self) -> None:
        """
        Embedding: default normal init is fine.
        LSTM: Xavier on input-hidden, orthogonal on hidden-hidden, biases zero
              with forget-gate bias = 1 for better long-range gradient flow.
        Head: Xavier (works well for GELU; no Kaiming-ReLU over-scaling).
        """
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                # PyTorch LSTM bias layout per layer: [i, f, g, o] of size
                # hidden_size each. Set forget gate (slice [hidden : 2*hidden])
                # to 1 in BOTH bias_ih and bias_hh.
                h = self.hidden_dim
                p.data[h : 2 * h].fill_(1.0)

        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    # Encode / pool                                                      #
    # ------------------------------------------------------------------ #

    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Cast input to integer tokens in [0, num_genotypes - 1].

        Defensive against soft-imputed dosages (e.g. 0.7 -> 1) and stray values
        from preprocessing. For sub-integer imputed dosages, consider using a
        linear projection of the float dosage instead of a discrete embedding.
        """
        if not x.is_floating_point() and x.dtype == torch.long:
            return x.clamp(0, self.num_genotypes - 1)
        return x.round().long().clamp(0, self.num_genotypes - 1)

    def _pool(self, output: torch.Tensor) -> torch.Tensor:
        """
        Pool a (B, L, H) BiLSTM output along the sequence axis.

        Returns:
            (B, H) for unimodal pools, (B, 2H) for 'mean_max'.
        """
        if self.pool_type == "mean":
            return output.mean(dim=1)
        if self.pool_type == "max":
            return output.max(dim=1).values
        if self.pool_type == "last":
            return output[:, -1, :]
        # mean_max
        return torch.cat([output.mean(dim=1), output.max(dim=1).values], dim=-1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed -> BiLSTM -> pool. Returns pooled representation.
        """
        tokens = self._to_tokens(x)
        embedded = self.embedding(tokens)         # (B, L, hidden_dim)
        output, _ = self.lstm(embedded)           # (B, L, hidden_dim * 2)
        return self._pool(output)

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_snps), values interpreted as dosage tokens in
               [0, num_genotypes - 1].
        Returns:
            (B, output_dim) raw continuous predictions.
        """
        return self.head(self._encode(x))

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled BiLSTM representation, before the head."""
        return self._encode(x)

    def get_snp_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-SNP saliency via gradient of summed prediction w.r.t. the BiLSTM
        output at each position, averaged over batch and hidden dimensions.

        Uses torch.autograd.grad rather than .backward() so that .grad on
        model parameters is NOT modified (safe to call mid-training).

        Returns:
            (n_snps,) importance scores.
        """
        tokens = self._to_tokens(x)
        embedded = self.embedding(tokens)
        output, _ = self.lstm(embedded)           # (B, L, 2H), requires_grad=True
        pooled = self._pool(output)
        prediction = self.head(pooled)

        grad = torch.autograd.grad(
            outputs=prediction.sum(),
            inputs=output,
            create_graph=False,
            retain_graph=False,
        )[0]
        return grad.abs().mean(dim=(0, 2))