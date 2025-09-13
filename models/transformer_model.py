"""
Transformer-based model for Polygenic Risk Score prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    
    Adds positional information to the input embeddings to help the model
    understand the sequential nature of SNP positions along the genome.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Dimension of the model embeddings
            dropout: Dropout probability
            max_len: Maximum sequence length to support
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for the sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and transpose
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (not a parameter, but moves with the model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Input with positional encoding added
        """
        # Add positional encoding to input
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer-based model for genotype data.
    
    This model treats SNPs as tokens in a sequence and uses self-attention
    mechanisms to capture long-range dependencies and interactions between
    distant SNPs. Particularly useful for capturing epistatic effects.
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
        max_seq_length: int = 10000
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
        """
        super(TransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        
        # Embedding layer to project each SNP to d_model dimension
        self.embedding = nn.Linear(1, d_model)
        
        # Positional encoding to add position information
        self.positional_encoding = PositionalEncoding(
            d_model, dropout_rate, max_seq_length
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True  # Use batch first format
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers
        )
        
        # Global pooling to aggregate sequence information
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Output layers
        self.fc = nn.Linear(d_model, d_model // 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(d_model // 2, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, n_snps)
            
        Returns:
            Output tensor of shape (batch_size, n_phenotypes)
        """
        # Reshape input: (batch_size, seq_len) -> (batch_size, seq_len, 1)
        x = x.unsqueeze(-1)
        
        # Project each SNP to d_model dimension
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling: (batch_size, seq_len, d_model) -> (batch_size, d_model)
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.pooling(x).squeeze(-1)  # (batch_size, d_model)
        
        # Apply output layers
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        output = self.output_layer(x)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for interpretability.
        
        Note: This requires modifying the forward pass to return attention weights,
        which is not implemented in the current version.
        
        Args:
            x: Input tensor of shape (batch_size, n_snps)
            
        Returns:
            Attention weights
        """
        # This would require modifying the transformer to return attention weights
        raise NotImplementedError(
            "Attention weight extraction not implemented. "
            "Requires modification of transformer encoder."
        )
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the transformer embeddings before the output layers.
        
        Args:
            x: Input tensor of shape (batch_size, n_snps)
            
        Returns:
            Embeddings tensor of shape (batch_size, d_model)
        """
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        
        # Global pooling
        x = x.transpose(1, 2)
        x = self.pooling(x).squeeze(-1)
        
        return x