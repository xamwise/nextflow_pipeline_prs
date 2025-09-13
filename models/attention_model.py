"""
Attention-based model for Polygenic Risk Score prediction.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class AttentionModel(nn.Module):
    """
    Attention-based model with self-attention mechanism for SNP importance.
    
    This model uses multi-head self-attention to learn which SNPs are most
    important for phenotype prediction. It can provide interpretable attention
    weights that indicate the relative importance of different genomic regions.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        n_attention_heads: int = 4,
        dropout_rate: float = 0.3
    ):
        """
        Initialize the Attention model.
        
        Args:
            input_dim: Number of input features (SNPs)
            output_dim: Number of output features (phenotypes)
            hidden_dim: Hidden dimension for projections
            attention_dim: Dimension for attention computation (not used in current implementation)
            n_attention_heads: Number of attention heads
            dropout_rate: Dropout probability for regularization
        """
        super(AttentionModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_attention_heads
        
        # Initial projection to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head self-attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_attention_heads,
            dropout=dropout_rate,
            batch_first=True  # Use batch first format
        )
        
        # Feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Layer normalization for stable training
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Output projection layers
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Store attention weights for interpretability
        self.last_attention_weights = None
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, n_snps)
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor of shape (batch_size, n_phenotypes)
            Optionally returns attention weights if return_attention=True
        """
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Add sequence dimension: (batch_size, hidden_dim) -> (batch_size, 1, hidden_dim)
        x = x.unsqueeze(1)
        
        # Apply self-attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        
        # Store attention weights for interpretability
        self.last_attention_weights = attn_weights
        
        # Apply feed-forward network with residual connection
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        
        # Remove sequence dimension: (batch_size, 1, hidden_dim) -> (batch_size, hidden_dim)
        x = x.squeeze(1)
        
        # Generate output predictions
        output = self.output_projection(x)
        
        if return_attention:
            return output, attn_weights
        return output
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get the last computed attention weights.
        
        Returns:
            Attention weights from the last forward pass, or None if not available
        """
        return self.last_attention_weights
    
    def get_snp_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute SNP importance scores based on attention and gradients.
        
        Args:
            x: Input tensor of shape (batch_size, n_snps)
            
        Returns:
            SNP importance scores of shape (n_snps,)
        """
        # Enable gradient computation
        x.requires_grad = True
        
        # Forward pass with attention weights
        output, attn_weights = self.forward(x, return_attention=True)
        
        # Calculate gradients with respect to input
        grad = torch.autograd.grad(
            outputs=output.sum(),
            inputs=x,
            create_graph=False
        )[0]
        
        # Combine gradient information with attention weights
        # Note: This is a simplified importance measure
        gradient_importance = grad.abs().mean(dim=0)
        
        # If we have multi-head attention, average across heads
        if attn_weights is not None:
            # attn_weights shape: (batch_size, n_heads, 1, 1)
            # We need to process this appropriately
            attention_importance = attn_weights.mean(dim=(0, 1, 2, 3))
            # Expand to match gradient dimension if needed
            if attention_importance.numel() == 1:
                attention_importance = attention_importance.expand(gradient_importance.shape)
        else:
            attention_importance = torch.ones_like(gradient_importance)
        
        # Combine both importance measures
        combined_importance = gradient_importance * attention_importance
        
        return combined_importance
    
    def get_hidden_representations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get hidden representations after attention layers.
        
        Args:
            x: Input tensor of shape (batch_size, n_snps)
            
        Returns:
            Hidden representations of shape (batch_size, hidden_dim)
        """
        x = self.input_projection(x)
        x = x.unsqueeze(1)
        
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        
        x = x.squeeze(1)
        
        return x