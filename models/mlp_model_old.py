"""
Multi-Layer Perceptron model for Polygenic Risk Score prediction.
"""

import torch
import torch.nn as nn
from typing import List


class SimpleMLP(nn.Module):
    """
    Multi-Layer Perceptron model for genotype data.
    
    This model uses fully connected layers with batch normalization and dropout
    for regularization. Suitable for learning complex non-linear relationships
    between SNPs and phenotypes.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [1024, 512, 256],
        dropout_rate: float = 0.3,
        activation: str = 'relu',
        batch_norm: bool = True
    ):
        """
        Initialize the MLP model.
        
        Args:
            input_dim: Number of input features (SNPs)
            output_dim: Number of output features (phenotypes)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability for regularization
            activation: Activation function ('relu', 'elu', 'leaky_relu')
            batch_norm: Whether to use batch normalization
        """
        super(SimpleMLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization (if enabled)
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation function
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            # Dropout for regularization
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming initialization for ReLU activation
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                # Initialize batch norm parameters
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, n_snps)
            
        Returns:
            Output tensor of shape (batch_size, n_phenotypes)
        """
        # Extract features through hidden layers
        features = self.feature_extractor(x)
        
        # Generate output predictions
        output = self.output_layer(features)
        
        return output
    
    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature importance scores using gradient-based method.
        
        Args:
            x: Input tensor of shape (batch_size, n_snps)
            
        Returns:
            Feature importance scores of shape (n_snps,)
        """
        x.requires_grad = True
        output = self.forward(x)
        
        # Calculate gradients with respect to input
        grad = torch.autograd.grad(
            outputs=output.sum(),
            inputs=x,
            create_graph=False
        )[0]
        
        # Average absolute gradients across batch
        importance = grad.abs().mean(dim=0)
        
        return importance