"""
1D Convolutional Neural Network model for Polygenic Risk Score prediction.
"""

import torch
import torch.nn as nn
from typing import List


class SimpleCNNModel(nn.Module):
    """
    1D Convolutional Neural Network model for genotype data.
    
    This model treats SNPs as a sequence and uses 1D convolutions to capture
    local patterns and interactions between nearby SNPs. Useful for detecting
    haplotype effects and local genomic patterns.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        channels: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [7, 5, 3],
        pool_sizes: List[int] = [2, 2, 2],
        fc_dims: List[int] = [256, 128],
        dropout_rate: float = 0.3
    ):
        """
        Initialize the CNN model.
        
        Args:
            input_dim: Number of input features (SNPs)
            output_dim: Number of output features (phenotypes)
            channels: List of channel sizes for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            pool_sizes: List of pooling sizes for each layer
            fc_dims: List of fully connected layer dimensions
            dropout_rate: Dropout probability for regularization
        """
        super(SimpleCNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Validate input parameters
        assert len(channels) == len(kernel_sizes) == len(pool_sizes), \
            "channels, kernel_sizes, and pool_sizes must have the same length"
        
        # Build convolutional layers
        conv_layers = []
        in_channels = 1  # Start with single channel (raw genotype values)
        current_dim = input_dim
        
        for i, (out_channels, kernel_size, pool_size) in enumerate(
            zip(channels, kernel_sizes, pool_sizes)
        ):
            # Convolutional layer with padding to preserve dimension
            conv_layers.append(
                nn.Conv1d(
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    padding=kernel_size//2
                )
            )
            # Batch normalization for stable training
            conv_layers.append(nn.BatchNorm1d(out_channels))
            # ReLU activation
            conv_layers.append(nn.ReLU())
            # Max pooling to reduce dimension
            conv_layers.append(nn.MaxPool1d(pool_size))
            # Dropout for regularization
            conv_layers.append(nn.Dropout(dropout_rate))
            
            in_channels = out_channels
            current_dim = current_dim // pool_size
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate flattened dimension after convolutions
        self.flatten_dim = channels[-1] * current_dim
        
        # Build fully connected layers
        fc_layers = []
        prev_dim = self.flatten_dim
        
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(prev_dim, fc_dim))
            fc_layers.append(nn.BatchNorm1d(fc_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout_rate))
            prev_dim = fc_dim
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Kaiming initialization for conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Kaiming initialization for linear layers
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
        # Add channel dimension: (batch_size, n_snps) -> (batch_size, 1, n_snps)
        x = x.unsqueeze(1)
        
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = self.fc_layers(x)
        
        # Generate output predictions
        output = self.output_layer(x)
        
        return output
    
    def get_conv_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract convolutional features before the fully connected layers.
        
        Args:
            x: Input tensor of shape (batch_size, n_snps)
            
        Returns:
            Convolutional features tensor
        """
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        return x
    
    def get_receptive_field(self) -> int:
        """
        Calculate the receptive field of the CNN.
        
        Returns:
            Size of the receptive field in terms of input SNPs
        """
        receptive_field = 1
        for kernel_size, pool_size in zip(
            self.kernel_sizes if hasattr(self, 'kernel_sizes') else [7, 5, 3],
            self.pool_sizes if hasattr(self, 'pool_sizes') else [2, 2, 2]
        ):
            receptive_field = receptive_field + (kernel_size - 1)
            receptive_field = receptive_field * pool_size
        return receptive_field