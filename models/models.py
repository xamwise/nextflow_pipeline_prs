"""
Main module for deep learning models for Polygenic Risk Score prediction.

This module imports all individual model architectures and provides
a factory function to create models based on configuration.
"""

from typing import Dict, Any
import torch.nn as nn

# Import all model architectures
from .mlp_model import SimpleMLP
from .cnn_model import SimpleCNNModel
from .transformer_model import TransformerModel
from .attention_model import AttentionModel
from .ensemble_model import GenotypeEnsembleModel
from .bayesian_model import BayesianNeuralNetwork

# Re-export models for convenience
__all__ = [
    'SimpleMLP',
    'SimpleCNNModel', 
    'TransformerModel',
    'AttentionModel',
    'GenotypeEnsembleModel',
    'BayesianNeuralNetwork',
    'create_model'
]


def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create a model based on configuration.
    
    Args:
        config: Model configuration dictionary containing:
            - model_type: Type of model to create ('mlp', 'cnn', 'transformer', 'attention')
            - input_dim: Number of input features (SNPs)
            - output_dim: Number of output features (phenotypes)
            - Other model-specific parameters
            
    Returns:
        Initialized PyTorch model
        
    Raises:
        ValueError: If unknown model type is specified
        
    Example:
        >>> config = {
        ...     'model_type': 'mlp',
        ...     'input_dim': 100000,
        ...     'output_dim': 1,
        ...     'hidden_dims': [1024, 512, 256],
        ...     'dropout_rate': 0.3
        ... }
        >>> model = create_model(config)
    """
    model_type = config.get('model_type', '').lower()
    
    # Extract common parameters
    input_dim = config.get('input_dim')
    output_dim = config.get('output_dim')
    
    if not input_dim or not output_dim:
        raise ValueError("input_dim and output_dim must be specified in config")
    
    # Create model based on type
    if model_type == 'mlp':
        return SimpleMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=config.get('hidden_dims', [1024, 512, 256]),
            dropout_rate=config.get('dropout_rate', 0.3),
            activation=config.get('activation', 'relu'),
            batch_norm=config.get('batch_norm', True)
        )
    
    elif model_type == 'cnn':
        return SimpleCNNModel(
            input_dim=input_dim,
            output_dim=output_dim,
            channels=config.get('channels', [32, 64, 128]),
            kernel_sizes=config.get('kernel_sizes', [7, 5, 3]),
            pool_sizes=config.get('pool_sizes', [2, 2, 2]),
            fc_dims=config.get('fc_dims', [256, 128]),
            dropout_rate=config.get('dropout_rate', 0.3)
        )
    
    elif model_type == 'transformer':
        return TransformerModel(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=config.get('d_model', 256),
            n_heads=config.get('n_heads', 8),
            n_layers=config.get('n_layers', 4),
            d_ff=config.get('d_ff', 1024),
            dropout_rate=config.get('dropout_rate', 0.1),
            max_seq_length=config.get('max_seq_length', max(10000, input_dim))
        )
    
    elif model_type == 'attention':
        return AttentionModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config.get('hidden_dim', 256),
            attention_dim=config.get('attention_dim', 128),
            n_attention_heads=config.get('n_attention_heads', 4),
            dropout_rate=config.get('dropout_rate', 0.3)
        )
    
    elif model_type == 'bayesian' or model_type == 'bnn':
        return BayesianNeuralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=config.get('hidden_dims', [512, 256, 128]),
            prior_var=config.get('prior_var', 1.0),
            activation=config.get('activation', 'relu'),
            dropout_rate=config.get('dropout_rate', 0.1),
            n_samples=config.get('n_samples', 10)
        )
    
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available types: mlp, cnn, transformer, attention, bayesian/bnn"
        )


def get_model_info(model_type: str) -> Dict[str, Any]:
    """
    Get information about a specific model type.
    
    Args:
        model_type: Type of model ('mlp', 'cnn', 'transformer', 'attention')
        
    Returns:
        Dictionary with model information including default parameters
    """
    model_info = {
        'mlp': {
            'class': SimpleMLP,
            'description': 'Multi-Layer Perceptron with batch normalization and dropout',
            'default_params': {
                'hidden_dims': [1024, 512, 256],
                'dropout_rate': 0.3,
                'activation': 'relu',
                'batch_norm': True
            }
        },
        'cnn': {
            'class': SimpleCNNModel,
            'description': '1D CNN for sequential SNP data',
            'default_params': {
                'channels': [32, 64, 128],
                'kernel_sizes': [7, 5, 3],
                'pool_sizes': [2, 2, 2],
                'fc_dims': [256, 128],
                'dropout_rate': 0.3
            }
        },
        'transformer': {
            'class': TransformerModel,
            'description': 'Transformer with self-attention for long-range SNP interactions',
            'default_params': {
                'd_model': 256,
                'n_heads': 8,
                'n_layers': 4,
                'd_ff': 1024,
                'dropout_rate': 0.1
            }
        },
        'attention': {
            'class': AttentionModel,
            'description': 'Attention-based model for interpretable SNP importance',
            'default_params': {
                'hidden_dim': 256,
                'attention_dim': 128,
                'n_attention_heads': 4,
                'dropout_rate': 0.3
            }
        },
        'bayesian': {
            'class': BayesianNeuralNetwork,
            'description': 'Bayesian Neural Network with uncertainty quantification',
            'default_params': {
                'hidden_dims': [512, 256, 128],
                'prior_var': 1.0,
                'activation': 'relu',
                'dropout_rate': 0.1,
                'n_samples': 10
            }
        },
        'bnn': {  # Alias for bayesian
            'class': BayesianNeuralNetwork,
            'description': 'Bayesian Neural Network with uncertainty quantification',
            'default_params': {
                'hidden_dims': [512, 256, 128],
                'prior_var': 1.0,
                'activation': 'relu',
                'dropout_rate': 0.1,
                'n_samples': 10
            }
        }
    }
    
    if model_type not in model_info:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_info[model_type]


def list_available_models() -> list:
    """
    List all available model types.
    
    Returns:
        List of available model type strings
    """
    return ['mlp', 'cnn', 'transformer', 'attention', 'bayesian', 'bnn']