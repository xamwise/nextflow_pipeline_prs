import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
import math


class SimpleMLP(nn.Module):
    """
    Multi-Layer Perceptron model for genotype data.
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
        super(SimpleMLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        output = self.output_layer(features)
        return output


class SimpleCNNModel(nn.Module):
    """
    1D Convolutional Neural Network model for genotype data.
    Treats SNPs as a sequence.
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
        super(SimpleCNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build convolutional layers
        conv_layers = []
        in_channels = 1
        current_dim = input_dim
        
        for i, (out_channels, kernel_size, pool_size) in enumerate(
            zip(channels, kernel_sizes, pool_sizes)
        ):
            conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
            )
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(pool_size))
            conv_layers.append(nn.Dropout(dropout_rate))
            
            in_channels = out_channels
            current_dim = current_dim // pool_size
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate flattened dimension
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
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add channel dimension
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        # Output
        output = self.output_layer(x)
        return output


class TransformerModel(nn.Module):
    """
    Transformer-based model for genotype data.
    Treats SNPs as tokens in a sequence.
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
        super(TransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        
        # Embedding layer to project SNPs to d_model dimension
        self.embedding = nn.Linear(1, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout_rate, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Global pooling
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Output layers
        self.fc = nn.Linear(d_model, d_model // 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(d_model // 2, output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input: (batch_size, seq_len) -> (batch_size, seq_len, 1)
        x = x.unsqueeze(-1)
        
        # Embedding
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global pooling: (batch_size, seq_len, d_model) -> (batch_size, d_model)
        x = x.transpose(1, 2)
        x = self.pooling(x).squeeze(-1)
        
        # Output layers
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        output = self.output_layer(x)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class AttentionModel(nn.Module):
    """
    Attention-based model with self-attention mechanism for SNP importance.
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
        super(AttentionModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initial projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input
        x = self.input_projection(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        
        # Global pooling
        x = x.squeeze(1)
        
        # Output
        output = self.output_projection(x)
        return output


class GenotypeEnsembleModel(nn.Module):
    """
    Ensemble model combining multiple architectures.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        ensemble_method: str = 'average',
        weights: Optional[List[float]] = None
    ):
        super(GenotypeEnsembleModel, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        
        if weights is not None:
            assert len(weights) == len(models)
            self.weights = torch.tensor(weights)
        else:
            self.weights = torch.ones(len(models)) / len(models)
        
        if ensemble_method == 'learned':
            # Learn ensemble weights
            self.ensemble_weights = nn.Parameter(torch.ones(len(models)) / len(models))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        outputs = torch.stack(outputs, dim=0)
        
        if self.ensemble_method == 'average':
            return torch.mean(outputs, dim=0)
        elif self.ensemble_method == 'weighted':
            weights = self.weights.to(x.device)
            weights = weights.view(-1, 1, 1)
            return torch.sum(outputs * weights, dim=0)
        elif self.ensemble_method == 'learned':
            weights = F.softmax(self.ensemble_weights, dim=0)
            weights = weights.view(-1, 1, 1)
            return torch.sum(outputs * weights, dim=0)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")


def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create a model based on configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized model
    """
    model_type = config['model_type']
    
    if model_type == 'mlp':
        return SimpleMLP(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            hidden_dims=config.get('hidden_dims', [1024, 512, 256]),
            dropout_rate=config.get('dropout_rate', 0.3),
            activation=config.get('activation', 'relu'),
            batch_norm=config.get('batch_norm', True)
        )
    elif model_type == 'cnn':
        return SimpleCNNModel(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            channels=config.get('channels', [32, 64, 128]),
            kernel_sizes=config.get('kernel_sizes', [7, 5, 3]),
            pool_sizes=config.get('pool_sizes', [2, 2, 2]),
            fc_dims=config.get('fc_dims', [256, 128]),
            dropout_rate=config.get('dropout_rate', 0.3)
        )
    elif model_type == 'transformer':
        return TransformerModel(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            d_model=config.get('d_model', 256),
            n_heads=config.get('n_heads', 8),
            n_layers=config.get('n_layers', 4),
            d_ff=config.get('d_ff', 1024),
            dropout_rate=config.get('dropout_rate', 0.1)
        )
    elif model_type == 'attention':
        return AttentionModel(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            hidden_dim=config.get('hidden_dim', 256),
            attention_dim=config.get('attention_dim', 128),
            n_attention_heads=config.get('n_attention_heads', 4),
            dropout_rate=config.get('dropout_rate', 0.3)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")