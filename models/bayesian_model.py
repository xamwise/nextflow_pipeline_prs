"""
Bayesian Neural Network model for Polygenic Risk Score prediction with uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with weight uncertainty.
    
    Implements a linear layer where weights and biases are distributions
    rather than point estimates, using the reparameterization trick for
    variational inference.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_var: float = 1.0,
        bias: bool = True
    ):
        """
        Initialize Bayesian linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            prior_var: Prior variance for weight distributions
            bias: Whether to include bias parameters
        """
        super(BayesianLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Prior parameters (assume Gaussian prior)
        self.prior_mean = 0.0
        self.prior_var = prior_var
        self.prior_log_var = math.log(prior_var)
        
        # Variational parameters for weights
        self.weight_mean = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.weight_log_var = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        
        # Variational parameters for bias
        if bias:
            self.bias_mean = nn.Parameter(torch.Tensor(out_features))
            self.bias_log_var = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mean', None)
            self.register_parameter('bias_log_var', None)
        
        # Initialize parameters
        self.reset_parameters()
        
        # For KL divergence calculation
        self.kl_div = 0
    
    def reset_parameters(self):
        """Initialize parameters using appropriate distributions."""
        # Initialize means using Xavier/Glorot initialization
        stdv = 1. / math.sqrt(self.weight_mean.size(1))
        self.weight_mean.data.uniform_(-stdv, stdv)
        self.weight_log_var.data.fill_(-5)  # Start with low variance
        
        if self.use_bias:
            self.bias_mean.data.uniform_(-stdv, stdv)
            self.bias_log_var.data.fill_(-5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with weight sampling.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Sample weights using reparameterization trick
        if self.training:
            # During training, sample weights
            weight_std = torch.exp(0.5 * self.weight_log_var)
            weight_eps = torch.randn_like(weight_std)
            weight = self.weight_mean + weight_eps * weight_std
            
            if self.use_bias:
                bias_std = torch.exp(0.5 * self.bias_log_var)
                bias_eps = torch.randn_like(bias_std)
                bias = self.bias_mean + bias_eps * bias_std
            else:
                bias = None
            
            # Calculate KL divergence for this layer
            self.kl_div = self.calculate_kl_div()
        else:
            # During inference, use mean weights
            weight = self.weight_mean
            bias = self.bias_mean if self.use_bias else None
            self.kl_div = 0
        
        return F.linear(x, weight, bias)
    
    def calculate_kl_div(self) -> torch.Tensor:
        """
        Calculate KL divergence between posterior and prior distributions.
        
        Returns:
            KL divergence loss term
        """
        # KL divergence for weights
        kl_weight = 0.5 * torch.sum(
            torch.exp(self.weight_log_var) / self.prior_var +
            (self.weight_mean - self.prior_mean) ** 2 / self.prior_var -
            1 - self.weight_log_var + self.prior_log_var
        )
        
        # KL divergence for bias
        kl_bias = 0
        if self.use_bias:
            kl_bias = 0.5 * torch.sum(
                torch.exp(self.bias_log_var) / self.prior_var +
                (self.bias_mean - self.prior_mean) ** 2 / self.prior_var -
                1 - self.bias_log_var + self.prior_log_var
            )
        
        return kl_weight + kl_bias


class BayesianNeuralNetwork(nn.Module):
    """
    Bayesian Neural Network for PRS prediction with uncertainty quantification.
    
    This model provides not only predictions but also uncertainty estimates,
    which is crucial for clinical decision-making and risk assessment.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        prior_var: float = 1.0,
        activation: str = 'relu',
        dropout_rate: float = 0.1,
        n_samples: int = 10
    ):
        """
        Initialize Bayesian Neural Network.
        
        Args:
            input_dim: Number of input features (SNPs)
            output_dim: Number of output features (phenotypes)
            hidden_dims: List of hidden layer dimensions
            prior_var: Prior variance for Bayesian layers
            activation: Activation function ('relu', 'elu', 'leaky_relu')
            dropout_rate: Additional dropout for regularization
            n_samples: Number of samples for prediction during inference
        """
        super(BayesianNeuralNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_samples = n_samples
        
        # Build Bayesian layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Add Bayesian linear layer
            layers.append(BayesianLinear(prev_dim, hidden_dim, prior_var))
            
            # Add activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            # Add dropout for additional regularization
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.ModuleList(layers)
        
        # Output layer (also Bayesian)
        self.output_layer = BayesianLinear(prev_dim, output_dim, prior_var)
        
        # Store all Bayesian layers for KL calculation
        self.bayesian_layers = [
            layer for layer in self.feature_extractor 
            if isinstance(layer, BayesianLinear)
        ]
        self.bayesian_layers.append(self.output_layer)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_uncertainty: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the Bayesian network.
        
        Args:
            x: Input tensor of shape (batch_size, n_snps)
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            If return_uncertainty is False:
                Output tensor of shape (batch_size, n_phenotypes)
            If return_uncertainty is True:
                Tuple of (mean predictions, uncertainty)
        """
        if not return_uncertainty or self.training:
            # Single forward pass
            h = x
            for layer in self.feature_extractor:
                h = layer(h)
            output = self.output_layer(h)
            return output
        else:
            # Multiple forward passes for uncertainty estimation
            return self.predict_with_uncertainty(x)
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor, 
        n_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            x: Input tensor of shape (batch_size, n_snps)
            n_samples: Number of forward passes (default: self.n_samples)
            
        Returns:
            Tuple of:
                - Mean predictions of shape (batch_size, n_phenotypes)
                - Uncertainty (std) of shape (batch_size, n_phenotypes)
        """
        if n_samples is None:
            n_samples = self.n_samples
        
        # Set to training mode to sample weights
        was_training = self.training
        self.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                h = x
                for layer in self.feature_extractor:
                    if isinstance(layer, BayesianLinear):
                        h = layer(h)
                    else:
                        # For non-Bayesian layers (activation, dropout)
                        if isinstance(layer, nn.Dropout):
                            h = layer(h) if was_training else h
                        else:
                            h = layer(h)
                
                output = self.output_layer(h)
                predictions.append(output)
        
        # Restore original training mode
        self.train(was_training)
        
        # Stack predictions: (n_samples, batch_size, n_phenotypes)
        predictions = torch.stack(predictions)
        
        # Calculate mean and uncertainty
        mean_prediction = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_prediction, uncertainty
    
    def get_kl_divergence(self) -> torch.Tensor:
        """
        Calculate total KL divergence for all Bayesian layers.
        
        Returns:
            Total KL divergence loss
        """
        kl_div = 0
        for layer in self.bayesian_layers:
            kl_div += layer.kl_div
        return kl_div
    
    def elbo_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        n_batches: int = 1,
        kl_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Calculate Evidence Lower Bound (ELBO) loss.
        
        Args:
            predictions: Model predictions
            targets: Target values
            n_batches: Total number of batches in dataset (for KL scaling)
            kl_weight: Weight for KL divergence term
            
        Returns:
            ELBO loss (negative log likelihood + KL divergence)
        """
        # Negative log likelihood (assuming Gaussian likelihood)
        nll = F.mse_loss(predictions, targets, reduction='mean')
        
        # KL divergence (scaled by dataset size)
        kl_div = self.get_kl_divergence() / n_batches
        
        # Total loss
        loss = nll + kl_weight * kl_div
        
        return loss
    
    def get_weight_statistics(self) -> dict:
        """
        Get statistics about weight distributions.
        
        Returns:
            Dictionary with mean and variance statistics for each layer
        """
        stats = {}
        for i, layer in enumerate(self.bayesian_layers):
            layer_stats = {
                'weight_mean': layer.weight_mean.mean().item(),
                'weight_std': torch.exp(0.5 * layer.weight_log_var).mean().item(),
                'weight_min': layer.weight_mean.min().item(),
                'weight_max': layer.weight_mean.max().item()
            }
            if layer.use_bias:
                layer_stats.update({
                    'bias_mean': layer.bias_mean.mean().item(),
                    'bias_std': torch.exp(0.5 * layer.bias_log_var).mean().item()
                })
            stats[f'layer_{i}'] = layer_stats
        return stats
    
    def enable_dropout_inference(self):
        """Enable dropout during inference for MC Dropout uncertainty."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def get_epistemic_uncertainty(
        self, 
        x: torch.Tensor, 
        n_samples: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get epistemic (model) uncertainty using multiple forward passes.
        
        Args:
            x: Input tensor
            n_samples: Number of forward passes
            
        Returns:
            Tuple of (mean predictions, epistemic uncertainty)
        """
        return self.predict_with_uncertainty(x, n_samples)
    
    def get_aleatoric_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get aleatoric (data) uncertainty.
        
        Note: This requires modifying the output layer to predict both
        mean and variance. Currently returns zeros as placeholder.
        
        Args:
            x: Input tensor
            
        Returns:
            Aleatoric uncertainty (currently placeholder)
        """
        # This would require the model to output both mean and variance
        # For now, return placeholder
        with torch.no_grad():
            output = self.forward(x)
        return torch.zeros_like(output)
    
    def calibrate_uncertainty(
        self, 
        val_loader: torch.utils.data.DataLoader,
        n_bins: int = 10
    ) -> dict:
        """
        Calibrate uncertainty estimates using validation data.
        
        Args:
            val_loader: Validation data loader
            n_bins: Number of bins for calibration
            
        Returns:
            Calibration statistics
        """
        all_means = []
        all_stds = []
        all_targets = []
        
        self.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                mean, std = self.predict_with_uncertainty(inputs)
                all_means.append(mean)
                all_stds.append(std)
                all_targets.append(targets)
        
        all_means = torch.cat(all_means)
        all_stds = torch.cat(all_stds)
        all_targets = torch.cat(all_targets)
        
        # Calculate calibration statistics
        errors = (all_means - all_targets).abs()
        
        # Bin by uncertainty
        calibration_stats = []
        for i in range(n_bins):
            low = i * (1.0 / n_bins)
            high = (i + 1) * (1.0 / n_bins)
            
            # Find samples in this uncertainty range
            quantile_low = torch.quantile(all_stds, low)
            quantile_high = torch.quantile(all_stds, high)
            
            mask = (all_stds >= quantile_low) & (all_stds < quantile_high)
            if mask.any():
                bin_errors = errors[mask].mean().item()
                bin_std = all_stds[mask].mean().item()
                calibration_stats.append({
                    'bin': i,
                    'expected_std': bin_std,
                    'observed_error': bin_errors,
                    'n_samples': mask.sum().item()
                })
        
        return calibration_stats