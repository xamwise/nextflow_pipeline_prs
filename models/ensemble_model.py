"""
Ensemble model for Polygenic Risk Score prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any


class GenotypeEnsembleModel(nn.Module):
    """
    Ensemble model combining multiple architectures.
    
    This model combines predictions from multiple models using various
    ensemble strategies (averaging, weighted averaging, or learned weights).
    Ensemble methods often provide better generalization and robustness
    compared to individual models.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        ensemble_method: str = 'average',
        weights: Optional[List[float]] = None
    ):
        """
        Initialize the Ensemble model.
        
        Args:
            models: List of PyTorch models to ensemble
            ensemble_method: Method for combining predictions 
                           ('average', 'weighted', 'learned')
            weights: Optional list of weights for weighted averaging
                    (must sum to 1 for weighted method)
        """
        super(GenotypeEnsembleModel, self).__init__()
        
        if not models:
            raise ValueError("At least one model must be provided for ensemble")
        
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        self.n_models = len(models)
        
        # Initialize weights based on ensemble method
        if ensemble_method == 'weighted':
            if weights is not None:
                if len(weights) != len(models):
                    raise ValueError(
                        f"Number of weights ({len(weights)}) must match "
                        f"number of models ({len(models)})"
                    )
                # Normalize weights to sum to 1
                weight_sum = sum(weights)
                self.weights = torch.tensor(
                    [w / weight_sum for w in weights], 
                    dtype=torch.float32
                )
            else:
                # Equal weights if not specified
                self.weights = torch.ones(len(models), dtype=torch.float32) / len(models)
        
        elif ensemble_method == 'learned':
            # Learn ensemble weights during training
            self.ensemble_weights = nn.Parameter(
                torch.ones(len(models), dtype=torch.float32) / len(models)
            )
        
        elif ensemble_method == 'average':
            # Simple averaging - equal weights
            self.weights = torch.ones(len(models), dtype=torch.float32) / len(models)
        
        else:
            raise ValueError(
                f"Unknown ensemble method: {ensemble_method}. "
                f"Choose from 'average', 'weighted', or 'learned'"
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ensemble.
        
        Args:
            x: Input tensor of shape (batch_size, n_snps)
            
        Returns:
            Ensemble output tensor of shape (batch_size, n_phenotypes)
        """
        # Collect predictions from all models
        outputs = []
        for model in self.models:
            model_output = model(x)
            outputs.append(model_output)
        
        # Stack outputs: (n_models, batch_size, n_phenotypes)
        outputs = torch.stack(outputs, dim=0)
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == 'average':
            # Simple averaging
            ensemble_output = torch.mean(outputs, dim=0)
        
        elif self.ensemble_method == 'weighted':
            # Weighted averaging with fixed weights
            weights = self.weights.to(x.device)
            weights = weights.view(-1, 1, 1)  # Shape for broadcasting
            ensemble_output = torch.sum(outputs * weights, dim=0)
        
        elif self.ensemble_method == 'learned':
            # Weighted averaging with learned weights
            weights = F.softmax(self.ensemble_weights, dim=0)
            weights = weights.view(-1, 1, 1)  # Shape for broadcasting
            ensemble_output = torch.sum(outputs * weights, dim=0)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return ensemble_output
    
    def get_individual_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get predictions from each individual model in the ensemble.
        
        Args:
            x: Input tensor of shape (batch_size, n_snps)
            
        Returns:
            List of prediction tensors from each model
        """
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        return predictions
    
    def get_model_weights(self) -> torch.Tensor:
        """
        Get the current ensemble weights.
        
        Returns:
            Tensor of model weights
        """
        if self.ensemble_method == 'learned':
            return F.softmax(self.ensemble_weights, dim=0)
        else:
            return self.weights
    
    def set_model_weights(self, weights: List[float]):
        """
        Manually set ensemble weights (only for 'weighted' method).
        
        Args:
            weights: New weights for the ensemble
        """
        if self.ensemble_method != 'weighted':
            raise ValueError(
                "Can only set weights manually for 'weighted' ensemble method"
            )
        
        if len(weights) != self.n_models:
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of models ({self.n_models})"
            )
        
        # Normalize weights to sum to 1
        weight_sum = sum(weights)
        self.weights = torch.tensor(
            [w / weight_sum for w in weights], 
            dtype=torch.float32
        )
    
    def add_model(self, model: nn.Module, weight: float = None):
        """
        Add a new model to the ensemble.
        
        Args:
            model: New model to add
            weight: Weight for the new model (for weighted ensemble)
        """
        self.models.append(model)
        self.n_models += 1
        
        if self.ensemble_method == 'average':
            # Recalculate equal weights
            self.weights = torch.ones(self.n_models, dtype=torch.float32) / self.n_models
        
        elif self.ensemble_method == 'weighted':
            # Add new weight and renormalize
            if weight is None:
                weight = 1.0 / self.n_models
            
            current_weights = self.weights.tolist()
            current_weights.append(weight)
            weight_sum = sum(current_weights)
            self.weights = torch.tensor(
                [w / weight_sum for w in current_weights],
                dtype=torch.float32
            )
        
        elif self.ensemble_method == 'learned':
            # Reinitialize learnable weights
            self.ensemble_weights = nn.Parameter(
                torch.ones(self.n_models, dtype=torch.float32) / self.n_models
            )
    
    def remove_model(self, index: int):
        """
        Remove a model from the ensemble.
        
        Args:
            index: Index of the model to remove
        """
        if index >= self.n_models:
            raise ValueError(f"Index {index} out of range for {self.n_models} models")
        
        del self.models[index]
        self.n_models -= 1
        
        if self.n_models == 0:
            raise ValueError("Cannot remove last model from ensemble")
        
        # Update weights
        if self.ensemble_method in ['average', 'weighted']:
            if self.ensemble_method == 'weighted':
                # Remove weight and renormalize
                weights_list = self.weights.tolist()
                del weights_list[index]
                weight_sum = sum(weights_list)
                self.weights = torch.tensor(
                    [w / weight_sum for w in weights_list],
                    dtype=torch.float32
                )
            else:
                # Recalculate equal weights
                self.weights = torch.ones(self.n_models, dtype=torch.float32) / self.n_models
        
        elif self.ensemble_method == 'learned':
            # Reinitialize learnable weights
            self.ensemble_weights = nn.Parameter(
                torch.ones(self.n_models, dtype=torch.float32) / self.n_models
            )