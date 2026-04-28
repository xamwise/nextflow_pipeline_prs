"""
Multi-Layer Perceptron model for Polygenic Risk Score prediction (regression).

Forward returns raw continuous predictions of shape (batch_size, output_dim).
Pair with nn.MSELoss, nn.HuberLoss, or nn.GaussianNLLLoss externally.
"""

from typing import List

import torch
import torch.nn as nn


class SimpleMLP_R(nn.Module):
    """
    MLP for continuous phenotype prediction from genotype data.

    Hidden layers: Linear -> [BatchNorm1d] -> activation -> Dropout.
    Output layer:  Linear with no activation (raw real-valued predictions).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: List[int] = [1024, 512, 256],
        dropout_rate: float = 0.3,
        activation: str = "relu",
        batch_norm: bool = True,
    ):
        """
        Args:
            input_dim: number of input SNPs.
            output_dim: number of continuous targets (typically 1).
            hidden_dims: widths of hidden layers.
            dropout_rate: dropout probability after each hidden block.
            activation: 'relu' | 'elu' | 'leaky_relu'.
            batch_norm: whether to apply BatchNorm1d after each hidden Linear.
        """
        super().__init__()

        if output_dim < 1:
            raise ValueError("output_dim must be >= 1 for regression.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation

        # Match the init nonlinearity tag to the chosen activation so that
        # Kaiming gain is consistent with what the activation actually does.
        # (ELU has no dedicated mode; 'relu' is a reasonable proxy.)
        init_nonlinearity = {
            "relu": "relu",
            "leaky_relu": "leaky_relu",
            "elu": "relu",
        }.get(activation)
        if init_nonlinearity is None:
            raise ValueError(f"Unknown activation: {activation}")
        self._init_nonlinearity = init_nonlinearity

        # Build hidden layers
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self._make_activation(activation))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

        self._initialize_weights()

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        if name == "elu":
            return nn.ELU()
        if name == "leaky_relu":
            return nn.LeakyReLU()
        raise ValueError(f"Unknown activation: {name}")

    def _initialize_weights(self) -> None:
        """
        Hidden Linear: Kaiming init matched to the chosen activation.
        Output Linear: Xavier init -- no non-linearity follows the head, so
                       a ReLU-tuned Kaiming gain would over-scale weights by
                       sqrt(2) and produce needlessly large initial losses.
        BatchNorm:     weight=1, bias=0.
        """
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity=self._init_nonlinearity
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            (batch_size, output_dim) -- raw continuous predictions.
        """
        features = self.feature_extractor(x)
        return self.output_layer(features)

    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gradient-based saliency: |d(sum_outputs)/dx| averaged over the batch.

        Clones+detaches the input so the caller's tensor is not mutated.

        Args:
            x: (batch_size, input_dim)
        Returns:
            (input_dim,) importance scores.
        """
        x = x.clone().detach().requires_grad_(True)
        output = self.forward(x)

        grad = torch.autograd.grad(
            outputs=output.sum(),
            inputs=x,
            create_graph=False,
        )[0]

        return grad.abs().mean(dim=0)