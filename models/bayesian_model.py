"""
Bayesian Neural Network model for Polygenic Risk Score prediction with uncertainty quantification.

Improvements over baseline:
- Spike-and-slab (scale mixture of Gaussians) prior for sparsity
- Heteroscedastic output for proper aleatoric uncertainty
- Local reparameterization trick for lower-variance gradients
- Deterministic input compression for high-dimensional SNP data
- Binary classification support (BCE likelihood)
- Correct KL scaling by dataset size
- Proper calibration metrics (coverage-based)
- Bayesian layers registered via ModuleList
- Explicit sampling flag decoupled from training mode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer using the local reparameterization trick
    with a scale mixture of Gaussians prior (spike-and-slab approximation).

    Prior: pi * N(0, sigma1^2) + (1 - pi) * N(0, sigma2^2)
    where sigma1 is large (slab) and sigma2 is small (spike).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_pi: float = 0.5,
        prior_sigma1: float = 1.0,
        prior_sigma2: float = 0.002,
        bias: bool = True,
        init_log_var: float = -3.0,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Scale-mixture-of-Gaussians prior parameters
        self.prior_pi = prior_pi
        self.prior_sigma1 = prior_sigma1
        self.prior_sigma2 = prior_sigma2
        self.prior_log_sigma1 = math.log(prior_sigma1)
        self.prior_log_sigma2 = math.log(prior_sigma2)

        # Variational posterior parameters for weights
        self.weight_mean = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_log_var = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias_mean = nn.Parameter(torch.empty(out_features))
            self.bias_log_var = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_mean", None)
            self.register_parameter("bias_log_var", None)

        self.reset_parameters(init_log_var)

        # KL divergence cached after each forward pass
        self.register_buffer('_kl_div_default', torch.tensor(0.0))
        self.kl_div = self._kl_div_default

        # Explicit flag: sample weights even in eval mode
        self._sample_weights = True

    def reset_parameters(self, init_log_var: float = -3.0):
        stdv = 1.0 / math.sqrt(self.in_features)
        self.weight_mean.data.uniform_(-stdv, stdv)
        self.weight_log_var.data.fill_(init_log_var)
        if self.use_bias:
            self.bias_mean.data.uniform_(-stdv, stdv)
            self.bias_log_var.data.fill_(init_log_var)

    # ------------------------------------------------------------------
    # Prior & posterior log-prob helpers
    # ------------------------------------------------------------------

    def _log_gaussian(self, x: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
        return -0.5 * math.log(2 * math.pi) - math.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2)

    def _log_scale_mixture_prior(self, x: torch.Tensor) -> torch.Tensor:
        """Log probability under the scale-mixture-of-Gaussians prior."""
        log_p1 = self._log_gaussian(x, 0.0, self.prior_sigma1) + math.log(self.prior_pi)
        log_p2 = self._log_gaussian(x, 0.0, self.prior_sigma2) + math.log(1.0 - self.prior_pi)
        return torch.logaddexp(log_p1, log_p2)

    def _log_posterior(self, x: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Log probability under the variational posterior q(w)."""
        var = torch.exp(log_var)
        return -0.5 * (math.log(2 * math.pi) + log_var + (x - mean) ** 2 / var)

    # ------------------------------------------------------------------
    # Forward with local reparameterization trick
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._sample_weights:
            return self._forward_local_reparam(x)
        else:
            return self._forward_deterministic(x)

    def _forward_local_reparam(self, x: torch.Tensor) -> torch.Tensor:
        """
        Local reparameterization trick: instead of sampling weight matrices,
        sample the pre-activation outputs directly.  This gives the same
        expected gradient but with lower variance.

        act_mean = x @ W_mean^T + b_mean
        act_var  = x^2 @ exp(W_log_var)^T + exp(b_log_var)   [element-wise]
        act      = act_mean + sqrt(act_var) * eps
        """
        weight_var = torch.exp(self.weight_log_var)

        act_mean = F.linear(x, self.weight_mean, self.bias_mean if self.use_bias else None)

        act_var = F.linear(x.pow(2), weight_var)
        if self.use_bias:
            act_var = act_var + torch.exp(self.bias_log_var)

        act_std = torch.sqrt(act_var + 1e-8)
        eps = torch.randn_like(act_mean)
        output = act_mean + act_std * eps

        # Compute KL via sampled weights (sample once for the KL term)
        self.kl_div = self._compute_kl()

        return output

    def _forward_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        self.kl_div = torch.tensor(0.0, device=x.device)
        return F.linear(x, self.weight_mean, self.bias_mean if self.use_bias else None)

    def _compute_kl(self) -> torch.Tensor:
        """
        Monte-Carlo KL: KL(q || p) ≈ log q(w) - log p(w) with a single sample.
        """
        # Sample weights from posterior
        weight_std = torch.exp(0.5 * self.weight_log_var)
        weight = self.weight_mean + weight_std * torch.randn_like(weight_std)

        log_q = self._log_posterior(weight, self.weight_mean, self.weight_log_var).sum()
        log_p = self._log_scale_mixture_prior(weight).sum()

        if self.use_bias:
            bias_std = torch.exp(0.5 * self.bias_log_var)
            bias = self.bias_mean + bias_std * torch.randn_like(bias_std)
            log_q = log_q + self._log_posterior(bias, self.bias_mean, self.bias_log_var).sum()
            log_p = log_p + self._log_scale_mixture_prior(bias).sum()

        return log_q - log_p


class InputCompressor(nn.Module):
    """
    Deterministic dimensionality reduction for high-dimensional SNP inputs.
    Reduces parameter count before Bayesian layers are applied.
    """

    def __init__(
        self,
        input_dim: int,
        compressed_dim: int,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for i in range(n_layers):
            out = compressed_dim if i == n_layers - 1 else max(compressed_dim, dim // 4)
            layers.extend([
                nn.Linear(dim, out),
                nn.BatchNorm1d(out),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            dim = out
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BayesianNeuralNetwork(nn.Module):
    """
    Bayesian Neural Network for PRS prediction with:
    - Deterministic input compression for p >> n genomic data
    - Scale-mixture-of-Gaussians (spike-and-slab) prior for sparsity
    - Local reparameterization trick
    - Heteroscedastic output (learned aleatoric uncertainty)
    - Binary classification and regression support
    - Coverage-based calibration
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: List[int] = [256, 128],
        compressed_dim: int = 512,
        compressor_layers: int = 1,
        compressor_dropout: float = 0.1,
        prior_pi: float = 0.5,
        prior_sigma1: float = 1.0,
        prior_sigma2: float = 0.002,
        activation: str = "relu",
        n_samples: int = 10,
        task: str = "regression",
        init_log_var: float = -3.0,
    ):
        """
        Args:
            input_dim: Number of input SNPs.
            output_dim: Number of phenotypes / classes.
            hidden_dims: Bayesian hidden layer sizes.
            compressed_dim: Dimension after deterministic compression.
            compressor_layers: Layers in the deterministic compressor.
            compressor_dropout: Dropout in compressor.
            prior_pi: Mixture weight for slab component.
            prior_sigma1: Slab std.
            prior_sigma2: Spike std.
            activation: 'relu' | 'elu' | 'leaky_relu'.
            n_samples: MC samples at inference.
            task: 'regression' (heteroscedastic) or 'binary'.
            init_log_var: Initial log-variance for Bayesian layers.
        """
        super().__init__()
        self.output_dim = output_dim
        self.n_samples = n_samples
        self.task = task

        # --- Deterministic input compressor ---
        self.compressor = InputCompressor(
            input_dim, compressed_dim, compressor_layers, compressor_dropout
        )

        # --- Bayesian hidden layers ---
        _act = {"relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}
        if activation not in _act:
            raise ValueError(f"Unknown activation: {activation}")

        bayesian_layers: list[nn.Module] = []
        prev = compressed_dim
        for h in hidden_dims:
            bayesian_layers.append(
                BayesianLinear(prev, h, prior_pi, prior_sigma1, prior_sigma2,
                               init_log_var=init_log_var)
            )
            bayesian_layers.append(_act[activation]())
            prev = h

        self.hidden = nn.ModuleList(bayesian_layers)

        # --- Output layer ---
        if task == "regression":
            # Heteroscedastic: predict mean and log-variance
            self.output_layer = BayesianLinear(
                prev, output_dim * 2, prior_pi, prior_sigma1, prior_sigma2,
                init_log_var=init_log_var,
            )
        elif task == "binary":
            # Logits for binary classification
            self.output_layer = BayesianLinear(
                prev, output_dim, prior_pi, prior_sigma1, prior_sigma2,
                init_log_var=init_log_var,
            )
        else:
            raise ValueError(f"task must be 'regression' or 'binary', got '{task}'")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bayesian_modules(self):
        for m in self.modules():
            if isinstance(m, BayesianLinear):
                yield m

    def set_sampling(self, enabled: bool):
        """Enable or disable weight sampling in all Bayesian layers."""
        for m in self._bayesian_modules():
            m._sample_weights = enabled

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """One forward pass (compressed → hidden → output)."""
        h = self.compressor(x)
        for layer in self.hidden:
            h = layer(h)
        return self.output_layer(h)

    def _split_output(self, raw: torch.Tensor):
        """Split raw output into mean + log_var (regression) or logits (binary)."""
        if self.task == "regression":
            mean, log_var = raw.chunk(2, dim=-1)
            return mean, log_var
        else:
            return raw, None

    def forward(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = False,
    ):
        """
        Args:
            x: (batch, n_snps)
            return_uncertainty: if True, run MC sampling and return
                (mean, epistemic_std, aleatoric_std).

        Returns:
            Training / no uncertainty:
                regression → (pred_mean, pred_log_var)
                binary     → logits
            With uncertainty:
                (mean, epistemic_std, aleatoric_std)
        """
        if not return_uncertainty:
            raw = self._forward_single(x)
            if self.task == "regression":
                return self._split_output(raw)
            return raw

        return self.predict_with_uncertainty(x)

    # ------------------------------------------------------------------
    # Uncertainty estimation
    # ------------------------------------------------------------------

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MC forward passes → (mean, epistemic_std, aleatoric_std).

        Epistemic uncertainty: std of the means across MC samples.
        Aleatoric uncertainty: mean of per-sample predicted std (regression)
                               or 0 (binary, where it's implicit in the logit).
        """
        n = n_samples or self.n_samples
        self.set_sampling(True)
        self.compressor.eval()  # deterministic compressor stays in eval

        means = []
        aleatoric_vars = []

        with torch.no_grad():
            for _ in range(n):
                raw = self._forward_single(x)
                if self.task == "regression":
                    m, lv = raw.chunk(2, dim=-1)
                    means.append(m)
                    aleatoric_vars.append(torch.exp(lv))
                else:
                    means.append(torch.sigmoid(raw))

        means = torch.stack(means)  # (n, batch, dim)
        epistemic_std = means.std(dim=0)
        mean_pred = means.mean(dim=0)

        if self.task == "regression":
            aleatoric_std = torch.stack(aleatoric_vars).mean(dim=0).sqrt()
        else:
            aleatoric_std = torch.zeros_like(mean_pred)

        return mean_pred, epistemic_std, aleatoric_std

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def get_kl_divergence(self) -> torch.Tensor:
        return sum(m.kl_div for m in self._bayesian_modules())

    def elbo_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        n_datapoints: int,
        kl_weight: float = 1.0,
        pred_log_var: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        ELBO = NLL + (kl_weight / n_datapoints) * KL.

        Args:
            predictions: predicted means (regression) or logits (binary).
            targets: ground truth.
            n_datapoints: total training set size for correct KL scaling.
            kl_weight: additional KL multiplier (for warm-up schedules).
            pred_log_var: predicted log-variance per sample (regression only).
        """
        if self.task == "regression":
            if pred_log_var is not None:
                # Heteroscedastic Gaussian NLL
                precision = torch.exp(-pred_log_var)
                nll = 0.5 * (pred_log_var + precision * (predictions - targets) ** 2).mean()
            else:
                nll = F.mse_loss(predictions, targets)
        else:
            nll = F.binary_cross_entropy_with_logits(
                predictions, targets.float(), reduction="mean"
            )

        kl = self.get_kl_divergence()
        return nll + (kl_weight / n_datapoints) * kl

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_weight_statistics(self) -> Dict[str, dict]:
        stats = {}
        for i, layer in enumerate(self._bayesian_modules()):
            s = {
                "weight_mean_avg": layer.weight_mean.mean().item(),
                "weight_std_avg": torch.exp(0.5 * layer.weight_log_var).mean().item(),
                "weight_mean_range": (layer.weight_mean.min().item(), layer.weight_mean.max().item()),
            }
            if layer.use_bias:
                s["bias_mean_avg"] = layer.bias_mean.mean().item()
                s["bias_std_avg"] = torch.exp(0.5 * layer.bias_log_var).mean().item()
            stats[f"bayesian_layer_{i}"] = s
        return stats

    @torch.no_grad()
    def calibrate_uncertainty(
        self,
        val_loader: torch.utils.data.DataLoader,
        confidence_levels: Optional[List[float]] = None,
        n_samples: int = 50,
        device: Optional[torch.device] = None,
    ) -> Dict[str, object]:
        """
        Coverage-based calibration: for each nominal confidence level,
        compute the fraction of true values falling inside the predicted
        credible interval.

        Returns dict with:
            - 'coverage': {level: observed_coverage}
            - 'mean_epistemic_std': average epistemic uncertainty
            - 'mean_aleatoric_std': average aleatoric uncertainty (regression)
        """
        if confidence_levels is None:
            confidence_levels = [0.5, 0.8, 0.9, 0.95, 0.99]

        all_means, all_epist, all_aleat, all_targets = [], [], [], []

        for inputs, targets in val_loader:
            if device is not None:
                inputs, targets = inputs.to(device), targets.to(device)
            mean, ep, al = self.predict_with_uncertainty(inputs, n_samples)
            all_means.append(mean)
            all_epist.append(ep)
            all_aleat.append(al)
            all_targets.append(targets)

        means = torch.cat(all_means)
        epist = torch.cat(all_epist)
        aleat = torch.cat(all_aleat)
        targets = torch.cat(all_targets)

        # Total predictive std (epistemic + aleatoric in quadrature)
        total_std = torch.sqrt(epist ** 2 + aleat ** 2 + 1e-8)

        coverage = {}
        for level in confidence_levels:
            # z-score for symmetric credible interval
            from scipy.stats import norm as _norm
            z = _norm.ppf(0.5 + level / 2.0)
            lower = means - z * total_std
            upper = means + z * total_std
            frac = ((targets >= lower) & (targets <= upper)).float().mean().item()
            coverage[level] = frac

        return {
            "coverage": coverage,
            "mean_epistemic_std": epist.mean().item(),
            "mean_aleatoric_std": aleat.mean().item(),
        }