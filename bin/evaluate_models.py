import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
import logging
from typing import List, Dict, Any, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, precision_recall_curve, auc
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wandb
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bin.genotype_dataset import GenotypeDataModule
from bin.models import create_model, GenotypeEnsembleModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PRSEvaluator:
    """
    Comprehensive evaluator for Polygenic Risk Score models.
    """
    
    def __init__(
        self,
        models: List[str],
        data_module: GenotypeDataModule,
        device: str = 'cuda',
        use_wandb: bool = True
    ):
        """
        Initialize the PRS evaluator.
        
        Args:
            models: List of model checkpoint paths
            data_module: Data module for loading test data
            device: Device to use for evaluation
            use_wandb: Whether to log to Weights & Biases
        """
        self.model_paths = models
        self.data_module = data_module
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_wandb = use_wandb
        
        if use_wandb:
            wandb.init(project="prs-evaluation", name="model_evaluation")
        
        # Load models
        self.models = []
        self.configs = []
        self.fold_metrics = []
        
        for model_path in models:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract configuration
            config = checkpoint['config']['model']
            config['input_dim'] = data_module.get_input_dim()
            config['output_dim'] = data_module.get_output_dim()
            
            # Create and load model
            model = create_model(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            self.models.append(model)
            self.configs.append(config)
            
            # Store fold-specific metrics if available
            if 'training_history' in checkpoint:
                self.fold_metrics.append(checkpoint['training_history'])
        
        logger.info(f"Loaded {len(self.models)} models for PRS evaluation")
    
    def evaluate_single_model(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        return_features: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Evaluate a single model for PRS prediction.
        
        Args:
            model: Model to evaluate
            data_loader: Data loader for evaluation
            return_features: Whether to return intermediate features
            
        Returns:
            Tuple of (predictions, targets, metrics)
        """
        model.eval()
        all_predictions = []
        all_targets = []
        all_features = []
        
        with torch.no_grad():
            for genotypes, phenotypes in data_loader:
                genotypes = genotypes.to(self.device)
                phenotypes = phenotypes.to(self.device)
                
                # Get predictions
                predictions = model(genotypes)
                
                # Optionally extract features for visualization
                if return_features and hasattr(model, 'feature_extractor'):
                    features = model.feature_extractor(genotypes)
                    all_features.append(features.cpu().numpy())
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(phenotypes.cpu().numpy())
        
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        
        # Calculate PRS-specific metrics
        metrics = self.calculate_prs_metrics(predictions, targets)
        
        if return_features and all_features:
            features = np.concatenate(all_features)
            return predictions, targets, metrics, features
        
        return predictions, targets, metrics
    
    def calculate_prs_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        threshold_percentiles: List[int] = [80, 90, 95, 99]
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for PRS evaluation.
        
        Args:
            predictions: Model predictions (PRS scores)
            targets: Ground truth values
            threshold_percentiles: Percentiles for risk stratification
            
        Returns:
            Dictionary of PRS-specific metrics
        """
        metrics = {}
        
        # Basic regression metrics
        metrics['mse'] = float(mean_squared_error(targets, predictions))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        metrics['mae'] = float(mean_absolute_error(targets, predictions))
        metrics['r2'] = float(r2_score(targets, predictions))
        
        # Correlation metrics (important for PRS)
        if predictions.shape[1] == 1:
            pearson_r, pearson_p = stats.pearsonr(
                predictions.flatten(),
                targets.flatten()
            )
            spearman_r, spearman_p = stats.spearmanr(
                predictions.flatten(),
                targets.flatten()
            )
            metrics['pearson_r'] = float(pearson_r)
            metrics['pearson_p'] = float(pearson_p)
            metrics['spearman_r'] = float(spearman_r)
            metrics['spearman_p'] = float(spearman_p)
        
        # Risk stratification metrics
        predictions_flat = predictions.flatten()
        targets_flat = targets.flatten()
        
        for percentile in threshold_percentiles:
            threshold = np.percentile(predictions_flat, percentile)
            high_risk = predictions_flat >= threshold
            
            # Odds ratio calculation
            if len(np.unique(targets_flat)) == 2:  # Binary outcome
                a = np.sum((high_risk == 1) & (targets_flat == 1))
                b = np.sum((high_risk == 1) & (targets_flat == 0))
                c = np.sum((high_risk == 0) & (targets_flat == 1))
                d = np.sum((high_risk == 0) & (targets_flat == 0))
                
                if b > 0 and c > 0:
                    odds_ratio = (a * d) / (b * c)
                    metrics[f'odds_ratio_p{percentile}'] = float(odds_ratio)
            
            # Mean outcome in high vs low risk groups
            if np.sum(high_risk) > 0:
                mean_high = np.mean(targets_flat[high_risk])
                mean_low = np.mean(targets_flat[~high_risk])
                metrics[f'mean_diff_p{percentile}'] = float(mean_high - mean_low)
                metrics[f'risk_ratio_p{percentile}'] = float(mean_high / (mean_low + 1e-8))
        
        # Percentile-based performance
        deciles = np.percentile(predictions_flat, [10, 20, 30, 40, 50, 60, 70, 80, 90])
        metrics['decile_spread'] = float(deciles[-1] - deciles[0])
        
        # Variance explained
        metrics['variance_explained'] = float(
            1 - np.var(targets_flat - predictions_flat) / np.var(targets_flat)
        )
        
        # Calibration metrics
        calibration = self.calculate_calibration_metrics(predictions_flat, targets_flat)
        metrics.update(calibration)
        
        return metrics
    
    def calculate_calibration_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Calculate calibration metrics for PRS.
        
        Args:
            predictions: PRS predictions
            targets: True outcomes
            n_bins: Number of bins for calibration
            
        Returns:
            Dictionary of calibration metrics
        """
        metrics = {}
        
        # Bin predictions
        bins = np.linspace(predictions.min(), predictions.max(), n_bins + 1)
        bin_indices = np.digitize(predictions, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Calculate calibration
        expected = []
        observed = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                expected.append(np.mean(predictions[mask]))
                observed.append(np.mean(targets[mask]))
        
        if len(expected) > 1:
            # Calibration slope
            slope, intercept, r_value, _, _ = stats.linregress(expected, observed)
            metrics['calibration_slope'] = float(slope)
            metrics['calibration_intercept'] = float(intercept)
            metrics['calibration_r2'] = float(r_value ** 2)
            
            # Expected Calibration Error (ECE)
            ece = np.mean(np.abs(np.array(expected) - np.array(observed)))
            metrics['expected_calibration_error'] = float(ece)
        
        return metrics
    
    def evaluate_ensemble(self) -> Dict[str, Any]:
        """
        Evaluate ensemble of models (cross-validation folds).
        
        Returns:
            Dictionary with ensemble evaluation results
        """
        test_loader = self.data_module.test_dataloader()
        
        # Get predictions from all models
        all_model_predictions = []
        all_model_metrics = []
        
        for i, model in enumerate(self.models):
            predictions, targets, metrics = self.evaluate_single_model(model, test_loader)
            all_model_predictions.append(predictions)
            all_model_metrics.append(metrics)
            
            logger.info(f"Fold {i} - R²: {metrics['r2']:.4f}, "
                       f"Pearson r: {metrics.get('pearson_r', 0):.4f}")
        
        # Ensemble predictions (average)
        ensemble_predictions = np.mean(all_model_predictions, axis=0)
        
        # Calculate ensemble metrics
        ensemble_metrics = self.calculate_prs_metrics(ensemble_predictions, targets)
        
        # Create ensemble model for saving
        ensemble_model = GenotypeEnsembleModel(
            models=self.models,
            ensemble_method='average'
        )
        
        results = {
            'individual_metrics': all_model_metrics,
            'ensemble_metrics': ensemble_metrics,
            'ensemble_predictions': ensemble_predictions,
            'targets': targets,
            'individual_predictions': all_model_predictions
        }
        
        # Log to W&B
        if self.use_wandb:
            wandb.log({
                'ensemble_r2': ensemble_metrics['r2'],
                'ensemble_pearson_r': ensemble_metrics.get('pearson_r', 0),
                'mean_fold_r2': np.mean([m['r2'] for m in all_model_metrics]),
                'std_fold_r2': np.std([m['r2'] for m in all_model_metrics])
            })
        
        return results
    
    def create_evaluation_report(
        self,
        results: Dict[str, Any],
        output_path: str
    ):
        """
        Create comprehensive HTML evaluation report for PRS models.
        
        Args:
            results: Evaluation results
            output_path: Path to save HTML report
        """
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'PRS Distribution', 'Predicted vs Actual', 'Calibration Plot',
                'Risk Stratification', 'Fold Performance', 'Feature Importance',
                'Decile Analysis', 'ROC-like Curve', 'Model Comparison'
            ),
            specs=[
                [{'type': 'histogram'}, {'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'box'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'table'}]
            ]
        )
        
        ensemble_pred = results['ensemble_predictions'].flatten()
        targets = results['targets'].flatten()
        
        # 1. PRS Distribution
        fig.add_trace(
            go.Histogram(x=ensemble_pred, name='PRS Scores', nbinsx=50),
            row=1, col=1
        )
        
        # 2. Predicted vs Actual
        fig.add_trace(
            go.Scatter(
                x=targets, y=ensemble_pred,
                mode='markers', name='Predictions',
                marker=dict(size=3, opacity=0.5)
            ),
            row=1, col=2
        )
        
        # Add diagonal line
        min_val = min(targets.min(), ensemble_pred.min())
        max_val = max(targets.max(), ensemble_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', name='Perfect Calibration',
                line=dict(dash='dash', color='red')
            ),
            row=1, col=2
        )
        
        # 3. Calibration Plot
        n_bins = 10
        bins = np.linspace(ensemble_pred.min(), ensemble_pred.max(), n_bins + 1)
        bin_indices = np.digitize(ensemble_pred, bins) - 1
        expected = []
        observed = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                expected.append(np.mean(ensemble_pred[mask]))
                observed.append(np.mean(targets[mask]))
        
        fig.add_trace(
            go.Scatter(x=expected, y=observed, mode='markers+lines', name='Calibration'),
            row=1, col=3
        )
        
        # 4. Risk Stratification
        percentiles = [80, 90, 95, 99]
        risk_ratios = []
        for p in percentiles:
            key = f'risk_ratio_p{p}'
            if key in results['ensemble_metrics']:
                risk_ratios.append(results['ensemble_metrics'][key])
            else:
                risk_ratios.append(1.0)
        
        fig.add_trace(
            go.Bar(x=[f'Top {100-p}%' for p in percentiles], y=risk_ratios, name='Risk Ratio'),
            row=2, col=1
        )
        
        # 5. Fold Performance
        fold_r2s = [m['r2'] for m in results['individual_metrics']]
        fig.add_trace(
            go.Box(y=fold_r2s, name='Fold R²'),
            row=2, col=2
        )
        
        # 6. Decile Analysis
        deciles = np.percentile(ensemble_pred, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        decile_means = []
        for i in range(len(deciles) - 1):
            mask = (ensemble_pred >= deciles[i]) & (ensemble_pred < deciles[i + 1])
            decile_means.append(np.mean(targets[mask]))
        
        fig.add_trace(
            go.Bar(x=[f'D{i+1}' for i in range(9)], y=decile_means, name='Mean Outcome'),
            row=3, col=1
        )
        
        # 7. Percentile plot
        percentiles_range = np.arange(0, 101, 5)
        thresholds = np.percentile(ensemble_pred, percentiles_range)
        mean_outcomes = []
        
        for threshold in thresholds:
            high_risk = ensemble_pred >= threshold
            if np.sum(high_risk) > 0:
                mean_outcomes.append(np.mean(targets[high_risk]))
            else:
                mean_outcomes.append(np.nan)
        
        fig.add_trace(
            go.Scatter(
                x=percentiles_range, y=mean_outcomes,
                mode='lines+markers', name='Mean Outcome by Percentile'
            ),
            row=3, col=2
        )
        
        # 8. Model Comparison Table
        comparison_data = []
        for i, metrics in enumerate(results['individual_metrics']):
            comparison_data.append([
                f'Fold {i}',
                f"{metrics['r2']:.4f}",
                f"{metrics.get('pearson_r', 0):.4f}",
                f"{metrics['rmse']:.4f}"
            ])
        
        # Add ensemble
        ensemble_metrics = results['ensemble_metrics']
        comparison_data.append([
            'Ensemble',
            f"{ensemble_metrics['r2']:.4f}",
            f"{ensemble_metrics.get('pearson_r', 0):.4f}",
            f"{ensemble_metrics['rmse']:.4f}"
        ])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Model', 'R²', 'Pearson r', 'RMSE']),
                cells=dict(values=list(zip(*comparison_data)))
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            title='Polygenic Risk Score Model Evaluation Report',
            height=1200,
            showlegend=True
        )
        
        # Save HTML
        fig.write_html(output_path)
        logger.info(f"Evaluation report saved to {output_path}")
        
        # Log to W&B
        if self.use_wandb:
            wandb.log({"evaluation_report": wandb.Html(open(output_path).read())})
    
    def save_predictions(self, results: Dict[str, Any], output_path: str):
        """
        Save predictions to CSV for downstream analysis.
        
        Args:
            results: Evaluation results
            output_path: Path to save predictions
        """
        df = pd.DataFrame({
            'sample_id': range(len(results['targets'])),
            'true_value': results['targets'].flatten(),
            'prs_score': results['ensemble_predictions'].flatten()
        })
        
        # Add individual fold predictions
        for i, pred in enumerate(results['individual_predictions']):
            df[f'fold_{i}_prediction'] = pred.flatten()
        
        # Add percentile ranks
        df['prs_percentile'] = df['prs_score'].rank(pct=True) * 100
        
        # Add risk categories
        df['risk_category'] = pd.cut(
            df['prs_percentile'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['Very Low', 'Low', 'Average', 'High', 'Very High']
        )
        
        df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
        # Log to W&B
        if self.use_wandb:
            wandb.log({"predictions_table": wandb.Table(dataframe=df.head(100))})


def main():
    parser = argparse.ArgumentParser(description='Evaluate PRS models')
    parser.add_argument('--models', nargs='+', required=True, help='Model checkpoint paths')
    parser.add_argument('--metrics', nargs='+', required=True, help='Metrics JSON paths')
    parser.add_argument('--genotype_file', type=str, required=True)
    parser.add_argument('--phenotype_file', type=str, required=True)
    parser.add_argument('--indices_file', type=str, required=True)
    parser.add_argument('--output_report', type=str, required=True)
    parser.add_argument('--output_predictions', type=str, required=True)
    parser.add_argument('--output_comparison', type=str, required=True)
    parser.add_argument('--wandb_project', type=str, default='prs-evaluation')
    
    args = parser.parse_args()
    
    # Initialize W&B
    wandb.init(project=args.wandb_project, name='final_evaluation')
    
    # Create data module
    data_module = GenotypeDataModule(
        h5_file=args.genotype_file,
        phenotype_file=args.phenotype_file,
        indices_file=args.indices_file,
        batch_size=32,
        num_workers=4
    )
    
    # Create evaluator
    evaluator = PRSEvaluator(
        models=args.models,
        data_module=data_module,
        use_wandb=True
    )
    
    # Evaluate ensemble
    results = evaluator.evaluate_ensemble()
    
    # Create report
    evaluator.create_evaluation_report(results, args.output_report)
    
    # Save predictions
    evaluator.save_predictions(results, args.output_predictions)
    
    # Save comparison
    comparison = {
        'individual_metrics': results['individual_metrics'],
        'ensemble_metrics': results['ensemble_metrics'],
        'model_configs': evaluator.configs
    }
    
    with open(args.output_comparison, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Final W&B logging
    wandb.log({
        'final_ensemble_r2': results['ensemble_metrics']['r2'],
        'final_ensemble_pearson': results['ensemble_metrics'].get('pearson_r', 0),
        'calibration_slope': results['ensemble_metrics'].get('calibration_slope', 1.0),
        'expected_calibration_error': results['ensemble_metrics'].get('expected_calibration_error', 0)
    })
    
    wandb.finish()
    logger.info("Evaluation completed successfully")


if __name__ == '__main__':
    main()