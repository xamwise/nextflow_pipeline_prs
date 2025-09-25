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

from nagelkerke_r2 import calculate_nagelkerke_r2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bin.genotype_dataset import GenotypeDataModule
from models import create_model, GenotypeEnsembleModel

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
        
    def _detect_task_type(self, targets: np.ndarray) -> str:
        """
        Detect task type based on target data.
        
        Returns:
            'binary', 'multiclass', or 'regression'
        """
        unique_values = np.unique(targets)
        n_unique = len(unique_values)
        
        # Check if all values are integers and in a small range
        if np.allclose(targets, targets.astype(int)):
            if n_unique == 2:
                return 'binary'
            elif n_unique <= 20:  # Configurable threshold
                return 'multiclass'
    
        return 'regression'
    
    def calculate_nagelkerke_r2(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate Nagelkerke R2 for binary classification."""
        return calculate_nagelkerke_r2(predictions, targets)
    
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
        Calculate comprehensive metrics for PRS evaluation based on task type.
        """
        metrics = {}
        
        # Detect task type
        task_type = self._detect_task_type(targets)
        metrics['task_type'] = task_type
        
        predictions_flat = predictions.flatten()
        targets_flat = targets.flatten()
        
        if task_type == 'regression':
            # Regression metrics
            metrics['mse'] = float(mean_squared_error(targets_flat, predictions_flat))
            metrics['rmse'] = float(np.sqrt(metrics['mse']))
            metrics['mae'] = float(mean_absolute_error(targets_flat, predictions_flat))
            metrics['r2'] = float(r2_score(targets_flat, predictions_flat))
            
            # Correlation metrics
            pearson_r, pearson_p = stats.pearsonr(predictions_flat, targets_flat)
            spearman_r, spearman_p = stats.spearmanr(predictions_flat, targets_flat)
            metrics['pearson_r'] = float(pearson_r)
            metrics['pearson_p'] = float(pearson_p)
            metrics['spearman_r'] = float(spearman_r)
            metrics['spearman_p'] = float(spearman_p)
            
            # Variance explained
            metrics['variance_explained'] = float(
                1 - np.var(targets_flat - predictions_flat) / np.var(targets_flat)
            )
            
            # Risk stratification for continuous outcomes
            for percentile in threshold_percentiles:
                threshold = np.percentile(predictions_flat, percentile)
                high_risk = predictions_flat >= threshold
                
                if np.sum(high_risk) > 0:
                    mean_high = np.mean(targets_flat[high_risk])
                    mean_low = np.mean(targets_flat[~high_risk])
                    metrics[f'mean_diff_p{percentile}'] = float(mean_high - mean_low)
                    metrics[f'risk_ratio_p{percentile}'] = float(mean_high / (mean_low + 1e-8))
        
        elif task_type == 'binary':
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score, 
                roc_auc_score, average_precision_score
            )
            
            # Convert logits to probabilities if needed
            if predictions_flat.min() < 0 or predictions_flat.max() > 1:
                probs = 1 / (1 + np.exp(-predictions_flat))  # sigmoid
            else:
                probs = predictions_flat
                
            pred_labels = (probs > 0.5).astype(int)
            targets_int = targets_flat.astype(int)
            
            # Classification metrics
            metrics['accuracy'] = float(accuracy_score(targets_int, pred_labels))
            metrics['precision'] = float(precision_score(targets_int, pred_labels, average='binary', zero_division=0))
            metrics['recall'] = float(recall_score(targets_int, pred_labels, average='binary', zero_division=0))
            metrics['f1'] = float(f1_score(targets_int, pred_labels, average='binary', zero_division=0))
            
            # AUC metrics
            try:
                metrics['auc_roc'] = float(roc_auc_score(targets_int, probs))
                metrics['auc_pr'] = float(average_precision_score(targets_int, probs))
            except ValueError:
                metrics['auc_roc'] = 0.0
                metrics['auc_pr'] = 0.0
            
            # Pseudo-R² metrics
            metrics['nagelkerke_r2'] = self.calculate_nagelkerke_r2(predictions, targets)
            
            # Risk stratification for binary outcomes
            for percentile in threshold_percentiles:
                threshold = np.percentile(probs, percentile)
                high_risk = probs >= threshold
                
                if np.sum(high_risk) > 0 and np.sum(~high_risk) > 0:
                    # Odds ratio calculation
                    a = np.sum((high_risk == 1) & (targets_int == 1))
                    b = np.sum((high_risk == 1) & (targets_int == 0))
                    c = np.sum((high_risk == 0) & (targets_int == 1))
                    d = np.sum((high_risk == 0) & (targets_int == 0))
                    
                    if b > 0 and c > 0:
                        odds_ratio = (a * d) / (b * c)
                        metrics[f'odds_ratio_p{percentile}'] = float(odds_ratio)
                    
                    # Risk ratio
                    risk_high = a / (a + b) if (a + b) > 0 else 0
                    risk_low = c / (c + d) if (c + d) > 0 else 0
                    metrics[f'risk_ratio_p{percentile}'] = float(risk_high / (risk_low + 1e-8))
        
        elif task_type == 'multiclass':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            pred_labels = np.argmax(predictions, axis=1)
            targets_int = targets_flat.astype(int)
            
            metrics['accuracy'] = float(accuracy_score(targets_int, pred_labels))
            metrics['precision'] = float(precision_score(targets_int, pred_labels, average='weighted', zero_division=0))
            metrics['recall'] = float(recall_score(targets_int, pred_labels, average='weighted', zero_division=0))
            metrics['f1'] = float(f1_score(targets_int, pred_labels, average='weighted', zero_division=0))
        
        # Calibration metrics (for all task types)
        calibration = self.calculate_calibration_metrics(predictions_flat, targets_flat, task_type)
        metrics.update(calibration)
        
        # Percentile-based performance
        deciles = np.percentile(predictions_flat, [10, 20, 30, 40, 50, 60, 70, 80, 90])
        metrics['decile_spread'] = float(deciles[-1] - deciles[0])
        
        return metrics
    
    def calculate_calibration_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        task_type: str,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """Calculate calibration metrics based on task type."""
        metrics = {}
        
        if task_type == 'binary':
            # For binary classification, use probability calibration
            if predictions.min() < 0 or predictions.max() > 1:
                probs = 1 / (1 + np.exp(-predictions))  # sigmoid
            else:
                probs = predictions
                
            # Use sklearn's calibration tools
            from sklearn.calibration import calibration_curve
            
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    targets.astype(int), probs, n_bins=n_bins
                )
                
                # Brier score
                metrics['brier_score'] = float(np.mean((probs - targets) ** 2))
                
                # Expected Calibration Error
                ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                metrics['expected_calibration_error'] = float(ece)
                
            except ValueError:
                metrics['brier_score'] = 1.0
                metrics['expected_calibration_error'] = 1.0
        
        else:
            # For regression, use traditional calibration slope
            bins = np.linspace(predictions.min(), predictions.max(), n_bins + 1)
            bin_indices = np.digitize(predictions, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            expected = []
            observed = []
            
            for i in range(n_bins):
                mask = bin_indices == i
                if np.sum(mask) > 0:
                    expected.append(np.mean(predictions[mask]))
                    observed.append(np.mean(targets[mask]))
            
            if len(expected) > 1:
                slope, intercept, r_value, _, _ = stats.linregress(expected, observed)
                metrics['calibration_slope'] = float(slope)
                metrics['calibration_intercept'] = float(intercept)
                metrics['calibration_r2'] = float(r_value ** 2)
                
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
            
            if task_type is None:
                task_type = metrics.get('task_type', 'regression')
            
            # Task-specific logging
            if task_type == 'regression':
                logger.info(f"Fold {i} - R²: {metrics['r2']:.4f}, "
                        f"Pearson r: {metrics.get('pearson_r', 0):.4f}")
            elif task_type == 'binary':
                logger.info(f"Fold {i} - Accuracy: {metrics['accuracy']:.4f}, "
                        f"AUC: {metrics.get('auc_roc', 0):.4f}, "
                        f"Nagelkerke R²: {metrics.get('nagelkerke_r2', 0):.4f}")
            else:  # multiclass
                logger.info(f"Fold {i} - Accuracy: {metrics['accuracy']:.4f}, "
                        f"F1: {metrics.get('f1', 0):.4f}")
        
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
        Create comprehensive HTML evaluation report for PRS models, adapted for different task types.
        
        Args:
            results: Evaluation results
            output_path: Path to save HTML report
        """
        ensemble_pred = results['ensemble_predictions'].flatten()
        targets = results['targets'].flatten()
        ensemble_metrics = results['ensemble_metrics']
        
        # Detect task type
        task_type = ensemble_metrics.get('task_type', 'regression')
        
        # Create task-specific subplot configuration
        if task_type == 'regression':
            subplot_titles = (
                'PRS Distribution', 'Predicted vs Actual', 'Calibration Plot',
                'Risk Stratification', 'Fold Performance', 'Correlation Plot',
                'Decile Analysis', 'Residual Plot', 'Model Comparison'
            )
            specs = [
                [{'type': 'histogram'}, {'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'box'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'table'}]
            ]
        elif task_type == 'binary':
            subplot_titles = (
                'Score Distribution', 'ROC Curve', 'Precision-Recall Curve',
                'Calibration Plot', 'Fold Performance', 'Risk Stratification',
                'Decile Analysis', 'Confusion Matrix', 'Model Comparison'
            )
            specs = [
                [{'type': 'histogram'}, {'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'box'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'heatmap'}, {'type': 'table'}]
            ]
        else:  # multiclass
            subplot_titles = (
                'Class Distribution', 'Confusion Matrix', 'Per-Class Performance',
                'Prediction Confidence', 'Fold Performance', 'Class Probabilities',
                'Calibration by Class', 'Feature Importance', 'Model Comparison'
            )
            specs = [
                [{'type': 'bar'}, {'type': 'heatmap'}, {'type': 'bar'}],
                [{'type': 'histogram'}, {'type': 'box'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'table'}]
            ]
        
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=subplot_titles,
            specs=specs
        )
        
        if task_type == 'regression':
            self._create_regression_plots(fig, ensemble_pred, targets, results)
        elif task_type == 'binary':
            self._create_binary_plots(fig, ensemble_pred, targets, results)
        else:  # multiclass
            self._create_multiclass_plots(fig, ensemble_pred, targets, results)
        
        # Update layout with task-specific title
        fig.update_layout(
            title=f'Polygenic Risk Score Model Evaluation Report - {task_type.title()} Task',
            height=1200,
            showlegend=True
        )
        
        # Save HTML
        fig.write_html(output_path)
        logger.info(f"Evaluation report saved to {output_path}")
        
        # Log to W&B
        if self.use_wandb:
            wandb.log({"evaluation_report": wandb.Html(open(output_path).read())})

    def _create_regression_plots(self, fig, ensemble_pred, targets, results):
        """Create plots specific to regression tasks."""
        ensemble_metrics = results['ensemble_metrics']
        
        # 1. PRS Distribution
        fig.add_trace(
            go.Histogram(x=ensemble_pred, name='PRS Scores', nbinsx=50, opacity=0.7),
            row=1, col=1
        )
        
        # 2. Predicted vs Actual
        fig.add_trace(
            go.Scatter(
                x=targets, y=ensemble_pred,
                mode='markers', name='Predictions',
                marker=dict(size=3, opacity=0.6),
                text=[f'Pred: {p:.3f}<br>True: {t:.3f}' for p, t in zip(ensemble_pred, targets)],
                hovertemplate='%{text}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Perfect prediction line
        min_val = min(targets.min(), ensemble_pred.min())
        max_val = max(targets.max(), ensemble_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', name='Perfect Prediction',
                line=dict(dash='dash', color='red', width=2)
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
        
        if expected:
            fig.add_trace(
                go.Scatter(
                    x=expected, y=observed, 
                    mode='markers+lines', name='Calibration',
                    marker=dict(size=8), line=dict(width=3)
                ),
                row=1, col=3
            )
            
            # Perfect calibration line
            cal_min, cal_max = min(expected), max(expected)
            fig.add_trace(
                go.Scatter(
                    x=[cal_min, cal_max], y=[cal_min, cal_max],
                    mode='lines', name='Perfect Calibration',
                    line=dict(dash='dash', color='red', width=2)
                ),
                row=1, col=3
            )
        
        # 4. Risk Stratification
        percentiles = [80, 90, 95, 99]
        risk_ratios = []
        percentile_labels = []
        
        for p in percentiles:
            key = f'risk_ratio_p{p}'
            if key in ensemble_metrics:
                risk_ratios.append(ensemble_metrics[key])
                percentile_labels.append(f'Top {100-p}%')
        
        if risk_ratios:
            fig.add_trace(
                go.Bar(
                    x=percentile_labels, y=risk_ratios, 
                    name='Risk Ratio',
                    marker=dict(color='lightblue')
                ),
                row=2, col=1
            )
        
        # 5. Fold Performance
        fold_r2s = [m['r2'] for m in results['individual_metrics']]
        fig.add_trace(
            go.Box(y=fold_r2s, name='Fold R²', boxpoints='all'),
            row=2, col=2
        )
        
        # 6. Correlation Plot with trend line
        fig.add_trace(
            go.Scatter(
                x=targets, y=ensemble_pred,
                mode='markers', name='Correlation',
                marker=dict(size=3, opacity=0.6, color='blue')
            ),
            row=2, col=3
        )
        
        # Add trend line
        z = np.polyfit(targets, ensemble_pred, 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(
                x=np.sort(targets), y=p(np.sort(targets)),
                mode='lines', name=f'Trend (r={ensemble_metrics.get("pearson_r", 0):.3f})',
                line=dict(color='red', width=2)
            ),
            row=2, col=3
        )
        
        # 7. Decile Analysis
        deciles = np.percentile(ensemble_pred, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        decile_means = []
        
        for i in range(len(deciles) - 1):
            mask = (ensemble_pred >= deciles[i]) & (ensemble_pred < deciles[i + 1])
            if np.sum(mask) > 0:
                decile_means.append(np.mean(targets[mask]))
            else:
                decile_means.append(0)
        
        fig.add_trace(
            go.Bar(
                x=[f'D{i+1}' for i in range(len(decile_means))], 
                y=decile_means, name='Mean Outcome by Decile',
                marker=dict(color='lightgreen')
            ),
            row=3, col=1
        )
        
        # 8. Residual Plot
        residuals = targets - ensemble_pred
        fig.add_trace(
            go.Scatter(
                x=ensemble_pred, y=residuals,
                mode='markers', name='Residuals',
                marker=dict(size=3, opacity=0.6, color='purple')
            ),
            row=3, col=2
        )
        
        # Add zero line
        fig.add_trace(
            go.Scatter(
                x=[ensemble_pred.min(), ensemble_pred.max()], y=[0, 0],
                mode='lines', name='Zero Line',
                line=dict(dash='dash', color='black', width=1)
            ),
            row=3, col=2
        )
        
        # 9. Model Comparison Table
        self._add_model_comparison_table(fig, results, 'regression')

    def _create_binary_plots(self, fig, ensemble_pred, targets, results):
        """Create plots specific to binary classification tasks."""
        from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
        
        ensemble_metrics = results['ensemble_metrics']
        
        # Convert predictions to probabilities if needed
        if ensemble_pred.min() < 0 or ensemble_pred.max() > 1:
            probs = 1 / (1 + np.exp(-ensemble_pred))  # sigmoid
        else:
            probs = ensemble_pred
        
        pred_labels = (probs > 0.5).astype(int)
        targets_int = targets.astype(int)
        
        # 1. Score Distribution by Class
        pos_scores = probs[targets_int == 1]
        neg_scores = probs[targets_int == 0]
        
        fig.add_trace(
            go.Histogram(
                x=neg_scores, name='Class 0', nbinsx=30, 
                opacity=0.7, marker_color='red'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(
                x=pos_scores, name='Class 1', nbinsx=30, 
                opacity=0.7, marker_color='blue'
            ),
            row=1, col=1
        )
        
        # 2. ROC Curve
        if len(np.unique(targets_int)) > 1:
            fpr, tpr, _ = roc_curve(targets_int, probs)
            auc_score = ensemble_metrics.get('auc_roc', 0)
            
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines', name=f'ROC (AUC={auc_score:.3f})',
                    line=dict(width=3)
                ),
                row=1, col=2
            )
            
            # Random classifier line
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines', name='Random',
                    line=dict(dash='dash', color='gray')
                ),
                row=1, col=2
            )
        
        # 3. Precision-Recall Curve
        if len(np.unique(targets_int)) > 1:
            precision, recall, _ = precision_recall_curve(targets_int, probs)
            auc_pr = ensemble_metrics.get('auc_pr', 0)
            
            fig.add_trace(
                go.Scatter(
                    x=recall, y=precision,
                    mode='lines', name=f'PR (AUC={auc_pr:.3f})',
                    line=dict(width=3)
                ),
                row=1, col=3
            )
        
        # 4. Calibration Plot (Reliability Diagram)
        from sklearn.calibration import calibration_curve
        
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                targets_int, probs, n_bins=10
            )
            
            fig.add_trace(
                go.Scatter(
                    x=mean_predicted_value, y=fraction_of_positives,
                    mode='markers+lines', name='Calibration',
                    marker=dict(size=8), line=dict(width=3)
                ),
                row=2, col=1
            )
            
            # Perfect calibration line
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines', name='Perfect Calibration',
                    line=dict(dash='dash', color='red')
                ),
                row=2, col=1
            )
        except ValueError:
            pass
        
        # 5. Fold Performance (multiple metrics)
        metrics_to_plot = ['accuracy', 'auc_roc', 'nagelkerke_r2']
        fold_values = {metric: [] for metric in metrics_to_plot}
        
        for fold_metrics in results['individual_metrics']:
            for metric in metrics_to_plot:
                fold_values[metric].append(fold_metrics.get(metric, 0))
        
        for metric in metrics_to_plot:
            if fold_values[metric]:
                fig.add_trace(
                    go.Box(y=fold_values[metric], name=metric.replace('_', ' ').title()),
                    row=2, col=2
                )
        
        # 6. Risk Stratification (Odds Ratios)
        percentiles = [80, 90, 95, 99]
        odds_ratios = []
        percentile_labels = []
        
        for p in percentiles:
            key = f'odds_ratio_p{p}'
            if key in ensemble_metrics:
                odds_ratios.append(ensemble_metrics[key])
                percentile_labels.append(f'Top {100-p}%')
        
        if odds_ratios:
            fig.add_trace(
                go.Bar(
                    x=percentile_labels, y=odds_ratios,
                    name='Odds Ratio',
                    marker=dict(color='orange')
                ),
                row=2, col=3
            )
        
        # 7. Decile Analysis (Event Rates)
        deciles = np.percentile(probs, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        event_rates = []
        
        for i in range(len(deciles) - 1):
            mask = (probs >= deciles[i]) & (probs < deciles[i + 1])
            if np.sum(mask) > 0:
                event_rates.append(np.mean(targets_int[mask]))
            else:
                event_rates.append(0)
        
        fig.add_trace(
            go.Bar(
                x=[f'D{i+1}' for i in range(len(event_rates))],
                y=event_rates, name='Event Rate by Decile',
                marker=dict(color='lightcoral')
            ),
            row=3, col=1
        )
        
        # 8. Confusion Matrix
        cm = confusion_matrix(targets_int, pred_labels)
        
        fig.add_trace(
            go.Heatmap(
                z=cm, x=['Pred 0', 'Pred 1'], y=['True 1', 'True 0'],
                colorscale='Blues', showscale=True,
                text=cm, texttemplate='%{text}', textfont={"size": 16}
            ),
            row=3, col=2
        )
        
        # 9. Model Comparison Table
        self._add_model_comparison_table(fig, results, 'binary')

    def _create_multiclass_plots(self, fig, ensemble_pred, targets, results):
        """Create plots specific to multiclass classification tasks."""
        from sklearn.metrics import confusion_matrix
        
        ensemble_metrics = results['ensemble_metrics']
        n_classes = ensemble_pred.shape[1] if len(ensemble_pred.shape) > 1 else len(np.unique(targets))
        
        pred_labels = np.argmax(ensemble_pred, axis=1) if len(ensemble_pred.shape) > 1 else ensemble_pred
        targets_int = targets.astype(int)
        
        # 1. Class Distribution
        unique, counts = np.unique(targets_int, return_counts=True)
        fig.add_trace(
            go.Bar(x=[f'Class {i}' for i in unique], y=counts, name='True Class Distribution'),
            row=1, col=1
        )
        
        unique_pred, counts_pred = np.unique(pred_labels, return_counts=True)
        fig.add_trace(
            go.Bar(x=[f'Class {i}' for i in unique_pred], y=counts_pred, name='Predicted Class Distribution'),
            row=1, col=1
        )
        
        # 2. Confusion Matrix
        cm = confusion_matrix(targets_int, pred_labels)
        
        fig.add_trace(
            go.Heatmap(
                z=cm, 
                x=[f'Pred {i}' for i in range(cm.shape[1])],
                y=[f'True {i}' for i in range(cm.shape[0])],
                colorscale='Blues', showscale=True,
                text=cm, texttemplate='%{text}', textfont={"size": 12}
            ),
            row=1, col=2
        )
        
        # 3. Per-Class Performance
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precisions = precision_score(targets_int, pred_labels, average=None, zero_division=0)
        recalls = recall_score(targets_int, pred_labels, average=None, zero_division=0)
        f1s = f1_score(targets_int, pred_labels, average=None, zero_division=0)
        
        classes = [f'Class {i}' for i in range(len(precisions))]
        
        fig.add_trace(go.Bar(x=classes, y=precisions, name='Precision'), row=1, col=3)
        fig.add_trace(go.Bar(x=classes, y=recalls, name='Recall'), row=1, col=3)
        fig.add_trace(go.Bar(x=classes, y=f1s, name='F1-Score'), row=1, col=3)
        
        # 4. Prediction Confidence
        if len(ensemble_pred.shape) > 1:
            max_probs = np.max(ensemble_pred, axis=1)
            fig.add_trace(
                go.Histogram(x=max_probs, name='Max Probability', nbinsx=30),
                row=2, col=1
            )
        
        # 5. Fold Performance
        fold_accuracies = [m['accuracy'] for m in results['individual_metrics']]
        fold_f1s = [m['f1'] for m in results['individual_metrics']]
        
        fig.add_trace(go.Box(y=fold_accuracies, name='Accuracy'), row=2, col=2)
        fig.add_trace(go.Box(y=fold_f1s, name='F1-Score'), row=2, col=2)
        
        # 6. Class Probabilities (if available)
        if len(ensemble_pred.shape) > 1:
            for class_idx in range(min(n_classes, 5)):  # Limit to 5 classes for readability
                class_probs = ensemble_pred[:, class_idx]
                fig.add_trace(
                    go.Histogram(
                        x=class_probs, name=f'Class {class_idx} Probs',
                        nbinsx=20, opacity=0.7
                    ),
                    row=2, col=3
                )
        
        # 7-9. Placeholder plots for consistency
        fig.add_trace(go.Scatter(x=[0], y=[0], name='Placeholder'), row=3, col=1)
        fig.add_trace(go.Scatter(x=[0], y=[0], name='Placeholder'), row=3, col=2)
        
        # Model Comparison Table
        self._add_model_comparison_table(fig, results, 'multiclass')

    def _add_model_comparison_table(self, fig, results, task_type):
        """Add model comparison table based on task type."""
        comparison_data = []
        
        if task_type == 'regression':
            headers = ['Model', 'R²', 'Pearson r', 'RMSE', 'MAE']
            for i, metrics in enumerate(results['individual_metrics']):
                comparison_data.append([
                    f'Fold {i}',
                    f"{metrics['r2']:.4f}",
                    f"{metrics.get('pearson_r', 0):.4f}",
                    f"{metrics['rmse']:.4f}",
                    f"{metrics['mae']:.4f}"
                ])
            
            ensemble_metrics = results['ensemble_metrics']
            comparison_data.append([
                'Ensemble',
                f"{ensemble_metrics['r2']:.4f}",
                f"{ensemble_metrics.get('pearson_r', 0):.4f}",
                f"{ensemble_metrics['rmse']:.4f}",
                f"{ensemble_metrics['mae']:.4f}"
            ])
            
        elif task_type == 'binary':
            headers = ['Model', 'Accuracy', 'AUC-ROC', 'Nagelkerke R²', 'F1']
            for i, metrics in enumerate(results['individual_metrics']):
                comparison_data.append([
                    f'Fold {i}',
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics.get('auc_roc', 0):.4f}",
                    f"{metrics.get('nagelkerke_r2', 0):.4f}",
                    f"{metrics['f1']:.4f}"
                ])
            
            ensemble_metrics = results['ensemble_metrics']
            comparison_data.append([
                'Ensemble',
                f"{ensemble_metrics['accuracy']:.4f}",
                f"{ensemble_metrics.get('auc_roc', 0):.4f}",
                f"{ensemble_metrics.get('nagelkerke_r2', 0):.4f}",
                f"{ensemble_metrics['f1']:.4f}"
            ])
            
        else:  # multiclass
            headers = ['Model', 'Accuracy', 'F1-Score', 'Precision', 'Recall']
            for i, metrics in enumerate(results['individual_metrics']):
                comparison_data.append([
                    f'Fold {i}',
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['f1']:.4f}",
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}"
                ])
            
            ensemble_metrics = results['ensemble_metrics']
            comparison_data.append([
                'Ensemble',
                f"{ensemble_metrics['accuracy']:.4f}",
                f"{ensemble_metrics['f1']:.4f}",
                f"{ensemble_metrics['precision']:.4f}",
                f"{ensemble_metrics['recall']:.4f}"
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(values=headers),
                cells=dict(values=list(zip(*comparison_data)))
            ),
            row=3, col=3
        )
    
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