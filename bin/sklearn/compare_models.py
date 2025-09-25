#!/usr/bin/env python3
"""
Compare performance of multiple sklearn models.
Generates comprehensive comparison report with statistical tests.
"""

import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import glob
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_metrics_files(metrics_files: List[str]) -> Dict[str, Dict]:
    """
    Load metrics from multiple JSON files.
    
    Args:
        metrics_files: List of paths to metrics JSON files
        
    Returns:
        Dictionary mapping model names to metrics
    """
    all_metrics = {}
    
    for metrics_file in metrics_files:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Extract model name from filename or metrics
        model_name = metrics.get('model_type', Path(metrics_file).stem.replace('test_metrics_', ''))
        all_metrics[model_name] = metrics
        
    logger.info(f"Loaded metrics for {len(all_metrics)} models: {list(all_metrics.keys())}")
    
    return all_metrics


def load_predictions_files(predictions_files: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Load predictions from multiple CSV files.
    
    Args:
        predictions_files: List of paths to predictions CSV files
        
    Returns:
        Dictionary mapping model names to prediction DataFrames
    """
    all_predictions = {}
    
    for pred_file in predictions_files:
        df = pd.read_csv(pred_file)
        
        # Extract model name from filename
        model_name = Path(pred_file).stem.replace('test_predictions_', '')
        all_predictions[model_name] = df
        
    logger.info(f"Loaded predictions for {len(all_predictions)} models")
    
    return all_predictions


def perform_statistical_tests(predictions_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Perform statistical tests to compare model predictions.
    
    Args:
        predictions_dict: Dictionary of model predictions
        
    Returns:
        DataFrame with statistical test results
    """
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)
    
    if n_models < 2:
        logger.warning("Need at least 2 models for comparison")
        return pd.DataFrame()
    
    # Initialize results matrix
    results = []
    
    # Pairwise comparisons
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i >= j:
                continue
            
            pred1 = predictions_dict[model1]
            pred2 = predictions_dict[model2]
            
            # Ensure same samples
            common_idx = set(pred1['sample_idx']) & set(pred2['sample_idx'])
            pred1_common = pred1[pred1['sample_idx'].isin(common_idx)].sort_values('sample_idx')
            pred2_common = pred2[pred2['sample_idx'].isin(common_idx)].sort_values('sample_idx')
            
            errors1 = np.abs(pred1_common['error'].values)
            errors2 = np.abs(pred2_common['error'].values)
            
            # Paired t-test on absolute errors
            t_stat, p_value_t = stats.ttest_rel(errors1, errors2)
            
            # Wilcoxon signed-rank test (non-parametric)
            try:
                w_stat, p_value_w = stats.wilcoxon(errors1, errors2)
            except:
                w_stat, p_value_w = np.nan, np.nan
            
            # DeLong test for R² (approximation using bootstrap)
            r2_diff = []
            for _ in range(1000):
                idx = np.random.choice(len(errors1), len(errors1), replace=True)
                y_true = pred1_common['true_value'].values[idx]
                y_pred1 = pred1_common['predicted_value'].values[idx]
                y_pred2 = pred2_common['predicted_value'].values[idx]
                
                from sklearn.metrics import r2_score
                r2_1 = r2_score(y_true, y_pred1)
                r2_2 = r2_score(y_true, y_pred2)
                r2_diff.append(r2_1 - r2_2)
            
            r2_diff = np.array(r2_diff)
            p_value_r2 = 2 * min(
                (r2_diff > 0).mean(),
                (r2_diff < 0).mean()
            )
            
            results.append({
                'model1': model1,
                'model2': model2,
                'mean_error_diff': errors1.mean() - errors2.mean(),
                't_statistic': t_stat,
                'p_value_ttest': p_value_t,
                'wilcoxon_statistic': w_stat,
                'p_value_wilcoxon': p_value_w,
                'p_value_r2_bootstrap': p_value_r2,
                'significant_05': p_value_t < 0.05
            })
    
    return pd.DataFrame(results)


def create_comparison_table(metrics_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create comparison table of all metrics.
    
    Args:
        metrics_dict: Dictionary of model metrics
        
    Returns:
        DataFrame with comparison table
    """
    rows = []
    
    for model_name, metrics in metrics_dict.items():
        row = {
            'Model': model_name,
            'R²': metrics.get('r2', np.nan),
            'RMSE': metrics.get('rmse', np.nan),
            'MAE': metrics.get('mae', np.nan),
            'Pearson r': metrics.get('pearson_r', np.nan),
            'Spearman r': metrics.get('spearman_r', np.nan),
            'Explained Variance': metrics.get('explained_variance', np.nan),
            'Max Error': metrics.get('max_error', np.nan)
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by R² descending
    df = df.sort_values('R²', ascending=False)
    
    # Add rank column
    df['Rank'] = range(1, len(df) + 1)
    
    # Reorder columns
    cols = ['Rank', 'Model'] + [c for c in df.columns if c not in ['Rank', 'Model']]
    df = df[cols]
    
    return df


def create_comparison_plots(metrics_dict: Dict[str, Dict], predictions_dict: Dict[str, pd.DataFrame],
                          output_file: str):
    """
    Create comparison plots.
    
    Args:
        metrics_dict: Dictionary of model metrics
        predictions_dict: Dictionary of model predictions
        output_file: Output PDF file
    """
    n_models = len(metrics_dict)
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    
    # 1. Metrics comparison bar plot
    ax = axes[0, 0]
    metrics_df = create_comparison_table(metrics_dict)
    x = np.arange(len(metrics_df))
    width = 0.35
    
    ax.bar(x - width/2, metrics_df['R²'], width, label='R²')
    ax.bar(x + width/2, 1 - metrics_df['RMSE']/metrics_df['RMSE'].max(), width, label='1-Normalized RMSE')
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Error distribution boxplot
    ax = axes[0, 1]
    error_data = []
    labels = []
    
    for model_name, pred_df in predictions_dict.items():
        error_data.append(np.abs(pred_df['error'].values))
        labels.append(model_name)
    
    bp = ax.boxplot(error_data, labels=labels)
    ax.set_xlabel('Model')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Error Distribution Comparison')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # 3. Correlation heatmap
    ax = axes[1, 0]
    
    # Create correlation matrix of predictions
    pred_matrix = []
    model_names = []
    
    for model_name, pred_df in predictions_dict.items():
        pred_matrix.append(pred_df['predicted_value'].values)
        model_names.append(model_name)
    
    if len(pred_matrix) > 1:
        pred_matrix = np.array(pred_matrix).T
        corr_matrix = np.corrcoef(pred_matrix.T)
        
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=0.8, vmax=1)
        ax.set_xticks(np.arange(len(model_names)))
        ax.set_yticks(np.arange(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_yticklabels(model_names)
        ax.set_title('Prediction Correlation Matrix')
        
        # Add text annotations
        for i in range(len(model_names)):
            for j in range(len(model_names)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=ax)
    
    # 4. Metric comparison radar chart
    ax = axes[1, 1]
    
    metrics_to_plot = ['R²', 'Pearson r', 'Spearman r', 'Explained Variance']
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    ax = plt.subplot(3, 2, 4, projection='polar')
    
    for model_name in list(metrics_dict.keys())[:5]:  # Limit to 5 models for clarity
        values = []
        for metric in metrics_to_plot:
            if metric == 'R²':
                values.append(metrics_dict[model_name].get('r2', 0))
            elif metric == 'Pearson r':
                values.append(abs(metrics_dict[model_name].get('pearson_r', 0)))
            elif metric == 'Spearman r':
                values.append(abs(metrics_dict[model_name].get('spearman_r', 0)))
            elif metric == 'Explained Variance':
                values.append(metrics_dict[model_name].get('explained_variance', 0))
        
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Metric Comparison')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    # 5. Cumulative error distribution
    ax = axes[2, 0]
    
    for model_name, pred_df in predictions_dict.items():
        errors = np.sort(np.abs(pred_df['error'].values))
        cumulative = np.arange(1, len(errors) + 1) / len(errors)
        ax.plot(errors, cumulative, label=model_name, linewidth=2)
    
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Best model summary
    ax = axes[2, 1]
    ax.axis('off')
    
    # Find best model
    best_model = metrics_df.iloc[0]['Model']
    best_metrics = metrics_dict[best_model]
    
    summary_text = f"""
    Best Model: {best_model}
    
    Performance Metrics:
    • R² Score: {best_metrics.get('r2', 0):.4f}
    • RMSE: {best_metrics.get('rmse', 0):.4f}
    • MAE: {best_metrics.get('mae', 0):.4f}
    • Pearson r: {best_metrics.get('pearson_r', 0):.4f}
    • Max Error: {best_metrics.get('max_error', 0):.4f}
    
    Total Models Compared: {n_models}
    """
    
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center')
    
    plt.suptitle('Model Comparison Report', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison plots saved to {output_file}")


def create_html_report(metrics_dict, predictions_dict, comparison_table, 
                      output_file, best_model):
    """Create comprehensive HTML report."""
    
    # Create visualizations
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # ... existing code ...
    
    # Build HTML - Fix the f-string formatting issue
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Comparison Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .best-model { background-color: #e8f5e9; font-weight: bold; }
            .plot-container { margin: 20px 0; }
            .metrics-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
        </style>
    </head>
    <body>
        <h1>Model Comparison Report</h1>
    """
    
    # Add best model summary
    html_content += f"""
        <h2>Best Model</h2>
        <p>The best performing model is <strong>{best_model['model']}</strong> based on {best_model['selection_metric']}.</p>
    """
    
    # Add comparison table
    html_content += """
        <h2>Performance Metrics Comparison</h2>
        <table>
            <thead>
                <tr>
    """
    
    # Add table headers
    for col in comparison_table.columns:
        html_content += f"<th>{col}</th>"
    html_content += "</tr></thead><tbody>"
    
    # Add table rows
    for idx, row in comparison_table.iterrows():
        row_class = "best-model" if idx == best_model['model'] else ""
        html_content += f'<tr class="{row_class}">'
        for val in row.values:
            if isinstance(val, float):
                html_content += f"<td>{val:.4f}</td>"
            else:
                html_content += f"<td>{val}</td>"
        html_content += "</tr>"
    
    html_content += """
        </tbody>
        </table>
    """
    
    # Add interactive plots using Plotly
    if metrics_dict:
        # Create comparison plot
        models = list(metrics_dict.keys())
        
        # Determine if classification or regression
        first_metrics = metrics_dict[models[0]]
        is_classification = 'accuracy' in first_metrics or 'roc_auc' in first_metrics
        
        if is_classification:
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        else:
            metrics_to_plot = ['r2', 'rmse', 'mae', 'pearson_r']
        
        # Filter available metrics
        available_metrics = [m for m in metrics_to_plot if m in first_metrics]
        
        # Create bar chart
        fig = go.Figure()
        for metric in available_metrics:
            values = [metrics_dict[model].get(metric, 0) for model in models]
            fig.add_trace(go.Bar(name=metric, x=models, y=values))
        
        fig.update_layout(
            title='Model Performance Comparison',
            barmode='group',
            xaxis_title='Model',
            yaxis_title='Score',
            height=500
        )
        
        # Convert to JSON for embedding
        import json
        plot_json = json.dumps(fig.to_dict())
        
        html_content += f"""
        <h2>Performance Visualization</h2>
        <div id="performance-plot" class="plot-container"></div>
        
        <script>
            var plotData = {plot_json};
            Plotly.newPlot('performance-plot', plotData.data, plotData.layout);
        </script>
        """
    
    # Add model details
    html_content += """
        <h2>Model Details</h2>
        <div class="metrics-grid">
    """
    
    for model_name, metrics in metrics_dict.items():
        html_content += f"""
        <div>
            <h3>{model_name}</h3>
            <ul>
        """
        
        # Show key metrics
        key_metrics = []
        if 'accuracy' in metrics:
            key_metrics = ['accuracy', 'f1', 'roc_auc', 'pr_auc']
        else:
            key_metrics = ['r2', 'rmse', 'mae', 'pearson_r']
        
        for metric in key_metrics:
            if metric in metrics:
                html_content += f"<li>{metric}: {metrics[metric]:.4f}</li>"
        
        html_content += """
            </ul>
        </div>
        """
    
    html_content += """
        </div>
        
        <h2>Additional Information</h2>
        <ul>
            <li>Number of models compared: """ + str(len(metrics_dict)) + """</li>
            <li>Task type: """ + ("Classification" if is_classification else "Regression") + """</li>
            <li>Number of test samples: """ + str(metrics_dict[models[0]].get('n_test_samples', 'N/A')) + """</li>
        </ul>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(output_file, 'w') as f:
        f.write(html_content)


def select_best_model(metrics_dict: Dict, criterion: str = 'r2') -> Tuple[str, Dict]:
    """
    Select the best model based on a criterion.
    
    Args:
        metrics_dict: Dictionary of model metrics
        criterion: Metric to use for selection
        
    Returns:
        Tuple of (model_name, metrics)
    """
    best_model = None
    best_value = -np.inf if criterion != 'rmse' else np.inf
    
    for model_name, metrics in metrics_dict.items():
        value = metrics.get(criterion, np.nan)
        
        if criterion in ['rmse', 'mae', 'max_error']:
            # Lower is better
            if value < best_value:
                best_value = value
                best_model = model_name
        else:
            # Higher is better
            if value > best_value:
                best_value = value
                best_model = model_name
    
    return best_model, metrics_dict[best_model]


def main():
    parser = argparse.ArgumentParser(description='Compare sklearn models')
    parser.add_argument('--metrics_files', nargs='+', required=True,
                       help='Metrics JSON files from model evaluation')
    parser.add_argument('--predictions_files', nargs='+', required=True,
                       help='Predictions CSV files from model evaluation')
    parser.add_argument('--output_report', type=str, required=True,
                       help='Output HTML report')
    parser.add_argument('--output_table', type=str, required=True,
                       help='Output CSV comparison table')
    parser.add_argument('--output_plots', type=str, required=True,
                       help='Output PDF plots')
    parser.add_argument('--output_best', type=str, required=True,
                       help='Output JSON with best model info')
    parser.add_argument('--selection_criterion', type=str, default='r2',
                       help='Criterion for best model selection')
    
    args = parser.parse_args()
    
    # Load data
    metrics_dict = load_metrics_files(args.metrics_files)
    predictions_dict = load_predictions_files(args.predictions_files)
    
    # Create comparison table
    comparison_table = create_comparison_table(metrics_dict)
    comparison_table.to_csv(args.output_table, index=False)
    logger.info(f"Comparison table saved to {args.output_table}")
    
    # Perform statistical tests
    stat_tests = perform_statistical_tests(predictions_dict)
    
    # Create plots
    create_comparison_plots(metrics_dict, predictions_dict, args.output_plots)
    
    # Create HTML report
    create_html_report(metrics_dict, predictions_dict, comparison_table, 
                      stat_tests, args.output_report)
    
    # Select best model
    best_model, best_metrics = select_best_model(metrics_dict, args.selection_criterion)
    
    best_model_info = {
        'best_model': best_model,
        'selection_criterion': args.selection_criterion,
        'metrics': best_metrics,
        'comparison_summary': {
            'n_models': len(metrics_dict),
            'all_models': list(metrics_dict.keys()),
            'ranking': comparison_table[['Rank', 'Model', 'R²']].to_dict('records')
        }
    }
    
    with open(args.output_best, 'w') as f:
        json.dump(best_model_info, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)
    print(comparison_table.to_string())
    print("\n" + "="*60)
    print(f"Best Model: {best_model}")
    print(f"Selection Criterion: {args.selection_criterion}")
    print(f"Best {args.selection_criterion}: {best_metrics.get(args.selection_criterion, 'N/A'):.4f}")
    print("="*60)


if __name__ == '__main__':
    main()