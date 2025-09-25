#!/usr/bin/env python3
"""
Generate comprehensive report for sklearn pipeline results.
Combines model comparison, feature importance, CV results, and ensemble performance.
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_comparison_report(comparison_file: str) -> dict:
    """Load model comparison results."""
    if not comparison_file or not Path(comparison_file).exists():
        logger.warning(f"Comparison file not found: {comparison_file}")
        return {}
    
    # If HTML file, extract key metrics from a summary JSON that should exist
    if comparison_file.endswith('.html'):
        # Look for accompanying JSON file
        json_file = comparison_file.replace('.html', '_summary.json')
        if Path(json_file).exists():
            with open(json_file, 'r') as f:
                return json.load(f)
        else:
            return {'status': 'HTML report available'}
    else:
        with open(comparison_file, 'r') as f:
            return json.load(f)


def load_feature_importance(features_file: str) -> pd.DataFrame:
    """Load feature importance data."""
    if not features_file or not Path(features_file).exists():
        logger.warning(f"Features file not found: {features_file}")
        return pd.DataFrame()
    
    return pd.read_csv(features_file)


def load_cv_results(cv_results_files: list) -> dict:
    """Load cross-validation results from multiple models."""
    all_results = {}
    
    for cv_file in cv_results_files:
        if not Path(cv_file).exists():
            continue
            
        with open(cv_file, 'r') as f:
            results = json.load(f)
        
        model_type = results.get('model_type', Path(cv_file).stem.replace('cv_results_', ''))
        all_results[model_type] = results
    
    return all_results


def load_ensemble_performance(ensemble_file: str) -> dict:
    """Load ensemble performance metrics."""
    if not ensemble_file or not Path(ensemble_file).exists():
        logger.warning(f"Ensemble file not found: {ensemble_file}")
        return {}
    
    with open(ensemble_file, 'r') as f:
        return json.load(f)


def create_summary_statistics(cv_results: dict, ensemble_perf: dict) -> dict:
    """Create summary statistics across all models."""
    summary = {
        'n_models': len(cv_results),
        'models': list(cv_results.keys()),
        'best_single_model': None,
        'best_single_r2': -np.inf,
        'ensemble_r2': ensemble_perf.get('best_val_r2', None),
        'average_r2': 0,
        'average_mse': 0
    }
    
    r2_scores = []
    mse_scores = []
    
    for model_name, results in cv_results.items():
        val_metrics = results.get('cv_val_metrics', {})
        
        if 'r2' in val_metrics:
            r2_mean = val_metrics['r2'].get('mean', 0)
            r2_scores.append(r2_mean)
            
            if r2_mean > summary['best_single_r2']:
                summary['best_single_r2'] = r2_mean
                summary['best_single_model'] = model_name
        
        if 'mse' in val_metrics:
            mse_scores.append(val_metrics['mse'].get('mean', 0))
    
    if r2_scores:
        summary['average_r2'] = np.mean(r2_scores)
    if mse_scores:
        summary['average_mse'] = np.mean(mse_scores)
    
    return summary


def create_performance_table(cv_results: dict) -> pd.DataFrame:
    """Create performance comparison table."""
    rows = []
    
    for model_name, results in cv_results.items():
        val_metrics = results.get('cv_val_metrics', {})
        
        row = {
            'Model': model_name,
            'R² (mean)': val_metrics.get('r2', {}).get('mean', np.nan),
            'R² (std)': val_metrics.get('r2', {}).get('std', np.nan),
            'MSE (mean)': val_metrics.get('mse', {}).get('mean', np.nan),
            'MSE (std)': val_metrics.get('mse', {}).get('std', np.nan),
            'MAE (mean)': val_metrics.get('mae', {}).get('mean', np.nan),
            'Pearson r': val_metrics.get('pearson_r', {}).get('mean', np.nan),
            'N Features': results.get('n_features', np.nan),
            'N Folds': results.get('n_folds', 5)
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('R² (mean)', ascending=False)
    df['Rank'] = range(1, len(df) + 1)
    
    # Reorder columns
    cols = ['Rank', 'Model'] + [c for c in df.columns if c not in ['Rank', 'Model']]
    return df[cols]


def create_pdf_report(cv_results: dict, feature_importance: pd.DataFrame,
                      ensemble_perf: dict, summary: dict, output_pdf: str):
    """Create comprehensive PDF report."""
    
    with PdfPages(output_pdf) as pdf:
        # Page 1: Title and Summary
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Sklearn Pipeline Report', fontsize=20, fontweight='bold')
        
        # Add timestamp
        plt.text(0.5, 0.9, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ha='center', transform=fig.transFigure, fontsize=10)
        
        # Summary text
        summary_text = f"""
        Executive Summary
        ═══════════════════════════════════════
        
        Total Models Evaluated: {summary['n_models']}
        Models: {', '.join(summary['models'])}
        
        Best Single Model: {summary['best_single_model']}
        Best Single Model R²: {summary['best_single_r2']:.4f}
        Average R² Across Models: {summary['average_r2']:.4f}
        
        Ensemble Performance:
        Ensemble R² (if available): {summary.get('ensemble_r2', 'N/A')}
        """
        
        plt.text(0.1, 0.5, summary_text, transform=fig.transFigure, 
                fontsize=12, verticalalignment='center', family='monospace')
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Performance Table
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('tight')
        ax.axis('off')
        
        # Create performance table
        perf_table = create_performance_table(cv_results)
        
        # Convert to display format
        display_data = []
        for _, row in perf_table.head(10).iterrows():  # Show top 10
            display_row = [
                f"{row['Rank']:.0f}",
                row['Model'],
                f"{row['R² (mean)']:.4f} ± {row['R² (std)']:.4f}",
                f"{row['MSE (mean)']:.4f}",
                f"{row['MAE (mean)']:.4f}",
                f"{row['Pearson r']:.4f}"
            ]
            display_data.append(display_row)
        
        table = ax.table(
            cellText=display_data,
            colLabels=['Rank', 'Model', 'R² (mean±std)', 'MSE', 'MAE', 'Pearson r'],
            cellLoc='center',
            loc='center',
            colWidths=[0.08, 0.25, 0.2, 0.15, 0.15, 0.15]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style the header
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 3: Performance Visualization
        if len(cv_results) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            
            # Extract metrics for visualization
            models = []
            r2_means = []
            r2_stds = []
            mse_means = []
            mae_means = []
            
            for model_name, results in cv_results.items():
                val_metrics = results.get('cv_val_metrics', {})
                models.append(model_name)
                r2_means.append(val_metrics.get('r2', {}).get('mean', 0))
                r2_stds.append(val_metrics.get('r2', {}).get('std', 0))
                mse_means.append(val_metrics.get('mse', {}).get('mean', 0))
                mae_means.append(val_metrics.get('mae', {}).get('mean', 0))
            
            # R² comparison
            ax = axes[0, 0]
            x_pos = np.arange(len(models))
            ax.bar(x_pos, r2_means, yerr=r2_stds, capsize=5, alpha=0.7)
            ax.set_xlabel('Model')
            ax.set_ylabel('R² Score')
            ax.set_title('R² Score Comparison')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # MSE comparison
            ax = axes[0, 1]
            ax.bar(x_pos, mse_means, alpha=0.7, color='orange')
            ax.set_xlabel('Model')
            ax.set_ylabel('MSE')
            ax.set_title('Mean Squared Error Comparison')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # MAE comparison
            ax = axes[1, 0]
            ax.bar(x_pos, mae_means, alpha=0.7, color='green')
            ax.set_xlabel('Model')
            ax.set_ylabel('MAE')
            ax.set_title('Mean Absolute Error Comparison')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Ensemble comparison (if available)
            ax = axes[1, 1]
            if ensemble_perf:
                ensemble_data = []
                labels = []
                
                for result in ensemble_perf.get('all_results', []):
                    ensemble_data.append(result.get('val_r2', 0))
                    labels.append(result.get('method', 'Unknown'))
                
                if ensemble_data:
                    ax.bar(range(len(ensemble_data)), ensemble_data, alpha=0.7, color='purple')
                    ax.set_xlabel('Ensemble Method')
                    ax.set_ylabel('R² Score')
                    ax.set_title('Ensemble Methods Comparison')
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No ensemble data available', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'No ensemble data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
            
            plt.suptitle('Model Performance Visualizations', fontsize=14, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Page 4: Feature Importance (if available)
        if not feature_importance.empty:
            fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
            
            # Get top features from different methods
            n_top = 20
            
            ax = axes[0]
            if 'univariate' in feature_importance.columns:
                top_features = feature_importance.nlargest(n_top, 'univariate')
                ax.barh(range(len(top_features)), top_features['univariate'])
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels([f'Feature {i}' for i in top_features.index], fontsize=8)
                ax.set_xlabel('Univariate Score')
                ax.set_title(f'Top {n_top} Features (Univariate)')
                ax.invert_yaxis()
            else:
                ax.text(0.5, 0.5, 'No univariate scores available',
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
            
            ax = axes[1]
            if 'rf' in feature_importance.columns:
                top_features = feature_importance.nlargest(n_top, 'rf')
                ax.barh(range(len(top_features)), top_features['rf'], color='green')
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels([f'Feature {i}' for i in top_features.index], fontsize=8)
                ax.set_xlabel('Random Forest Importance')
                ax.set_title(f'Top {n_top} Features (Random Forest)')
                ax.invert_yaxis()
            else:
                ax.text(0.5, 0.5, 'No RF importance available',
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
            
            plt.suptitle('Feature Importance Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Add metadata page
        d = pdf.infodict()
        d['Title'] = 'Sklearn Pipeline Report'
        d['Author'] = 'Sklearn Pipeline'
        d['Subject'] = 'Model Performance Report'
        d['Keywords'] = 'Machine Learning, Sklearn, Genotype Analysis'
        d['CreationDate'] = datetime.now()
    
    logger.info(f"PDF report saved to {output_pdf}")


def create_html_report(cv_results: dict, feature_importance: pd.DataFrame,
                       ensemble_perf: dict, summary: dict, output_html: str):
    """Create comprehensive HTML report."""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sklearn Pipeline Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                padding: 30px;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
                margin-bottom: 30px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
                border-left: 4px solid #4CAF50;
                padding-left: 10px;
            }}
            .summary {{
                background: #f8f9fa;
                border-left: 5px solid #4CAF50;
                padding: 20px;
                margin: 20px 0;
                border-radius: 5px;
            }}
            .metric {{
                display: inline-block;
                background: white;
                padding: 10px 20px;
                margin: 10px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .metric-label {{
                font-size: 12px;
                color: #7f8c8d;
                text-transform: uppercase;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th {{
                background: #4CAF50;
                color: white;
                padding: 12px;
                text-align: left;
            }}
            td {{
                padding: 10px;
                border-bottom: 1px solid #ecf0f1;
            }}
            tr:nth-child(even) {{
                background: #f8f9fa;
            }}
            tr:hover {{
                background: #e8f5e9;
            }}
            .best-model {{
                background: #c8e6c9 !important;
                font-weight: bold;
            }}
            .timestamp {{
                color: #95a5a6;
                font-size: 12px;
                text-align: right;
                margin-top: 20px;
            }}
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div class="container">
            <h1>📊 Sklearn Pipeline Report</h1>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-label">Total Models</div>
                        <div class="metric-value">{summary['n_models']}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Best Model</div>
                        <div class="metric-value">{summary['best_single_model']}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Best R²</div>
                        <div class="metric-value">{summary['best_single_r2']:.4f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Average R²</div>
                        <div class="metric-value">{summary['average_r2']:.4f}</div>
                    </div>
                </div>
            </div>
            
            <h2>Model Performance Comparison</h2>
    """
    
    # Add performance table
    perf_table = create_performance_table(cv_results)
    
    html_content += """
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>R² (mean ± std)</th>
                        <th>MSE</th>
                        <th>MAE</th>
                        <th>Pearson r</th>
                        <th>N Features</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for idx, row in perf_table.iterrows():
        row_class = 'best-model' if idx == 0 else ''
        html_content += f"""
                    <tr class="{row_class}">
                        <td>{row['Rank']:.0f}</td>
                        <td>{row['Model']}</td>
                        <td>{row['R² (mean)']:.4f} ± {row['R² (std)']:.4f}</td>
                        <td>{row['MSE (mean)']:.4f}</td>
                        <td>{row['MAE (mean)']:.4f}</td>
                        <td>{row['Pearson r']:.4f}</td>
                        <td>{row['N Features']:.0f}</td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
    """
    
    # Add visualizations
    html_content += """
            <h2>Performance Visualizations</h2>
            <div id="performance-chart"></div>
            
            <script>
    """
    
    # Prepare data for Plotly
    models = list(cv_results.keys())
    r2_means = [cv_results[m].get('cv_val_metrics', {}).get('r2', {}).get('mean', 0) for m in models]
    
    html_content += f"""
                var data = [{{
                    x: {models},
                    y: {r2_means},
                    type: 'bar',
                    marker: {{
                        color: {r2_means},
                        colorscale: 'Viridis'
                    }}
                }}];
                
                var layout = {{
                    title: 'Model R² Scores',
                    xaxis: {{title: 'Model'}},
                    yaxis: {{title: 'R² Score'}},
                    height: 400
                }};
                
                Plotly.newPlot('performance-chart', data, layout);
            </script>
    """
    
    # Add ensemble section if available
    if ensemble_perf:
        html_content += f"""
            <h2>Ensemble Performance</h2>
            <div class="summary">
                <p><strong>Best Ensemble Method:</strong> {ensemble_perf.get('best_method', 'N/A')}</p>
                <p><strong>Ensemble R²:</strong> {ensemble_perf.get('best_val_r2', 'N/A')}</p>
                <p><strong>Number of Base Models:</strong> {len(models)}</p>
            </div>
        """
    
    # Add timestamp
    html_content += f"""
            <div class="timestamp">
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_html, 'w') as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved to {output_html}")


def main():
    parser = argparse.ArgumentParser(description='Generate sklearn pipeline report')
    parser.add_argument('--comparison', type=str, default=None,
                       help='Model comparison report')
    parser.add_argument('--features', type=str, default=None,
                       help='Feature importance CSV')
    parser.add_argument('--cv_results', nargs='+', default=[],
                       help='CV results JSON files')
    parser.add_argument('--ensemble', type=str, default=None,
                       help='Ensemble performance JSON')
    parser.add_argument('--output_html', type=str, required=True,
                       help='Output HTML report')
    parser.add_argument('--output_pdf', type=str, required=True,
                       help='Output PDF summary')
    
    args = parser.parse_args()
    
    logger.info("Generating sklearn pipeline report...")
    
    # Load all data
    comparison = load_comparison_report(args.comparison) if args.comparison else {}
    feature_importance = load_feature_importance(args.features) if args.features else pd.DataFrame()
    cv_results = load_cv_results(args.cv_results) if args.cv_results else {}
    ensemble_perf = load_ensemble_performance(args.ensemble) if args.ensemble else {}
    
    # Create summary statistics
    summary = create_summary_statistics(cv_results, ensemble_perf)
    
    # Generate reports
    create_html_report(cv_results, feature_importance, ensemble_perf, summary, args.output_html)
    create_pdf_report(cv_results, feature_importance, ensemble_perf, summary, args.output_pdf)
    
    # Print summary
    print("\n" + "="*60)
    print("Sklearn Pipeline Report Generated")
    print("="*60)
    print(f"Total Models: {summary['n_models']}")
    print(f"Best Model: {summary['best_single_model']}")
    print(f"Best R²: {summary['best_single_r2']:.4f}")
    print(f"Average R²: {summary['average_r2']:.4f}")
    if ensemble_perf:
        print(f"Ensemble R²: {ensemble_perf.get('best_val_r2', 'N/A')}")
    print("="*60)
    print(f"Reports saved:")
    print(f"  - HTML: {args.output_html}")
    print(f"  - PDF: {args.output_pdf}")
    print("="*60)


if __name__ == '__main__':
    main()