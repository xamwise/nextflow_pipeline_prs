#!/usr/bin/env python3
"""
Generate comprehensive report for sklearn pipeline results.
Combines model comparison, feature importance, CV results, and ensemble performance.
Supports both classification and regression tasks.
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


# ── Helpers ──────────────────────────────────────────────────────────────────

def detect_task_type(cv_results: dict) -> str:
    """Detect classification vs regression from CV result payloads."""
    for model_name, results in cv_results.items():
        if results.get('is_classification', False):
            return 'classification'
        if results.get('task_type', '').lower() == 'classification':
            return 'classification'
        val_metrics = results.get('cv_val_metrics', {})
        if 'accuracy' in val_metrics or 'roc_auc' in val_metrics or 'f1' in val_metrics:
            return 'classification'
    return 'regression'


# ── Classification metric keys (higher-is-better unless noted) ───────────────
_CLF_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc',
                'specificity', 'sensitivity']
_REG_METRICS = ['r2', 'mse', 'mae', 'pearson_r']

# Primary sort metric per task type
_PRIMARY = {'classification': 'f1', 'regression': 'r2'}
_PRIMARY_LABEL = {'classification': 'F1', 'regression': 'R²'}


# ── Loading ──────────────────────────────────────────────────────────────────

def load_comparison_report(comparison_file: str) -> dict:
    """Load model comparison results."""
    if not comparison_file or not Path(comparison_file).exists():
        logger.warning(f"Comparison file not found: {comparison_file}")
        return {}

    if comparison_file.endswith('.html'):
        json_file = comparison_file.replace('.html', '_summary.json')
        if Path(json_file).exists():
            with open(json_file, 'r') as f:
                return json.load(f)
        return {'status': 'HTML report available'}

    with open(comparison_file, 'r') as f:
        return json.load(f)


def load_feature_importance(features_file: str) -> pd.DataFrame:
    """Load feature importance data (returns empty DataFrame if missing)."""
    if not features_file or not Path(features_file).exists():
        logger.info("Feature importance file not provided or not found — skipping.")
        return pd.DataFrame()
    return pd.read_csv(features_file)


def load_cv_results(cv_results_files: list) -> dict:
    """Load cross-validation results from multiple models."""
    all_results = {}
    for cv_file in (cv_results_files or []):
        if not Path(cv_file).exists():
            continue
        with open(cv_file, 'r') as f:
            results = json.load(f)
        model_type = results.get('model_type',
                                 Path(cv_file).stem.replace('cv_results_', ''))
        all_results[model_type] = results
    return all_results


def load_ensemble_performance(ensemble_file: str) -> dict:
    """Load ensemble performance metrics (returns {} if missing)."""
    if not ensemble_file or not Path(ensemble_file).exists():
        logger.info("Ensemble file not provided or not found — skipping.")
        return {}
    with open(ensemble_file, 'r') as f:
        return json.load(f)


# ── Summary & table ─────────────────────────────────────────────────────────

def create_summary_statistics(cv_results: dict, ensemble_perf: dict,
                              task_type: str) -> dict:
    """Create summary statistics across all models."""
    primary = _PRIMARY[task_type]
    primary_label = _PRIMARY_LABEL[task_type]

    summary = {
        'task_type': task_type,
        'n_models': len(cv_results),
        'models': list(cv_results.keys()),
        'primary_metric': primary,
        'primary_metric_label': primary_label,
        'best_single_model': None,
        'best_single_score': -np.inf,
        'average_primary': 0,
    }

    # Ensemble headline — works for both task types
    if ensemble_perf:
        summary['ensemble_score'] = ensemble_perf.get(
            f'best_val_{primary}',
            ensemble_perf.get('best_val_r2',
                              ensemble_perf.get('best_val_f1', None)))
    else:
        summary['ensemble_score'] = None

    scores = []
    for model_name, results in cv_results.items():
        val_metrics = results.get('cv_val_metrics', {})
        score = val_metrics.get(primary, {}).get('mean', np.nan)
        if np.isnan(score):
            continue
        scores.append(score)
        if score > summary['best_single_score']:
            summary['best_single_score'] = score
            summary['best_single_model'] = model_name

    if scores:
        summary['average_primary'] = float(np.mean(scores))

    return summary


def create_performance_table(cv_results: dict, task_type: str) -> pd.DataFrame:
    """Create performance comparison table appropriate for the task type."""
    rows = []

    if task_type == 'classification':
        for model_name, results in cv_results.items():
            vm = results.get('cv_val_metrics', {})
            row = {
                'Model': model_name,
                'Accuracy (mean)': vm.get('accuracy', {}).get('mean', np.nan),
                'Accuracy (std)': vm.get('accuracy', {}).get('std', np.nan),
                'Precision (mean)': vm.get('precision', {}).get('mean', np.nan),
                'Recall (mean)': vm.get('recall', {}).get('mean', np.nan),
                'F1 (mean)': vm.get('f1', {}).get('mean', np.nan),
                'F1 (std)': vm.get('f1', {}).get('std', np.nan),
                'ROC AUC (mean)': vm.get('roc_auc', {}).get('mean', np.nan),
                'PR AUC (mean)': vm.get('pr_auc', {}).get('mean', np.nan),
                'N Features': results.get('n_features', np.nan),
                'N Folds': results.get('n_folds', 5),
            }
            rows.append(row)
        sort_col = 'F1 (mean)'
    else:
        for model_name, results in cv_results.items():
            vm = results.get('cv_val_metrics', {})
            row = {
                'Model': model_name,
                'R² (mean)': vm.get('r2', {}).get('mean', np.nan),
                'R² (std)': vm.get('r2', {}).get('std', np.nan),
                'MSE (mean)': vm.get('mse', {}).get('mean', np.nan),
                'MSE (std)': vm.get('mse', {}).get('std', np.nan),
                'MAE (mean)': vm.get('mae', {}).get('mean', np.nan),
                'Pearson r': vm.get('pearson_r', {}).get('mean', np.nan),
                'N Features': results.get('n_features', np.nan),
                'N Folds': results.get('n_folds', 5),
            }
            rows.append(row)
        sort_col = 'R² (mean)'

    df = pd.DataFrame(rows)
    df = df.sort_values(sort_col, ascending=False, na_position='last')
    df['Rank'] = range(1, len(df) + 1)
    cols = ['Rank', 'Model'] + [c for c in df.columns if c not in ['Rank', 'Model']]
    return df[cols]


# ── PDF report ───────────────────────────────────────────────────────────────

def _pdf_title_page(pdf, summary):
    """Page 1: title + executive summary."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Sklearn Pipeline Report', fontsize=20, fontweight='bold')
    plt.text(0.5, 0.92,
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             ha='center', transform=fig.transFigure, fontsize=10)

    label = summary['primary_metric_label']
    ens_line = (f"Ensemble {label}: {summary['ensemble_score']:.4f}"
                if summary.get('ensemble_score') is not None else
                "Ensemble: not available")

    text = f"""
    Executive Summary  ({summary['task_type'].title()})
    {'═' * 45}

    Total Models Evaluated: {summary['n_models']}
    Models: {', '.join(summary['models'])}

    Best Single Model: {summary['best_single_model']}
    Best {label}: {summary['best_single_score']:.4f}
    Average {label}: {summary['average_primary']:.4f}

    {ens_line}
    """
    plt.text(0.1, 0.5, text, transform=fig.transFigure,
             fontsize=12, verticalalignment='center', family='monospace')
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def _pdf_performance_table(pdf, perf_table, task_type):
    """Page 2: performance table."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('tight')
    ax.axis('off')

    display_data = []
    if task_type == 'classification':
        col_labels = ['Rank', 'Model', 'Accuracy', 'Precision', 'Recall',
                      'F1 (mean±std)', 'ROC AUC']
        for _, row in perf_table.head(10).iterrows():
            display_data.append([
                f"{row['Rank']:.0f}",
                row['Model'],
                f"{row.get('Accuracy (mean)', np.nan):.4f}",
                f"{row.get('Precision (mean)', np.nan):.4f}",
                f"{row.get('Recall (mean)', np.nan):.4f}",
                f"{row.get('F1 (mean)', np.nan):.4f} ± {row.get('F1 (std)', np.nan):.4f}",
                f"{row.get('ROC AUC (mean)', np.nan):.4f}",
            ])
        col_widths = [0.06, 0.22, 0.12, 0.12, 0.12, 0.2, 0.12]
    else:
        col_labels = ['Rank', 'Model', 'R² (mean±std)', 'MSE', 'MAE', 'Pearson r']
        for _, row in perf_table.head(10).iterrows():
            display_data.append([
                f"{row['Rank']:.0f}",
                row['Model'],
                f"{row['R² (mean)']:.4f} ± {row['R² (std)']:.4f}",
                f"{row['MSE (mean)']:.4f}",
                f"{row['MAE (mean)']:.4f}",
                f"{row['Pearson r']:.4f}",
            ])
        col_widths = [0.08, 0.25, 0.2, 0.15, 0.15, 0.15]

    table = ax.table(cellText=display_data, colLabels=col_labels,
                     cellLoc='center', loc='center', colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def _pdf_performance_vis(pdf, cv_results, ensemble_perf, task_type):
    """Page 3: bar charts of key metrics + optional ensemble panel."""
    if not cv_results:
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    models = list(cv_results.keys())
    x_pos = np.arange(len(models))

    def _extract(metric):
        return [cv_results[m].get('cv_val_metrics', {}).get(metric, {}).get('mean', 0)
                for m in models]

    def _extract_std(metric):
        return [cv_results[m].get('cv_val_metrics', {}).get(metric, {}).get('std', 0)
                for m in models]

    if task_type == 'classification':
        metric_specs = [
            ('accuracy', 'Accuracy', None, 'tab:blue'),
            ('f1', 'F1 Score', None, 'tab:orange'),
            ('precision', 'Precision', 'tab:green', 'tab:green'),
        ]
    else:
        metric_specs = [
            ('r2', 'R² Score', None, 'tab:blue'),
            ('mse', 'MSE', None, 'tab:orange'),
            ('mae', 'MAE', None, 'tab:green'),
        ]

    for idx, (metric, label, _, color) in enumerate(metric_specs[:3]):
        ax = axes[idx // 2, idx % 2]
        vals = _extract(metric)
        stds = _extract_std(metric)
        ax.bar(x_pos, vals, yerr=stds, capsize=5, alpha=0.7, color=color)
        ax.set_xlabel('Model')
        ax.set_ylabel(label)
        ax.set_title(f'{label} Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

    # Fourth panel: ensemble or recall/pearson
    ax = axes[1, 1]
    if ensemble_perf and ensemble_perf.get('all_results'):
        ens_data, ens_labels = [], []
        primary = _PRIMARY[task_type]
        for r in ensemble_perf['all_results']:
            val = r.get(f'val_{primary}', r.get('val_r2', r.get('val_f1', 0)))
            ens_data.append(val)
            ens_labels.append(r.get('method', '?'))
        ax.bar(range(len(ens_data)), ens_data, alpha=0.7, color='purple')
        ax.set_ylabel(_PRIMARY_LABEL[task_type])
        ax.set_title('Ensemble Methods Comparison')
        ax.set_xticks(range(len(ens_labels)))
        ax.set_xticklabels(ens_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    else:
        # Use recall (clf) or pearson_r (reg) as fallback fourth chart
        fallback = 'recall' if task_type == 'classification' else 'pearson_r'
        fallback_label = 'Recall' if task_type == 'classification' else 'Pearson r'
        vals = _extract(fallback)
        ax.bar(x_pos, vals, alpha=0.7, color='purple')
        ax.set_xlabel('Model')
        ax.set_ylabel(fallback_label)
        ax.set_title(f'{fallback_label} Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Model Performance Visualisations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def _pdf_feature_importance(pdf, feature_importance):
    """Page 4 (optional): feature importance plots."""
    if feature_importance.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    n_top = 20

    panels = [
        ('univariate', 'Univariate Score', 'tab:blue'),
        ('rf', 'Random Forest Importance', 'tab:green'),
    ]

    for ax, (col, label, color) in zip(axes, panels):
        if col in feature_importance.columns:
            top = feature_importance.nlargest(n_top, col)
            ax.barh(range(len(top)), top[col], color=color)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels([f'Feature {i}' for i in top.index], fontsize=8)
            ax.set_xlabel(label)
            ax.set_title(f'Top {n_top} Features ({label})')
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, f'No {label} data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    plt.suptitle('Feature Importance Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_pdf_report(cv_results: dict, feature_importance: pd.DataFrame,
                      ensemble_perf: dict, summary: dict, output_pdf: str,
                      task_type: str):
    """Create comprehensive PDF report."""
    perf_table = create_performance_table(cv_results, task_type)

    with PdfPages(output_pdf) as pdf:
        _pdf_title_page(pdf, summary)
        _pdf_performance_table(pdf, perf_table, task_type)
        _pdf_performance_vis(pdf, cv_results, ensemble_perf, task_type)
        _pdf_feature_importance(pdf, feature_importance)

        d = pdf.infodict()
        d['Title'] = 'Sklearn Pipeline Report'
        d['Author'] = 'Sklearn Pipeline'
        d['Subject'] = f'Model Performance Report ({task_type.title()})'
        d['CreationDate'] = datetime.now()

    logger.info(f"PDF report saved to {output_pdf}")


# ── HTML report ──────────────────────────────────────────────────────────────

def create_html_report(cv_results: dict, feature_importance: pd.DataFrame,
                       ensemble_perf: dict, summary: dict, output_html: str,
                       task_type: str):
    """Create comprehensive HTML report for either task type."""
    label = summary['primary_metric_label']
    best_score = summary['best_single_score']
    avg_score = summary['average_primary']

    # ── Header ──
    html = [f"""<!DOCTYPE html>
<html><head><title>Sklearn Pipeline Report</title>
<style>
body{{font-family:'Segoe UI',Tahoma,Arial,sans-serif;margin:0;padding:20px;
     background:linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%);min-height:100vh}}
.container{{max-width:1200px;margin:0 auto;background:#fff;border-radius:10px;
           box-shadow:0 10px 30px rgba(0,0,0,.1);padding:30px}}
h1{{color:#2c3e50;border-bottom:3px solid #4CAF50;padding-bottom:10px;margin-bottom:30px}}
h2{{color:#34495e;margin-top:30px;border-left:4px solid #4CAF50;padding-left:10px}}
.summary{{background:#f8f9fa;border-left:5px solid #4CAF50;padding:20px;margin:20px 0;border-radius:5px}}
.metric{{display:inline-block;background:#fff;padding:10px 20px;margin:10px;border-radius:5px;
        box-shadow:0 2px 5px rgba(0,0,0,.1)}}
.metric-label{{font-size:12px;color:#7f8c8d;text-transform:uppercase}}
.metric-value{{font-size:24px;font-weight:bold;color:#2c3e50}}
table{{width:100%;border-collapse:collapse;margin:20px 0}}
th{{background:#4CAF50;color:#fff;padding:12px;text-align:left}}
td{{padding:10px;border-bottom:1px solid #ecf0f1}}
tr:nth-child(even){{background:#f8f9fa}}
tr:hover{{background:#e8f5e9}}
.best-model{{background:#c8e6c9!important;font-weight:bold}}
.timestamp{{color:#95a5a6;font-size:12px;text-align:right;margin-top:20px}}
</style>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head><body><div class="container">
<h1>Sklearn Pipeline Report &mdash; {task_type.title()}</h1>

<div class="summary"><h2>Executive Summary</h2>
<div class="metrics">
  <div class="metric"><div class="metric-label">Total Models</div>
    <div class="metric-value">{summary['n_models']}</div></div>
  <div class="metric"><div class="metric-label">Best Model</div>
    <div class="metric-value">{summary['best_single_model']}</div></div>
  <div class="metric"><div class="metric-label">Best {label}</div>
    <div class="metric-value">{best_score:.4f}</div></div>
  <div class="metric"><div class="metric-label">Average {label}</div>
    <div class="metric-value">{avg_score:.4f}</div></div>
</div></div>
"""]

    # ── Performance table ──
    perf_table = create_performance_table(cv_results, task_type)

    if task_type == 'classification':
        col_defs = [
            ('Rank', 'Rank', '.0f'),
            ('Model', 'Model', None),
            ('Accuracy (mean)', 'Accuracy', '.4f'),
            ('Precision (mean)', 'Precision', '.4f'),
            ('Recall (mean)', 'Recall', '.4f'),
            ('F1 (mean)', 'F1', '.4f'),
            ('F1 (std)', 'F1 std', '.4f'),
            ('ROC AUC (mean)', 'ROC AUC', '.4f'),
            ('PR AUC (mean)', 'PR AUC', '.4f'),
            ('N Features', 'N Features', '.0f'),
        ]
    else:
        col_defs = [
            ('Rank', 'Rank', '.0f'),
            ('Model', 'Model', None),
            ('R² (mean)', 'R²', '.4f'),
            ('R² (std)', 'R² std', '.4f'),
            ('MSE (mean)', 'MSE', '.4f'),
            ('MAE (mean)', 'MAE', '.4f'),
            ('Pearson r', 'Pearson r', '.4f'),
            ('N Features', 'N Features', '.0f'),
        ]

    html.append("<h2>Model Performance Comparison</h2><table><thead><tr>")
    for _, hdr, _ in col_defs:
        html.append(f"<th>{hdr}</th>")
    html.append("</tr></thead><tbody>")

    best_name = summary['best_single_model']
    for _, row in perf_table.iterrows():
        cls = ' class="best-model"' if row['Model'] == best_name else ''
        html.append(f"<tr{cls}>")
        for key, _, fmt in col_defs:
            val = row.get(key, np.nan)
            if fmt and not (isinstance(val, float) and np.isnan(val)):
                html.append(f"<td>{val:{fmt}}</td>")
            elif isinstance(val, float) and np.isnan(val):
                html.append("<td>—</td>")
            else:
                html.append(f"<td>{val}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")

    # ── Plotly chart ──
    models = list(cv_results.keys())
    primary = _PRIMARY[task_type]
    primary_vals = [
        cv_results[m].get('cv_val_metrics', {}).get(primary, {}).get('mean', 0)
        for m in models
    ]
    html.append(f"""
<h2>Performance Visualisation</h2>
<div id="perf-chart"></div>
<script>
Plotly.newPlot('perf-chart', [{{
  x: {json.dumps(models)},
  y: {json.dumps(primary_vals)},
  type: 'bar',
  marker: {{color: {json.dumps(primary_vals)}, colorscale: 'Viridis'}}
}}], {{
  title: 'Model {label} Scores',
  xaxis: {{title: 'Model'}},
  yaxis: {{title: '{label}'}},
  height: 400
}});
</script>
""")

    # ── Ensemble (optional) ──
    if ensemble_perf:
        ens_score = summary.get('ensemble_score', 'N/A')
        if isinstance(ens_score, float):
            ens_score = f"{ens_score:.4f}"
        html.append(f"""
<h2>Ensemble Performance</h2>
<div class="summary">
  <p><strong>Best Ensemble Method:</strong> {ensemble_perf.get('best_method', 'N/A')}</p>
  <p><strong>Ensemble {label}:</strong> {ens_score}</p>
  <p><strong>Base Models:</strong> {len(models)}</p>
</div>
""")

    # ── Footer ──
    n_test = 'N/A'
    if cv_results:
        first = next(iter(cv_results.values()))
        n_test = first.get('n_test_samples', first.get('n_samples', 'N/A'))

    html.append(f"""
<h2>Additional Information</h2>
<ul>
  <li>Models compared: {len(models)}</li>
  <li>Task type: {task_type.title()}</li>
  <li>Samples: {n_test}</li>
</ul>
<div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</div></body></html>""")

    with open(output_html, 'w') as f:
        f.write('\n'.join(html))
    logger.info(f"HTML report saved to {output_html}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate sklearn pipeline report')
    parser.add_argument('--comparison', type=str, default=None,
                        help='Model comparison report (optional)')
    parser.add_argument('--features', type=str, default=None,
                        help='Feature importance CSV (optional)')
    parser.add_argument('--cv_results', nargs='+', default=[],
                        help='CV results JSON files')
    parser.add_argument('--ensemble', type=str, default=None,
                        help='Ensemble performance JSON (optional)')
    parser.add_argument('--output_html', type=str, required=True,
                        help='Output HTML report')
    parser.add_argument('--output_pdf', type=str, required=True,
                        help='Output PDF summary')

    args = parser.parse_args()
    logger.info("Generating sklearn pipeline report...")

    # Load all data (features & ensemble are gracefully optional)
    comparison = load_comparison_report(args.comparison) if args.comparison else {}
    feature_importance = load_feature_importance(args.features)
    cv_results = load_cv_results(args.cv_results)
    ensemble_perf = load_ensemble_performance(args.ensemble)

    # Detect task type
    task_type = detect_task_type(cv_results)
    logger.info(f"Detected task type: {task_type}")

    # Summary
    summary = create_summary_statistics(cv_results, ensemble_perf, task_type)

    # Generate reports
    create_html_report(cv_results, feature_importance, ensemble_perf,
                       summary, args.output_html, task_type)
    create_pdf_report(cv_results, feature_importance, ensemble_perf,
                      summary, args.output_pdf, task_type)

    # Console summary
    label = summary['primary_metric_label']
    print("\n" + "=" * 60)
    print(f"Sklearn Pipeline Report Generated  ({task_type.title()})")
    print("=" * 60)
    print(f"Total Models: {summary['n_models']}")
    print(f"Best Model: {summary['best_single_model']}")
    print(f"Best {label}: {summary['best_single_score']:.4f}")
    print(f"Average {label}: {summary['average_primary']:.4f}")
    if ensemble_perf and summary.get('ensemble_score') is not None:
        print(f"Ensemble {label}: {summary['ensemble_score']:.4f}")
    print("=" * 60)
    print(f"Reports saved:")
    print(f"  HTML: {args.output_html}")
    print(f"  PDF:  {args.output_pdf}")
    print("=" * 60)


if __name__ == '__main__':
    main()