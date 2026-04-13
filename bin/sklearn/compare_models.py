#!/usr/bin/env python3
"""
Compare performance of multiple sklearn models.
Generates comprehensive comparison report with statistical tests.
Supports both regression and classification tasks.
"""

import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import glob
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def detect_task_type(metrics_dict: Dict[str, Dict]) -> str:
    """
    Detect whether the loaded metrics come from a classification or regression task.

    Returns:
        'classification' or 'regression'
    """
    for model_name, metrics in metrics_dict.items():
        if metrics.get('is_classification', False):
            return 'classification'
        if metrics.get('task_type', '').lower() == 'classification':
            return 'classification'
        if 'accuracy' in metrics or 'roc_auc' in metrics or 'true_positives' in metrics:
            return 'classification'
    return 'regression'


# ── Loading ──────────────────────────────────────────────────────────────────

def load_metrics_files(metrics_files: List[str]) -> Dict[str, Dict]:
    """Load metrics from multiple JSON files."""
    all_metrics = {}

    for metrics_file in metrics_files:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        model_name = metrics.get('model_type',
                                 Path(metrics_file).stem.replace('test_metrics_', ''))
        all_metrics[model_name] = metrics

    logger.info(f"Loaded metrics for {len(all_metrics)} models: {list(all_metrics.keys())}")
    return all_metrics


def load_predictions_files(predictions_files: List[str]) -> Dict[str, pd.DataFrame]:
    """Load predictions from multiple CSV files."""
    all_predictions = {}

    for pred_file in predictions_files:
        df = pd.read_csv(pred_file)
        model_name = Path(pred_file).stem.replace('test_predictions_', '')
        all_predictions[model_name] = df

    logger.info(f"Loaded predictions for {len(all_predictions)} models")
    return all_predictions


# ── Comparison table ─────────────────────────────────────────────────────────

def create_comparison_table(metrics_dict: Dict[str, Dict],
                            task_type: str) -> pd.DataFrame:
    """
    Create a ranked comparison table of all metrics.

    Args:
        metrics_dict: model-name → metrics mapping
        task_type: 'classification' or 'regression'

    Returns:
        Sorted DataFrame with a Rank column.
    """
    rows = []

    for model_name, metrics in metrics_dict.items():
        if task_type == 'classification':
            row = {
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', np.nan),
                'Precision': metrics.get('precision', np.nan),
                'Recall': metrics.get('recall', np.nan),
                'F1': metrics.get('f1', np.nan),
                'Specificity': metrics.get('specificity', np.nan),
                'Sensitivity': metrics.get('sensitivity', np.nan),
                'ROC AUC': metrics.get('roc_auc', np.nan),
                'PR AUC': metrics.get('pr_auc', np.nan),
                'Model Agreement': metrics.get('model_agreement', np.nan),
                'N Models': metrics.get('n_models', np.nan),
            }
        else:
            row = {
                'Model': model_name,
                'R²': metrics.get('r2', np.nan),
                'RMSE': metrics.get('rmse', np.nan),
                'MAE': metrics.get('mae', np.nan),
                'Pearson r': metrics.get('pearson_r', np.nan),
                'Spearman r': metrics.get('spearman_r', np.nan),
                'Explained Variance': metrics.get('explained_variance', np.nan),
                'Max Error': metrics.get('max_error', np.nan),
            }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort: classification by F1, regression by R²
    sort_col = 'F1' if task_type == 'classification' else 'R²'
    df = df.sort_values(sort_col, ascending=False, na_position='last')
    df['Rank'] = range(1, len(df) + 1)

    cols = ['Rank', 'Model'] + [c for c in df.columns if c not in ['Rank', 'Model']]
    return df[cols]


# ── Statistical tests ────────────────────────────────────────────────────────

def _align_predictions(pred1: pd.DataFrame, pred2: pd.DataFrame,
                       id_col: str = 'sample_idx') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return two DataFrames restricted & sorted to common sample indices."""
    common = set(pred1[id_col]) & set(pred2[id_col])
    p1 = pred1[pred1[id_col].isin(common)].sort_values(id_col).reset_index(drop=True)
    p2 = pred2[pred2[id_col].isin(common)].sort_values(id_col).reset_index(drop=True)
    return p1, p2


def _mcnemar_test(y_true: np.ndarray,
                  y_pred1: np.ndarray,
                  y_pred2: np.ndarray) -> Tuple[float, float]:
    """
    McNemar's test for paired nominal data.

    Compares whether two classifiers have the same error rate on the same
    test set.

    Returns:
        (chi2_statistic, p_value)
    """
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)

    # b = model1 correct & model2 wrong, c = model1 wrong & model2 correct
    b = int(np.sum(correct1 & ~correct2))
    c = int(np.sum(~correct1 & correct2))

    # Use continuity-corrected form; fall back if b+c is too small
    if b + c < 25:
        # Exact binomial test (two-sided)
        p_value = stats.binom_test(b, b + c, 0.5) if (b + c) > 0 else 1.0
        chi2 = np.nan
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return chi2, p_value


def _bootstrap_metric_diff(y_true, scores1, scores2, metric_fn,
                           n_boot: int = 2000) -> float:
    """Two-sided bootstrap p-value for metric(scores1) − metric(scores2)."""
    rng = np.random.default_rng(42)
    diffs = np.empty(n_boot)
    n = len(y_true)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        diffs[i] = metric_fn(y_true[idx], scores1[idx]) - metric_fn(y_true[idx], scores2[idx])
    p = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
    return float(p)


def perform_statistical_tests(predictions_dict: Dict[str, pd.DataFrame],
                              task_type: str) -> pd.DataFrame:
    """
    Pairwise statistical comparisons between models.

    Classification: McNemar's test + bootstrap AUC comparison.
    Regression:     Paired t-test + Wilcoxon + bootstrap R² comparison.
    """
    model_names = list(predictions_dict.keys())
    if len(model_names) < 2:
        logger.warning("Need at least 2 models for comparison")
        return pd.DataFrame()

    results = []

    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i >= j:
                continue

            pred1, pred2 = _align_predictions(
                predictions_dict[model1], predictions_dict[model2])

            if task_type == 'classification':
                y_true = pred1['true_value'].values
                y_pred1 = pred1['predicted_value'].values
                y_pred2 = pred2['predicted_value'].values

                chi2, p_mcnemar = _mcnemar_test(y_true, y_pred1, y_pred2)

                # Bootstrap AUC if probability columns exist
                p_auc_boot = np.nan
                if 'predicted_probability' in pred1.columns and 'predicted_probability' in pred2.columns:
                    from sklearn.metrics import roc_auc_score
                    try:
                        p_auc_boot = _bootstrap_metric_diff(
                            y_true,
                            pred1['predicted_probability'].values,
                            pred2['predicted_probability'].values,
                            roc_auc_score,
                        )
                    except Exception as e:
                        logger.warning(f"Bootstrap AUC failed for {model1} vs {model2}: {e}")

                acc1 = (y_pred1 == y_true).mean()
                acc2 = (y_pred2 == y_true).mean()

                results.append({
                    'model1': model1,
                    'model2': model2,
                    'accuracy_diff': acc1 - acc2,
                    'mcnemar_chi2': chi2,
                    'p_value_mcnemar': p_mcnemar,
                    'p_value_auc_bootstrap': p_auc_boot,
                    'significant_05': p_mcnemar < 0.05,
                })

            else:  # regression
                errors1 = np.abs(pred1['error'].values)
                errors2 = np.abs(pred2['error'].values)

                t_stat, p_t = stats.ttest_rel(errors1, errors2)

                try:
                    w_stat, p_w = stats.wilcoxon(errors1, errors2)
                except Exception:
                    w_stat, p_w = np.nan, np.nan

                from sklearn.metrics import r2_score
                p_r2 = _bootstrap_metric_diff(
                    pred1['true_value'].values,
                    pred1['predicted_value'].values,
                    pred2['predicted_value'].values,
                    r2_score,
                )

                results.append({
                    'model1': model1,
                    'model2': model2,
                    'mean_error_diff': errors1.mean() - errors2.mean(),
                    't_statistic': t_stat,
                    'p_value_ttest': p_t,
                    'wilcoxon_statistic': w_stat,
                    'p_value_wilcoxon': p_w,
                    'p_value_r2_bootstrap': p_r2,
                    'significant_05': p_t < 0.05,
                })

    return pd.DataFrame(results)


# ── Plots ────────────────────────────────────────────────────────────────────

def _plot_classification(metrics_dict, predictions_dict, axes):
    """Fill *axes* (3×2 grid) with classification-specific plots."""
    model_names = list(metrics_dict.keys())
    comparison_df = create_comparison_table(metrics_dict, 'classification')

    # 1 ── Grouped bar chart of key metrics
    ax = axes[0, 0]
    bar_metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Specificity']
    bar_metrics = [m for m in bar_metrics if m in comparison_df.columns
                   and comparison_df[m].notna().any()]
    x = np.arange(len(comparison_df))
    width = 0.8 / max(len(bar_metrics), 1)

    for k, metric in enumerate(bar_metrics):
        ax.bar(x + k * width, comparison_df[metric].values, width, label=metric)

    ax.set_xticks(x + width * len(bar_metrics) / 2)
    ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Classification Metrics Comparison')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # 2 ── Confusion-matrix heat-maps (small multiples)
    ax = axes[0, 1]
    ax.set_title('Confusion Matrix Summary')
    cm_data = []
    for model in model_names:
        m = metrics_dict[model]
        tp = m.get('true_positives', 0)
        fp = m.get('false_positives', 0)
        fn = m.get('false_negatives', 0)
        tn = m.get('true_negatives', 0)
        cm_data.append([tn, fp, fn, tp])

    cm_arr = np.array(cm_data)  # (n_models, 4)
    bar_labels = ['TN', 'FP', 'FN', 'TP']
    x = np.arange(len(model_names))
    width = 0.8 / 4
    colors = ['#4CAF50', '#FF9800', '#F44336', '#2196F3']

    for k, (label, color) in enumerate(zip(bar_labels, colors)):
        ax.bar(x + k * width, cm_arr[:, k], width, label=label, color=color)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3 ── ROC curves (if probabilities available)
    ax = axes[1, 0]
    roc_plotted = False
    for model_name, pred_df in predictions_dict.items():
        if 'predicted_probability' in pred_df.columns and 'true_value' in pred_df.columns:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(pred_df['true_value'], pred_df['predicted_probability'])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC={roc_auc:.3f})')
            roc_plotted = True

    if roc_plotted:
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No probability\ncolumn available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('ROC Curves (unavailable)')

    # 4 ── Precision-Recall curves
    ax = axes[1, 1]
    pr_plotted = False
    for model_name, pred_df in predictions_dict.items():
        if 'predicted_probability' in pred_df.columns and 'true_value' in pred_df.columns:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            precision_vals, recall_vals, _ = precision_recall_curve(
                pred_df['true_value'], pred_df['predicted_probability'])
            ap = average_precision_score(pred_df['true_value'],
                                         pred_df['predicted_probability'])
            ax.plot(recall_vals, precision_vals, linewidth=2,
                    label=f'{model_name} (AP={ap:.3f})')
            pr_plotted = True

    if pr_plotted:
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision–Recall Curves')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No probability\ncolumn available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('PR Curves (unavailable)')

    # 5 ── Radar chart of per-model metrics
    ax_polar = plt.subplot(3, 2, 5, projection='polar')
    radar_metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Specificity']
    radar_metrics = [m for m in radar_metrics if m in comparison_df.columns
                     and comparison_df[m].notna().any()]
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    for _, row in comparison_df.iterrows():
        values = [row[m] if pd.notna(row[m]) else 0 for m in radar_metrics]
        values += values[:1]
        ax_polar.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
        ax_polar.fill(angles, values, alpha=0.15)

    ax_polar.set_xticks(angles[:-1])
    ax_polar.set_xticklabels(radar_metrics, fontsize=8)
    ax_polar.set_ylim(0, 1)
    ax_polar.set_title('Multi-Metric Radar', pad=20)
    ax_polar.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=7)
    ax_polar.grid(True)

    # 6 ── Best model summary
    ax = axes[2, 1]
    ax.axis('off')
    best_row = comparison_df.iloc[0]
    best_name = best_row['Model']
    bm = metrics_dict[best_name]

    summary_lines = [
        f"Best Model: {best_name}",
        "",
        "Performance Metrics:",
        f"  Accuracy:    {bm.get('accuracy', 0):.4f}",
        f"  Precision:   {bm.get('precision', 0):.4f}",
        f"  Recall:      {bm.get('recall', 0):.4f}",
        f"  F1:          {bm.get('f1', 0):.4f}",
        f"  Specificity: {bm.get('specificity', 0):.4f}",
    ]
    if 'roc_auc' in bm:
        summary_lines.append(f"  ROC AUC:     {bm['roc_auc']:.4f}")
    if 'pr_auc' in bm:
        summary_lines.append(f"  PR AUC:      {bm['pr_auc']:.4f}")
    summary_lines += [
        "",
        f"Total Models Compared: {len(metrics_dict)}",
        f"Test Samples: {bm.get('n_test_samples', 'N/A')}",
    ]

    ax.text(0.1, 0.5, '\n'.join(summary_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment='center', family='monospace')


def _plot_regression(metrics_dict, predictions_dict, axes):
    """Fill *axes* (3×2 grid) with regression-specific plots."""
    comparison_df = create_comparison_table(metrics_dict, 'regression')

    # 1 ── Metrics comparison bar plot
    ax = axes[0, 0]
    x = np.arange(len(comparison_df))
    width = 0.35
    ax.bar(x - width / 2, comparison_df['R²'], width, label='R²')
    ax.bar(x + width / 2,
           1 - comparison_df['RMSE'] / comparison_df['RMSE'].max(),
           width, label='1−Norm RMSE')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2 ── Error distribution box-plot
    ax = axes[0, 1]
    error_data, labels = [], []
    for name, df in predictions_dict.items():
        error_data.append(np.abs(df['error'].values))
        labels.append(name)
    ax.boxplot(error_data, labels=labels)
    ax.set_ylabel('Absolute Error')
    ax.set_title('Error Distribution Comparison')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3)

    # 3 ── Prediction correlation heat-map
    ax = axes[1, 0]
    pred_matrix, model_names = [], []
    for name, df in predictions_dict.items():
        pred_matrix.append(df['predicted_value'].values)
        model_names.append(name)
    if len(pred_matrix) > 1:
        corr = np.corrcoef(pred_matrix)
        im = ax.imshow(corr, cmap='coolwarm', vmin=0.8, vmax=1)
        ax.set_xticks(np.arange(len(model_names)))
        ax.set_yticks(np.arange(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_yticklabels(model_names)
        for ii in range(len(model_names)):
            for jj in range(len(model_names)):
                ax.text(jj, ii, f'{corr[ii, jj]:.3f}', ha='center', va='center')
        plt.colorbar(im, ax=ax)
    ax.set_title('Prediction Correlation Matrix')

    # 4 ── Radar chart
    ax_polar = plt.subplot(3, 2, 4, projection='polar')
    radar_metrics = ['R²', 'Pearson r', 'Spearman r', 'Explained Variance']
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]
    key_map = {'R²': 'r2', 'Pearson r': 'pearson_r',
               'Spearman r': 'spearman_r', 'Explained Variance': 'explained_variance'}
    for model_name in list(metrics_dict.keys())[:5]:
        values = [abs(metrics_dict[model_name].get(key_map[m], 0)) for m in radar_metrics]
        values += values[:1]
        ax_polar.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax_polar.fill(angles, values, alpha=0.25)
    ax_polar.set_xticks(angles[:-1])
    ax_polar.set_xticklabels(radar_metrics, fontsize=8)
    ax_polar.set_ylim(0, 1)
    ax_polar.set_title('Multi-Metric Comparison', pad=20)
    ax_polar.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=7)
    ax_polar.grid(True)

    # 5 ── Cumulative error distribution
    ax = axes[2, 0]
    for name, df in predictions_dict.items():
        errors = np.sort(np.abs(df['error'].values))
        cum = np.arange(1, len(errors) + 1) / len(errors)
        ax.plot(errors, cum, label=name, linewidth=2)
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6 ── Best model summary
    ax = axes[2, 1]
    ax.axis('off')
    best_name = comparison_df.iloc[0]['Model']
    bm = metrics_dict[best_name]
    summary = '\n'.join([
        f"Best Model: {best_name}",
        "",
        "Performance Metrics:",
        f"  R² Score:    {bm.get('r2', 0):.4f}",
        f"  RMSE:        {bm.get('rmse', 0):.4f}",
        f"  MAE:         {bm.get('mae', 0):.4f}",
        f"  Pearson r:   {bm.get('pearson_r', 0):.4f}",
        f"  Max Error:   {bm.get('max_error', 0):.4f}",
        "",
        f"Total Models Compared: {len(metrics_dict)}",
    ])
    ax.text(0.1, 0.5, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='center', family='monospace')


def create_comparison_plots(metrics_dict: Dict[str, Dict],
                            predictions_dict: Dict[str, pd.DataFrame],
                            output_file: str,
                            task_type: str):
    """
    Create comparison plots dispatched by task type.
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    if task_type == 'classification':
        _plot_classification(metrics_dict, predictions_dict, axes)
    else:
        _plot_regression(metrics_dict, predictions_dict, axes)

    plt.suptitle(f'Model Comparison Report ({task_type.title()})',
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Comparison plots saved to {output_file}")


# ── HTML report ──────────────────────────────────────────────────────────────

def create_html_report(metrics_dict, predictions_dict, comparison_table,
                       stat_tests, output_file, task_type: str):
    """Create comprehensive HTML report for either task type."""
    import plotly.graph_objects as go

    models = list(metrics_dict.keys())

    # ── Determine best model ──
    if task_type == 'classification':
        sort_metric = 'F1'
        default_criterion = 'f1'
    else:
        sort_metric = 'R²'
        default_criterion = 'r2'

    best_model_name = comparison_table.iloc[0]['Model']
    best_info = {
        'model': best_model_name,
        'selection_metric': sort_metric,
    }

    # ── Build HTML ──
    html = ["""<!DOCTYPE html>
<html><head><title>Model Comparison Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body{font-family:Arial,sans-serif;margin:20px;color:#333}
h1,h2{color:#333}
table{border-collapse:collapse;width:100%;margin:20px 0}
th,td{border:1px solid #ddd;padding:8px;text-align:left}
th{background:#f2f2f2}
.best-model{background:#e8f5e9;font-weight:bold}
.plot-container{margin:20px 0}
.metrics-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:20px}
</style></head><body>"""]

    html.append(f"<h1>Model Comparison Report — {task_type.title()}</h1>")
    html.append(f"<h2>Best Model</h2><p>Best performing model is "
                f"<strong>{best_model_name}</strong> (by {sort_metric}).</p>")

    # ── Comparison table ──
    html.append("<h2>Performance Metrics Comparison</h2><table><thead><tr>")
    for col in comparison_table.columns:
        html.append(f"<th>{col}</th>")
    html.append("</tr></thead><tbody>")

    for _, row in comparison_table.iterrows():
        cls = ' class="best-model"' if row['Model'] == best_model_name else ''
        html.append(f"<tr{cls}>")
        for val in row.values:
            if isinstance(val, float):
                html.append(f"<td>{val:.4f}</td>")
            else:
                html.append(f"<td>{val}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")

    # ── Statistical tests table ──
    if stat_tests is not None and not stat_tests.empty:
        html.append("<h2>Pairwise Statistical Tests</h2><table><thead><tr>")
        for col in stat_tests.columns:
            html.append(f"<th>{col}</th>")
        html.append("</tr></thead><tbody>")
        for _, row in stat_tests.iterrows():
            html.append("<tr>")
            for val in row.values:
                if isinstance(val, float):
                    html.append(f"<td>{val:.4f}</td>")
                elif isinstance(val, bool):
                    html.append(f"<td>{'Yes' if val else 'No'}</td>")
                else:
                    html.append(f"<td>{val}</td>")
            html.append("</tr>")
        html.append("</tbody></table>")

    # ── Plotly bar chart ──
    if task_type == 'classification':
        plot_metrics = ['accuracy', 'precision', 'recall', 'f1',
                        'specificity', 'roc_auc', 'pr_auc']
    else:
        plot_metrics = ['r2', 'rmse', 'mae', 'pearson_r']

    available = [m for m in plot_metrics if m in metrics_dict[models[0]]]
    fig = go.Figure()
    for metric in available:
        fig.add_trace(go.Bar(
            name=metric,
            x=models,
            y=[metrics_dict[m].get(metric, 0) for m in models]))
    fig.update_layout(title='Performance Comparison', barmode='group',
                      xaxis_title='Model', yaxis_title='Score', height=500)
    plot_json = json.dumps(fig.to_dict())

    html.append(f"""<h2>Performance Visualisation</h2>
<div id="perf-plot" class="plot-container"></div>
<script>Plotly.newPlot('perf-plot',{plot_json}.data,{plot_json}.layout);</script>""")

    # ── Per-model detail cards ──
    html.append('<h2>Model Details</h2><div class="metrics-grid">')
    for model_name, metrics in metrics_dict.items():
        html.append(f"<div><h3>{model_name}</h3><ul>")
        for metric in available:
            if metric in metrics:
                html.append(f"<li>{metric}: {metrics[metric]:.4f}</li>")
        html.append("</ul></div>")
    html.append("</div>")

    # ── Footer ──
    n_test = metrics_dict[models[0]].get('n_test_samples', 'N/A')
    html.append(f"""<h2>Additional Information</h2><ul>
<li>Number of models compared: {len(models)}</li>
<li>Task type: {task_type.title()}</li>
<li>Test samples: {n_test}</li>
</ul></body></html>""")

    with open(output_file, 'w') as f:
        f.write('\n'.join(html))
    logger.info(f"HTML report saved to {output_file}")


# ── Best model selection ─────────────────────────────────────────────────────

# Metrics where lower is better
_LOWER_IS_BETTER = {'rmse', 'mae', 'max_error',
                     'false_positives', 'false_negatives'}


def select_best_model(metrics_dict: Dict, criterion: str = 'r2') -> Tuple[str, Dict]:
    """
    Select the best model based on a criterion.

    Automatically handles higher-is-better vs lower-is-better metrics.
    """
    best_model = None
    lower = criterion in _LOWER_IS_BETTER
    best_value = np.inf if lower else -np.inf

    for model_name, metrics in metrics_dict.items():
        value = metrics.get(criterion, np.nan)
        if np.isnan(value):
            continue

        if lower:
            if value < best_value:
                best_value = value
                best_model = model_name
        else:
            if value > best_value:
                best_value = value
                best_model = model_name

    return best_model, metrics_dict[best_model]


# ── Main ─────────────────────────────────────────────────────────────────────

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
    parser.add_argument('--selection_criterion', type=str, default=None,
                        help='Criterion for best model selection '
                             '(default: f1 for classification, r2 for regression)')

    args = parser.parse_args()

    # Load data
    metrics_dict = load_metrics_files(args.metrics_files)
    predictions_dict = load_predictions_files(args.predictions_files)

    # Auto-detect task type
    task_type = detect_task_type(metrics_dict)
    logger.info(f"Detected task type: {task_type}")

    # Default selection criterion
    criterion = args.selection_criterion
    if criterion is None:
        criterion = 'f1' if task_type == 'classification' else 'r2'

    # Create comparison table
    comparison_table = create_comparison_table(metrics_dict, task_type)
    comparison_table.to_csv(args.output_table, index=False)
    logger.info(f"Comparison table saved to {args.output_table}")

    # Perform statistical tests
    stat_tests = perform_statistical_tests(predictions_dict, task_type)

    # Create plots
    create_comparison_plots(metrics_dict, predictions_dict,
                            args.output_plots, task_type)

    # Create HTML report
    create_html_report(metrics_dict, predictions_dict, comparison_table,
                       stat_tests, args.output_report, task_type)

    # Select best model
    best_model, best_metrics = select_best_model(metrics_dict, criterion)

    best_model_info = {
        'best_model': best_model,
        'task_type': task_type,
        'selection_criterion': criterion,
        'metrics': best_metrics,
        'comparison_summary': {
            'n_models': len(metrics_dict),
            'all_models': list(metrics_dict.keys()),
            'ranking': comparison_table[['Rank', 'Model']].to_dict('records'),
        },
    }

    with open(args.output_best, 'w') as f:
        json.dump(best_model_info, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Model Comparison Summary  ({task_type.title()})")
    print("=" * 60)
    print(comparison_table.to_string(index=False))
    print("\n" + "=" * 60)
    print(f"Best Model: {best_model}")
    print(f"Selection Criterion: {criterion}")
    print(f"Best {criterion}: {best_metrics.get(criterion, 'N/A'):.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()