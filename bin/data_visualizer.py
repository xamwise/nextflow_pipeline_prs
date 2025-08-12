"""
Visualize genotype data and create comprehensive reports.
"""

import numpy as np
import pandas as pd
import h5py
import json
import argparse
from pathlib import Path
import logging
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from typing import Dict

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def load_data_subset(
    genotype_file: str,
    phenotype_file: str,
    max_samples: int = 1000,
    max_snps: int = 5000
) -> Tuple:
    """
    Load a subset of data for visualization.
    
    Args:
        genotype_file: Path to HDF5 genotype file
        phenotype_file: Path to phenotype CSV file
        max_samples: Maximum number of samples to load
        max_snps: Maximum number of SNPs to load
        
    Returns:
        Tuple of (genotypes, phenotypes, snp_info, sample_info)
    """
    with h5py.File(genotype_file, 'r') as f:
        n_samples = f['genotypes'].shape[0]
        n_snps = f['genotypes'].shape[1]
        
        # Sample subset for visualization
        sample_size = min(max_samples, n_samples)
        snp_size = min(max_snps, n_snps)
        
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        snp_indices = np.random.choice(n_snps, snp_size, replace=False)
        
        # Load genotype subset
        genotypes = f['genotypes'][sample_indices][:, snp_indices]
        
        # Load SNP information if available
        snp_info = None
        if 'snp_info/snp' in f:
            snp_names = f['snp_info/snp'][snp_indices]
            if 'snp_info/chrom' in f:
                chroms = f['snp_info/chrom'][snp_indices]
                positions = f['snp_info/pos'][snp_indices]
                snp_info = pd.DataFrame({
                    'snp': snp_names,
                    'chrom': chroms,
                    'pos': positions
                })
        
        # Load sample information if available
        sample_info = None
        if 'sample_info/iid' in f:
            iids = f['sample_info/iid'][sample_indices]
            sample_info = pd.DataFrame({'iid': iids})
    
    # Load phenotypes
    phenotypes = pd.read_csv(phenotype_file)
    phenotypes_subset = phenotypes.iloc[sample_indices]
    
    return genotypes, phenotypes_subset, snp_info, sample_info


def create_basic_visualizations(
    genotypes: np.ndarray,
    phenotypes: pd.DataFrame,
    stats: Dict,
    output_dir: Path
):
    """
    Create basic visualizations of genotype data.
    
    Args:
        genotypes: Genotype matrix subset
        phenotypes: Phenotype dataframe
        stats: Data statistics dictionary
        output_dir: Output directory for plots
    """
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Genotype distribution
    ax1 = plt.subplot(3, 3, 1)
    plt.hist(genotypes.flatten(), bins=50, edgecolor='black', alpha=0.7)
    plt.title('Genotype Value Distribution')
    plt.xlabel('Genotype Value')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(genotypes.flatten()), color='red', linestyle='--', label='Mean')
    plt.legend()
    
    # 2. Missing data pattern
    ax2 = plt.subplot(3, 3, 2)
    missing_pattern = np.isnan(genotypes[:100, :100])
    sns.heatmap(missing_pattern, cbar=True, cmap='RdYlBu', ax=ax2, 
                xticklabels=False, yticklabels=False)
    plt.title('Missing Data Pattern (100x100 subset)')
    
    # 3. Sample-wise statistics
    ax3 = plt.subplot(3, 3, 3)
    sample_means = np.nanmean(genotypes, axis=1)
    plt.hist(sample_means, bins=30, edgecolor='black', alpha=0.7)
    plt.title('Sample-wise Mean Genotype')
    plt.xlabel('Mean Genotype')
    plt.ylabel('Number of Samples')
    
    # 4. SNP-wise statistics
    ax4 = plt.subplot(3, 3, 4)
    snp_means = np.nanmean(genotypes, axis=0)
    plt.hist(snp_means, bins=30, edgecolor='black', alpha=0.7)
    plt.title('SNP-wise Mean Genotype')
    plt.xlabel('Mean Genotype')
    plt.ylabel('Number of SNPs')
    
    # 5. Minor allele frequency
    ax5 = plt.subplot(3, 3, 5)
    maf = np.minimum(snp_means/2, 1 - snp_means/2)
    plt.hist(maf, bins=30, edgecolor='black', alpha=0.7)
    plt.title('Minor Allele Frequency Distribution')
    plt.xlabel('MAF')
    plt.ylabel('Number of SNPs')
    
    # 6. Phenotype distribution
    ax6 = plt.subplot(3, 3, 6)
    phenotype_values = phenotypes.iloc[:, 0].values
    
    # Check if binary or continuous
    unique_vals = np.unique(phenotype_values[~np.isnan(phenotype_values)])
    if len(unique_vals) == 2:
        # Binary phenotype
        counts = pd.Series(phenotype_values).value_counts()
        plt.bar(counts.index, counts.values, edgecolor='black')
        plt.title('Phenotype Distribution (Binary)')
        plt.xlabel('Phenotype Value')
        plt.ylabel('Count')
    else:
        # Continuous phenotype
        plt.hist(phenotype_values, bins=30, edgecolor='black', alpha=0.7)
        plt.title('Phenotype Distribution (Continuous)')
        plt.xlabel('Phenotype Value')
        plt.ylabel('Frequency')
    
    # 7. PCA visualization
    ax7 = plt.subplot(3, 3, 7)
    logger.info("Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    
    # Handle missing values for PCA
    genotypes_imputed = genotypes.copy()
    for i in range(genotypes_imputed.shape[1]):
        col = genotypes_imputed[:, i]
        genotypes_imputed[np.isnan(col), i] = np.nanmean(col)
    
    pca_result = pca.fit_transform(genotypes_imputed)
    
    # Color by phenotype if possible
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                         c=phenotype_values, cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Phenotype')
    plt.title('PCA of Genotypes')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    
    # 8. SNP correlation heatmap
    ax8 = plt.subplot(3, 3, 8)
    n_snps_corr = min(50, genotypes.shape[1])
    corr_subset = np.corrcoef(genotypes[:, :n_snps_corr].T)
    sns.heatmap(corr_subset, cmap='coolwarm', center=0, ax=ax8,
                cbar_kws={'shrink': 0.8}, xticklabels=False, yticklabels=False)
    plt.title(f'SNP Correlation Matrix ({n_snps_corr} SNPs)')
    
    # 9. Data quality metrics
    ax9 = plt.subplot(3, 3, 9)
    metrics_text = f"""Data Quality Metrics:
    
    Total Samples: {stats['n_samples']:,}
    Total SNPs: {stats['n_snps']:,}
    Original SNPs: {stats.get('n_snps_original', stats['n_snps']):,}
    Filtered SNPs: {stats.get('n_snps_filtered', 0):,}
    Missing Rate: {stats['missing_rate']:.2%}
    Mean Genotype: {stats['mean_genotype']:.3f}
    Std Genotype: {stats['std_genotype']:.3f}
    Min Value: {stats['min_genotype']:.3f}
    Max Value: {stats['max_genotype']:.3f}
    MAF Threshold: {stats.get('maf_threshold', 'N/A')}
    """
    ax9.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
             fontfamily='monospace')
    ax9.axis('off')
    
    plt.suptitle('Genotype Data Overview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'data_overview.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Basic visualizations saved to {output_path}")


def create_interactive_visualizations(
    genotypes: np.ndarray,
    phenotypes: pd.DataFrame,
    stats: Dict,
    output_dir: Path
):
    """
    Create interactive visualizations using Plotly.
    
    Args:
        genotypes: Genotype matrix subset
        phenotypes: Phenotype dataframe
        stats: Data statistics dictionary
        output_dir: Output directory for plots
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Genotype Distribution', 'Phenotype vs Mean Genotype',
                       'MAF Distribution', 'Sample Quality', 'PCA Plot', 'Statistics'),
        specs=[[{'type': 'histogram'}, {'type': 'scatter'}, {'type': 'histogram'}],
               [{'type': 'box'}, {'type': 'scatter'}, {'type': 'table'}]]
    )
    
    # 1. Genotype distribution
    fig.add_trace(
        go.Histogram(x=genotypes.flatten(), nbinsx=50, name='Genotypes',
                    marker_color='blue', opacity=0.7),
        row=1, col=1
    )
    
    # 2. Phenotype vs mean genotype
    sample_means = np.nanmean(genotypes, axis=1)
    phenotype_values = phenotypes.iloc[:, 0].values
    
    fig.add_trace(
        go.Scatter(x=sample_means, y=phenotype_values, mode='markers',
                  marker=dict(size=5, opacity=0.5, color=phenotype_values,
                             colorscale='Viridis', showscale=True),
                  name='Samples'),
        row=1, col=2
    )
    
    # 3. MAF distribution
    snp_means = np.nanmean(genotypes, axis=0)
    maf = np.minimum(snp_means/2, 1 - snp_means/2)
    
    fig.add_trace(
        go.Histogram(x=maf, nbinsx=30, name='MAF',
                    marker_color='green', opacity=0.7),
        row=1, col=3
    )
    
    # 4. Sample quality boxplot
    sample_missing = np.mean(np.isnan(genotypes), axis=1)
    
    fig.add_trace(
        go.Box(y=sample_missing, name='Missing Rate',
               marker_color='orange'),
        row=2, col=1
    )
    
    # 5. PCA scatter
    logger.info("Computing PCA for interactive plot...")
    pca = PCA(n_components=2, random_state=42)
    
    # Impute missing values
    genotypes_imputed = genotypes.copy()
    for i in range(genotypes_imputed.shape[1]):
        col = genotypes_imputed[:, i]
        genotypes_imputed[np.isnan(col), i] = np.nanmean(col)
    
    pca_result = pca.fit_transform(genotypes_imputed)
    
    fig.add_trace(
        go.Scatter(x=pca_result[:, 0], y=pca_result[:, 1], mode='markers',
                  marker=dict(size=5, color=phenotype_values,
                             colorscale='Viridis', showscale=True),
                  text=[f'Sample {i}' for i in range(len(pca_result))],
                  name='PCA'),
        row=2, col=2
    )
    
    # 6. Statistics table
    stats_data = []
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                stats_data.append([key.replace('_', ' ').title(), f'{value:.4f}'])
            else:
                stats_data.append([key.replace('_', ' ').title(), f'{value:,}'])
    
    fig.add_trace(
        go.Table(
            header=dict(values=['Metric', 'Value'],
                       fill_color='lightgray',
                       align='left'),
            cells=dict(values=list(zip(*stats_data)),
                      fill_color='white',
                      align='left')
        ),
        row=2, col=3
    )
    
    # Update layout
    fig.update_layout(
        title='Interactive Genotype Data Visualization',
        height=800,
        showlegend=False
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Genotype Value", row=1, col=1)
    fig.update_xaxes(title_text="Mean Genotype", row=1, col=2)
    fig.update_xaxes(title_text="MAF", row=1, col=3)
    fig.update_xaxes(title_text=f"PC1 ({pca.explained_variance_ratio_[0]:.2%})", row=2, col=2)
    
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Phenotype", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=3)
    fig.update_yaxes(title_text="Missing Rate", row=2, col=1)
    fig.update_yaxes(title_text=f"PC2 ({pca.explained_variance_ratio_[1]:.2%})", row=2, col=2)
    
    # Save HTML
    output_path = output_dir / 'interactive_viz.html'
    fig.write_html(output_path)
    logger.info(f"Interactive visualizations saved to {output_path}")


def create_quality_report(
    genotypes: np.ndarray,
    phenotypes: pd.DataFrame,
    stats: Dict,
    output_dir: Path
):
    """
    Create a PDF quality report.
    
    Args:
        genotypes: Genotype matrix subset
        phenotypes: Phenotype dataframe
        stats: Data statistics dictionary
        output_dir: Output directory for report
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    pdf_path = output_dir / 'visualization_report.pdf'
    
    with PdfPages(pdf_path) as pdf:
        # Page 1: Overview
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Genotype Data Quality Report', fontsize=16, fontweight='bold')
        
        # Summary text
        ax = fig.add_subplot(111)
        summary_text = f"""
        Dataset Overview
        ================
        Total Samples: {stats['n_samples']:,}
        Total SNPs: {stats['n_snps']:,}
        Missing Data Rate: {stats['missing_rate']:.2%}
        
        Genotype Statistics
        ==================
        Mean: {stats['mean_genotype']:.4f}
        Std Dev: {stats['std_genotype']:.4f}
        Range: [{stats['min_genotype']:.2f}, {stats['max_genotype']:.2f}]
        
        Quality Control
        ===============
        Original SNPs: {stats.get('n_snps_original', stats['n_snps']):,}
        Filtered SNPs: {stats.get('n_snps_filtered', 0):,}
        MAF Threshold: {stats.get('maf_threshold', 'N/A')}
        
        This report provides a comprehensive overview of the genotype data quality
        and characteristics for PRS model training.
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace')
        ax.axis('off')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Distributions
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Data Distributions', fontsize=14, fontweight='bold')
        
        # Genotype distribution
        axes[0, 0].hist(genotypes.flatten(), bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Genotype Values')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        
        # Phenotype distribution
        phenotype_values = phenotypes.iloc[:, 0].values
        axes[0, 1].hist(phenotype_values, bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[0, 1].set_title('Phenotype Distribution')
        axes[0, 1].set_xlabel('Phenotype Value')
        axes[0, 1].set_ylabel('Frequency')
        
        # Sample means
        sample_means = np.nanmean(genotypes, axis=1)
        axes[1, 0].hist(sample_means, bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 0].set_title('Sample Mean Genotypes')
        axes[1, 0].set_xlabel('Mean Genotype')
        axes[1, 0].set_ylabel('Number of Samples')
        
        # MAF distribution
        snp_means = np.nanmean(genotypes, axis=0)
        maf = np.minimum(snp_means/2, 1 - snp_means/2)
        axes[1, 1].hist(maf, bins=30, edgecolor='black', alpha=0.7, color='purple')
        axes[1, 1].set_title('Minor Allele Frequency')
        axes[1, 1].set_xlabel('MAF')
        axes[1, 1].set_ylabel('Number of SNPs')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    logger.info(f"PDF report saved to {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize genotype data for PRS pipeline')
    parser.add_argument('--genotype_file', type=str, required=True,
                        help='Path to HDF5 genotype file')
    parser.add_argument('--phenotype_file', type=str, required=True,
                        help='Path to phenotype CSV file')
    parser.add_argument('--stats_file', type=str, required=True,
                        help='Path to statistics JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for visualizations')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Maximum samples for visualization')
    parser.add_argument('--max_snps', type=int, default=5000,
                        help='Maximum SNPs for visualization')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load statistics
    with open(args.stats_file, 'r') as f:
        stats = json.load(f)
    
    # Load data subset
    logger.info("Loading data subset for visualization...")
    genotypes, phenotypes, snp_info, sample_info = load_data_subset(
        args.genotype_file,
        args.phenotype_file,
        args.max_samples,
        args.max_snps
    )
    
    logger.info(f"Loaded {genotypes.shape[0]} samples and {genotypes.shape[1]} SNPs")
    
    # Create visualizations
    logger.info("Creating basic visualizations...")
    create_basic_visualizations(genotypes, phenotypes, stats, output_dir)
    
    logger.info("Creating interactive visualizations...")
    create_interactive_visualizations(genotypes, phenotypes, stats, output_dir)
    
    logger.info("Creating quality report...")
    create_quality_report(genotypes, phenotypes, stats, output_dir)
    
    logger.info(f"All visualizations completed and saved to {output_dir}")


if __name__ == '__main__':
    main()