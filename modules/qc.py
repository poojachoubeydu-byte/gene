"""
Quality Control Module for Gene Dashboard
==========================================

This module provides comprehensive quality control analysis for RNA-seq data including:
- Library size distributions
- PCA and UMAP visualizations
- Sample distance heatmaps
- Outlier detection
- Data normalization checks

Author: Gene Dashboard Team
Version: 2.0.0
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import zscore
import logging
from typing import Tuple, Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class QualityControl:
    """Main class for RNA-seq quality control analysis."""

    def __init__(self, count_matrix: pd.DataFrame, metadata: Optional[pd.DataFrame] = None):
        """
        Initialize QC analyzer.

        Args:
            count_matrix: DataFrame with genes as rows, samples as columns
            metadata: Optional DataFrame with sample metadata
        """
        self.count_matrix = count_matrix.copy()
        self.metadata = metadata.copy() if metadata is not None else None
        self.gene_symbols = count_matrix.index.tolist()
        self.sample_ids = count_matrix.columns.tolist()

        # Pre-compute normalized data
        self._prepare_normalized_data()

    def _prepare_normalized_data(self):
        """Prepare normalized data for downstream analysis."""
        # Library size normalization (CPM)
        self.lib_sizes = self.count_matrix.sum(axis=0)
        self.cpm_matrix = self.count_matrix.div(self.lib_sizes, axis=1) * 1e6

        # Log2 transformation (add pseudocount)
        self.log_cpm_matrix = np.log2(self.cpm_matrix + 1)

        # Filter low-expression genes (at least 1 CPM in at least 2 samples)
        keep_genes = (self.cpm_matrix >= 1).sum(axis=1) >= 2
        self.filtered_matrix = self.log_cpm_matrix.loc[keep_genes]

        logger.info(f"QC: Filtered to {len(self.filtered_matrix)} genes from {len(self.count_matrix)}")

    def get_library_size_plot(self) -> go.Figure:
        """Generate library size distribution plot."""
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=self.sample_ids,
            y=self.lib_sizes.values,
            name='Library Size',
            marker_color='lightblue'
        ))

        fig.update_layout(
            title="Library Size Distribution",
            xaxis_title="Sample",
            yaxis_title="Total Counts",
            template='simple_white',
            showlegend=False
        )

        # Add median line
        median_size = np.median(self.lib_sizes)
        fig.add_hline(y=median_size, line_dash="dash", line_color="red",
                     annotation_text=f"Median: {median_size:,.0f}")

        return fig

    def get_pca_plot(self, n_components: int = 2) -> go.Figure:
        """Generate PCA plot of samples."""
        if len(self.filtered_matrix.columns) < 3:
            # Not enough samples for meaningful PCA
            fig = go.Figure()
            fig.update_layout(
                title="PCA Plot (Insufficient samples)",
                template='simple_white'
            )
            return fig

        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.filtered_matrix.T)

        # Perform PCA
        pca = PCA(n_components=min(n_components, len(self.filtered_matrix.columns)-1))
        pca_result = pca.fit_transform(scaled_data)

        # Create DataFrame for plotting
        pca_df = pd.DataFrame(
            pca_result[:, :2],
            columns=['PC1', 'PC2'],
            index=self.filtered_matrix.columns
        )

        # Add metadata if available
        if self.metadata is not None:
            pca_df = pca_df.join(self.metadata.set_index('sample_id'))

        # Calculate variance explained
        var_explained = pca.explained_variance_ratio_ * 100

        # Create plot
        if 'condition' in pca_df.columns:
            fig = px.scatter(
                pca_df, x='PC1', y='PC2', color='condition',
                title=f"PCA Plot (PC1: {var_explained[0]:.1f}%, PC2: {var_explained[1]:.1f}%)",
                template='simple_white'
            )
        else:
            fig = px.scatter(
                pca_df, x='PC1', y='PC2',
                title=f"PCA Plot (PC1: {var_explained[0]:.1f}%, PC2: {var_explained[1]:.1f}%)",
                template='simple_white'
            )

        fig.update_traces(marker=dict(size=8))
        return fig

    def get_umap_plot(self) -> go.Figure:
        """Generate UMAP plot of samples."""
        try:
            from umap import UMAP
        except ImportError:
            logger.warning("UMAP not available, falling back to PCA")
            return self.get_pca_plot()

        if len(self.filtered_matrix.columns) < 3:
            return self.get_pca_plot()

        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.filtered_matrix.T)

        # Perform UMAP
        umap_model = UMAP(n_components=2, random_state=42)
        umap_result = umap_model.fit_transform(scaled_data)

        # Create DataFrame for plotting
        umap_df = pd.DataFrame(
            umap_result,
            columns=['UMAP1', 'UMAP2'],
            index=self.filtered_matrix.columns
        )

        # Add metadata if available
        if self.metadata is not None:
            umap_df = umap_df.join(self.metadata.set_index('sample_id'))

        # Create plot
        if 'condition' in umap_df.columns:
            fig = px.scatter(
                umap_df, x='UMAP1', y='UMAP2', color='condition',
                title="UMAP Plot",
                template='simple_white'
            )
        else:
            fig = px.scatter(
                umap_df, x='UMAP1', y='UMAP2',
                title="UMAP Plot",
                template='simple_white'
            )

        fig.update_traces(marker=dict(size=8))
        return fig

    def get_sample_distance_heatmap(self) -> go.Figure:
        """Generate sample distance heatmap."""
        if len(self.filtered_matrix.columns) < 2:
            fig = go.Figure()
            fig.update_layout(
                title="Sample Distance Heatmap (Insufficient samples)",
                template='simple_white'
            )
            return fig

        # Calculate Euclidean distances
        distances = euclidean_distances(self.filtered_matrix.T)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=distances,
            x=self.filtered_matrix.columns,
            y=self.filtered_matrix.columns,
            colorscale='RdBu_r',
            colorbar=dict(title='Euclidean Distance')
        ))

        fig.update_layout(
            title="Sample Distance Heatmap",
            xaxis_title="Sample",
            yaxis_title="Sample",
            template='simple_white'
        )

        return fig

    def detect_outliers(self, method: str = 'zscore', threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect outlier samples based on expression profiles.

        Args:
            method: Detection method ('zscore', 'isolation_forest', 'local_outlier_factor')
            threshold: Threshold for outlier detection

        Returns:
            Dictionary with outlier information
        """
        if len(self.filtered_matrix.columns) < 3:
            return {'outliers': [], 'method': method, 'message': 'Insufficient samples for outlier detection'}

        outliers = []

        if method == 'zscore':
            # Calculate z-scores for each sample's expression profile
            sample_means = self.filtered_matrix.mean(axis=0)
            sample_stds = self.filtered_matrix.std(axis=0)

            z_scores = np.abs((sample_means - sample_means.mean()) / sample_means.std())
            outlier_indices = np.where(z_scores > threshold)[0]
            outliers = [self.filtered_matrix.columns[i] for i in outlier_indices]

        elif method == 'isolation_forest':
            try:
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(self.filtered_matrix.T)
                outliers = [self.filtered_matrix.columns[i] for i in np.where(outlier_labels == -1)[0]]
            except ImportError:
                logger.warning("IsolationForest not available")
                return self.detect_outliers('zscore', threshold)

        elif method == 'local_outlier_factor':
            try:
                from sklearn.neighbors import LocalOutlierFactor
                lof = LocalOutlierFactor(contamination=0.1)
                outlier_labels = lof.fit_predict(self.filtered_matrix.T)
                outliers = [self.filtered_matrix.columns[i] for i in np.where(outlier_labels == -1)[0]]
            except ImportError:
                logger.warning("LocalOutlierFactor not available")
                return self.detect_outliers('zscore', threshold)

        return {
            'outliers': outliers,
            'method': method,
            'threshold': threshold,
            'total_samples': len(self.filtered_matrix.columns),
            'outlier_percentage': len(outliers) / len(self.filtered_matrix.columns) * 100
        }

    def get_qc_summary(self) -> Dict[str, Any]:
        """Generate comprehensive QC summary."""
        summary = {
            'total_samples': len(self.sample_ids),
            'total_genes': len(self.count_matrix),
            'filtered_genes': len(self.filtered_matrix),
            'library_sizes': {
                'min': int(self.lib_sizes.min()),
                'max': int(self.lib_sizes.max()),
                'median': int(self.lib_sizes.median()),
                'mean': int(self.lib_sizes.mean())
            },
            'expression_stats': {
                'mean_cpm': float(self.cpm_matrix.mean().mean()),
                'median_cpm': float(self.cpm_matrix.median().median()),
                'zeros_percentage': float((self.count_matrix == 0).sum().sum() / self.count_matrix.size * 100)
            }
        }

        # Outlier detection
        outlier_info = self.detect_outliers()
        summary['outliers'] = outlier_info

        # Condition balance (if metadata available)
        if self.metadata is not None:
            condition_counts = self.metadata['condition'].value_counts().to_dict()
            summary['condition_balance'] = condition_counts

        return summary

    def get_qc_report_plots(self) -> List[go.Figure]:
        """Generate all QC plots for reporting."""
        plots = []

        # Library size plot
        plots.append(self.get_library_size_plot())

        # PCA plot
        plots.append(self.get_pca_plot())

        # UMAP plot (if available)
        try:
            plots.append(self.get_umap_plot())
        except:
            pass

        # Sample distance heatmap
        plots.append(self.get_sample_distance_heatmap())

        return plots