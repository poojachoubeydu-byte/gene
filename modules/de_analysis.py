"""
Differential Expression Analysis Module for Gene Dashboard
===========================================================

This module provides multiple differential expression analysis engines:
- pyDESeq2: DESeq2 implementation in Python
- EdgeR-like: Python implementation of edgeR methodology
- Limma-like: Linear modeling approach

Supports configurable contrasts, covariates, and FDR correction.

Author: Gene Dashboard Team
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import logging
from typing import Tuple, Optional, Dict, Any, List, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DEAnalysisEngine(ABC):
    """Abstract base class for differential expression analysis engines."""

    @abstractmethod
    def run_de_analysis(self, count_matrix: pd.DataFrame, metadata: pd.DataFrame,
                       design_formula: str, contrast: str) -> pd.DataFrame:
        """Run differential expression analysis."""
        pass

    @abstractmethod
    def get_engine_name(self) -> str:
        """Return engine name."""
        pass


class PyDESeq2Engine(DEAnalysisEngine):
    """DESeq2 implementation using pyDESeq2."""

    def get_engine_name(self) -> str:
        return "pyDESeq2"

    def run_de_analysis(self, count_matrix: pd.DataFrame, metadata: pd.DataFrame,
                       design_formula: str, contrast: str) -> pd.DataFrame:
        """Run DE analysis using pyDESeq2."""
        try:
            from pydeseq2 import DeseqDataSet, DefaultInference
            from pydeseq2.utils import load_example_data
        except ImportError:
            logger.error("pyDESeq2 not available")
            raise ImportError("pyDESeq2 is required for this engine")

        try:
            # Prepare data
            counts = count_matrix.T  # pyDESeq2 expects samples x genes
            metadata = metadata.set_index('sample_id')

            # Ensure sample order matches
            counts = counts.loc[metadata.index]

            # Create DESeqDataSet
            dds = DeseqDataSet(
                counts=counts,
                metadata=metadata,
                design_factors=design_formula.split('~')[1].strip().split('+')
            )

            # Run DESeq2
            dds.deseq2()

            # Get results
            if ':' in contrast:
                # Interaction contrast
                condition1, condition2 = contrast.split(':')
                res = dds.results(condition1.strip(), condition2.strip())
            else:
                # Simple contrast
                res = dds.results(contrast)

            # Convert to standard format
            results_df = res.reset_index()
            results_df = results_df.rename(columns={
                results_df.columns[0]: 'gene_symbol',
                'log2FoldChange': 'log2_fold_change',
                'padj': 'adjusted_p_value',
                'pvalue': 'p_value',
                'baseMean': 'base_mean'
            })

            # Ensure required columns exist
            required_cols = ['gene_symbol', 'log2_fold_change', 'adjusted_p_value', 'p_value', 'base_mean']
            for col in required_cols:
                if col not in results_df.columns:
                    results_df[col] = np.nan

            return results_df[required_cols]

        except Exception as e:
            logger.error(f"pyDESeq2 analysis failed: {e}")
            raise


class EdgeRLikeEngine(DEAnalysisEngine):
    """EdgeR-like differential expression analysis in Python."""

    def get_engine_name(self) -> str:
        return "EdgeR-like"

    def run_de_analysis(self, count_matrix: pd.DataFrame, metadata: pd.DataFrame,
                       design_formula: str, contrast: str) -> pd.DataFrame:
        """Run DE analysis using EdgeR-like methodology."""
        try:
            # This is a simplified implementation
            # In practice, you'd want to use rpy2 to call edgeR or implement full edgeR

            # For now, implement a basic negative binomial GLM approach
            return self._simple_nb_glm(count_matrix, metadata, contrast)

        except Exception as e:
            logger.error(f"EdgeR-like analysis failed: {e}")
            raise

    def _simple_nb_glm(self, count_matrix: pd.DataFrame, metadata: pd.DataFrame,
                      contrast: str) -> pd.DataFrame:
        """Simple negative binomial GLM implementation."""
        # Parse contrast (e.g., "condition_treated_vs_control")
        if '_vs_' in contrast:
            condition_col, contrast_vals = contrast.split('_vs_')
            treated_val, control_val = contrast_vals.split('_')
        else:
            raise ValueError("Contrast must be in format: condition_treated_vs_control")

        # Get sample groups
        treated_samples = metadata[metadata[condition_col] == treated_val]['sample_id'].tolist()
        control_samples = metadata[metadata[condition_col] == control_val]['sample_id'].tolist()

        if not treated_samples or not control_samples:
            raise ValueError("No samples found for contrast groups")

        # Calculate statistics for each gene
        results = []

        for gene in count_matrix.index:
            treated_counts = count_matrix.loc[gene, treated_samples]
            control_counts = count_matrix.loc[gene, control_samples]

            # Simple fold change calculation
            treated_mean = treated_counts.mean()
            control_mean = control_counts.mean()

            if control_mean == 0:
                log2fc = np.inf if treated_mean > 0 else 0
            else:
                log2fc = np.log2(treated_mean / control_mean)

            # T-test (not ideal for count data, but simple)
            try:
                t_stat, p_val = stats.ttest_ind(treated_counts, control_counts)
            except:
                t_stat, p_val = 0, 1

            results.append({
                'gene_symbol': gene,
                'log2_fold_change': log2fc,
                'p_value': p_val,
                'base_mean': (treated_mean + control_mean) / 2
            })

        results_df = pd.DataFrame(results)

        # Apply BH correction
        _, adj_p = multipletests(results_df['p_value'], method='fdr_bh')[:2]
        results_df['adjusted_p_value'] = adj_p

        return results_df


class LimmaLikeEngine(DEAnalysisEngine):
    """Limma-like linear modeling approach."""

    def get_engine_name(self) -> str:
        return "Limma-like"

    def run_de_analysis(self, count_matrix: pd.DataFrame, metadata: pd.DataFrame,
                       design_formula: str, contrast: str) -> pd.DataFrame:
        """Run DE analysis using limma-like approach."""
        try:
            # For simplicity, use log-transformed data and linear modeling
            # In practice, you'd implement voom or similar normalization

            # Log transform counts
            log_counts = np.log2(count_matrix + 1)

            # Simple linear model implementation
            return self._simple_linear_model(log_counts, metadata, contrast)

        except Exception as e:
            logger.error(f"Limma-like analysis failed: {e}")
            raise

    def _simple_linear_model(self, log_counts: pd.DataFrame, metadata: pd.DataFrame,
                           contrast: str) -> pd.DataFrame:
        """Simple linear model for DE analysis."""
        # Parse contrast
        if '_vs_' in contrast:
            condition_col, contrast_vals = contrast.split('_vs_')
            treated_val, control_val = contrast_vals.split('_')
        else:
            raise ValueError("Contrast must be in format: condition_treated_vs_control")

        # Create design matrix
        treated_samples = metadata[metadata[condition_col] == treated_val]['sample_id'].tolist()
        control_samples = metadata[metadata[condition_col] == control_val]['sample_id'].tolist()

        if not treated_samples or not control_samples:
            raise ValueError("No samples found for contrast groups")

        # Calculate fold changes and p-values
        results = []

        for gene in log_counts.index:
            treated_expr = log_counts.loc[gene, treated_samples]
            control_expr = log_counts.loc[gene, control_samples]

            # Mean difference (log2 fold change)
            log2fc = treated_expr.mean() - control_expr.mean()

            # T-test
            try:
                t_stat, p_val = stats.ttest_ind(treated_expr, control_expr)
            except:
                t_stat, p_val = 0, 1

            results.append({
                'gene_symbol': gene,
                'log2_fold_change': log2fc,
                'p_value': p_val,
                'base_mean': log_counts.loc[gene].mean()
            })

        results_df = pd.DataFrame(results)

        # Apply BH correction
        _, adj_p = multipletests(results_df['p_value'], method='fdr_bh')[:2]
        results_df['adjusted_p_value'] = adj_p

        return results_df


class DEAanalyzer:
    """Main class for differential expression analysis."""

    def __init__(self):
        self.engines = {
            'pydeseq2': PyDESeq2Engine(),
            'edger_like': EdgeRLikeEngine(),
            'limma_like': LimmaLikeEngine()
        }

    def run_analysis(self, count_matrix: pd.DataFrame, metadata: pd.DataFrame,
                    engine: str = 'pydeseq2', design_formula: str = '~ condition',
                    contrast: str = 'condition_treated_vs_control',
                    lfc_threshold: float = 0.0, padj_threshold: float = 0.05) -> Dict[str, Any]:
        """
        Run differential expression analysis.

        Args:
            count_matrix: Genes x samples DataFrame
            metadata: Sample metadata DataFrame
            engine: Analysis engine to use
            design_formula: Design formula for the model
            contrast: Contrast specification
            lfc_threshold: Log2 fold change threshold
            padj_threshold: Adjusted p-value threshold

        Returns:
            Dictionary with results and metadata
        """

        if engine not in self.engines:
            raise ValueError(f"Unknown engine: {engine}")

        logger.info(f"Running DE analysis with {engine} engine")

        # Run analysis
        de_results = self.engines[engine].run_de_analysis(
            count_matrix, metadata, design_formula, contrast
        )

        # Apply thresholds
        de_results['significant'] = (
            (de_results['log2_fold_change'].abs() >= lfc_threshold) &
            (de_results['adjusted_p_value'] <= padj_threshold)
        )

        # Summary statistics
        summary = {
            'total_genes': len(de_results),
            'upregulated': (de_results['significant'] & (de_results['log2_fold_change'] > 0)).sum(),
            'downregulated': (de_results['significant'] & (de_results['log2_fold_change'] < 0)).sum(),
            'engine': engine,
            'design_formula': design_formula,
            'contrast': contrast,
            'lfc_threshold': lfc_threshold,
            'padj_threshold': padj_threshold
        }

        return {
            'results': de_results,
            'summary': summary,
            'engine_info': {
                'name': self.engines[engine].get_engine_name(),
                'version': '2.0.0'
            }
        }

    def get_available_engines(self) -> List[str]:
        """Get list of available analysis engines."""
        return list(self.engines.keys())

    def validate_inputs(self, count_matrix: pd.DataFrame, metadata: pd.DataFrame) -> List[str]:
        """Validate inputs for DE analysis."""
        errors = []

        # Check count matrix
        if count_matrix.empty:
            errors.append("Count matrix is empty")

        # Check metadata
        if metadata.empty:
            errors.append("Metadata is empty")

        # Check sample overlap
        count_samples = set(count_matrix.columns)
        meta_samples = set(metadata['sample_id'])

        if not count_samples.issubset(meta_samples):
            missing = count_samples - meta_samples
            errors.append(f"Samples in count matrix not found in metadata: {missing}")

        # Check conditions
        if 'condition' not in metadata.columns:
            errors.append("Metadata must contain 'condition' column")

        condition_counts = metadata['condition'].value_counts()
        if len(condition_counts) < 2:
            errors.append("At least 2 different conditions required")

        return errors