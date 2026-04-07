"""
Pathway Analysis Module for Gene Dashboard
===========================================

This module provides comprehensive pathway enrichment analysis:
- GO (Gene Ontology) enrichment
- KEGG pathway analysis
- Reactome pathway analysis
- Custom GMT file support
- Offline annotation databases

Supports multiple species and statistical methods.

Author: Gene Dashboard Team
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact, hypergeom
from statsmodels.stats.multitest import multipletests
import logging
from typing import Tuple, Optional, Dict, Any, List, Set
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class PathwayDatabase:
    """Container for pathway gene set databases."""

    def __init__(self):
        self.databases = {}
        self._load_builtin_databases()

    def _load_builtin_databases(self):
        """Load built-in pathway databases."""
        # KEGG pathways (simplified example - in practice load from files)
        self.databases['kegg'] = {
            'Cell Cycle': ['CDK1', 'CCNB1', 'TOP2A', 'MKI67', 'CDK2', 'CCNA2', 'CCNE1', 'RB1', 'E2F1'],
            'Apoptosis': ['BAX', 'CASP3', 'BCL2', 'TP53', 'FAS', 'TNF', 'TRADD', 'FADD', 'CASP8'],
            'DNA Repair': ['BRCA1', 'BRCA2', 'RAD51', 'PARP1', 'ATM', 'ATR', 'CHEK1', 'CHEK2'],
            'p53 Signaling': ['TP53', 'CDKN1A', 'MDM2', 'BAX', 'PUMA', 'BBC3', 'CDK2', 'CCNE1'],
            'TGF-beta Signaling': ['TGFB1', 'SMAD2', 'SMAD3', 'SMAD4', 'MYC', 'JUN', 'FOS'],
            'Wnt Signaling': ['CTNNB1', 'APC', 'GSK3B', 'AXIN1', 'TCF7L2', 'LEF1', 'MYC'],
            'MAPK Signaling': ['MAPK1', 'MAPK3', 'RAF1', 'BRAF', 'HRAS', 'KRAS', 'NRAS', 'MEK1', 'MEK2'],
            'PI3K-AKT Signaling': ['PIK3CA', 'PIK3CB', 'AKT1', 'AKT2', 'PTEN', 'MTOR', 'RHEB'],
            'Hedgehog Signaling': ['SHH', 'PTCH1', 'SMO', 'GLI1', 'GLI2', 'GLI3', 'SUFU'],
            'Notch Signaling': ['NOTCH1', 'NOTCH2', 'DLL1', 'JAG1', 'RBPJ', 'HES1', 'HEY1']
        }

        # GO Biological Process (simplified)
        self.databases['go_bp'] = {
            'Cell Cycle Process': ['CDK1', 'CCNB1', 'TOP2A', 'MKI67', 'CDK2', 'CCNA2', 'RB1'],
            'Apoptotic Process': ['BAX', 'CASP3', 'BCL2', 'TP53', 'FAS', 'TNF', 'CASP8'],
            'DNA Repair': ['BRCA1', 'BRCA2', 'RAD51', 'PARP1', 'ATM', 'ATR', 'XRCC1'],
            'Cell Proliferation': ['MYC', 'CCND1', 'CDK4', 'CDK6', 'RB1', 'E2F1', 'PCNA'],
            'Inflammatory Response': ['TNF', 'IL6', 'IL1B', 'NFKB1', 'RELA', 'STAT3', 'JAK2'],
            'Immune Response': ['IFNG', 'CD4', 'CD8A', 'CD28', 'CTLA4', 'PDCD1', 'LAG3'],
            'Metabolic Process': ['HK1', 'PFKL', 'ALDOA', 'GAPDH', 'PGK1', 'ENO1', 'LDHA'],
            'Signal Transduction': ['EGFR', 'ERBB2', 'SRC', 'ABL1', 'JAK2', 'STAT1', 'STAT3'],
            'Transcription': ['TP53', 'MYC', 'JUN', 'FOS', 'SP1', 'NFYA', 'CREB1'],
            'Translation': ['EIF4E', 'EIF4G1', 'RPS6KB1', 'MTOR', 'EIF2AK3', 'PERK']
        }

        # Reactome pathways (simplified)
        self.databases['reactome'] = {
            'Cell Cycle': ['CDK1', 'CCNB1', 'TOP2A', 'MKI67', 'CDK2', 'CCNA2', 'RB1', 'E2F1'],
            'Apoptosis': ['BAX', 'CASP3', 'BCL2', 'TP53', 'FAS', 'TNF', 'TRADD', 'FADD'],
            'DNA Repair': ['BRCA1', 'BRCA2', 'RAD51', 'PARP1', 'ATM', 'ATR', 'CHEK1', 'XRCC1'],
            'Immune System': ['IFNG', 'TNF', 'IL6', 'IL1B', 'NFKB1', 'RELA', 'STAT1', 'IRF3'],
            'Signal Transduction': ['EGFR', 'ERBB2', 'SRC', 'ABL1', 'RAS', 'RAF1', 'MEK1', 'ERK1'],
            'Metabolism': ['HK1', 'PFKL', 'ALDOA', 'GAPDH', 'PGK1', 'ENO1', 'LDHA', 'PDHA1'],
            'Gene Expression': ['TP53', 'MYC', 'JUN', 'FOS', 'SP1', 'TBP', 'TFIIA', 'TFIIB'],
            'Cell-Cell Communication': ['NOTCH1', 'DLL1', 'JAG1', 'RBPJ', 'HES1', 'HEY1'],
            'Developmental Biology': ['WNT1', 'FZD1', 'CTNNB1', 'APC', 'GSK3B', 'AXIN1'],
            'Disease': ['TP53', 'BRCA1', 'BRCA2', 'APC', 'MLH1', 'MSH2', 'PTEN', 'RB1']
        }

    def get_database(self, name: str) -> Dict[str, List[str]]:
        """Get a pathway database by name."""
        return self.databases.get(name, {})

    def get_available_databases(self) -> List[str]:
        """Get list of available databases."""
        return list(self.databases.keys())

    def load_custom_gmt(self, filepath: str, name: str):
        """Load custom GMT file."""
        try:
            pathways = {}
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        pathway_name = parts[0]
                        genes = parts[2:]  # Skip description
                        pathways[pathway_name] = genes
            self.databases[name] = pathways
            logger.info(f"Loaded custom GMT: {name} with {len(pathways)} pathways")
        except Exception as e:
            logger.error(f"Failed to load GMT file: {e}")
            raise


class EnrichmentAnalyzer:
    """Performs pathway enrichment analysis."""

    def __init__(self, database: str = 'kegg', species: str = 'human'):
        self.database_name = database
        self.species = species
        self.db = PathwayDatabase()
        self.pathways = self.db.get_database(database)

        if not self.pathways:
            raise ValueError(f"Database '{database}' not found")

        logger.info(f"Initialized enrichment analyzer for {species} using {database}")

    def run_enrichment(self, gene_list: List[str], background_size: int = 20000,
                      method: str = 'fisher', correction: str = 'fdr_bh') -> pd.DataFrame:
        """
        Run pathway enrichment analysis.

        Args:
            gene_list: List of genes to test for enrichment
            background_size: Total number of genes in background
            method: Statistical test ('fisher', 'hypergeometric')
            correction: Multiple testing correction method

        Returns:
            DataFrame with enrichment results
        """
        # Normalize gene list
        gene_set = set(g.strip().upper() for g in gene_list if g.strip())

        if not gene_set:
            return pd.DataFrame()

        results = []

        for pathway_name, pathway_genes in self.pathways.items():
            pathway_set = set(g.upper() for g in pathway_genes)

            # Calculate contingency table
            overlap = gene_set.intersection(pathway_set)
            overlap_count = len(overlap)

            if overlap_count == 0:
                continue

            # Fisher exact test or hypergeometric test
            if method == 'fisher':
                # Contingency table: [[in_pathway_and_selected, in_pathway_not_selected],
                #                     [not_in_pathway_selected, not_in_pathway_not_selected]]
                table = [
                    [overlap_count, len(pathway_set) - overlap_count],
                    [len(gene_set) - overlap_count, background_size - len(pathway_set) - (len(gene_set) - overlap_count)]
                ]
                odds_ratio, p_value = fisher_exact(table, alternative='greater')
            elif method == 'hypergeometric':
                # Hypergeometric test
                M = background_size  # total genes
                n = len(pathway_set)  # genes in pathway
                N = len(gene_set)    # selected genes
                k = overlap_count    # overlap

                p_value = hypergeom.sf(k - 1, M, n, N)  # survival function for enrichment
                odds_ratio = (k / N) / (n / M) if N > 0 and n > 0 else 1
            else:
                raise ValueError(f"Unknown method: {method}")

            results.append({
                'pathway': pathway_name,
                'database': self.database_name,
                'p_value': p_value,
                'adjusted_p_value': p_value,  # Will be corrected below
                'odds_ratio': odds_ratio,
                'overlap_count': overlap_count,
                'pathway_size': len(pathway_set),
                'query_size': len(gene_set),
                'background_size': background_size,
                'genes': ';'.join(sorted(list(overlap))),
                'enrichment_score': -np.log10(p_value) if p_value > 0 else 300
            })

        if not results:
            return pd.DataFrame()

        results_df = pd.DataFrame(results)

        # Apply multiple testing correction
        if len(results_df) > 1:
            _, adj_p = multipletests(results_df['p_value'], method=correction)[:2]
            results_df['adjusted_p_value'] = adj_p

        # Sort by adjusted p-value
        results_df = results_df.sort_values('adjusted_p_value')

        return results_df

    def get_enriched_pathways(self, gene_list: List[str], padj_threshold: float = 0.05,
                            top_n: Optional[int] = None) -> pd.DataFrame:
        """Get significantly enriched pathways."""
        results = self.run_enrichment(gene_list)

        if results.empty:
            return results

        # Filter by significance
        significant = results[results['adjusted_p_value'] <= padj_threshold]

        # Limit to top N if specified
        if top_n and len(significant) > top_n:
            significant = significant.head(top_n)

        return significant

    def get_pathway_genes(self, pathway_name: str) -> List[str]:
        """Get genes in a specific pathway."""
        return self.pathways.get(pathway_name, [])


class MultiDatabaseEnrichment:
    """Run enrichment analysis across multiple databases."""

    def __init__(self, databases: List[str] = None, species: str = 'human'):
        if databases is None:
            databases = ['kegg', 'go_bp', 'reactome']

        self.analyzers = {}
        for db in databases:
            try:
                self.analyzers[db] = EnrichmentAnalyzer(db, species)
            except ValueError:
                logger.warning(f"Could not load database: {db}")

    def run_multi_enrichment(self, gene_list: List[str], padj_threshold: float = 0.05,
                           combine_results: bool = True) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Run enrichment across multiple databases."""
        results = {}

        for db_name, analyzer in self.analyzers.items():
            db_results = analyzer.get_enriched_pathways(gene_list, padj_threshold)
            results[db_name] = db_results

        if combine_results:
            # Combine all results
            combined = pd.concat(results.values(), ignore_index=True)
            combined = combined.sort_values('adjusted_p_value')
            return combined

        return results

    def get_database_info(self) -> Dict[str, int]:
        """Get information about loaded databases."""
        info = {}
        for db_name, analyzer in self.analyzers.items():
            info[db_name] = len(analyzer.pathways)
        return info


# Utility functions for pathway analysis
def calculate_enrichment_score(p_value: float, odds_ratio: float) -> float:
    """Calculate combined enrichment score."""
    if p_value <= 0:
        return 300  # Cap at very significant
    return -np.log10(p_value) * np.log2(odds_ratio + 1)


def filter_pathways_by_size(results: pd.DataFrame, min_size: int = 5, max_size: int = 500) -> pd.DataFrame:
    """Filter pathways by size."""
    return results[(results['pathway_size'] >= min_size) & (results['pathway_size'] <= max_size)]


def get_leading_edge_genes(pathway_results: pd.DataFrame, deg_results: pd.DataFrame,
                          pathway_name: str) -> List[str]:
    """Get leading edge genes for a pathway from DEG results."""
    if pathway_name not in pathway_results['pathway'].values:
        return []

    pathway_row = pathway_results[pathway_results['pathway'] == pathway_name].iloc[0]
    genes_val = pathway_row.get('genes') if hasattr(pathway_row, 'get') else pathway_row['genes']
    if not genes_val or pd.isna(genes_val):
        return []
    pathway_genes = set(str(genes_val).split(';'))

    # Get DEG genes that are in the pathway
    deg_pathway = deg_results[deg_results['gene_symbol'].str.upper().isin(pathway_genes)]

    # Sort by significance and return top genes
    leading_edge = deg_pathway.nsmallest(10, 'adjusted_p_value')['gene_symbol'].tolist()

    return leading_edge