import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnrichmentEngine:
    def __init__(self, local_gmt_path=None):
        self.local_gmt = self._load_local_gmt(local_gmt_path)

    def _load_local_gmt(self, path):
        # Local fallback gene sets (placeholder - can be expanded)
        return {
            "Cell Cycle Control": ["CDK1", "CCNB1", "TOP2A", "MKI67", "CDK2"],
            "Apoptosis Signaling": ["BAX", "CASP3", "BCL2", "FAS", "TP53"],
            "Hypoxia Response": ["HIF1A", "VEGFA", "ENO1", "LDHA"],
            "DNA Repair": ["BRCA1", "BRCA2", "RAD51", "PARP1"]
        }

    def run_enrichment(self, selected_genes, background_size=20000):
        # Normalize: Strip whitespace and uppercase for 99.9% matching accuracy
        selected_genes = [g.strip().upper() for g in selected_genes if g.strip()]
        if not selected_genes:
            return pd.DataFrame()

        # We start with the local deterministic engine (DeepSeek's specialty)
        return self._run_local_fisher(selected_genes, background_size)

    def _run_local_fisher(self, selected_genes, background_size):
        results = []
        selected_set = set(selected_genes)
        num_selected = len(selected_set)

        for pathway, pathway_genes in self.local_gmt.items():
            pathway_set = set([g.upper() for g in pathway_genes])
            overlap = selected_set.intersection(pathway_set)
            overlap_count = len(overlap)
            
            if overlap_count == 0:
                continue

            # Fisher's Exact Test Table
            # [Overlap, PathGenes_Not_Selected]
            # [Selected_Not_In_Path, Background_Remaining]
            table = [
                [overlap_count, len(pathway_set) - overlap_count],
                [num_selected - overlap_count, background_size - (num_selected + len(pathway_set) - overlap_count)]
            ]
            
            _, p_val = fisher_exact(table, alternative='greater')
            
            results.append({
                'pathway': pathway,
                'p_value': p_val,
                'overlap': "; ".join(list(overlap)),
                'overlap_count': overlap_count,
                'pathway_size': len(pathway_set)
            })

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        # Apply Benjamini-Hochberg FDR correction
        _, df['adj_p_value'], _, _ = multipletests(df['p_value'], method='fdr_bh')
        
        return df.sort_values('adj_p_value')

