"""
Pathway Enrichment Module — Apex Bioinformatics Platform
=========================================================
Provides offline-capable enrichment via built-in gene sets plus
Fisher-exact testing with FDR correction.

Statistical upgrades (v3.0):
  • Odds ratio + 95 % CI  (Wald method) for every pathway row
  • Gene ratio  (overlap / query size)  — like clusterProfiler output
  • Activation Z-score  (IPA-style directional scoring per pathway)
  • Effect-weighted enrichment score (Σ |LFC| × −log10p for overlap genes)
  • Exposed background size in public API (default 20 000)
  • Exposed padj threshold in public API (default 0.05 for strict mode)
  • Public helpers: list_databases(), get_pathway_genes()
"""

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Built-in gene-set database
# ─────────────────────────────────────────────────────────────────────────────

_KEGG: Dict[str, List[str]] = {
    "Cell Cycle":             ["CDK1","CCNB1","TOP2A","MKI67","CDK2","CCNA2","CCNE1","RB1","E2F1","CDK4","CDK6","CCND1","CDC20","BUB1","PLK1","AURKA","AURKB"],
    "Apoptosis":              ["BAX","CASP3","BCL2","TP53","FAS","TNF","TRADD","FADD","CASP8","CASP9","PARP1","CYCS","BID","MCL1","BCL2L1","XIAP","BIRC5"],
    "DNA Repair":             ["BRCA1","BRCA2","RAD51","PARP1","ATM","ATR","CHEK1","CHEK2","XRCC1","MLH1","MSH2","ERCC1","FANCD2","PALB2","NBN","MRE11"],
    "p53 Signaling":          ["TP53","CDKN1A","MDM2","BAX","BBC3","CDK2","CCNE1","RB1","E2F1","GADD45A","PUMA","PERP","SESN1","SESN2","TIGAR","DDB2"],
    "TGF-β Signaling":        ["TGFB1","SMAD2","SMAD3","SMAD4","MYC","JUN","FOS","TGFBR1","TGFBR2","ACVR1","ACVR2A","ID1","ID2","ID3","BAMBI","SMURF1"],
    "Wnt Signaling":          ["CTNNB1","APC","GSK3B","AXIN1","TCF7L2","LEF1","MYC","WNT1","FZD1","DVL1","LRP5","LRP6","CCND1","MMP7","CD44","AXIN2"],
    "MAPK Signaling":         ["MAPK1","MAPK3","RAF1","BRAF","HRAS","KRAS","NRAS","MAP2K1","MAP2K2","MAP3K1","DUSP1","DUSP6","EGR1","MYC","JUN","FOS"],
    "PI3K-AKT Signaling":     ["PIK3CA","PIK3CB","AKT1","AKT2","AKT3","PTEN","MTOR","RHEB","PDK1","TSC1","TSC2","RPS6KB1","EIF4EBP1","GSK3B","FOXO1","MDM2"],
    "Hedgehog Signaling":     ["SHH","PTCH1","SMO","GLI1","GLI2","GLI3","SUFU","HHIP","DISP1","BOC","GAS1","IHH","DHH"],
    "Notch Signaling":        ["NOTCH1","NOTCH2","NOTCH3","NOTCH4","DLL1","DLL3","DLL4","JAG1","JAG2","RBPJ","HES1","HES5","HEY1","HEY2","MAML1","NICD"],
    "HIF-1 Signaling":        ["HIF1A","ARNT","VEGFA","EPO","SLC2A1","LDHA","ENO1","PGAM1","VHL","EGLN1","EGLN2","EGLN3","EP300","CREBBP","HSP90AA1"],
    "mTOR Signaling":         ["MTOR","RPTOR","RICTOR","AKT1","TSC1","TSC2","RHEB","RPS6KB1","EIF4EBP1","ULK1","ATG13","DEPTOR","MLST8","AKT1S1"],
    "JAK-STAT Signaling":     ["JAK1","JAK2","JAK3","TYK2","STAT1","STAT2","STAT3","STAT4","STAT5A","STAT5B","STAT6","SOCS1","SOCS3","IFNG","IL6","IL2"],
    "NF-κB Signaling":        ["NFKB1","RELA","NFKB2","RELB","REL","IKBKA","IKBKB","NFKBIA","NFKBIB","TNFRSF1A","IL1B","TNF","LTA","TRADD","TRAF2"],
    "Autophagy":              ["ULK1","ATG1","ATG5","ATG7","ATG12","ATG16L1","BECN1","PIK3C3","AMBRA1","RUBCN","SQSTM1","MAP1LC3A","LAMP1","RAB7A","TFEB"],
    "EMT / Invasion":         ["CDH1","CDH2","VIM","FN1","TWIST1","SNAI1","SNAI2","ZEB1","ZEB2","MMP2","MMP9","TGFB1","WNT5A","AXL","FOXC2","ITGB6"],
    "VEGF Signaling":         ["VEGFA","VEGFB","VEGFC","VEGFD","KDR","FLT1","FLT4","NRP1","NRP2","PDGFRA","PDGFRB","ANGPT1","ANGPT2","TIE1","TEK"],
    "DNA Damage Response":    ["ATM","ATR","CHEK1","CHEK2","TP53","BRCA1","H2AFX","MDC1","53BP1","RNF8","RNF168","PARP1","XRCC4","LIG4","NHEJ1"],
    "Oxidative Phosphorylation":["MT-CO1","MT-ND1","ATP5F1A","NDUFS1","SDHA","UQCRC1","COX4I1","ATP6V1A","CYCS","TFAM","NRF1","PPARGC1A"],
    "Glycolysis / Gluconeogenesis":["HK1","HK2","GPI","PFKL","PFKM","ALDOA","GAPDH","PGK1","PGAM1","ENO1","PKM","LDHA","PDK1","PDHB","PCK1","G6PC"],
    "Fatty Acid Metabolism":  ["FASN","ACACA","ACACB","CPT1A","CPT2","HADHA","ACADM","ACSL1","ACSL4","ELOVL1","FADS1","FADS2","SCD","PPARA","PPARG"],
    "Amino Acid Metabolism":  ["SLC7A5","ASNS","PHGDH","PSAT1","PSPH","GOT1","GOT2","GLUD1","GLS","GLS2","ARG1","ASS1","ODC1","SMOX","AMD1"],
    "Proteasome":             ["PSMA1","PSMB1","PSMB5","PSMC1","PSMD1","UBA1","UBB","UBC","UBE2C","CDC34","CUL1","SKP1","FBXW7","BTRC","KEAP1"],
    "Spliceosome":            ["SF3B1","SF3A1","U2AF1","U2AF2","SRSF1","SRSF2","HNRNPA1","HNRNPC","PTBP1","RBM10","DDX5","DHX9","PRPF8","SNRNP70"],
    "Ribosome":               ["RPS6","RPS14","RPS19","RPL5","RPL11","EIF4E","EIF4G1","EIF2AK3","EIF2S1","EIF3A","RPS6KA1","MTOR"],
}

_GO_BP: Dict[str, List[str]] = {
    "Cell Proliferation":          ["MYC","CCND1","CDK4","CDK6","RB1","E2F1","PCNA","EGFR","ERBB2","MKI67","TOP2A","BIRC5","CDK2","CDK1"],
    "Apoptotic Process":           ["BAX","CASP3","BCL2","TP53","FAS","TNF","CASP8","CASP9","BID","CYCS","APAF1","XIAP","MCL1","BCL2L1"],
    "Inflammatory Response":       ["TNF","IL6","IL1B","IL8","NFKB1","RELA","STAT3","JAK2","CXCL10","CCL2","PTGS2","MCP1","IRF1","IRF3"],
    "Immune Response":             ["IFNG","CD4","CD8A","CD28","CTLA4","PDCD1","LAG3","TIGIT","FOXP3","IL2","IL10","TGFB1","CD274","HAVCR2"],
    "DNA Repair":                  ["BRCA1","BRCA2","RAD51","PARP1","ATM","ATR","XRCC1","XRCC4","MLH1","MSH2","ERCC1","PALB2","NBN","FANCD2"],
    "Cell Cycle Process":          ["CDK1","CCNB1","TOP2A","MKI67","CDK2","CCNA2","RB1","PCNA","CHEK1","CHEK2","PLK1","AURKA","BUB1","MAD2L1"],
    "Transcription Regulation":    ["TP53","MYC","JUN","FOS","SP1","NFYA","CREB1","RELA","YAP1","TWIST1","SNAI1","ZEB1","E2F1","KLF4"],
    "Translation":                 ["EIF4E","EIF4G1","RPS6KB1","MTOR","EIF2AK3","PERK","RPS14","RPL5","EIF3A","EIF2S1","EIF4A1","EIF6"],
    "Angiogenesis":                ["VEGFA","KDR","FLT1","NRP1","ANGPT1","ANGPT2","TIE1","PDGFRA","PDGFRB","HIF1A","MMP2","MMP9","COL4A1","FGF2"],
    "Signal Transduction":         ["EGFR","ERBB2","SRC","ABL1","JAK2","STAT1","STAT3","AKT1","MAPK1","MAPK3","PIK3CA","MTOR","RAS","RAF1"],
    "Cell Migration":              ["MMP2","MMP9","CDH1","CDH2","VIM","FN1","ITGB1","ITGAV","RAC1","RHOA","CDC42","CXCR4","CXCL12","TWIST1"],
    "Oxidative Stress Response":   ["SOD1","SOD2","CAT","GPX1","GPX4","TXNRD1","HMOX1","NQO1","NFE2L2","KEAP1","PRDX1","PRDX3","MT1A","SQSTM1"],
    "Metabolic Process":           ["HK1","PFKL","ALDOA","GAPDH","PGK1","ENO1","LDHA","FASN","ACACA","G6PD","IDH1","IDH2","ACLY","SUCLA2"],
    "Epigenetic Regulation":       ["EZH2","DNMT3A","DNMT3B","TET1","TET2","KDM5C","KDM6A","HDAC1","HDAC2","EP300","CREBBP","SETD2","ARID1A","SMARCA4"],
    "p53 Pathway":                 ["TP53","CDKN1A","MDM2","BAX","GADD45A","PUMA","BBC3","SESN1","TIGAR","RRM2B","DDB2","XPC","POLK","PERP"],
    "Ubiquitin Proteasome":        ["UBA1","UBB","UBC","UBE2C","FBXW7","BTRC","CUL1","SKP1","PSMB5","PSMC1","MDM2","SIAH1","CHIP","USP7"],
    "Autophagy Regulation":        ["BECN1","ATG5","ATG7","ATG12","ULK1","SQSTM1","MAP1LC3A","TFEB","MTOR","AKT1","PIK3C3","AMBRA1","NBR1"],
    "Cell Senescence":             ["TP53","RB1","CDKN1A","CDKN2A","CDKN2B","IL6","IL8","MMP3","HMGA2","NF1","PTEN","LMNB1","SAT1"],
    "Chromatin Remodeling":        ["SMARCA4","SMARCB1","ARID1A","ARID2","PBRM1","BRD4","BRD2","MBD3","NuRD","CHD4","CHD7","RBBP4","SMARCC1"],
    "RNA Processing":              ["SF3B1","U2AF1","SRSF2","HNRNPA1","HNRNPC","PTBP1","RBM10","DICER1","DROSHA","DGCR8","AGO2","XRN1","DCP1A"],
}

_REACTOME: Dict[str, List[str]] = {
    "Cell Cycle":                  ["CDK1","CCNB1","TOP2A","MKI67","CDK2","CCNA2","RB1","E2F1","CDC20","BUB1","PLK1","AURKA","AURKB","MAD2L1"],
    "Apoptosis":                   ["BAX","CASP3","BCL2","TP53","FAS","TNF","TRADD","FADD","CASP8","CASP9","PARP1","CYCS","BID","XIAP","MCL1"],
    "DNA Repair":                  ["BRCA1","BRCA2","RAD51","PARP1","ATM","ATR","CHEK1","CHEK2","XRCC1","MLH1","MSH2","ERCC1","PALB2","FANCD2","NBN"],
    "Immune System":               ["IFNG","TNF","IL6","IL1B","NFKB1","RELA","STAT1","IRF3","CD274","PDCD1","CTLA4","LAG3","TIGIT","HAVCR2"],
    "Signal Transduction":         ["EGFR","ERBB2","SRC","ABL1","RAS","RAF1","MEK1","ERK1","AKT1","PIK3CA","MTOR","JAK1","JAK2","STAT3"],
    "Metabolism":                  ["HK1","PFKL","ALDOA","GAPDH","PGK1","ENO1","LDHA","FASN","ACACA","IDH1","IDH2","ACLY","SUCLA2","GLUD1"],
    "Gene Expression":             ["TP53","MYC","JUN","FOS","SP1","TBP","TFIIA","TFIIB","EP300","CREBBP","BRD4","YAP1","TAZ","WWTR1"],
    "Transcriptional Regulation by Small Molecules":
                                   ["RXRA","PPARA","PPARG","PPARD","RARA","RARB","NR3C1","NR3C2","ESR1","AR","THR1","VDR"],
    "Extracellular Matrix Organisation":
                                   ["COL1A1","COL3A1","COL4A1","FN1","VTN","LAMA1","LAMB1","LAMC1","ITGB1","ITGAV","MMP2","MMP9","TGFB1","CTGF"],
    "Developmental Biology":       ["WNT1","FZD1","CTNNB1","APC","GSK3B","AXIN1","SHH","PTCH1","SMO","NOTCH1","DLL1","JAG1","RBPJ","HES1"],
    "PI3K/AKT Activation":         ["PIK3CA","PIK3CB","AKT1","AKT2","PTEN","MTOR","PDK1","TSC1","TSC2","RHEB","RPS6KB1","EIF4EBP1","FOXO1","MDM2"],
    "VEGF Ligand-Receptor Interactions":
                                   ["VEGFA","VEGFB","VEGFC","KDR","FLT1","FLT4","NRP1","NRP2","PDGFRA","PDGFRB","ANGPT1","ANGPT2"],
    "MAPK Signaling":              ["MAPK1","MAPK3","MAP2K1","MAP2K2","RAF1","BRAF","KRAS","HRAS","NRAS","MAP3K1","DUSP1","EGR1","FOS","JUN"],
    "TGF-β Receptor Signaling":    ["TGFB1","TGFBR1","TGFBR2","SMAD2","SMAD3","SMAD4","ACVR1","BMPR1A","BMP4","NODAL","LEFTY1","SMURF1","SKI"],
    "Cytokine Signaling in Immune System":
                                   ["IL6","IL2","IL10","IL12A","IL15","IFNG","TNF","TGFB1","JAK1","JAK2","STAT1","STAT3","SOCS1","SOCS3"],
    "Chromatin Organization":      ["EZH2","DNMT3A","SETD2","KDM6A","KDM5C","SMARCA4","ARID1A","EP300","HDAC1","HDAC2","BRD4","SIRT1","MBD3"],
    "RNA Metabolism":              ["SF3B1","U2AF1","SRSF2","HNRNPA1","DDX5","DHX9","DICER1","AGO2","XRN1","EXOSC3","DCP2","RNMT","PAPOLA"],
    "Programmed Cell Death":       ["BAX","CASP3","CASP8","CASP9","PARP1","BCL2","BCL2L1","MCL1","BID","CYCS","APAF1","XIAP","BIRC5","RIPK1","MLKL"],
    "p53-Dependent G1/S DNA damage checkpoint":
                                   ["TP53","CDKN1A","MDM2","CDK2","CCNE1","E2F1","RB1","GADD45A","CHEK1","CHEK2","ATM","PMAIP1","BBC3"],
    "ESR-mediated signalling":     ["ESR1","ESR2","SP1","ERBB2","SRC","PIK3CA","AKT1","MTOR","CCND1","CDK4","CDK6","RB1","E2F1","MYC"],
}

_ALL_DATABASES = {"kegg": _KEGG, "go_bp": _GO_BP, "reactome": _REACTOME}


def list_databases() -> List[str]:
    """Return available database names."""
    return list(_ALL_DATABASES.keys())


def get_pathway_genes(database: str, pathway: str) -> List[str]:
    """Return gene list for a specific pathway in a database."""
    return _ALL_DATABASES.get(database, {}).get(pathway, [])


# ─────────────────────────────────────────────────────────────────────────────
# Odds-ratio helper  (Wald 95 % CI, log-space)
# ─────────────────────────────────────────────────────────────────────────────

def _odds_ratio_ci(a: int, b: int, c: int, d: int, ci: float = 0.95):
    """
    Compute odds ratio and its Wald 95 % confidence interval.

    OR = (a*d) / (b*c)
    95 % CI: exp(log(OR) ± z × SE)
    SE = sqrt(1/a + 1/b + 1/c + 1/d)

    Returns (or_val, or_lo, or_hi) — all rounded to 3 dp.
    All zero-cell corrections: add 0.5 to each cell if any cell == 0.
    """
    # Haldane-Anscombe correction for zero cells
    if 0 in (a, b, c, d):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    or_val = (a * d) / (b * c)
    se     = math.sqrt(1/a + 1/b + 1/c + 1/d)
    z      = 1.959964  # z_{0.975} for 95 % CI
    lo     = math.exp(math.log(or_val) - z * se)
    hi     = math.exp(math.log(or_val) + z * se)
    return round(or_val, 3), round(lo, 3), round(hi, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Core enrichment engine  (v3.0 — with OR, CI, gene ratio, activation z-score)
# ─────────────────────────────────────────────────────────────────────────────

def _run_enrichment(
    gene_list: List[str],
    pathway_db: Dict[str, List[str]],
    db_name: str,
    background: int = 20_000,
    correction: str = "fdr_bh",
    gene_lfc_map: Optional[Dict[str, float]] = None,
    gene_nlp_map: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Core Fisher-exact pathway enrichment with extended output columns.

    New columns vs v2:
        odds_ratio          — OR from the 2×2 Fisher table
        or_ci_low           — Lower bound of Wald 95 % CI on OR
        or_ci_high          — Upper bound of Wald 95 % CI on OR
        gene_ratio          — overlap_count / query_size  (like clusterProfiler)
        activation_zscore   — (N_up − N_down) / √N_tot   (IPA-style)
        direction           — 'Activated' | 'Inhibited' | 'Inconclusive'
        effect_weighted_score — Σ |LFC| × −log10(padj) for overlap genes

    Args:
        gene_list      : Query gene symbols (upper-case).
        pathway_db     : {pathway_name: [gene_symbols]}.
        db_name        : Label for the 'database' column.
        background     : Estimated genome-wide gene count for Fisher table.
        correction     : Multiple-testing method (statsmodels multipletests).
        gene_lfc_map   : {GENE: log2FC} — enables directional scoring.
        gene_nlp_map   : {GENE: −log10(padj)} — enables EWS.
    """
    gene_set = {g.strip().upper() for g in gene_list if g.strip()}
    if not gene_set:
        return pd.DataFrame()

    lfc_map = gene_lfc_map or {}
    nlp_map = gene_nlp_map or {}

    rows = []
    for pw, pw_genes in pathway_db.items():
        pw_set  = {g.upper() for g in pw_genes}
        overlap = gene_set & pw_set
        n_over  = len(overlap)
        if n_over == 0:
            continue

        # 2×2 contingency table
        a = n_over
        b = len(pw_set) - n_over
        c = len(gene_set) - n_over
        d = max(0, background - a - b - c)

        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")

        # Odds ratio + Wald 95 % CI
        or_val, or_lo, or_hi = _odds_ratio_ci(a, b, c, max(1, d))

        # Gene ratio (clusterProfiler convention)
        gene_ratio = round(n_over / max(1, len(gene_set)), 4)

        # Activation z-score (IPA-style)
        overlap_genes = sorted(overlap)
        lfc_arr = np.array([lfc_map[g] for g in overlap_genes if g in lfc_map])
        if len(lfc_arr) > 0:
            n_up   = int((lfc_arr > 0).sum())
            n_dn   = int((lfc_arr < 0).sum())
            n_tot  = n_up + n_dn
            az     = round((n_up - n_dn) / math.sqrt(n_tot), 3) if n_tot > 0 else 0.0
            if az >= 2.0:
                direction = "Activated"
            elif az <= -2.0:
                direction = "Inhibited"
            else:
                direction = "Inconclusive"
            # Effect-weighted score
            ews = round(sum(
                abs(lfc_map.get(g, 0.0)) * nlp_map.get(g, 0.0)
                for g in overlap_genes if g in lfc_map
            ), 2)
        else:
            az, direction, ews = 0.0, "Inconclusive", 0.0

        rows.append({
            "pathway":               pw,
            "database":              db_name,
            "p_value":               float(p),
            "adjusted_p_value":      float(p),   # placeholder — corrected below
            "overlap_count":         n_over,
            "pathway_size":          len(pw_set),
            "query_size":            len(gene_set),
            "gene_ratio":            gene_ratio,
            "odds_ratio":            or_val,
            "or_ci_low":             or_lo,
            "or_ci_high":            or_hi,
            "enrichment_score":      (-np.log10(p) if p > 0 else 300.0),
            "activation_zscore":     az,
            "direction":             direction,
            "effect_weighted_score": ews,
            "genes":                 ";".join(overlap_genes),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if len(df) > 1:
        _, adj = multipletests(df["p_value"], method=correction)[:2]
        df["adjusted_p_value"] = adj
    df = df.sort_values("adjusted_p_value").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public API  (backward-compatible; enriches app.py imports unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class EnrichmentAnalyzer:
    """
    Single-database enrichment wrapper.

    v3.0 changes:
    • padj_threshold default tightened to 0.05 (was 0.5 — too permissive).
    • Accepts gene_lfc_map for activation z-score computation.
    • Exposes background parameter.
    """

    def __init__(self, database: str = "kegg", species: str = "human"):
        if database not in _ALL_DATABASES:
            raise ValueError(
                f"Database '{database}' not available. Choose from {list(_ALL_DATABASES)}"
            )
        self.database_name = database
        self.pathways      = _ALL_DATABASES[database]

    def run_enrichment(
        self,
        gene_list: List[str],
        padj_threshold: float = 0.05,
        background: int = 20_000,
        strict: bool = True,
        gene_lfc_map: Optional[Dict[str, float]] = None,
        gene_nlp_map: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        # Normalise gene symbols here so callers don't need to
        gene_list = [g.strip().upper() for g in gene_list if g and g.strip()]
        df = _run_enrichment(
            gene_list, self.pathways, self.database_name,
            background=background,
            gene_lfc_map=gene_lfc_map,
            gene_nlp_map=gene_nlp_map,
        )
        if df.empty:
            return df
        # Strict mode: filter on BH-adjusted p-value (robust, conservative)
        # Relaxed mode: filter on nominal p-value (more results, exploratory)
        filter_col = "adjusted_p_value" if strict else "p_value"
        return df[df[filter_col] <= padj_threshold].reset_index(drop=True)

    def get_enriched_pathways(
        self,
        gene_list: List[str],
        padj_threshold: float = 0.05,
        top_n: Optional[int] = None,
        background: int = 20_000,
        strict: bool = True,
        gene_lfc_map: Optional[Dict[str, float]] = None,
        gene_nlp_map: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        df = self.run_enrichment(
            gene_list, padj_threshold, background, strict, gene_lfc_map, gene_nlp_map
        )
        return df.head(top_n) if (top_n and not df.empty) else df


class MultiDatabaseEnrichment:
    """
    Multi-database enrichment aggregator.

    v3.0 changes:
    • padj_threshold default tightened to 0.05.
    • Forwards gene_lfc_map and gene_nlp_map to each sub-analyzer.
    • Exposes background parameter.
    """

    def __init__(self, databases: List[str] = None, species: str = "human"):
        dbs = databases or ["kegg", "go_bp", "reactome"]
        self.analyzers = {
            d: EnrichmentAnalyzer(d, species)
            for d in dbs if d in _ALL_DATABASES
        }

    def run_multi_enrichment(
        self,
        gene_list: List[str],
        padj_threshold: float = 0.05,
        background: int = 20_000,
        strict: bool = True,
        gene_lfc_map: Optional[Dict[str, float]] = None,
        gene_nlp_map: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        parts = [
            a.run_enrichment(
                gene_list, padj_threshold, background, strict, gene_lfc_map, gene_nlp_map
            )
            for a in self.analyzers.values()
        ]
        parts = [p for p in parts if not p.empty]
        if not parts:
            return pd.DataFrame()
        combined = pd.concat(parts, ignore_index=True).sort_values("adjusted_p_value")
        return combined.reset_index(drop=True)
