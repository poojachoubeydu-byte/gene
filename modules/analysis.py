"""
Analysis Module — Apex Bioinformatics Platform
================================================
All heavy computation lives here. Every function is pure (no shared mutable
state) to be safe for concurrent Gunicorn workers.
"""

import hashlib
import logging
import uuid
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def df_fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of a DataFrame — used as cache key."""
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=False).values.tobytes()
    ).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Meta-Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_meta_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite priority score (0–100):
        50 % normalised −log10(padj)
        50 % normalised |log2FC|
    Genes with baseMean (expression level) get a small boost.
    """
    df = df.copy()
    sig  = -np.log10(df["padj"].clip(lower=1e-300))
    sig_n = _minmax(sig)
    lfc_n = _minmax(df["log2FC"].abs())

    score = (sig_n * 0.5 + lfc_n * 0.5) * 100

    # Optional baseMean boost (up to +10 points)
    if "baseMean" in df.columns:
        bm = pd.to_numeric(df["baseMean"], errors="coerce").fillna(0)
        bm_n = _minmax(np.log1p(bm))
        score = np.clip(score + bm_n * 10, 0, 100)

    df["meta_score"] = score.round(1)
    return df


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 3-D PCA + K-Means
# ─────────────────────────────────────────────────────────────────────────────

def run_pca_3d(df: pd.DataFrame, n_clusters: int = 4) -> dict:
    if len(df) < 6:
        return {"error": "Minimum 6 genes required for PCA."}

    df = df.copy()
    df["-log10p"] = -np.log10(df["padj"].clip(lower=1e-300))

    features = ["log2FC", "-log10p"]
    if "baseMean" in df.columns:
        df["log_bm"] = np.log1p(
            pd.to_numeric(df["baseMean"], errors="coerce").fillna(0)
        )
        features.append("log_bm")

    X = df[features].fillna(0).values
    X_sc = StandardScaler().fit_transform(X)

    n_comp = min(3, len(features), X_sc.shape[0])
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(X_sc)
    if coords.shape[1] < 3:
        coords = np.hstack([coords, np.zeros((len(coords), 3 - coords.shape[1]))])

    n_clust = min(n_clusters, max(2, len(df) // 3))
    labels = KMeans(n_clusters=n_clust, random_state=42, n_init=10).fit_predict(coords)

    var = ([round(float(v) * 100, 1) for v in pca.explained_variance_ratio_]
           + [0.0] * (3 - n_comp))

    return {
        "pc1": coords[:, 0].tolist(),
        "pc2": coords[:, 1].tolist(),
        "pc3": coords[:, 2].tolist(),
        "clusters": labels.tolist(),
        "genes": df["symbol"].tolist(),
        "lfc": df["log2FC"].tolist(),
        "explained_variance": var,
        "n_clusters": n_clust,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WGCNA-lite  (correlation-based co-expression network)
# ─────────────────────────────────────────────────────────────────────────────

def run_wgcna_lite(df: pd.DataFrame, n_clusters: int = 5, max_genes: int = 120) -> dict:
    import networkx as nx

    _empty = {
        "modules": {}, "graph": nx.Graph(),
        "labels": {}, "n_modules": 0, "n_edges": 0, "error": None,
    }

    if len(df) < 6:
        _empty["error"] = "Minimum 6 genes required for network."
        return _empty

    df = df.copy()
    df["-log10p"] = -np.log10(df["padj"].clip(lower=1e-300))
    df["net_score"] = (df["log2FC"].abs() * df["-log10p"]).fillna(0)
    df = df.nlargest(min(max_genes, len(df)), "net_score").reset_index(drop=True)

    X = df[["log2FC", "-log10p"]].fillna(0).values
    corr = np.corrcoef(X)

    link = linkage(X, method="ward")
    n_clust = min(n_clusters, max(2, len(df) // 3))
    labels = fcluster(link, n_clust, criterion="maxclust")

    genes = df["symbol"].tolist()
    G = nx.Graph()
    G.add_nodes_from(genes)

    thr = 0.90 if len(genes) > 50 else 0.70
    edges = [
        (genes[i], genes[j], float(corr[i, j]))
        for i in range(len(genes))
        for j in range(i + 1, len(genes))
        if abs(corr[i, j]) > thr
    ]
    edges.sort(key=lambda x: abs(x[2]), reverse=True)
    for s, t, w in edges[:3000]:
        G.add_edge(s, t, weight=w)

    modules = {}
    for gene, lbl in zip(genes, labels):
        modules.setdefault(int(lbl), []).append(gene)

    return {
        "modules": modules,
        "graph": G,
        "labels": dict(zip(genes, labels.tolist())),
        "n_modules": n_clust,
        "n_edges": G.number_of_edges(),
        "lfc_values": dict(zip(df["symbol"], df["log2FC"])),
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GSEA Pre-ranked  (per-call temp dir so workers don't collide)
# ─────────────────────────────────────────────────────────────────────────────

def run_gsea_preranked(df: pd.DataFrame) -> dict:
    if len(df) < 15:
        return {"error": "GSEA requires ≥ 15 genes.", "results": pd.DataFrame()}

    try:
        import gseapy as gp

        rank = (
            df.copy()
            .assign(
                rank_metric=lambda d: np.sign(d["log2FC"])
                * -np.log10(d.get("pvalue", d["padj"]).clip(lower=1e-300))
            )[["symbol", "rank_metric"]]
            .dropna()
            .drop_duplicates("symbol")
            .set_index("symbol")["rank_metric"]
            .sort_values(ascending=False)
        )

        # Unique temp directory per call — avoids worker clashes
        tmp = tempfile.mkdtemp(prefix=f"gsea_{uuid.uuid4().hex[:8]}_")
        try:
            pre = gp.prerank(
                rnk=rank,
                gene_sets=["KEGG_2021_Human", "Reactome_2022"],
                outdir=tmp,
                min_size=5,
                max_size=500,
                permutation_num=100,
                seed=42,
                verbose=False,
            )
            res = pre.res2d.copy()
            res = res[res["FDR q-val"] < 0.25].copy()
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

        return {"results": res, "error": None}

    except Exception as e:
        log.warning(f"GSEA: {e}")
        return {"error": str(e), "results": pd.DataFrame()}


# ─────────────────────────────────────────────────────────────────────────────
# Biomarker Score  (extended Meta-Score with clinical annotation bonus)
# ─────────────────────────────────────────────────────────────────────────────

def compute_biomarker_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extended score that adds a clinical bonus for genes with:
      • An FDA-approved drug target  (+15 pts)
      • COSMIC Tier-1 cancer gene    (+10 pts)
      • Established biomarker        (+5 pts)

    Returns df with columns: symbol, log2FC, padj, meta_score, biomarker_score,
                              is_drug_target, cancer_role, biomarker_type
    """
    from data.gene_annotations import (
        DRUG_GENE_INTERACTIONS,
        CANCER_GENE_CENSUS,
        ESTABLISHED_BIOMARKERS,
    )

    df = compute_meta_score(df).copy()

    bonus       = pd.Series(0.0, index=df.index)
    drug_flag   = []
    cancer_role = []
    bm_type     = []

    for _, row in df.iterrows():
        sym = str(row["symbol"]).strip().upper()

        # Drug target bonus
        drugs = DRUG_GENE_INTERACTIONS.get(sym, [])
        fda_drugs = [d for d in drugs if d.get("fda")]
        if fda_drugs:
            bonus[_] += 15
            drug_flag.append(f"{len(fda_drugs)} drug(s)")
        else:
            drug_flag.append("")

        # Cancer gene bonus
        cgc = CANCER_GENE_CENSUS.get(sym)
        if cgc:
            bonus[_] += 10 if cgc["tier"] == 1 else 5
            cancer_role.append(cgc["role"])
        else:
            cancer_role.append("")

        # Known biomarker bonus
        bm = ESTABLISHED_BIOMARKERS.get(sym)
        if bm:
            bonus[_] += 5
            bm_type.append(bm["type"])
        else:
            bm_type.append("")

    df["biomarker_score"] = np.clip(df["meta_score"] + bonus, 0, 100).round(1)
    df["is_drug_target"]  = drug_flag
    df["cancer_role"]     = cancer_role
    df["biomarker_type"]  = bm_type
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Pathway Crosstalk
# ─────────────────────────────────────────────────────────────────────────────

def compute_pathway_crosstalk(enr_df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """
    Given an enrichment result DataFrame (must have 'pathway' and 'genes' columns),
    compute a Jaccard-similarity matrix between the top pathways' gene sets.
    Returns a square DataFrame suitable for heatmap plotting.
    """
    if enr_df is None or enr_df.empty or "genes" not in enr_df.columns:
        return pd.DataFrame()

    top = enr_df.head(top_n).copy()
    top = top[top["genes"].notna() & (top["genes"] != "")]
    if top.empty:
        return pd.DataFrame()

    labels = top["pathway"].str[:40].tolist()
    gene_sets = [
        set(str(g).upper() for g in row["genes"].split(";") if g.strip())
        for _, row in top.iterrows()
    ]

    n = len(labels)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            u = gene_sets[i] | gene_sets[j]
            inter = gene_sets[i] & gene_sets[j]
            mat[i, j] = len(inter) / len(u) if u else 0.0

    return pd.DataFrame(mat, index=labels, columns=labels)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-Insights generator  (rule-based, offline)
# ─────────────────────────────────────────────────────────────────────────────

def generate_insights(df: pd.DataFrame, enr_df: pd.DataFrame,
                      lfc_t: float = 1.0, p_t: float = 0.05) -> list[dict]:
    """
    Returns a list of insight dicts:
        { 'category': str, 'icon': str, 'title': str, 'body': str, 'severity': str }
    severity: 'primary' | 'success' | 'warning' | 'info' | 'danger'
    """
    from data.gene_annotations import (
        DRUG_GENE_INTERACTIONS,
        CANCER_GENE_CENSUS,
        ESTABLISHED_BIOMARKERS,
    )

    insights = []

    # ── 1. Dataset overview ──────────────────────────────────────────────────
    n_total = len(df)
    up   = df[(df["log2FC"] >  lfc_t) & (df["padj"] < p_t)]
    dn   = df[(df["log2FC"] < -lfc_t) & (df["padj"] < p_t)]
    n_up, n_dn = len(up), len(dn)
    bias = "up-regulated" if n_up > n_dn else ("down-regulated" if n_dn > n_up else "balanced")
    insights.append({
        "category": "Overview",
        "icon": "📊",
        "title": f"{n_up + n_dn} significant genes ({n_up} ↑ / {n_dn} ↓)",
        "body": (
            f"Out of {n_total} genes tested, {n_up + n_dn} pass the "
            f"|Log2FC| ≥ {lfc_t} & adj. p < {p_t} thresholds. "
            f"The expression profile is predominantly <strong>{bias}</strong>."
        ),
        "severity": "primary",
    })

    # ── 2. Top drivers ───────────────────────────────────────────────────────
    sig = pd.concat([up, dn]).copy()
    sig["_nlp"] = -np.log10(sig["padj"].clip(lower=1e-300))
    top3 = sig.nlargest(3, "_nlp")["symbol"].tolist()
    if top3:
        insights.append({
            "category": "Top Drivers",
            "icon": "🔬",
            "title": f"Most significant: {', '.join(top3)}",
            "body": (
                "These genes show the strongest statistical evidence of differential "
                "expression. They represent the highest-priority candidates for "
                "functional validation."
            ),
            "severity": "info",
        })

    # ── 3. Drug targets ──────────────────────────────────────────────────────
    all_sig = sig["symbol"].str.upper().tolist()
    druggable = [g for g in all_sig if g in DRUG_GENE_INTERACTIONS
                 and any(d["fda"] for d in DRUG_GENE_INTERACTIONS[g])]
    if druggable:
        example = druggable[0]
        ex_drugs = [d["drug"] for d in DRUG_GENE_INTERACTIONS[example] if d["fda"]][:2]
        insights.append({
            "category": "Drug Targets",
            "icon": "💊",
            "title": f"{len(druggable)} significant genes are FDA-approved drug targets",
            "body": (
                f"Druggable genes: <strong>{', '.join(druggable[:6])}</strong>"
                + (f" (+{len(druggable)-6} more)" if len(druggable) > 6 else "")
                + f".<br>Example: <em>{example}</em> is targeted by "
                f"<strong>{', '.join(ex_drugs)}</strong>. "
                "See the Drug Targets tab for full details."
            ),
            "severity": "success",
        })

    # ── 4. Oncogene / TSG balance ────────────────────────────────────────────
    up_onco  = [g for g in up["symbol"].str.upper() if CANCER_GENE_CENSUS.get(g, {}).get("role") == "Oncogene"]
    dn_tsg   = [g for g in dn["symbol"].str.upper() if CANCER_GENE_CENSUS.get(g, {}).get("role") == "TSG"]
    if up_onco or dn_tsg:
        lines = []
        if up_onco:
            lines.append(f"<strong>Up-regulated oncogenes:</strong> {', '.join(up_onco[:5])}")
        if dn_tsg:
            lines.append(f"<strong>Down-regulated TSGs:</strong> {', '.join(dn_tsg[:5])}")
        insights.append({
            "category": "Cancer Biology",
            "icon": "🧬",
            "title": f"Cancer gene pattern: {len(up_onco)} oncogenes ↑ / {len(dn_tsg)} TSGs ↓",
            "body": (
                " | ".join(lines)
                + "<br>This co-occurring pattern (oncogene activation + TSG suppression) "
                "is a hallmark of tumour progression and may indicate co-operative oncogenic signalling."
            ),
            "severity": "danger" if (up_onco and dn_tsg) else "warning",
        })

    # ── 5. Pathway themes ────────────────────────────────────────────────────
    if enr_df is not None and not enr_df.empty:
        top_pw = enr_df.head(5)["pathway"].tolist()
        db_cov = enr_df["database"].nunique() if "database" in enr_df.columns else 1
        insights.append({
            "category": "Pathway Enrichment",
            "icon": "🗂️",
            "title": f"{len(enr_df)} enriched pathways across {db_cov} database(s)",
            "body": (
                f"Top pathways: <strong>{' • '.join(top_pw[:3])}</strong>.<br>"
                "Check the Volcano & Pathway tab for bubble charts and full tables. "
                "Use lasso selection on the volcano plot to enrich any gene subset interactively."
            ),
            "severity": "info",
        })

    # ── 6. Established biomarkers ────────────────────────────────────────────
    bm_genes = [g for g in all_sig if g in ESTABLISHED_BIOMARKERS]
    if bm_genes:
        bm_info = ESTABLISHED_BIOMARKERS[bm_genes[0]]
        insights.append({
            "category": "Clinical Biomarkers",
            "icon": "🏥",
            "title": f"{len(bm_genes)} established clinical biomarker(s) among significant genes",
            "body": (
                f"Genes: <strong>{', '.join(bm_genes[:5])}</strong>.<br>"
                f"Example — <em>{bm_genes[0]}</em>: {bm_info['type']} biomarker "
                f"({bm_info['context']}, test: {bm_info['test']})."
            ),
            "severity": "warning",
        })

    # ── 7. Immune context ────────────────────────────────────────────────────
    immune_genes = {"PDCD1", "CD274", "CTLA4", "LAG3", "TIGIT", "HAVCR2",
                    "IFNG", "CD8A", "CD4", "FOXP3", "IL2", "IL10", "TGFB1"}
    sig_immune = [g for g in all_sig if g in immune_genes]
    if sig_immune:
        insights.append({
            "category": "Immune Context",
            "icon": "🛡️",
            "title": f"Immune checkpoint / activation genes detected: {', '.join(sig_immune[:4])}",
            "body": (
                "Differential expression of immune checkpoint genes suggests "
                "potential relevance to immunotherapy response or tumour microenvironment remodelling. "
                "Consider checkpoint inhibitor therapy context."
            ),
            "severity": "info",
        })

    # ── 8. Low gene count warning ────────────────────────────────────────────
    if n_up + n_dn < 5:
        insights.append({
            "category": "QC Warning",
            "icon": "⚠️",
            "title": "Very few significant genes — consider relaxing thresholds",
            "body": (
                f"Only {n_up + n_dn} genes pass current thresholds "
                f"(|Log2FC| ≥ {lfc_t}, p ≤ {p_t}). "
                "Try reducing the |Log2FC| cutoff to 0.5 or increasing the p-value cutoff to 0.1."
            ),
            "severity": "warning",
        })

    return insights
