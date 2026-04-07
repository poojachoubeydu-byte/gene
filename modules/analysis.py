"""
Analysis Module — Apex Bioinformatics Platform
================================================
All heavy computation lives here. Every function is pure (no shared mutable
state) to be safe for concurrent Gunicorn workers.

Statistical upgrades (v3.0):
  • compute_meta_score  — 60/40 stat/LFC weighting + multiplicative expression
                          confidence factor (library-size aware, ECF)
  • compute_activation_zscore — IPA-style directional z-score per pathway
  • run_gsea_preranked  — 1 000 permutations; improved S2N rank metric;
                          proper p-value column priority
  • run_wgcna_lite      — cosine-similarity adjacency + soft-power (β=6)
                          + degree-centrality hub detection
  • filter_low_expression — CPM-proxy pre-filter via baseMean
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
from scipy.stats import norm as _scipy_norm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
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


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Expression Pre-Filter  (library-size aware CPM proxy)
# ─────────────────────────────────────────────────────────────────────────────

def filter_low_expression(df: pd.DataFrame, min_basemean: float = 10.0) -> pd.DataFrame:
    """
    Remove genes below a minimum expression threshold before scoring.

    baseMean ≥ 10 ≈ CPM ≥ 0.5 in a typical 20 M-read library (DESeq2 default
    independent-filtering threshold). Genes below this are statistically
    unreliable: their log2FC estimates are inflated by count noise.

    Args:
        df            : DEG DataFrame with optional 'baseMean' column.
        min_basemean  : Minimum normalised count threshold (default: 10).

    Returns:
        Filtered DataFrame (returns df unchanged if baseMean absent).
    """
    if "baseMean" not in df.columns:
        return df
    bm = pd.to_numeric(df["baseMean"], errors="coerce").fillna(0)
    return df[bm >= min_basemean].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Meta-Score  (industry-calibrated, expression-aware)
# ─────────────────────────────────────────────────────────────────────────────

def compute_meta_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite priority score (0–100):
        55 % normalised −log10(padj)   — statistical evidence
        35 % normalised |log2FC|        — biological effect size
        10 % expression confidence ECF  — library-size aware baseMean signal

    Industry rationale
    ------------------
    • Statistical evidence is weighted highest because the Wald/LRT p-value
      encodes both effect size AND experimental precision (replicate noise).
      This follows IPA's and DESeq2's philosophy: rank by Wald statistic.
    • ECF (expression confidence factor) is multiplicative and sigmoid-shaped:
        ECF(baseMean) = 0.80 + 0.20 × sigmoid(1.5 × (log10(baseMean) − 1))
      → ECF ≈ 0.83 at baseMean=1 (very low count — penalised)
      → ECF ≈ 0.90 at baseMean=10 (typical threshold)
      → ECF ≈ 0.96 at baseMean=100
      → ECF ≈ 1.00 at baseMean≥1 000
      This prevents low-count genes with inflated LFC from dominating rankings
      (mirrors DESeq2 independent filtering + apeglm LFC shrinkage).
    """
    df = df.copy()

    sig   = -np.log10(df["padj"].clip(lower=1e-300))
    sig_n = _minmax(sig)
    lfc_n = _minmax(df["log2FC"].abs())

    score = (sig_n * 0.55 + lfc_n * 0.35) * 100

    # Multiplicative Expression Confidence Factor
    if "baseMean" in df.columns:
        bm    = pd.to_numeric(df["baseMean"], errors="coerce").fillna(1).clip(lower=1)
        log_bm = np.log10(bm)
        ecf   = 0.80 + 0.20 / (1 + np.exp(-1.5 * (log_bm - 1)))
        score = np.clip(score * ecf, 0, 100)
    else:
        # Without baseMean: allocate the 10% expression slot to stat weight
        score = np.clip(score / 0.90, 0, 100)

    df["meta_score"] = score.round(1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Activation Z-Score  (IPA-style directional pathway scoring)
# ─────────────────────────────────────────────────────────────────────────────

def compute_activation_zscore(
    df: pd.DataFrame,
    enr_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute an IPA-style Activation Z-score for each enriched pathway.

    IPA formula (simplified for gene-list input):
        z = (N_up − N_down) / √(N_up + N_down)

    where N_up / N_down = number of overlap genes with positive / negative LFC.

    Interpretation (IPA thresholds):
        z ≥  2.0  → pathway is predicted ACTIVATED    (≥ 95 % confidence)
        z ≤ −2.0  → pathway is predicted INHIBITED
        |z| < 2.0 → INCONCLUSIVE  (insufficient directional consistency)

    Additionally computes:
        effect_weighted_score  = Σ |log2FC_i| × −log10(padj_i) for overlap genes
        This combines effect size and significance into a single pathway-level
        evidence score — equivalent to the enrichment NES concept in GSEA.

    Returns:
        enr_df with three new columns:
            activation_zscore, direction, effect_weighted_score
    """
    if enr_df is None or enr_df.empty or "genes" not in enr_df.columns:
        return enr_df

    ref = df.copy()
    ref["_sym"] = ref["symbol"].str.strip().str.upper()
    ref["_nlp"] = -np.log10(ref["padj"].clip(lower=1e-300))

    lfc_map = ref.set_index("_sym")["log2FC"].to_dict()
    nlp_map = ref.set_index("_sym")["_nlp"].to_dict()

    def _score_pathway(gene_str: str) -> tuple:
        genes    = [g.strip().upper() for g in str(gene_str).split(";") if g.strip()]
        lfc_vals = np.fromiter(
            (lfc_map[g] for g in genes if g in lfc_map), dtype=float
        )
        if len(lfc_vals) == 0:
            return np.nan, "Inconclusive", 0.0

        n_up  = int((lfc_vals > 0).sum())
        n_dn  = int((lfc_vals < 0).sum())
        n_tot = n_up + n_dn
        z     = (n_up - n_dn) / np.sqrt(n_tot) if n_tot > 0 else 0.0
        ews   = float(np.sum(np.abs(lfc_vals) * np.fromiter(
            (nlp_map.get(g, 0.0) for g in genes if g in lfc_map), dtype=float
        )))
        direction = (
            "Activated" if z >= 2.0 else
            "Inhibited" if z <= -2.0 else
            "Inconclusive"
        )
        return round(float(z), 3), direction, round(ews, 2)

    results = enr_df["genes"].map(_score_pathway)
    out = enr_df.copy()
    out["activation_zscore"]     = results.map(lambda t: t[0])
    out["direction"]             = results.map(lambda t: t[1])
    out["effect_weighted_score"] = results.map(lambda t: t[2])
    return out


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
    pca    = PCA(n_components=n_comp)
    coords = pca.fit_transform(X_sc)
    if coords.shape[1] < 3:
        coords = np.hstack([coords, np.zeros((len(coords), 3 - coords.shape[1]))])

    n_clust = min(n_clusters, max(2, len(df) // 3))
    labels  = KMeans(n_clusters=n_clust, random_state=42, n_init=10).fit_predict(coords)

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
# WGCNA-lite  (cosine-similarity + soft-power adjacency + hub detection)
# ─────────────────────────────────────────────────────────────────────────────

def run_wgcna_lite(df: pd.DataFrame, n_clusters: int = 5, max_genes: int = 120) -> dict:
    """
    Co-expression network using cosine-similarity adjacency with soft-power
    thresholding (WGCNA default β = 6).

    Improvements over v2:
    • Cosine similarity between [log2FC, −log10p] unit vectors captures
      directional co-regulation more faithfully than raw Pearson correlation
      on 2-point vectors.
    • Soft-power adjacency: adj(i,j) = ((1 + sim(i,j))/2)^6
      suppresses weak edges non-linearly, producing scale-free-like networks.
    • Hub detection: degree centrality identifies most-connected genes in
      each module — reported in net-info panel.
    • Adaptive edge-density guard: prune to top-3 000 edges after applying
      adjacency threshold to keep graph renderable.
    """
    import networkx as nx

    _empty = {
        "modules": {}, "graph": nx.Graph(),
        "labels": {}, "n_modules": 0, "n_edges": 0, "error": None,
        "hubs": {},
    }

    if len(df) < 6:
        _empty["error"] = "Minimum 6 genes required for network."
        return _empty

    df = df.copy()
    df["-log10p"] = -np.log10(df["padj"].clip(lower=1e-300))
    df["net_score"] = (df["log2FC"].abs() * df["-log10p"]).fillna(0)
    df = df.nlargest(min(max_genes, len(df)), "net_score").reset_index(drop=True)

    # ── Feature matrix: unit-normalised [log2FC, −log10p] ─────────────────
    X_raw = df[["log2FC", "-log10p"]].fillna(0).values.astype(float)
    norms = np.linalg.norm(X_raw, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1e-9, norms)
    X_unit = X_raw / norms  # unit cosine vectors

    # ── Cosine similarity + soft-power adjacency (β=6) ────────────────────
    sim = cosine_similarity(X_unit)       # ∈ [−1, +1]
    adj = ((1.0 + sim) / 2.0) ** 6       # ∈ [0, 1]; suppresses weak links

    # Adaptive adjacency threshold
    #   > 50 genes → use 0.50 (higher bar) ; ≤ 50 → 0.25 (lower bar)
    thr = 0.50 if len(df) > 50 else 0.25
    np.fill_diagonal(adj, 0)              # no self-loops

    # ── Hierarchical clustering for module assignment ──────────────────────
    link = linkage(X_unit, method="ward")
    n_clust = min(n_clusters, max(2, len(df) // 3))
    labels  = fcluster(link, n_clust, criterion="maxclust")

    # ── Build NetworkX graph ───────────────────────────────────────────────
    genes = df["symbol"].tolist()
    G = nx.Graph()
    G.add_nodes_from(genes)

    edges = [
        (genes[i], genes[j], float(adj[i, j]))
        for i in range(len(genes))
        for j in range(i + 1, len(genes))
        if adj[i, j] > thr
    ]
    edges.sort(key=lambda x: x[2], reverse=True)
    for s, t, w in edges[:3000]:
        G.add_edge(s, t, weight=w)

    # ── Hub detection: top-degree node per module ──────────────────────────
    modules: dict = {}
    for gene, lbl in zip(genes, labels):
        modules.setdefault(int(lbl), []).append(gene)

    degree_map = dict(G.degree())
    hubs: dict = {}
    for mod_id, mod_genes in modules.items():
        if mod_genes:
            hub = max(mod_genes, key=lambda g: degree_map.get(g, 0))
            hubs[mod_id] = hub

    return {
        "modules":    modules,
        "graph":      G,
        "labels":     dict(zip(genes, labels.tolist())),
        "n_modules":  n_clust,
        "n_edges":    G.number_of_edges(),
        "lfc_values": dict(zip(df["symbol"], df["log2FC"])),
        "hubs":       hubs,
        "error":      None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GSEA Pre-ranked  (1 000 permutations; improved rank metric; worker-safe)
# ─────────────────────────────────────────────────────────────────────────────

def run_gsea_preranked(df: pd.DataFrame) -> dict:
    """
    GSEA pre-ranked enrichment with industry-standard parameters.

    Rank metric: signal-to-noise proxy
        rank = sign(log2FC) × −log10(p_nominal)

    Uses nominal p-value when available (column 'pvalue') — it is more
    discriminating as a rank metric than the adjusted p-value, which is
    compressed by multiple-testing correction.  Falls back to padj.

    Permutations: 1 000 (GSEAv4 / Broad Institute standard).
    FDR display threshold: 0.25 (GSEA community convention).
    """
    if len(df) < 15:
        return {"error": "GSEA requires ≥ 15 genes.", "results": pd.DataFrame()}

    try:
        import gseapy as gp

        # Prefer nominal p-value for ranking; fall back to padj
        p_col = "pvalue" if "pvalue" in df.columns else "padj"

        _df = df.copy()
        _df["symbol"] = _df["symbol"].astype(str).str.upper().str.strip()
        rank = (
            _df
            .assign(
                rank_metric=lambda d: (
                    np.sign(d["log2FC"])
                    * -np.log10(d[p_col].clip(lower=1e-300))
                )
            )[["symbol", "rank_metric"]]
            .dropna()
            .drop_duplicates("symbol")
            .set_index("symbol")["rank_metric"]
            .sort_values(ascending=False)
        )

        tmp = tempfile.mkdtemp(prefix=f"gsea_{uuid.uuid4().hex[:8]}_")
        try:
            pre = gp.prerank(
                rnk=rank,
                gene_sets=["KEGG_2021_Human", "Reactome_2022"],
                outdir=tmp,
                min_size=5,
                max_size=500,
                permutation_num=1000,   # was 100 — increased to GSEAv4 standard
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
# Biomarker Score  (meta-score + clinical annotation bonus)
# ─────────────────────────────────────────────────────────────────────────────

def compute_biomarker_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extended score adding clinical bonuses:
      • FDA-approved drug target  (+15 pts)
      • COSMIC Tier-1 cancer gene (+10 pts)
      • Established biomarker     (+5 pts)

    Returns df with: symbol, log2FC, padj, meta_score, biomarker_score,
                     is_drug_target, cancer_role, biomarker_type
    """
    from data.gene_annotations import (
        DRUG_GENE_INTERACTIONS,
        CANCER_GENE_CENSUS,
        ESTABLISHED_BIOMARKERS,
    )

    df = compute_meta_score(df).copy()

    # Vectorised lookup — O(n) dict access, no Python-level row loop
    syms = df["symbol"].str.strip().str.upper()

    def _drug_flag(sym: str) -> str:
        drugs = DRUG_GENE_INTERACTIONS.get(sym, [])
        fda   = [d for d in drugs if d.get("fda")]
        return f"{len(fda)} drug(s)" if fda else ""

    def _drug_bonus(sym: str) -> float:
        drugs = DRUG_GENE_INTERACTIONS.get(sym, [])
        return 15.0 if any(d.get("fda") for d in drugs) else 0.0

    def _cancer_role(sym: str) -> str:
        cgc = CANCER_GENE_CENSUS.get(sym)
        return cgc["role"] if cgc else ""

    def _cancer_bonus(sym: str) -> float:
        cgc = CANCER_GENE_CENSUS.get(sym)
        if not cgc:
            return 0.0
        return 10.0 if cgc["tier"] == 1 else 5.0

    def _bm_type(sym: str) -> str:
        bm = ESTABLISHED_BIOMARKERS.get(sym)
        return bm["type"] if bm else ""

    def _bm_bonus(sym: str) -> float:
        return 5.0 if ESTABLISHED_BIOMARKERS.get(sym) else 0.0

    bonus = (
        syms.map(_drug_bonus)
        + syms.map(_cancer_bonus)
        + syms.map(_bm_bonus)
    )

    df["biomarker_score"] = np.clip(df["meta_score"] + bonus, 0, 100).round(1)
    df["is_drug_target"]  = syms.map(_drug_flag).values
    df["cancer_role"]     = syms.map(_cancer_role).values
    df["biomarker_type"]  = syms.map(_bm_type).values
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Pathway Crosstalk
# ─────────────────────────────────────────────────────────────────────────────

def compute_pathway_crosstalk(enr_df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """
    Jaccard similarity matrix between top pathway gene sets.
    Returns a square DataFrame for heatmap plotting.
    """
    if enr_df is None or enr_df.empty or "genes" not in enr_df.columns:
        return pd.DataFrame()

    top = enr_df.head(top_n).copy()
    top = top[top["genes"].notna() & (top["genes"] != "")]
    if top.empty:
        return pd.DataFrame()

    labels    = top["pathway"].str[:40].tolist()
    gene_sets = [
        set(str(g).upper() for g in row["genes"].split(";") if g.strip())
        for _, row in top.iterrows()
    ]

    n   = len(labels)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            u     = gene_sets[i] | gene_sets[j]
            inter = gene_sets[i] & gene_sets[j]
            mat[i, j] = len(inter) / len(u) if u else 0.0

    return pd.DataFrame(mat, index=labels, columns=labels)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-Insights generator  (rule-based, offline; v3.0 includes pathway activity)
# ─────────────────────────────────────────────────────────────────────────────

def generate_insights(df: pd.DataFrame, enr_df: pd.DataFrame,
                      lfc_t: float = 1.0, p_t: float = 0.05) -> list[dict]:
    """
    Returns a list of insight dicts:
        { 'category', 'icon', 'title', 'body', 'severity' }
    severity: 'primary' | 'success' | 'warning' | 'info' | 'danger'

    v3.0 addition: Pathway Activity insight using activation z-scores.
    """
    from data.gene_annotations import (
        DRUG_GENE_INTERACTIONS,
        CANCER_GENE_CENSUS,
        ESTABLISHED_BIOMARKERS,
    )

    insights = []

    # ── 1. Dataset overview ──────────────────────────────────────────────────
    n_total = len(df)
    up  = df[(df["log2FC"] >  lfc_t) & (df["padj"] < p_t)]
    dn  = df[(df["log2FC"] < -lfc_t) & (df["padj"] < p_t)]
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
        example  = druggable[0]
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
    up_onco = [g for g in up["symbol"].str.upper()
               if CANCER_GENE_CENSUS.get(g, {}).get("role") == "Oncogene"]
    dn_tsg  = [g for g in dn["symbol"].str.upper()
               if CANCER_GENE_CENSUS.get(g, {}).get("role") == "TSG"]
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

        # ── 5b. Pathway Activity (activation z-score) ────────────────────
        if "activation_zscore" in enr_df.columns and "direction" in enr_df.columns:
            activated = enr_df[enr_df["direction"] == "Activated"].head(3)
            inhibited = enr_df[enr_df["direction"] == "Inhibited"].head(3)
            if not activated.empty or not inhibited.empty:
                lines = []
                if not activated.empty:
                    act_names = activated["pathway"].tolist()
                    act_z     = activated["activation_zscore"].tolist()
                    act_str   = ", ".join(
                        f"<em>{p}</em> (z={z:+.2f})"
                        for p, z in zip(act_names, act_z)
                    )
                    lines.append(f"<strong>Activated:</strong> {act_str}")
                if not inhibited.empty:
                    inh_names = inhibited["pathway"].tolist()
                    inh_z     = inhibited["activation_zscore"].tolist()
                    inh_str   = ", ".join(
                        f"<em>{p}</em> (z={z:+.2f})"
                        for p, z in zip(inh_names, inh_z)
                    )
                    lines.append(f"<strong>Inhibited:</strong> {inh_str}")
                insights.append({
                    "category": "Pathway Activity",
                    "icon": "⚡",
                    "title": "Predicted pathway activation states (IPA-style z-scores)",
                    "body": (
                        "<br>".join(lines)
                        + "<br><small>|z| ≥ 2 indicates statistically confident "
                        "directional activity. Based on the ratio of up- vs "
                        "down-regulated genes within each enriched pathway.</small>"
                    ),
                    "severity": "warning",
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
