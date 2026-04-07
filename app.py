"""
Apex Bioinformatics Platform
=============================
Multi-user RNA-seq analysis dashboard.
Safe for ≥ 5 concurrent lab users (Gunicorn / Dash).

Unique features not available in other free tools:
  • Interactive lasso → real-time multi-DB pathway enrichment
  • Drug Target Landscape (FDA drug overlay on DE results)
  • Cancer Gene Census overlay on Volcano (oncogenes ★ / TSGs ◆)
  • Pathway Crosstalk heatmap (Jaccard gene overlap between pathways)
  • Biomarker Priority Score (clinical + statistical composite)
  • Auto-Insights — rule-based biological narrative
  • 3-D PCA with K-means clustering
  • WGCNA-lite co-expression network
  • GSEA pre-ranked (per-call temp dir — worker-safe)
  • PDF report with all panels
"""

import base64
import io
import logging
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dash_table, dcc, html

# ── local modules ─────────────────────────────────────────────────────────────
from modules.analysis import (
    compute_biomarker_score,
    compute_meta_score,
    compute_pathway_crosstalk,
    generate_insights,
    run_gsea_preranked,
    run_pca_3d,
    run_wgcna_lite,
)
from modules.bio_context import fetch_gene_info
from modules.pathway import EnrichmentAnalyzer, MultiDatabaseEnrichment
from modules.plots import (
    blank,
    create_biomarker_score_chart,
    create_drug_target_chart,
    create_drug_type_donut,
    create_lfc_dist,
    create_ma_plot,
    create_meta_score_bar,
    create_network_graph,
    create_pathway_bar,
    create_pathway_bubble,
    create_pathway_crosstalk,
    create_pca_3d,
    create_pval_hist,
    create_rank_metric,
    create_top_heatmap,
    create_volcano_plot,
)
from modules.reports import generate_pdf_report
from data.gene_annotations import get_drug_targets, get_cancer_gene_info

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# App initialisation
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
    ],
    suppress_callback_exceptions=True,
    title="Apex Bioinformatics",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server   # Gunicorn entry point

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEMO_CSV   = "sample_deg_data.csv"
RICH_CSV   = "data/demo_dge.csv"
HOUSEKEEP  = {"ACTB", "GAPDH", "B2M", "RPLP0", "HPRT1", "GUSB", "TBP"}

DB_OPTIONS = [
    {"label": "KEGG Pathways",         "value": "kegg"},
    {"label": "GO Biological Process", "value": "go_bp"},
    {"label": "Reactome",              "value": "reactome"},
    {"label": "All Databases",         "value": "all"},
]

# Module-level enrichment analyzer cache (per-worker, stateless analyzers)
# Keyed by db-name; thread-safe because analyzers have no mutable state.
_ENR_ANALYZERS: dict = {}


def _get_analyzer(db_choice: str):
    if db_choice not in _ENR_ANALYZERS:
        if db_choice == "all":
            _ENR_ANALYZERS["all"] = MultiDatabaseEnrichment(["kegg", "go_bp", "reactome"])
        else:
            _ENR_ANALYZERS[db_choice] = EnrichmentAnalyzer(db_choice)
    return _ENR_ANALYZERS[db_choice]


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    _SYMBOL_EXACT = {"gene_name","gene","genename","hgnc_symbol","symbol","id",
                     "gene_id","gene_symbol","geneid","name"}
    _LFC_EXACT    = {"log2foldchange","log2fc","lfc","foldchange","log2_fold_change",
                     "log2_fc","fold_change"}
    _PADJ_EXACT   = {"padj","adj.p.val","adj_p_value","fdr","p_adj","p.adj","bh",
                     "p_value_adj","adjusted_pvalue","adj_pval","adjusted_p_value",
                     "qvalue"}
    _PVAL_EXACT   = {"pvalue","p.value","p_value","pval","nominal_p"}
    _BASE_EXACT   = {"basemean","base_mean","mean_expr","avgexpr","averageexpression",
                     "mean_count"}

    # Pass 1: exact match
    rn = {}
    for c in df.columns:
        cl = c.lower().strip().replace(" ", "_")
        if   cl in _SYMBOL_EXACT and "symbol" not in rn.values(): rn[c] = "symbol"
        elif cl in _LFC_EXACT    and "log2FC" not in rn.values(): rn[c] = "log2FC"
        elif cl in _PADJ_EXACT   and "padj"   not in rn.values(): rn[c] = "padj"
        elif cl in _PVAL_EXACT   and "pvalue" not in rn.values(): rn[c] = "pvalue"
        elif cl in _BASE_EXACT   and "baseMean" not in rn.values(): rn[c] = "baseMean"

    # Pass 2: substring match for compound names like "diffexp_log2fc_NORMAL-vs-DISEASE"
    for c in df.columns:
        if c in rn:
            continue
        cl = c.lower().replace("-", "_")
        if "symbol" not in rn.values():
            if any(p in cl for p in ("gene_symbol","gene_name","hgnc")):
                rn[c] = "symbol"
        if "log2FC" not in rn.values():
            if any(p in cl for p in ("log2fc","log2_fc","log2fold")):
                rn[c] = "log2FC"
        if "padj" not in rn.values():
            if any(p in cl for p in ("qvalue","padj","fdr","adj_p","p_adj")):
                rn[c] = "padj"

    df = df.rename(columns=rn)
    if not {"symbol", "log2FC", "padj"}.issubset(df.columns):
        cols = df.columns.tolist()
        df   = df.rename(columns={cols[0]: "symbol", cols[1]: "log2FC", cols[2]: "padj"})

    # If the symbol column contains numeric values (e.g. Entrez IDs), find the
    # first object-dtype column with short string values and use that instead.
    _reserved = {"log2FC", "padj", "pvalue", "baseMean"}
    if not pd.api.types.is_object_dtype(df["symbol"]):
        for c in df.columns:
            if c not in _reserved and c != "symbol" and pd.api.types.is_object_dtype(df[c]):
                df = df.rename(columns={"symbol": "__entrez_id", c: "symbol"})
                break

    # Always cast symbol to string so .str accessor never fails downstream
    df["symbol"] = df["symbol"].astype(str).str.strip()

    df["log2FC"] = pd.to_numeric(df["log2FC"], errors="coerce")
    # DESeq2 / edgeR output regularly has NA padj for low-count genes (independent
    # filtering). Coerce those to 1.0 (non-significant) so NO genes are dropped.
    padj_raw = pd.to_numeric(df["padj"], errors="coerce")
    df["padj"] = padj_raw.fillna(1.0).clip(lower=1e-300, upper=1.0)
    # Only drop rows where log2FC is truly missing (all-zero-count genes, uninformative)
    return df.dropna(subset=["log2FC"]).reset_index(drop=True)


def _load_demo() -> pd.DataFrame:
    for path in (RICH_CSV, DEMO_CSV):
        try:
            return _normalise(pd.read_csv(path))
        except Exception:
            continue
    # Fallback: built-in 30-gene panel covering diverse biology
    return pd.DataFrame({
        "symbol": [
            "TNF","IL6","TP53","MYC","EGFR","BRCA1","PTEN","VEGFA",
            "CDK4","STAT3","CCND1","E2F1","RB1","HIF1A","MMP9",
            "CASP3","BAX","CDKN1A","VHL","ESR1","ERBB2","ALK","KRAS",
            "BCL2","MTOR","JAK2","BRAF","NOTCH1","IDH1","FLT3",
        ],
        "log2FC": [
            4.0, 3.8, 2.5, 3.1, 1.9, -2.8, -2.5, 1.2, 1.6, 1.7,
            1.5, 2.4, -3.1, 2.3, 3.5, -2.6, -3.0, 2.1, -3.4, -2.7,
            2.8, 1.3, 2.2, -1.8, 1.4, 2.0, 2.6, 1.1, 1.8, 2.3,
        ],
        "padj": [
            5e-6, 3e-4, 1e-5, 1.2e-7, 2e-3, 2e-4, 9e-4, 4e-2, 9e-3, 1e-3,
            8e-3, 4e-4, 7e-5, 7e-4, 8e-5, 6e-4, 1e-4, 5e-5, 3e-5, 3e-4,
            5e-7, 1e-2, 3e-4, 8e-3, 5e-3, 4e-4, 2e-3, 2e-2, 1e-2, 5e-4,
        ],
    })


def _s2df(data) -> pd.DataFrame:
    """Reconstruct DataFrame from dcc.Store records. Never silently falls back
    to demo data — returns an empty (but schema-valid) DataFrame so callers can
    detect the no-data state without being misled by 50-gene demo values."""
    if data:
        try:
            return pd.DataFrame(data)
        except Exception:
            pass
    return pd.DataFrame(columns=["symbol", "log2FC", "padj"])


def _run_enrichment(
    sig_genes: list,
    db_choice: str,
    gene_lfc_map: dict | None = None,
    gene_nlp_map: dict | None = None,
) -> pd.DataFrame:
    """
    Run pathway enrichment and attach activation z-scores + effect-weighted
    scores when LFC/NLP maps are supplied.

    padj_threshold: 0.05 (v3.0, tightened from 0.5) — returns only
    statistically reliable pathways by default.  Downstream views still
    show all returned rows; users see fewer but higher-confidence results.
    """
    if not sig_genes:
        return pd.DataFrame()
    try:
        analyzer  = _get_analyzer(db_choice)
        lfc_map   = gene_lfc_map or {}
        nlp_map   = gene_nlp_map or {}
        if db_choice == "all":
            return analyzer.run_multi_enrichment(
                sig_genes,
                padj_threshold=0.05,
                gene_lfc_map=lfc_map,
                gene_nlp_map=nlp_map,
            )
        return analyzer.run_enrichment(
            sig_genes,
            padj_threshold=0.05,
            gene_lfc_map=lfc_map,
            gene_nlp_map=nlp_map,
        )
    except Exception as e:
        log.warning(f"Enrichment ({db_choice}): {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────────────────────────────────────

def _section_head(icon: str, title: str, subtitle: str = "") -> html.Div:
    return html.Div([
        html.Span(icon + " ", style={"color": "#f39c12"}),
        html.Strong(title, style={"fontSize": 11, "color": "#1a5276"}),
        *([] if not subtitle else [
            html.Br(),
            html.Small(subtitle, className="text-muted"),
        ]),
    ], className="px-2 pt-2 pb-1 border-bottom mb-2")


def _enr_table(enr: pd.DataFrame) -> html.Div:
    if enr is None or enr.empty:
        return html.P("No significant pathways.", className="text-muted small ps-1")
    cols = [c for c in ["pathway", "database", "overlap_count", "pathway_size",
                         "adjusted_p_value", "genes"]
            if c in enr.columns]
    d = enr[cols].head(20).copy()
    d["adjusted_p_value"] = d["adjusted_p_value"].map("{:.2e}".format)
    return dash_table.DataTable(
        data=d.to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in cols],
        style_table={"overflowX": "auto", "maxHeight": "320px", "overflowY": "auto"},
        style_cell={"fontSize": 11, "padding": "3px 8px", "textAlign": "left",
                    "maxWidth": "200px", "overflow": "hidden", "textOverflow": "ellipsis"},
        style_header={"backgroundColor": "#1a5276", "color": "white",
                      "fontWeight": "bold", "fontSize": 11},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f4f6f7"}],
        page_size=10,
        tooltip_delay=0,
        tooltip_duration=None,
        tooltip_data=[
            {c: {"value": str(rec[c]), "type": "markdown"} for c in cols}
            for rec in d.to_dict("records")
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

SIDEBAR = dbc.Col([
    html.Div([
        html.Div("🧬", style={"fontSize": 36}),
        html.H5("Apex Bioinformatics", className="mb-0 text-white fw-bold"),
        html.Small("v3.2.5 (Platinum Performance)", className="text-white-50"),
    ], className="p-3 text-center",
       style={"background": "linear-gradient(135deg,#1a5276,#117a65)"}),

    html.Div([
        _section_head("⚡", "Data & Thresholds"),

        html.Label("📂 Data Source", className="fw-bold small text-muted mb-1"),
        dcc.Upload(
            id="upload-data",
            children=dbc.Button("⬆ Upload DGE CSV", color="primary",
                                outline=True, size="sm", className="w-100"),
            accept=".csv",
        ),
        html.Div(id="upload-status",
                 className="small text-muted mt-1 px-1",
                 children="📊 Demo: 30-gene panel loaded"),

        html.Hr(className="my-2"),
        html.Label("⚙️ Analysis Thresholds", className="fw-bold small text-muted"),
        html.Label("|Log2FC| cutoff", className="small"),
        dcc.Slider(id="lfc-thresh", min=0.5, max=3.0, step=0.1, value=1.0,
                   marks={0.5: "0.5", 1: "1", 2: "2", 3: "3"},
                   tooltip={"placement": "bottom", "always_visible": False}),
        html.Label("Adj. p-value cutoff", className="small"),
        dcc.Slider(id="p-thresh", min=0.001, max=0.1, step=0.001, value=0.05,
                   marks={0.01: "0.01", 0.05: "0.05", 0.1: "0.1"},
                   tooltip={"placement": "bottom", "always_visible": False}),

        html.Hr(className="my-2"),
        _section_head("🗃️", "Pathway Database"),
        dcc.RadioItems(
            id="pathway-db",
            options=DB_OPTIONS,
            value="all",
            labelStyle={"display": "block", "fontSize": 12, "marginBottom": 3},
            inputStyle={"marginRight": 5},
        ),

        html.Hr(className="my-2"),
        html.Label("🔍 Gene Bio-Context", className="fw-bold small text-muted"),
        dbc.Input(id="gene-input", placeholder="e.g. TNF  IL6  TP53",
                  type="text", size="sm", debounce=True, className="mb-1"),
        dcc.Loading(
            html.Div(id="gene-panel",
                     style={"fontSize": 11, "maxHeight": 220, "overflowY": "auto"}),
            type="dot",
        ),

        html.Hr(className="my-2"),
        dbc.Button("🔄 Refresh Current Tab", id="refresh-tab-btn",
                   color="warning", outline=True, size="sm", className="w-100 mb-2"),
        dbc.Tooltip("Re-run analysis for the active tab even if results are cached",
                    target="refresh-tab-btn", placement="right"),

        html.Hr(className="my-2"),
        dbc.Button("📥 Download PDF Report", id="btn-pdf",
                   color="success", size="sm", className="w-100 mb-2"),
        dcc.Download(id="dl-pdf"),

        html.Div([
            html.Small("🔒 Data stays in your browser session.",
                       className="text-muted"),
        ], className="px-1 pb-2"),

    ], className="p-2"),
], width=3, className="bg-light border-end",
   style={"minHeight": "100vh", "position": "sticky", "top": 0, "overflowY": "auto"})


# ─────────────────────────────────────────────────────────────────────────────
# Main panel
# ─────────────────────────────────────────────────────────────────────────────

MAIN = dbc.Col([
    dcc.Store(id="store",       storage_type="memory"),  # per-session DEG data
    dcc.Store(id="enr-store",   storage_type="memory"),  # per-session enrichment results
    dcc.Store(id="bm-store",    storage_type="memory"),  # per-session biomarker scores
    # ── Result cache fingerprints (cleared on new data / refresh) ──────────────
    dcc.Store(id="data-fp",     storage_type="memory"),  # hash of current dataset
    dcc.Store(id="cache-pca",   storage_type="memory"),  # fp when PCA was last computed
    dcc.Store(id="cache-gsea",  storage_type="memory"),  # fp when GSEA was last computed
    dcc.Store(id="cache-net",   storage_type="memory"),  # fp when Network was last computed
    dcc.Store(id="cache-bm",    storage_type="memory"),  # fp when Biomarker was last computed

    # Summary banner
    html.Div(id="banner",
             className="d-flex flex-wrap gap-2 p-2 border-bottom bg-white"),

    dbc.Tabs([

        # ── 1. Meta-Score ─────────────────────────────────────────────────
        dbc.Tab(label="🏆 Meta-Score", tab_id="tab-meta", children=[
            dbc.Row([
                dbc.Col(dcc.Loading(
                    dcc.Graph(id="meta-bar", style={"height": 480}), type="circle",
                ), width=5),
                dbc.Col([
                    html.H6("Meta-Score Prioritization",
                            className="fw-bold mt-2 mb-0", style={"color": "#1a5276"}),
                    html.Small("Combined significance × effect size. Green ≥ 70.",
                               className="text-muted d-block mb-2"),
                    dcc.Loading(html.Div(id="meta-table"), type="dot"),
                    dbc.Button("⬇ Download CSV", id="btn-meta-csv", color="secondary",
                               outline=True, size="sm", className="mt-2"),
                    dcc.Download(id="dl-meta-csv"),
                ], width=7),
            ], className="mt-2 px-2"),
        ]),

        # ── 2. Volcano & Pathway ──────────────────────────────────────────
        dbc.Tab(label="🌋 Volcano & Pathway", tab_id="tab-volcano", children=[
            dbc.Row([
                dbc.Col(
                    dbc.Alert([
                        "💡 Click ", html.Strong("Lasso"), " or ",
                        html.Strong("Box Select"), " in the toolbar, "
                        "draw a selection → pathway enrichment updates for that gene subset.",
                    ], color="info", className="py-1 px-2 mb-1 small"),
                    width=8,
                ),
                dbc.Col(
                    html.Div(id="sel-counter",
                             className="text-muted small text-end align-self-center"),
                    width=4,
                ),
            ], className="px-2 pt-1"),
            dbc.Row([
                dbc.Col(dcc.Loading(
                    dcc.Graph(id="volcano", style={"height": 460},
                              config={"modeBarButtonsToAdd": ["lasso2d", "select2d"]}),
                    type="circle",
                ), width=6),
                dbc.Col(dcc.Loading(
                    dcc.Graph(id="bubble", style={"height": 430}), type="circle",
                ), width=6),
            ], className="mt-1 px-2"),
            dbc.Row([
                dbc.Col(dcc.Loading(
                    dcc.Graph(id="path-bar", style={"height": 400}), type="circle",
                ), width=6),
                dbc.Col([
                    dcc.Loading(html.Div(id="path-table"), type="dot"),
                    dbc.Button("⬇ Enrichment CSV", id="btn-enr-csv", color="secondary",
                               outline=True, size="sm", className="mt-2"),
                    dcc.Download(id="dl-enr-csv"),
                ], width=6),
            ], className="px-2 mt-1"),
            dbc.Row([
                dbc.Col(dcc.Loading(
                    dcc.Graph(id="crosstalk", style={"height": 440}), type="circle",
                ), width=12),
            ], className="px-2 mt-1"),
        ]),

        # ── 3. 3D PCA ─────────────────────────────────────────────────────
        dbc.Tab(label="🔵 3D PCA", tab_id="tab-pca", children=[
            dbc.Row([
                dbc.Col(dcc.Loading(
                    dcc.Graph(id="pca", style={"height": 540}), type="circle",
                ), width=12),
            ], className="mt-2 px-2"),
            html.Div(id="pca-info", className="small text-muted px-3 pb-2"),
        ]),

        # ── 4. Advanced Analytics ─────────────────────────────────────────
        dbc.Tab(label="📊 Advanced Analytics", tab_id="tab-adv", children=[
            dbc.Row([
                dbc.Col(dcc.Loading(
                    dcc.Graph(id="ma-plot",  style={"height": 340}), type="circle",
                ), width=6),
                dbc.Col(dcc.Loading(
                    dcc.Graph(id="heatmap",  style={"height": 340}), type="circle",
                ), width=6),
            ], className="mt-2 px-2"),
            dbc.Row([
                dbc.Col(dcc.Loading(
                    dcc.Graph(id="pval-hist", style={"height": 300}), type="circle",
                ), width=4),
                dbc.Col(dcc.Loading(
                    dcc.Graph(id="lfc-dist",  style={"height": 300}), type="circle",
                ), width=4),
                dbc.Col(dcc.Loading(
                    dcc.Graph(id="rank-plot", style={"height": 300}), type="circle",
                ), width=4),
            ], className="px-2 mt-1"),
        ]),

        # ── 5. GSEA ───────────────────────────────────────────────────────
        dbc.Tab(label="🧬 GSEA", tab_id="tab-gsea", children=[
            dbc.Alert(
                "GSEA pre-ranked (requires ≥ 15 genes + internet for gene sets). "
                "Each user runs in an isolated temp directory.",
                color="secondary", className="py-1 px-2 mb-1 small",
            ),
            dbc.Row([
                dbc.Col(dcc.Loading(
                    dcc.Graph(id="gsea-bar", style={"height": 400}), type="circle",
                ), width=7),
                dbc.Col(dcc.Loading(html.Div(id="gsea-table"), type="dot"), width=5),
            ], className="mt-2 px-2"),
        ]),

        # ── 6. Gene Network ───────────────────────────────────────────────
        dbc.Tab(label="🔗 Gene Network", tab_id="tab-net", children=[
            dbc.Row([
                dbc.Col(dcc.Loading(
                    dcc.Graph(id="network", style={"height": 520}), type="circle",
                ), width=12),
            ], className="mt-2 px-2"),
            html.Div(id="net-info", className="small text-muted px-3 pb-2"),
        ]),

        # ── 7. Drug Targets  ★ UNIQUE ─────────────────────────────────────
        dbc.Tab(label="💊 Drug Targets", tab_id="tab-drugs", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Alert([
                        "🎯 Significant DE genes mapped to ",
                        html.Strong("FDA-approved drugs"),
                        ". Upregulated targets = potential inhibitor candidates; "
                        "downregulated = potential activator candidates.",
                    ], color="success", className="py-1 px-2 mb-2 small"),
                    dcc.Loading(
                        dcc.Graph(id="drug-scatter", style={"height": 440}), type="circle",
                    ),
                ], width=8),
                dbc.Col([
                    dcc.Loading(
                        dcc.Graph(id="drug-donut", style={"height": 340}), type="circle",
                    ),
                    dcc.Loading(html.Div(id="drug-table"), type="dot"),
                    dbc.Button("⬇ Drug Targets CSV", id="btn-drug-csv", color="secondary",
                               outline=True, size="sm", className="mt-2"),
                    dcc.Download(id="dl-drug-csv"),
                ], width=4),
            ], className="mt-2 px-2"),
        ]),

        # ── 8. Biomarker Score  ★ UNIQUE ─────────────────────────────────
        dbc.Tab(label="🎯 Biomarker Score", tab_id="tab-bm", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Alert([
                        html.Strong("Biomarker Priority Score"),
                        " = Meta-Score + clinical bonus for FDA drug targets (+15), "
                        "COSMIC Tier-1 cancer genes (+10), established biomarkers (+5).",
                    ], color="warning", className="py-1 px-2 mb-2 small"),
                    dcc.Loading(
                        dcc.Graph(id="bm-chart", style={"height": 480}), type="circle",
                    ),
                    dbc.Button("⬇ Biomarker CSV", id="btn-bm-csv", color="secondary",
                               outline=True, size="sm", className="mt-1"),
                    dcc.Download(id="dl-bm-csv"),
                ], width=7),
                dbc.Col([
                    html.H6("Cancer Gene Annotations",
                            className="fw-bold mt-2 mb-1", style={"color": "#1a5276"}),
                    dcc.Loading(html.Div(id="cancer-table"), type="dot"),
                ], width=5),
            ], className="mt-2 px-2"),
        ]),

        # ── 9. Auto-Insights  ★ UNIQUE ───────────────────────────────────
        dbc.Tab(label="📋 Insights", tab_id="tab-insights", children=[
            dbc.Row([
                dbc.Col([
                    html.H5("🔬 Auto-Generated Biological Insights",
                            className="fw-bold mt-2 mb-1", style={"color": "#1a5276"}),
                    html.Small(
                        "Rule-based interpretation of your differential expression results. "
                        "No AI — fully reproducible, offline.",
                        className="text-muted d-block mb-3",
                    ),
                    dcc.Loading(html.Div(id="insights-panel"), type="circle"),
                ], width=12),
            ], className="px-3 mt-2"),
        ]),

    ], id="tabs", active_tab="tab-meta"),
], width=9)


app.layout = dbc.Container(
    dbc.Row([SIDEBAR, MAIN], className="g-0"),
    fluid=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

# ── Data ingest ───────────────────────────────────────────────────────────────
@app.callback(
    Output("store", "data"),
    Output("upload-status", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def cb_ingest(contents, fname):
    if contents is None:
        df = _load_demo()
        return df.to_dict("records"), f"📊 Demo dataset — {len(df)} genes loaded"
    try:
        _, b64 = contents.split(",", 1)
        raw = base64.b64decode(b64).decode("utf-8", errors="replace")
        df  = _normalise(pd.read_csv(io.StringIO(raw)))
        if df.empty:
            return dash.no_update, dbc.Alert(
                f"⚠️ {fname} parsed but 0 genes remained after normalisation. "
                "Check that your file has symbol / log2FC / padj columns.",
                color="warning", dismissable=True,
            )
        # Trim to essential columns only — keeps memory store lean for 17k+ genes
        _KEEP = ["symbol", "log2FC", "padj", "pvalue", "baseMean"]
        df = df[[c for c in _KEEP if c in df.columns]]
        log.info(f"Uploaded {fname}: {len(df)} genes, cols={df.columns.tolist()}")
        return df.to_dict("records"), dbc.Alert(
            f"✅ {fname} — {len(df):,} genes loaded", color="success", dismissable=True,
        )
    except Exception as e:
        log.error(f"cb_ingest error on {fname}: {e}")
        return dash.no_update, dbc.Alert(
            f"❌ Could not parse {fname}: {e}", color="danger", dismissable=True,
        )


# ── Data fingerprint: updated on every new upload, clears all tab caches ──────
@app.callback(
    Output("data-fp",    "data"),
    Output("cache-pca",  "data"),
    Output("cache-gsea", "data"),
    Output("cache-net",  "data"),
    Output("cache-bm",   "data"),
    Input("store", "data"),
)
def cb_update_fp(rec):
    if not rec:
        return None, None, None, None, None
    import hashlib, json as _json
    sample = rec[:20] if isinstance(rec, list) else []
    fp = hashlib.md5(
        _json.dumps(sample, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    return fp, None, None, None, None  # new fingerprint + wipe all caches


# ── Refresh button: force-clears all tab caches ────────────────────────────────
@app.callback(
    Output("cache-pca",  "data", allow_duplicate=True),
    Output("cache-gsea", "data", allow_duplicate=True),
    Output("cache-net",  "data", allow_duplicate=True),
    Output("cache-bm",   "data", allow_duplicate=True),
    Input("refresh-tab-btn", "n_clicks"),
    prevent_initial_call=True,
)
def cb_refresh_tabs(_):
    return None, None, None, None


# ── Summary banner ────────────────────────────────────────────────────────────
@app.callback(
    Output("banner", "children"),
    Input("store", "data"),
    Input("lfc-thresh", "value"),
    Input("p-thresh", "value"),
)
def cb_banner(rec, lfc_t, p_t):
    df = _s2df(rec)
    if df.empty:
        return [html.Span("Upload a DGE CSV to begin analysis",
                          className="text-muted small align-self-center")]
    n   = len(df)   # Total Genes  = full dataset regardless of thresholds
    up  = int(((df["log2FC"] >  lfc_t) & (df["padj"] < p_t)).sum())
    dn  = int(((df["log2FC"] < -lfc_t) & (df["padj"] < p_t)).sum())
    sig = up + dn   # Significant Genes = passing the UI sliders only

    # Count drug targets among significant genes
    sig_genes = df.loc[
        (df["log2FC"].abs() >= lfc_t) & (df["padj"] < p_t), "symbol"
    ].astype(str).str.upper().tolist()
    from data.gene_annotations import DRUG_GENE_INTERACTIONS
    n_drug = sum(1 for g in sig_genes if g in DRUG_GENE_INTERACTIONS)

    def card(lbl, val, bg):
        return html.Div([
            html.Div(str(val), style={"fontSize": 20, "fontWeight": "bold", "color": "white"}),
            html.Div(lbl,      style={"fontSize": 10, "color": "rgba(255,255,255,.85)"}),
        ], style={"background": bg, "borderRadius": 6, "padding": "6px 14px",
                  "minWidth": 95, "textAlign": "center"})

    return [
        card("Total Genes",      n,      "#2c3e50"),
        card("⬆ Up-regulated",   up,     "#c0392b"),
        card("⬇ Down-regulated", dn,     "#2980b9"),
        card("Significant Genes", sig,    "#27ae60"),
        card("💊 Drug Targets",  n_drug, "#8e44ad"),
        html.Div(f"|Log2FC| ≥ {lfc_t}  p ≤ {p_t}",
                 className="align-self-center small text-muted ms-2"),
    ]


# ── Meta-Score tab ────────────────────────────────────────────────────────────
@app.callback(
    Output("meta-bar", "figure"),
    Output("meta-table", "children"),
    Input("store", "data"),
)
def cb_meta(rec):
    try:
        df     = _s2df(rec)
        scored = compute_meta_score(df)
        fig    = create_meta_score_bar(scored)
        filt   = scored[~scored["symbol"].isin(HOUSEKEEP)].sort_values(
            "meta_score", ascending=False
        )
        d = filt[["symbol", "log2FC", "padj", "meta_score"]].head(25).copy()
        d["log2FC"]     = d["log2FC"].map("{:+.3f}".format)
        d["padj"]       = d["padj"].map("{:.2e}".format)
        d["meta_score"] = d["meta_score"].map("{:.1f}".format)
        tbl = dash_table.DataTable(
            data=d.to_dict("records"),
            columns=[{"name": h, "id": i} for h, i in
                     [("Gene", "symbol"), ("Log2FC", "log2FC"),
                      ("Adj.P", "padj"), ("Score", "meta_score")]],
            style_table={"maxHeight": "460px", "overflowY": "auto"},
            style_cell={"fontSize": 12, "padding": "4px 10px", "textAlign": "left"},
            style_header={"backgroundColor": "#1a5276", "color": "white",
                          "fontWeight": "bold"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f4f6f7"},
                {"if": {"filter_query": "{meta_score} >= 70",
                        "column_id": "meta_score"},
                 "backgroundColor": "#d5f5e3", "color": "#1e8449", "fontWeight": "bold"},
            ],
            page_size=25,
        )
        return fig, tbl
    except Exception as e:
        log.error(f"meta: {e}")
        return blank(str(e)), html.P(str(e), className="text-danger small")


# ── Volcano figure (redraws only on data/threshold changes, NOT on selection) ─
@app.callback(
    Output("volcano", "figure"),
    Input("store", "data"),
    Input("lfc-thresh", "value"),
    Input("p-thresh", "value"),
    Input("tabs", "active_tab"),
    prevent_initial_call=True,
)
def cb_volcano_fig(rec, lfc_t, p_t, active_tab):
    if active_tab != "tab-volcano":
        return dash.no_update
    try:
        return create_volcano_plot(_s2df(rec), "log2FC", "padj", "symbol", lfc_t, p_t)
    except Exception as e:
        log.error(f"volcano_fig: {e}")
        return blank(str(e))


# ── Enrichment (triggered by lasso OR data/threshold changes) ─────────────────
@app.callback(
    Output("bubble",     "figure"),
    Output("path-bar",   "figure"),
    Output("path-table", "children"),
    Output("enr-store",  "data"),
    Output("sel-counter","children"),
    Input("store",       "data"),
    Input("lfc-thresh",  "value"),
    Input("p-thresh",    "value"),
    Input("pathway-db",  "value"),
    Input("volcano",     "selectedData"),
    Input("tabs",        "active_tab"),
    prevent_initial_call=True,
)
def cb_volcano_enr(rec, lfc_t, p_t, db, sel, active_tab):
    if active_tab != "tab-volcano":
        return (dash.no_update,) * 5
    try:
        df      = _s2df(rec)
        lasso = bool(sel and sel.get("points"))

        if lasso:
            pts = sel["points"]
            # Primary: extract from customdata[0] (gene name stored there)
            genes = list(dict.fromkeys(
                str(pt["customdata"][0])
                for pt in pts
                if pt.get("customdata") and str(pt["customdata"][0]).strip()
                   and str(pt["customdata"][0]) not in ("0", "0.0", "nan", "")
            ))
            # Fallback: if customdata missing (e.g. from WebGL traces), match
            # selected x/y coordinates back to the dataframe
            if not genes:
                sel_x = {round(pt["x"], 6) for pt in pts if "x" in pt}
                sel_y = {round(pt["y"], 6) for pt in pts if "y" in pt}
                df_tmp = _s2df(rec).copy()
                df_tmp["_nlp"] = -np.log10(df_tmp["padj"].clip(lower=1e-300))
                genes = df_tmp[
                    df_tmp["log2FC"].round(6).isin(sel_x) |
                    df_tmp["_nlp"].round(6).isin(sel_y)
                ]["symbol"].astype(str).str.upper().tolist()
                genes = list(dict.fromkeys(genes))
                log.info(f"lasso: customdata missing, used xy-fallback → {len(genes)} genes")
            else:
                log.info(f"lasso: customdata → {len(genes)} genes")
        else:
            mask  = (df["log2FC"].abs() >= lfc_t) & (df["padj"] < p_t)
            genes = df.loc[mask, "symbol"].str.upper().tolist()

        # Build gene → LFC/NLP maps for activation z-score + effect-weighted score
        ref = df.copy()
        ref["_sym"] = ref["symbol"].astype(str).str.strip().str.upper()
        ref["_nlp"] = -np.log10(ref["padj"].clip(lower=1e-300))
        gene_lfc_map = ref.set_index("_sym")["log2FC"].to_dict()
        gene_nlp_map = ref.set_index("_sym")["_nlp"].to_dict()

        enr = _run_enrichment(genes, db, gene_lfc_map, gene_nlp_map)

        # Selection badge
        if lasso:
            chips = [
                dbc.Badge(g, color="primary", className="me-1", style={"fontSize": 10})
                for g in genes[:20]
            ]
            overflow = (
                [html.Small(f"…+{len(genes)-20} more", className="text-muted")]
                if len(genes) > 20 else []
            )
            counter = html.Div([
                html.Span(f"🔬 {len(genes)} genes selected  ",
                          className="fw-bold text-primary small"),
                *chips, *overflow,
                html.Br(),
                html.Small("Enrichment running on this subset ↓",
                           className="text-muted fst-italic"),
            ])
        else:
            counter = html.Small(
                f"Showing all {len(genes)} significant genes "
                "(lasso/box-select the volcano to analyse a subset)",
                className="text-muted",
            )

        return (
            create_pathway_bubble(enr),
            create_pathway_bar(enr),
            _enr_table(enr),
            enr.to_dict("records") if not enr.empty else [],
            counter,
        )
    except Exception as e:
        log.error(f"volcano_enr: {e}")
        emp = blank(str(e))
        return emp, emp, html.P(str(e), className="text-danger"), [], ""


# ── Crosstalk heatmap (driven by enr-store so it's a separate callback) ────────
@app.callback(
    Output("crosstalk", "figure"),
    Input("enr-store",  "data"),
    Input("tabs",       "active_tab"),
    prevent_initial_call=True,
)
def cb_crosstalk(records, active_tab):
    if active_tab != "tab-volcano":
        return dash.no_update
    try:
        import pandas as pd
        from modules.analysis import compute_pathway_crosstalk
        enr = pd.DataFrame(records) if records else pd.DataFrame()
        ct_df = compute_pathway_crosstalk(enr)
        return create_pathway_crosstalk(ct_df)
    except Exception as e:
        log.error(f"crosstalk: {e}")
        return blank(str(e))


# ── 3D PCA ────────────────────────────────────────────────────────────────────
@app.callback(
    Output("pca",      "figure"),
    Output("pca-info", "children"),
    Output("cache-pca", "data", allow_duplicate=True),
    Input("tabs",      "active_tab"),
    Input("store",     "data"),
    State("data-fp",   "data"),
    State("cache-pca", "data"),
    prevent_initial_call=True,
)
def cb_pca(active_tab, rec, fp, cache_fp):
    if active_tab != "tab-pca":
        return dash.no_update, dash.no_update, dash.no_update
    if cache_fp and cache_fp == fp:
        return dash.no_update, dash.no_update, dash.no_update  # already fresh
    try:
        res = run_pca_3d(_s2df(rec))
        if "error" in res:
            return blank(res["error"]), res["error"], None
        v = res["explained_variance"]
        return (
            create_pca_3d(res),
            (f"{res['n_clusters']} clusters  |  variance: "
             f"PC1 {v[0]:.1f}%  PC2 {v[1]:.1f}%  PC3 {v[2]:.1f}%  "
             f"|  {len(res['genes'])} genes"),
            fp,
        )
    except Exception as e:
        log.error(f"pca: {e}")
        return blank(str(e)), str(e), None


# ── Advanced Analytics ────────────────────────────────────────────────────────
@app.callback(
    Output("ma-plot",   "figure"),
    Output("heatmap",   "figure"),
    Output("pval-hist", "figure"),
    Output("lfc-dist",  "figure"),
    Output("rank-plot", "figure"),
    Input("store",      "data"),
    Input("lfc-thresh", "value"),
    Input("p-thresh",   "value"),
    Input("tabs",       "active_tab"),
    prevent_initial_call=True,
)
def cb_advanced(rec, lfc_t, p_t, active_tab):
    if active_tab != "tab-adv":
        return (dash.no_update,) * 5
    try:
        df = _s2df(rec)
        return (
            create_ma_plot(df),
            create_top_heatmap(df, lfc_t, p_t),
            create_pval_hist(df),
            create_lfc_dist(df),
            create_rank_metric(df),
        )
    except Exception as e:
        log.error(f"adv: {e}")
        f = blank(str(e))
        return f, f, f, f, f


# ── GSEA ──────────────────────────────────────────────────────────────────────
@app.callback(
    Output("gsea-bar",   "figure"),
    Output("gsea-table", "children"),
    Output("cache-gsea", "data", allow_duplicate=True),
    Input("tabs",  "active_tab"),
    Input("store", "data"),
    State("data-fp",    "data"),
    State("cache-gsea", "data"),
    prevent_initial_call=True,
)
def cb_gsea(active_tab, rec, fp, cache_fp):
    if active_tab != "tab-gsea":
        return dash.no_update, dash.no_update, dash.no_update
    if cache_fp and cache_fp == fp:
        return dash.no_update, dash.no_update, dash.no_update
    try:
        df  = _s2df(rec)
        res = run_gsea_preranked(df)
        if res.get("error"):
            return blank(f"GSEA: {res['error']}"), html.P(res["error"], className="text-muted small"), None
        results = res["results"]
        if results.empty:
            return blank("No GSEA results (FDR < 0.25)."), html.P("No results.", className="text-muted small"), None
        top = results.head(15)
        sc  = "NES" if "NES" in top.columns else top.columns[0]
        tm  = "Term" if "Term" in top.columns else top.columns[min(1, len(top.columns)-1)]
        fig = go.Figure(go.Bar(
            x=top[sc], y=top[tm].str[:50], orientation="h",
            marker=dict(
                color=top[sc], colorscale="RdBu", reversescale=True,
                cmin=-top[sc].abs().max(), cmax=top[sc].abs().max(),
                showscale=True, colorbar=dict(title="NES", thickness=12),
            ),
            hovertemplate="<b>%{y}</b><br>NES: %{x:.3f}<extra></extra>",
        ))
        fig.update_layout(
            template="simple_white", title="GSEA Pre-ranked — Top Pathways",
            xaxis_title="Normalized Enrichment Score (NES)",
            yaxis=dict(autorange="reversed"),
            height=400, margin=dict(l=10, r=10, t=45, b=30),
        )
        disp_cols = [c for c in ["Term", "NES", "FDR q-val", "Gene %"] if c in top.columns]
        tbl = dash_table.DataTable(
            data=top[disp_cols].to_dict("records"),
            columns=[{"name": c, "id": c} for c in disp_cols],
            style_table={"maxHeight": "340px", "overflowY": "auto"},
            style_cell={"fontSize": 11, "padding": "3px 8px"},
            style_header={"backgroundColor": "#1a5276", "color": "white", "fontWeight": "bold"},
            page_size=15,
        )
        return fig, tbl, fp
    except Exception as e:
        log.error(f"gsea: {e}")
        return blank(str(e)), html.P(str(e), className="text-danger small"), None


# ── Gene Network ──────────────────────────────────────────────────────────────
@app.callback(
    Output("network",  "figure"),
    Output("net-info", "children"),
    Output("cache-net", "data", allow_duplicate=True),
    Input("tabs",  "active_tab"),
    Input("store", "data"),
    State("data-fp",   "data"),
    State("cache-net", "data"),
    prevent_initial_call=True,
)
def cb_network(active_tab, rec, fp, cache_fp):
    if active_tab != "tab-net":
        return dash.no_update, dash.no_update, dash.no_update
    if cache_fp and cache_fp == fp:
        return dash.no_update, dash.no_update, dash.no_update
    try:
        df  = _s2df(rec)
        res = run_wgcna_lite(df)
        if res.get("error"):
            return blank(res["error"]), res["error"], None
        G = res["graph"]
        fig = create_network_graph(res)
        return (
            fig,
            (f"WGCNA-lite  |  {G.number_of_nodes()} nodes  "
             f"|  {res['n_edges']} edges  |  {res['n_modules']} modules"),
            fp,
        )
    except Exception as e:
        log.error(f"network: {e}")
        return blank(str(e)), str(e), None


# ── Drug Targets tab ──────────────────────────────────────────────────────────
@app.callback(
    Output("drug-scatter", "figure"),
    Output("drug-donut",   "figure"),
    Output("drug-table",   "children"),
    Input("store",       "data"),
    Input("lfc-thresh",  "value"),
    Input("p-thresh",    "value"),
    Input("tabs",        "active_tab"),
    prevent_initial_call=True,
)
def cb_drugs(rec, lfc_t, p_t, active_tab):
    if active_tab != "tab-drugs":
        return (dash.no_update,) * 3
    try:
        df = _s2df(rec)
        mask = (df["log2FC"].abs() >= lfc_t) & (df["padj"] < p_t)
        sig_genes = df.loc[mask, "symbol"].str.upper().tolist()

        drug_df = get_drug_targets(sig_genes)

        scatter = create_drug_target_chart(drug_df, df)
        donut   = create_drug_type_donut(drug_df)

        if drug_df.empty:
            tbl = html.P("No FDA-approved drug targets among significant genes.",
                         className="text-muted small ps-1 mt-2")
        else:
            fda = drug_df[drug_df["fda"] == True].copy() if "fda" in drug_df.columns else drug_df
            cols = [c for c in ["gene", "drug", "type", "indication", "fda"] if c in fda.columns]
            tbl = dash_table.DataTable(
                data=fda[cols].to_dict("records"),
                columns=[{"name": c.title(), "id": c} for c in cols],
                style_table={"maxHeight": "340px", "overflowY": "auto"},
                style_cell={"fontSize": 11, "padding": "3px 8px"},
                style_header={"backgroundColor": "#1a5276", "color": "white",
                              "fontWeight": "bold", "fontSize": 11},
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#f4f6f7"},
                    {"if": {"filter_query": "{fda} eq True"},
                     "backgroundColor": "#d5f5e3"},
                ],
                page_size=15,
            )
        return scatter, donut, tbl
    except Exception as e:
        log.error(f"drugs: {e}")
        return blank(str(e)), blank(""), html.P(str(e), className="text-danger small")


# ── Biomarker Score tab ───────────────────────────────────────────────────────
@app.callback(
    Output("bm-chart",     "figure"),
    Output("cancer-table", "children"),
    Output("bm-store",     "data"),
    Output("cache-bm",     "data", allow_duplicate=True),
    Input("tabs",       "active_tab"),
    Input("store",      "data"),
    State("data-fp",  "data"),
    State("cache-bm", "data"),
    prevent_initial_call=True,
)
def cb_biomarker(active_tab, rec, fp, cache_fp):
    if active_tab != "tab-bm":
        return (dash.no_update,) * 4
    if cache_fp and cache_fp == fp:
        return (dash.no_update,) * 4
    try:
        df     = _s2df(rec)
        bm_df  = compute_biomarker_score(df)
        fig    = create_biomarker_score_chart(bm_df)

        # Cancer gene table
        all_genes = df["symbol"].astype(str).str.upper().tolist()
        cg_df = get_cancer_gene_info(all_genes)
        if cg_df.empty:
            cancer_tbl = html.P("No COSMIC cancer genes found in this dataset.",
                                className="text-muted small ps-1")
        else:
            cancer_tbl = dash_table.DataTable(
                data=cg_df.to_dict("records"),
                columns=[{"name": c.replace("_"," ").title(), "id": c} for c in cg_df.columns],
                style_table={"maxHeight": "440px", "overflowY": "auto"},
                style_cell={"fontSize": 11, "padding": "3px 8px"},
                style_header={"backgroundColor": "#1a5276", "color": "white",
                              "fontWeight": "bold"},
                style_data_conditional=[
                    {"if": {"filter_query": "{role} eq 'Oncogene'"},
                     "backgroundColor": "#fdebd0"},
                    {"if": {"filter_query": "{role} eq 'TSG'"},
                     "backgroundColor": "#d6eaf8"},
                    {"if": {"filter_query": "{tier} eq 1"},
                     "fontWeight": "bold"},
                ],
                page_size=20,
            )
        return fig, cancer_tbl, bm_df.to_dict("records"), fp
    except Exception as e:
        log.error(f"biomarker: {e}")
        return blank(str(e)), html.P(str(e), className="text-danger small"), [], None


# ── Auto-Insights tab ─────────────────────────────────────────────────────────
@app.callback(
    Output("insights-panel", "children"),
    Input("store",       "data"),
    Input("enr-store",   "data"),
    Input("lfc-thresh",  "value"),
    Input("p-thresh",    "value"),
    Input("tabs",        "active_tab"),
    prevent_initial_call=True,
)
def cb_insights(rec, enr_rec, lfc_t, p_t, active_tab):
    if active_tab != "tab-insights":
        return dash.no_update
    try:
        df     = _s2df(rec)
        enr_df = pd.DataFrame(enr_rec) if enr_rec else pd.DataFrame()
        insights = generate_insights(df, enr_df, lfc_t, p_t)

        if not insights:
            return html.P("No insights generated yet. Run analyses first.",
                          className="text-muted")

        cards = []
        for ins in insights:
            cards.append(
                dbc.Card([
                    dbc.CardHeader([
                        html.Span(ins["icon"] + " ", style={"fontSize": 18}),
                        html.Strong(ins["title"]),
                        dbc.Badge(ins["category"], color="secondary",
                                  className="ms-2", style={"fontSize": 10}),
                    ]),
                    dbc.CardBody(
                        dcc.Markdown(
                            # Convert basic HTML tags to markdown-ish
                            ins["body"]
                            .replace("<strong>", "**").replace("</strong>", "**")
                            .replace("<em>", "*").replace("</em>", "*")
                            .replace("<br>", "\n\n")
                            .replace("<b>", "**").replace("</b>", "**"),
                            dangerously_allow_html=False,
                            style={"fontSize": 13},
                        )
                    ),
                ], color=ins["severity"], outline=True, className="mb-3",
                )
            )
        return html.Div(cards)
    except Exception as e:
        log.error(f"insights: {e}")
        return html.P(str(e), className="text-danger small")


# ── Gene bio-context ──────────────────────────────────────────────────────────
@app.callback(
    Output("gene-panel", "children"),
    Input("gene-input",  "value"),
    prevent_initial_call=True,
)
def cb_gene(gene):
    if not gene or len(gene.strip()) < 2:
        return ""
    info = fetch_gene_info(gene.strip().upper())
    if "error" in info:
        return dbc.Alert(info["error"], color="warning", className="p-1 small")
    parts = []
    if info.get("full_name"):
        parts.append(html.P([html.Strong("Name: "), info["full_name"]], className="mb-1"))
    if info.get("function"):
        fn = info["function"]
        parts.append(html.P([html.Strong("Function: "),
                              fn[:220] + ("…" if len(fn) > 220 else "")], className="mb-1"))
    if info.get("subcellular_location"):
        parts.append(html.P([html.Strong("Location: "),
                              info["subcellular_location"]], className="mb-1"))
    if info.get("disease_associations"):
        parts.append(html.P([html.Strong("Disease: "),
                              ", ".join(info["disease_associations"][:3])], className="mb-1"))
    if info.get("uniprot_id"):
        parts.append(html.Small(
            f"UniProt {info['uniprot_id']}  NCBI {info.get('ncbi_id','')}",
            className="text-muted",
        ))
    return html.Div(parts)


# ── PDF download ──────────────────────────────────────────────────────────────
@app.callback(
    Output("dl-pdf",   "data"),
    Input("btn-pdf",   "n_clicks"),
    State("volcano",   "figure"),
    State("pca",       "figure"),
    State("enr-store", "data"),
    State("bm-store",  "data"),
    State("store",     "data"),
    State("lfc-thresh","value"),
    State("p-thresh",  "value"),
    prevent_initial_call=True,
)
def cb_pdf(n, vol, pca_f, enr_recs, bm_recs, data_recs, lfc_t, p_t):
    if not n:
        return dash.no_update
    try:
        df     = _s2df(data_recs)
        enr_df = pd.DataFrame(enr_recs or [])
        bm_df  = pd.DataFrame(bm_recs  or [])
        drug_df = get_drug_targets(
            df.loc[(df["log2FC"].abs() >= lfc_t) & (df["padj"] < p_t),
                   "symbol"].str.upper().tolist()
        )

        up  = int(((df["log2FC"] >  lfc_t) & (df["padj"] < p_t)).sum())
        dn  = int(((df["log2FC"] < -lfc_t) & (df["padj"] < p_t)).sum())
        stats = {"total": len(df), "up": up, "down": dn, "sig": up + dn}

        insights = generate_insights(df, enr_df, lfc_t, p_t)

        pdf = generate_pdf_report(
            go.Figure(vol)   if vol   else None,
            enr_df,
            go.Figure(pca_f) if pca_f else None,
            drug_df=drug_df,
            insights=insights,
            summary_stats=stats,
        )
        fname = f"Apex_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        return dcc.send_bytes(pdf, fname)
    except Exception as e:
        log.error(f"pdf: {e}")
        return dash.no_update


# ── CSV Downloads ─────────────────────────────────────────────────────────────

@app.callback(
    Output("dl-meta-csv", "data"),
    Input("btn-meta-csv", "n_clicks"),
    State("store", "data"),
    prevent_initial_call=True,
)
def dl_meta_csv(n, rec):
    if not n:
        return dash.no_update
    df = compute_meta_score(_s2df(rec))
    cols = [c for c in ["symbol", "log2FC", "padj", "baseMean", "meta_score"] if c in df.columns]
    return dcc.send_data_frame(df[cols].to_csv, "meta_score.csv", index=False)


@app.callback(
    Output("dl-enr-csv", "data"),
    Input("btn-enr-csv", "n_clicks"),
    State("enr-store", "data"),
    prevent_initial_call=True,
)
def dl_enr_csv(n, rec):
    if not n or not rec:
        return dash.no_update
    return dcc.send_data_frame(pd.DataFrame(rec).to_csv, "enrichment_results.csv", index=False)


@app.callback(
    Output("dl-drug-csv", "data"),
    Input("btn-drug-csv", "n_clicks"),
    State("store", "data"),
    State("lfc-thresh", "value"),
    State("p-thresh", "value"),
    prevent_initial_call=True,
)
def dl_drug_csv(n, rec, lfc_t, p_t):
    if not n:
        return dash.no_update
    df = _s2df(rec)
    sig_genes = df.loc[
        (df["log2FC"].abs() >= lfc_t) & (df["padj"] < p_t), "symbol"
    ].str.upper().tolist()
    drug_df = get_drug_targets(sig_genes)
    return dcc.send_data_frame(drug_df.to_csv, "drug_targets.csv", index=False)


@app.callback(
    Output("dl-bm-csv", "data"),
    Input("btn-bm-csv", "n_clicks"),
    State("bm-store", "data"),
    prevent_initial_call=True,
)
def dl_bm_csv(n, rec):
    if not n or not rec:
        return dash.no_update
    return dcc.send_data_frame(pd.DataFrame(rec).to_csv, "biomarker_scores.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=7860)
