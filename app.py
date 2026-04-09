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
import os
from datetime import datetime

# ── Load .env for LOCAL development only ─────────────────────────────────────
# override=False means system/session env vars (HF Space secrets) always win.
# If no .env file exists, this is a no-op.
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass  # python-dotenv not installed — rely on os.environ directly

# ── Startup key check — visible in terminal before Dash server starts ─────────
_gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
_groq_key   = os.environ.get("GROQ_API_KEY",   "").strip()
print("=" * 55)
print(f"[Apex AI]  GEMINI_API_KEY : {'✅ Detected (primary)' if _gemini_key else '❌ Missing'}")
print(f"[Apex AI]  GROQ_API_KEY   : {'✅ Detected (fallback)' if _groq_key   else '❌ Missing'}")
print("=" * 55)


def _startup_badge():
    """Build the Research Copilot badge from os.environ at server start time.
    Called once during layout construction — no network calls, no callbacks.
    Returns a dbc.Badge reflecting the actual available AI engine."""
    import dash_bootstrap_components as _dbc
    if os.environ.get("GEMINI_API_KEY", "").strip():
        return _dbc.Badge("Online · Gemini 2.5 Flash", color="success",
                          style={"fontSize": 9})
    if os.environ.get("GROQ_API_KEY", "").strip():
        return _dbc.Badge("Online · Groq Llama 3", color="info",
                          style={"fontSize": 9})
    return _dbc.Badge("Offline · Rule-Based", color="secondary",
                      style={"fontSize": 9})

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
# Global symbol cleaner (mirrors modules/analysis._clean_sym)
# Strips whitespace, surrounding quotes, converts to UPPER.
# Applied at every boundary: upload, lasso, DB lookups.
# ─────────────────────────────────────────────────────────────────────────────

def _clean_sym(sym) -> str:
    return str(sym).strip().strip('"').strip("'").upper()

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

    # ── Numeric-dtype Entrez column: swap with the first string column ────────
    # Handles the case where the CSV has an integer Entrez-ID column that pandas
    # reads as int64 (e.g. a DESeq2 output with row names as Entrez IDs).
    _reserved = {"log2FC", "padj", "pvalue", "baseMean"}
    if not pd.api.types.is_object_dtype(df["symbol"]):
        for c in df.columns:
            if c not in _reserved and c != "symbol" and pd.api.types.is_object_dtype(df[c]):
                df = df.rename(columns={"symbol": "__entrez_id", c: "symbol"})
                break

    # Always cast symbol to string so .str accessor never fails downstream
    df["symbol"] = df["symbol"].map(_clean_sym)

    df["log2FC"] = pd.to_numeric(df["log2FC"], errors="coerce")
    # DESeq2 / edgeR output regularly has NA padj for low-count genes (independent
    # filtering). Coerce those to 1.0 (non-significant) so NO genes are dropped.
    padj_raw = pd.to_numeric(df["padj"], errors="coerce")
    df["padj"] = padj_raw.fillna(1.0).clip(lower=1e-300, upper=1.0)
    df = df.dropna(subset=["log2FC"]).reset_index(drop=True)

    # ── ID-type detection + translation (string Entrez / Ensembl IDs) ─────────
    # String Entrez IDs like "7157" have object dtype — the dtype check above
    # does NOT catch them.  Detect by content and translate to HGNC symbols so
    # every downstream enrichment DB lookup finds a match.
    try:
        from modules.id_mapper import detect_id_type, translate_to_symbols
        sym_vals = df["symbol"].tolist()
        _id_type = detect_id_type(sym_vals)
        if _id_type in ("entrez", "ensembl"):
            log.info(f"_normalise: detected {_id_type} IDs — running translation")
            unique_ids  = list(dict.fromkeys(sym_vals))          # preserve order, deduplicate
            trans_list, _, n_in, n_ok = translate_to_symbols(unique_ids)
            trans_map   = dict(zip(unique_ids, trans_list))
            df["symbol"] = df["symbol"].map(trans_map).fillna(df["symbol"])
            log.info(f"_normalise: translated {n_ok}/{n_in} {_id_type} IDs → gene symbols")
    except Exception as _te:
        log.warning(f"_normalise: ID translation skipped — {_te}")

    return df


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
    background: int = 20_000,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Run pathway enrichment and attach activation z-scores + effect-weighted
    scores when LFC/NLP maps are supplied.

    strict=True  → filter on BH-adjusted p-value ≤ 0.05 (conservative)
    strict=False → filter on nominal p-value ≤ 0.05 (relaxed / exploratory)

    background should be set to the actual number of genes in the loaded
    dataset so the Fisher 2×2 table reflects the real experiment, not a
    hardcoded genome-wide estimate.
    """
    if not sig_genes:
        return pd.DataFrame()

    # ── Normalise symbols ────────────────────────────────────────────────────
    sig_genes = [_clean_sym(g) for g in sig_genes if g and str(g).strip()]
    if not sig_genes:
        return pd.DataFrame()

    # ── ID translation safety net ────────────────────────────────────────────
    # Catches Entrez / Ensembl IDs that were not translated at upload time
    # (e.g. genes selected via lasso from a dataset that bypassed _normalise).
    id_type      = "symbol"
    n_translated = 0
    try:
        from modules.id_mapper import detect_id_type, translate_to_symbols
        id_type = detect_id_type(sig_genes)
        if id_type in ("entrez", "ensembl"):
            trans_list, id_type, n_in, n_translated = translate_to_symbols(sig_genes)
            if n_translated > 0:
                # Also re-key lfc/nlp maps so directional scoring works
                _trans_map   = dict(zip(sig_genes, trans_list))
                gene_lfc_map = {_trans_map.get(k, k): v
                                for k, v in (gene_lfc_map or {}).items()}
                gene_nlp_map = {_trans_map.get(k, k): v
                                for k, v in (gene_nlp_map or {}).items()}
                sig_genes    = trans_list
                log.info(f"_run_enrichment: translated {n_translated}/{n_in} "
                         f"{id_type} IDs → symbols")
    except Exception as _te:
        log.warning(f"_run_enrichment: ID translation skipped — {_te}")

    # ── Background floor ─────────────────────────────────────────────────────
    # Fisher's 2×2 table requires  background >= query_size + max_pathway_size.
    # If the uploaded dataset is tiny (demo / test), clamp to a safe minimum so
    # cell d never goes negative and the p-value is meaningful.
    MAX_PATHWAY_SIZE = 20          # largest gene set in the built-in databases
    background = max(background, len(sig_genes) + MAX_PATHWAY_SIZE, 500)

    try:
        analyzer = _get_analyzer(db_choice)
        lfc_map  = gene_lfc_map or {}
        nlp_map  = gene_nlp_map or {}
        _kwargs  = dict(
            padj_threshold=0.05,
            background=background,
            gene_lfc_map=lfc_map,
            gene_nlp_map=nlp_map,
        )
        if db_choice == "all":
            result = analyzer.run_multi_enrichment(sig_genes, strict=strict, **_kwargs)
            # Auto-retry: never leave user with blank — silently relax if strict gave nothing
            if result.empty and strict:
                log.info("Enrichment: strict mode 0 rows — retrying relaxed")
                result = analyzer.run_multi_enrichment(sig_genes, strict=False, **_kwargs)
                if not result.empty:
                    result["_mode"] = "relaxed"
        else:
            result = analyzer.run_enrichment(sig_genes, strict=strict, **_kwargs)
            if result.empty and strict:
                log.info("Enrichment: strict mode 0 rows — retrying relaxed")
                result = analyzer.run_enrichment(sig_genes, strict=False, **_kwargs)
                if not result.empty:
                    result["_mode"] = "relaxed"

        # ── Tag result with translation metadata so the UI can show alerts ──
        if not result.empty and id_type in ("entrez", "ensembl"):
            result["_id_type"]      = id_type
            result["_n_translated"] = n_translated
        elif result.empty and id_type in ("entrez", "ensembl"):
            # Even an empty DF needs the tag for the warning alert
            result = pd.DataFrame([{
                "_id_type": id_type, "_n_translated": n_translated,
            }])
            result = result.iloc[0:0]  # keep columns, drop the dummy row
            result["_id_type"]      = pd.Series(dtype=str)
            result["_n_translated"] = pd.Series(dtype=int)
            # Store in attrs — survives the empty-frame path
            result.attrs["_id_type"]      = id_type
            result.attrs["_n_translated"] = n_translated

        return result
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

    # Ordered preferred columns — skip any not in the result
    preferred = [
        "pathway", "database",
        "overlap_count", "gene_ratio",
        "adjusted_p_value", "odds_ratio",
        "direction", "activation_zscore",
        "effect_weighted_score", "genes",
    ]
    cols = [c for c in preferred if c in enr.columns]

    d = enr[cols].head(25).copy()

    # Format numerics
    if "adjusted_p_value"      in d.columns:
        d["adjusted_p_value"]      = d["adjusted_p_value"].map("{:.2e}".format)
    if "gene_ratio"            in d.columns:
        d["gene_ratio"]            = d["gene_ratio"].map(lambda v: f"{float(v)*100:.1f}%")
    if "odds_ratio"            in d.columns:
        d["odds_ratio"]            = d["odds_ratio"].map(lambda v: f"{float(v):.2f}")
    if "activation_zscore"     in d.columns:
        d["activation_zscore"]     = d["activation_zscore"].map(lambda v: f"{float(v):+.2f}")
    if "effect_weighted_score" in d.columns:
        d["effect_weighted_score"] = d["effect_weighted_score"].map(lambda v: f"{float(v):.1f}")
    if "genes"                 in d.columns:
        d["genes"]                 = d["genes"].str[:80]

    col_labels = {
        "pathway":               "Pathway",
        "database":              "DB",
        "overlap_count":         "Overlap",
        "gene_ratio":            "Gene Ratio",
        "adjusted_p_value":      "Adj. p",
        "odds_ratio":            "OR",
        "direction":             "Direction",
        "activation_zscore":     "z-score",
        "effect_weighted_score": "Effect Score",
        "genes":                 "Genes (overlap)",
    }

    # Conditional row coloring by direction
    dir_style = []
    if "direction" in d.columns:
        recs = d.to_dict("records")
        for i, rec in enumerate(recs):
            dirn = str(rec.get("direction", ""))
            if dirn == "Activated":
                dir_style.append({"if": {"row_index": i},
                                  "backgroundColor": "#fdf2f8", "color": "#922b21"})
            elif dirn == "Inhibited":
                dir_style.append({"if": {"row_index": i},
                                  "backgroundColor": "#ebf5fb", "color": "#1a5276"})

    return dash_table.DataTable(
        data=d.to_dict("records"),
        columns=[{"name": col_labels.get(c, c.replace("_", " ").title()), "id": c}
                 for c in cols],
        style_table={"overflowX": "auto", "maxHeight": "360px", "overflowY": "auto"},
        style_cell={"fontSize": 11, "padding": "3px 8px", "textAlign": "left",
                    "maxWidth": "180px", "overflow": "hidden",
                    "textOverflow": "ellipsis", "whiteSpace": "nowrap"},
        style_header={"backgroundColor": "#1a5276", "color": "white",
                      "fontWeight": "bold", "fontSize": 11},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
            *dir_style,
        ],
        page_size=10,
        sort_action="native",
        filter_action="native",
        tooltip_delay=0,
        tooltip_duration=None,
        tooltip_data=[
            {c: {"value": str(rec.get(c, "")), "type": "markdown"} for c in cols}
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
        html.Small("v3.7 — Research Copilot", className="text-white-50"),
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
        _section_head("🎯", "Enrichment Mode",
                      "Strict = FDR-corrected · Relaxed = nominal p"),
        dcc.RadioItems(
            id="enr-mode",
            options=[
                {"label": "Strict  (FDR adj. p < 0.05)", "value": "strict"},
                {"label": "Relaxed  (nominal p < 0.05)", "value": "relaxed"},
            ],
            value="strict",
            labelStyle={"display": "block", "fontSize": 12, "marginBottom": 3},
            inputStyle={"marginRight": 5},
        ),
        dbc.Alert(
            "💡 Use Relaxed for large selections (> 1 000 genes) where FDR "
            "correction is overly conservative.",
            color="info", className="py-1 px-2 mt-1 small",
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
        dbc.Tooltip(
            "Re-runs analysis for the active tab and re-reads your .env file "
            "so new API keys are picked up without restarting the server.",
            target="refresh-tab-btn", placement="right",
        ),
        # ── Key-missing warning (populated by cb_key_status on load/refresh) ──
        html.Div(id="key-warning-alert"),

        html.Hr(className="my-2"),
        dbc.Button("📥 Download PDF Report", id="btn-pdf",
                   color="success", size="sm", className="w-100 mb-2"),
        dcc.Download(id="dl-pdf"),

        html.Div([
            html.Small("🔒 Data stays in your browser session.",
                       className="text-muted"),
        ], className="px-1 pb-2"),

        html.Hr(className="my-2"),

        # ── 🔬 Deep-Analysis Research Consultant ─────────────────────────────
        html.Div([
            html.Div([
                html.Span("🔬 ", style={"fontSize": 15}),
                html.Strong("Research Consultant",
                            style={"fontSize": 11, "color": "#1a5276"}),
                # Badge pre-populated at server start; updated by cb_key_status.
                html.Div(
                    id="sidebar-ai-badge",
                    children=_startup_badge(),
                    className="ms-1",
                    style={"display": "inline-block"},
                ),
            ], className="d-flex align-items-center mb-2"),
            dbc.Button(
                "Deep Results Audit",
                id="btn-audit",
                color="primary",
                size="sm",
                className="w-100 mb-1",
                n_clicks=0,
            ),
            html.Small(
                "4-layer expert investigation · interactive drill-down",
                className="text-muted d-block",
                style={"fontSize": 10, "lineHeight": "1.4"},
            ),
        ], className="px-1 pb-2",
           style={"background": "rgba(26,82,118,0.06)",
                  "border": "1px solid rgba(26,82,118,0.18)",
                  "borderRadius": 6, "padding": "8px 8px 6px"}),

    ], className="p-2"),
], width=3, className="bg-light border-end",
   style={"minHeight": "100vh", "position": "sticky", "top": 0, "overflowY": "auto"})


# ─────────────────────────────────────────────────────────────────────────────
# Main panel
# ─────────────────────────────────────────────────────────────────────────────

MAIN = dbc.Col([
    dcc.Store(id="store",       storage_type="memory"),  # per-session DEG data (full master)
    dcc.Store(id="sig-store",   storage_type="memory"),  # pre-filtered significant genes only
    dcc.Store(id="enr-store",   storage_type="memory"),  # per-session enrichment results
    dcc.Store(id="bm-store",    storage_type="memory"),  # per-session biomarker scores
    # ── Result cache fingerprints (cleared on new data / threshold change / refresh) ──
    dcc.Store(id="data-fp",     storage_type="memory"),  # hash of dataset + thresholds
    dcc.Store(id="cache-pca",   storage_type="memory"),  # fp when PCA was last computed
    dcc.Store(id="cache-gsea",  storage_type="memory"),  # fp when GSEA was last computed
    dcc.Store(id="cache-net",   storage_type="memory"),  # fp when Network was last computed
    dcc.Store(id="cache-bm",    storage_type="memory"),  # fp when Biomarker was last computed
    # ── Deep-Analysis Consultant stores ──────────────────────────────────────
    dcc.Store(id="chat-thread-store", storage_type="memory"),  # Q&A investigation thread
    dcc.Store(id="audit-ctx-store",   storage_type="memory"),  # context snapshot for chat
    # ── Fires once on page load to show AI status badge before any analysis ──
    dcc.Interval(id="key-check-interval", interval=800, max_intervals=1, n_intervals=0),

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
                    dcc.Loading(
                        html.Div(id="sel-counter",
                                 className="text-muted small align-self-center"),
                        type="dot",
                    ),
                    width=12,
                ),
            ], className="px-2 pt-1"),
            # ── Enrichment status bar — visible while callback is running ──────
            dbc.Row([
                dbc.Col(
                    dcc.Loading(
                        html.Div(id="enr-status", className="small px-1"),
                        type="default",
                        color="#1a5276",
                    ),
                    width=12,
                ),
            ], className="px-2"),
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
                    dcc.Graph(id="ma-plot",   style={"height": 360}), type="circle",
                ), width=6),
                dbc.Col(dcc.Loading(
                    dcc.Graph(id="pval-hist", style={"height": 360}), type="circle",
                ), width=6),
            ], className="mt-2 px-2"),
            dbc.Row([
                dbc.Col(dcc.Loading(
                    dcc.Graph(id="heatmap", style={"minHeight": 500, "height": "auto"}),
                    type="circle",
                ), width=12),
            ], className="px-2 mt-2"),
            dbc.Row([
                dbc.Col(dcc.Loading(
                    dcc.Graph(id="lfc-dist",  style={"height": 300}), type="circle",
                ), width=6),
                dbc.Col(dcc.Loading(
                    dcc.Graph(id="rank-plot", style={"height": 300}), type="circle",
                ), width=6),
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

                    # ── Deep-Analysis Consultant notice ───────────────────
                    dbc.Alert(
                        [
                            html.Span("🔬 ", style={"fontSize": 14}),
                            "Use the ",
                            html.Strong("Deep Results Audit"),
                            " button in the left sidebar for a 4-layer AI investigation "
                            "(Statistical Topology → Pathway Synergy → Cluster Diagnostics "
                            "→ Literature Synthesis) with an interactive drill-down chat.",
                        ],
                        color="info", className="py-2 px-3 mb-3 small",
                    ),

                    # ── Rule-based Insights ───────────────────────────────
                    html.H5("🔬 Auto-Generated Biological Insights",
                            className="fw-bold mt-2 mb-1", style={"color": "#1a5276"}),
                    html.Small(
                        "Rule-based interpretation of your differential expression results. "
                        "Fully reproducible, offline.",
                        className="text-muted d-block mb-3",
                    ),
                    dcc.Loading(html.Div(id="insights-panel"), type="circle"),

                ], width=12),
            ], className="px-3 mt-2"),
        ]),

    ], id="tabs", active_tab="tab-meta",
       style={"minWidth": "900px"}),
], width=9, style={"overflowX": "auto", "maxWidth": "100%"})


# ─────────────────────────────────────────────────────────────────────────────
# Chat history renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_chat_history(thread: list[dict]) -> list:
    """Convert a list of {role, content} dicts into styled Dash elements."""
    if not thread:
        return [html.P("No follow-up questions yet.",
                       className="text-muted small fst-italic")]
    items = []
    for msg in thread:
        role    = msg.get("role", "")
        content = msg.get("content", "").strip()
        if not content:
            continue
        if role == "user":
            items.append(html.Div([
                html.Small("You", className="fw-bold text-primary d-block mb-1"),
                html.P(content, className="mb-0 small"),
            ], className="bg-light border rounded p-2 mb-2 ms-4",
               style={"borderLeft": "3px solid #2980b9"}))
        else:
            items.append(html.Div([
                html.Small("Consultant", className="fw-bold text-success d-block mb-1"),
                dcc.Markdown(content,
                             style={"fontSize": 12, "lineHeight": "1.65", "margin": 0}),
            ], className="border rounded p-2 mb-2 me-4 bg-white",
               style={"borderLeft": "3px solid #27ae60"}))
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Offcanvas — Deep-Analysis Research Consultant
# ─────────────────────────────────────────────────────────────────────────────

_AUDIT_OFFCANVAS = dbc.Offcanvas(
    id="audit-offcanvas",
    title=html.Span([
        "🔬 Deep-Analysis Research Consultant",
    ], style={"fontWeight": "bold", "fontSize": 15, "color": "#1a5276"}),
    placement="end",
    is_open=False,
    backdrop=False,      # leave the main panel interactive
    scrollable=True,
    style={"width": "26%", "minWidth": "380px", "maxWidth": "460px"},
    children=[
        # Model attribution badge
        html.Div(id="audit-model-badge", className="mb-3"),

        # ── Initial deep analysis result ─────────────────────────────────────
        dbc.Spinner(
            html.Div(
                id="audit-analysis-div",
                children=html.P(
                    "Click 'Deep Results Audit' in the sidebar to begin "
                    "the 4-layer investigation.",
                    className="text-muted small fst-italic",
                ),
                style={
                    "maxHeight": "42vh", "overflowY": "auto",
                    "lineHeight": "1.75", "fontSize": 13,
                    "paddingRight": 4,
                },
            ),
            color="primary", type="border",
        ),

        html.Hr(className="my-3"),

        # ── Investigation thread ─────────────────────────────────────────────
        html.Div([
            html.Strong("Investigation Thread",
                        style={"fontSize": 12, "color": "#1a5276"}),
            html.Small(" — drill down into specific genes, pathways, or mechanisms",
                       className="text-muted ms-1", style={"fontSize": 10}),
        ], className="mb-2"),

        dbc.Spinner(
            html.Div(
                id="audit-chat-history",
                style={
                    "maxHeight": "26vh", "overflowY": "auto",
                    "marginBottom": 8, "fontSize": 12,
                },
            ),
            color="secondary", size="sm", type="grow",
        ),

        # ── Chat input ───────────────────────────────────────────────────────
        dbc.InputGroup([
            dbc.Input(
                id="chat-input",
                placeholder=(
                    "e.g. 'Focus on the hemoglobin cluster — "
                    "oxidative stress or identity loss?'"
                ),
                type="text",
                size="sm",
                n_submit=0,
                style={"fontSize": 12},
            ),
            dbc.Button(
                "Send", id="chat-submit",
                color="primary", size="sm", n_clicks=0,
            ),
        ], className="mt-1"),
    ],
)


app.layout = dbc.Container([
    dbc.Row([SIDEBAR, MAIN], className="g-0"),
    _AUDIT_OFFCANVAS,
], fluid=True, style={"overflowX": "auto"})


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
        # Prune to three columns — mandatory for 50k-gene datasets on 16 GB RAM
        _KEEP = ["symbol", "log2FC", "padj"]
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


# ── Data fingerprint: updated on new upload OR threshold change ────────────────
# Including thresholds means lazy-tab caches (pca / gsea / net / bm) are busted
# whenever the user adjusts sliders, so revisiting those tabs reruns analysis
# on the correct significant subset rather than returning stale results.
@app.callback(
    Output("data-fp",    "data"),
    Output("cache-pca",  "data"),
    Output("cache-gsea", "data"),
    Output("cache-net",  "data"),
    Output("cache-bm",   "data"),
    Input("store",       "data"),
    Input("lfc-thresh",  "value"),
    Input("p-thresh",    "value"),
)
def cb_update_fp(rec, lfc_t, p_t):
    if not rec:
        return None, None, None, None, None
    import hashlib, json as _json
    sample = rec[:20] if isinstance(rec, list) else []
    fp = hashlib.md5(
        _json.dumps({"d": sample, "lfc": lfc_t, "p": p_t},
                    sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    return fp, None, None, None, None  # new fingerprint + wipe all lazy caches


# ── Significant-gene pre-filter (Significant-First core engine) ───────────────
# Runs on every data upload OR slider change; all analysis tabs consume this
# pre-filtered store instead of reconstructing the full 50k-row DataFrame.
@app.callback(
    Output("sig-store", "data"),
    Input("store",      "data"),
    Input("lfc-thresh", "value"),
    Input("p-thresh",   "value"),
)
def cb_sig_store(rec, lfc_t, p_t):
    df = _s2df(rec)
    if df.empty:
        return []
    lfc_t = lfc_t or 1.0
    p_t   = p_t   or 0.05
    sig = df[(df["log2FC"].abs() >= lfc_t) & (df["padj"] < p_t)]
    return sig.to_dict("records") if not sig.empty else []


# ── Refresh button: force-clears all tab caches + enrichment store ─────────────
@app.callback(
    Output("cache-pca",  "data", allow_duplicate=True),
    Output("cache-gsea", "data", allow_duplicate=True),
    Output("cache-net",  "data", allow_duplicate=True),
    Output("cache-bm",   "data", allow_duplicate=True),
    Output("enr-store",  "data", allow_duplicate=True),
    Input("refresh-tab-btn", "n_clicks"),
    prevent_initial_call=True,
)
def cb_refresh_tabs(_):
    # Clears all dcc.Store caches so every tab re-runs its analysis.
    # cb_key_status (also triggered by refresh-tab-btn) re-checks env vars
    # and updates the AI badge immediately.
    return None, None, None, None, []


# ── Startup key check: sets AI badge + key-warning alert on load / refresh ────
@app.callback(
    Output("sidebar-ai-badge",  "children", allow_duplicate=True),
    Output("key-warning-alert", "children"),
    Input("key-check-interval", "n_intervals"),
    Input("refresh-tab-btn",    "n_clicks"),
    prevent_initial_call=True,
)
def cb_key_status(_interval, _refresh):
    # Read live os.environ — no load_dotenv() here.
    # On HF Spaces, secrets are in os.environ from process start.
    # On local dev, startup load_dotenv(override=False) has already run.
    from modules.ai_summary import check_available_model
    label, color = check_available_model()
    badge = dbc.Badge(label, color=color, style={"fontSize": 9})

    # Warn when no key at all is present (Gemini is primary, Groq is fallback)
    gemini_set = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    groq_set   = bool(os.environ.get("GROQ_API_KEY",   "").strip())

    if not gemini_set and not groq_set:
        warn = dbc.Alert(
            [
                html.Strong("⚠️ API Keys Missing"),
                html.Br(),
                html.Small(
                    "Add GEMINI_API_KEY (primary) or GROQ_API_KEY (fallback) "
                    "to your .env file, then click Refresh — no server restart needed.",
                    className="text-muted",
                ),
                html.Br(),
                html.Small(
                    [
                        "Get a free Gemini key: ",
                        html.A("aistudio.google.com", href="#",
                               style={"pointerEvents": "none", "color": "inherit"}),
                    ],
                    className="text-muted",
                ),
            ],
            color="danger", className="py-1 px-2 mt-1 small",
        )
    elif not gemini_set:
        # Groq present but Gemini (primary) missing — soft notice only
        warn = dbc.Alert(
            html.Small(
                "💡 Add GEMINI_API_KEY for Gemini 2.5 Flash (primary AI). "
                "Currently using Groq as fallback.",
                className="text-muted",
            ),
            color="info", className="py-1 px-2 mt-1 small",
        )
    else:
        warn = None

    return badge, warn


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
    Input("sig-store", "data"),
)
def cb_meta(sig_rec):
    try:
        sig    = _s2df(sig_rec)
        scored = compute_meta_score(sig)
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


# ── Selection-to-Enrichment helpers ───────────────────────────────────────────

def _genes_from_selected(sel) -> list:
    """
    Extract gene symbols from Plotly selectedData.
    customdata[:, 0] holds the gene symbol for every trace in the volcano.
    Returns a de-duplicated list of clean uppercase symbols.
    """
    if not sel or not sel.get("points"):
        return []
    genes = []
    for pt in sel["points"]:
        cd = pt.get("customdata")
        if cd:
            sym = str(cd[0]).strip().strip('"').strip("'").upper()
            if sym and sym not in ("0", "0.0", "NAN", ""):
                genes.append(sym)
    return list(dict.fromkeys(genes))  # preserve order, deduplicate


def _genes_from_relayout(relayout, df: pd.DataFrame) -> list:
    """
    Extract gene symbols by filtering the FULL master dataframe with the
    box-selection coordinates stored in relayoutData.

    Plotly.js emits the selection rectangle in two formats depending on version:
      • Flat keys: relayoutData["selections[0].x0"], ..., .x1, .y0, .y1
      • Nested:    relayoutData["selections"][0]["x0"] …

    y-axis of the volcano is −log10(padj) so we derive _nlp from the store.
    Returns a de-duplicated list of clean uppercase symbols.
    """
    if not relayout:
        return []

    sel_obj = None
    # Form A: nested list (newer Plotly.js)
    sels = relayout.get("selections")
    if sels and isinstance(sels, list) and len(sels) > 0:
        sel_obj = sels[0]
    else:
        # Form B: flat dotted keys (older Plotly.js / Dash serialisation)
        x0 = relayout.get("selections[0].x0")
        x1 = relayout.get("selections[0].x1")
        y0 = relayout.get("selections[0].y0")
        y1 = relayout.get("selections[0].y1")
        if None not in (x0, x1, y0, y1):
            sel_obj = {"x0": x0, "x1": x1, "y0": y0, "y1": y1}

    if not sel_obj:
        return []

    try:
        lx0 = min(float(sel_obj["x0"]), float(sel_obj["x1"]))
        lx1 = max(float(sel_obj["x0"]), float(sel_obj["x1"]))
        ly0 = min(float(sel_obj["y0"]), float(sel_obj["y1"]))
        ly1 = max(float(sel_obj["y0"]), float(sel_obj["y1"]))
    except (KeyError, TypeError, ValueError):
        return []

    # Filter against the FULL dataframe (not the decimated 5 000-point plot)
    nlp = -np.log10(df["padj"].clip(lower=1e-300))
    mask = df["log2FC"].between(lx0, lx1) & nlp.between(ly0, ly1)
    return df.loc[mask, "symbol"].map(_clean_sym).tolist()


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


# ── Enrichment (global — triggered automatically by sig-store changes) ────────
@app.callback(
    Output("bubble",      "figure"),
    Output("path-bar",    "figure"),
    Output("path-table",  "children"),
    Output("enr-store",   "data"),
    Output("sel-counter", "children"),
    Output("enr-status",  "children"),
    Input("sig-store",    "data"),
    Input("pathway-db",   "value"),
    Input("enr-mode",     "value"),
    Input("tabs",         "active_tab"),
    State("store",        "data"),
    prevent_initial_call=True,
)
def cb_volcano_enr(sig_rec, db, enr_mode, active_tab, rec):
    if active_tab != "tab-volcano":
        return (dash.no_update,) * 6

    try:
        df         = _s2df(rec)
        background = max(len(df), 1)
        strict     = (enr_mode != "relaxed")

        # ── Gene extraction: all significant genes from sig-store ─────────
        genes = _s2df(sig_rec)["symbol"].map(_clean_sym).tolist()

        log.info(f"global enrichment  n_genes={len(genes)}")

        # ── Build LFC / NLP maps (use full df, not decimated plot) ───────
        ref = df.copy()
        ref["_sym"] = ref["symbol"].map(_clean_sym)
        ref["_nlp"] = -np.log10(ref["padj"].clip(lower=1e-300))
        gene_lfc_map = ref.set_index("_sym")["log2FC"].to_dict()
        gene_nlp_map = ref.set_index("_sym")["_nlp"].to_dict()

        enr = _run_enrichment(
            genes, db, gene_lfc_map, gene_nlp_map,
            background=background, strict=strict,
        )

        # ── Pull translation metadata from result ─────────────────────────
        enr_id_type      = enr.attrs.get("_id_type", "symbol")
        enr_n_translated = enr.attrs.get("_n_translated", 0)
        if "_id_type" in (enr.columns if not enr.empty else []):
            enr_id_type      = enr["_id_type"].iloc[0]  if len(enr) else enr_id_type
            enr_n_translated = enr["_n_translated"].iloc[0] if len(enr) else enr_n_translated
            # Drop internal columns before rendering
            enr = enr.drop(columns=[c for c in ["_id_type","_n_translated"]
                                    if c in enr.columns])

        # ── Status bar ───────────────────────────────────────────────────
        mode_label  = "Relaxed (nominal p)" if not strict else "Strict (FDR adj.)"
        auto_relaxed = (not enr.empty and enr.get("_mode", pd.Series()).eq("relaxed").any()
                        if "_mode" in enr.columns else False)
        n_paths = len(enr)
        status_color = "success" if n_paths > 0 else "warning"
        status_msg = (
            f"✅ {n_paths} pathway{'s' if n_paths != 1 else ''} found  "
            f"· {len(genes):,} genes  · {mode_label}"
            + ("  · ⚡ auto-relaxed" if auto_relaxed else "")
        ) if n_paths > 0 else (
            f"⚠️ No pathways found for {len(genes):,} genes ({mode_label})"
        )
        enr_status_alerts = [
            dbc.Alert(status_msg, color=status_color, className="py-1 px-2 mb-1 small"),
        ]

        # Numeric-ID conversion notice
        if enr_id_type in ("entrez", "ensembl"):
            id_label = "Entrez" if enr_id_type == "entrez" else "Ensembl"
            enr_status_alerts.append(dbc.Alert(
                f"⚠️ {id_label} IDs detected in selection — "
                f"auto-converted {enr_n_translated:,} IDs to gene symbols for analysis.",
                color="warning", className="py-1 px-2 mb-1 small",
            ))

        # Relaxed-mode suggestion when strict found nothing
        if n_paths == 0 and strict:
            enr_status_alerts.append(dbc.Alert(
                "💡 Tip: switch to Nominal P-value (Relaxed) mode in the sidebar "
                "to surface emerging trends that don't yet survive FDR correction.",
                color="info", className="py-1 px-2 mb-0 small",
            ))

        enr_status = html.Div(enr_status_alerts)

        # ── Large-selection warning ───────────────────────────────────────
        TOO_LARGE  = 2000
        warn_block = []
        if len(genes) > TOO_LARGE:
            warn_block = [dbc.Alert(
                f"⚠️ {len(genes):,} significant genes — enrichment statistical power "
                f"may be reduced above {TOO_LARGE:,}. Consider switching to Relaxed mode.",
                color="warning", className="py-1 px-2 mb-1 small",
            )]

        # ── Global enrichment counter badge ──────────────────────────────
        mode_badge = dbc.Badge(
            "Relaxed" if not strict else "Strict",
            color="warning" if not strict else "secondary",
            className="ms-2", style={"fontSize": 9},
        )
        counter = html.Div([
            *warn_block,
            html.Small(
                f"🌍 Global Enrichment · {len(genes):,} significant genes",
                className="text-muted fw-semibold",
            ),
            mode_badge,
        ])

        return (
            create_pathway_bubble(enr),
            create_pathway_bar(enr),
            _enr_table(enr),
            enr.to_dict("records") if not enr.empty else [],
            counter,
            enr_status,
        )

    except Exception as e:
        log.error(f"volcano_enr: {e}")
        emp = blank(str(e))
        err_status = dbc.Alert(f"❌ Error: {e}", color="danger",
                               className="py-1 px-2 mb-0 small")
        return emp, emp, html.P(str(e), className="text-danger small"), [], "", err_status


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
        ct_df = compute_pathway_crosstalk(enr, top_n=20)
        return create_pathway_crosstalk(ct_df)
    except Exception as e:
        log.error(f"crosstalk: {e}")
        return blank(str(e))


# ── 3D PCA ────────────────────────────────────────────────────────────────────
@app.callback(
    Output("pca",      "figure"),
    Output("pca-info", "children"),
    Output("cache-pca", "data", allow_duplicate=True),
    Input("tabs",       "active_tab"),
    State("sig-store",  "data"),
    State("data-fp",    "data"),
    State("cache-pca",  "data"),
    prevent_initial_call=True,
)
def cb_pca(active_tab, sig_rec, fp, cache_fp):
    if active_tab != "tab-pca":
        return dash.no_update, dash.no_update, dash.no_update
    if cache_fp and cache_fp == fp:
        return dash.no_update, dash.no_update, dash.no_update
    try:
        sig = _s2df(sig_rec)
        res = run_pca_3d(sig)
        if "error" in res:
            return blank(res["error"]), res["error"], None
        v = res["explained_variance"]
        return (
            create_pca_3d(res),
            (f"Sample-centric PCA  |  {res['n_genes']} genes as features  "
             f"|  variance: PC1 {v[0]:.1f}%  PC2 {v[1]:.1f}%  PC3 {v[2]:.1f}%"),
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
    Input("sig-store",  "data"),
    State("lfc-thresh", "value"),
    State("p-thresh",   "value"),
    Input("tabs",       "active_tab"),
    prevent_initial_call=True,
)
def cb_advanced(sig_rec, lfc_t, p_t, active_tab):
    if active_tab != "tab-adv":
        return (dash.no_update,) * 5
    lfc_t = lfc_t or 1.0
    p_t   = p_t   or 0.05
    try:
        plot_df = _s2df(sig_rec)
        return (
            create_ma_plot(plot_df),
            create_top_heatmap(plot_df, lfc_t, p_t),
            create_pval_hist(plot_df),
            create_lfc_dist(plot_df),
            create_rank_metric(plot_df),
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
    Input("tabs",        "active_tab"),
    State("sig-store",   "data"),
    State("data-fp",     "data"),
    State("cache-gsea",  "data"),
    prevent_initial_call=True,
)
def cb_gsea(active_tab, sig_rec, fp, cache_fp):
    if active_tab != "tab-gsea":
        return dash.no_update, dash.no_update, dash.no_update
    if cache_fp and cache_fp == fp:
        return dash.no_update, dash.no_update, dash.no_update
    try:
        sig = _s2df(sig_rec)
        res = run_gsea_preranked(sig)
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
    Input("tabs",       "active_tab"),
    State("sig-store",  "data"),
    State("data-fp",    "data"),
    State("cache-net",  "data"),
    prevent_initial_call=True,
)
def cb_network(active_tab, sig_rec, fp, cache_fp):
    if active_tab != "tab-net":
        return dash.no_update, dash.no_update, dash.no_update
    if cache_fp and cache_fp == fp:
        return dash.no_update, dash.no_update, dash.no_update
    try:
        net_df = _s2df(sig_rec)
        # Count candidate genes before the internal max_genes cap so we can warn early
        n_candidates = len(net_df)
        res = run_wgcna_lite(net_df)
        if res.get("error"):
            return blank(res["error"]), res["error"], None
        G   = res["graph"]

        # ── Safety valve: > 200 input genes → prune graph to top-100 by degree ──
        net_warning = None
        if n_candidates > 200:
            import networkx as nx
            degree_map = dict(G.degree())
            top100     = sorted(degree_map, key=degree_map.get, reverse=True)[:100]
            G_pruned   = G.subgraph(top100).copy()
            res        = {**res, "graph": G_pruned,
                          "n_edges": G_pruned.number_of_edges()}
            G          = G_pruned
            net_warning = dbc.Alert(
                f"⚠️ {n_candidates:,} genes in dataset — displaying top 100 "
                "high-degree nodes to prevent browser slowdown. "
                "Upload a pre-filtered significant-gene list for the full network.",
                color="warning", className="py-1 px-2 mb-1 small",
            )

        fig      = create_network_graph(res)
        info_txt = (f"WGCNA-lite  |  {G.number_of_nodes()} nodes  "
                    f"|  {res['n_edges']} edges  |  {res['n_modules']} modules")
        info_div = html.Div([net_warning, html.Small(info_txt, className="text-muted")]
                            if net_warning else html.Small(info_txt, className="text-muted"))
        return fig, info_div, fp
    except Exception as e:
        log.error(f"network: {e}")
        return blank(str(e)), str(e), None


# ── Drug Targets tab ──────────────────────────────────────────────────────────
@app.callback(
    Output("drug-scatter", "figure"),
    Output("drug-donut",   "figure"),
    Output("drug-table",   "children"),
    Input("sig-store",   "data"),
    Input("tabs",        "active_tab"),
    State("store",       "data"),
    prevent_initial_call=True,
)
def cb_drugs(sig_rec, active_tab, full_rec):
    if active_tab != "tab-drugs":
        return (dash.no_update,) * 3
    try:
        sig_df    = _s2df(sig_rec)
        sig_genes = sig_df["symbol"].map(_clean_sym).tolist()

        drug_df = get_drug_targets(sig_genes)

        # Fallback: if sig gives no drug hits, use top-100 up-regulated from full df
        fallback_used = False
        df = _s2df(full_rec)
        if drug_df.empty and not df.empty:
            top100 = (
                df[df["log2FC"] > 0]
                .nlargest(100, "log2FC")["symbol"]
                .map(_clean_sym).tolist()
            )
            drug_df = get_drug_targets(top100)
            fallback_used = drug_df is not None and not drug_df.empty

        scatter = create_drug_target_chart(drug_df, df)
        donut   = create_drug_type_donut(drug_df)

        if drug_df.empty:
            tbl = html.P(
                "No FDA-approved drug targets found. Try relaxing the LFC/p-value thresholds.",
                className="text-muted small ps-1 mt-2",
            )
        else:
            fda = drug_df[drug_df["fda"] == True].copy() if "fda" in drug_df.columns else drug_df
            cols = [c for c in ["gene", "drug", "type", "indication", "fda"] if c in fda.columns]
            fallback_alert = []
            if fallback_used:
                fallback_alert = [dbc.Alert(
                    "ℹ️ No drug targets in your filtered selection — showing Top 100 up-regulated genes instead.",
                    color="info", className="py-1 px-2 mb-2 small",
                )]
            tbl = html.Div([
                *fallback_alert,
                dash_table.DataTable(
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
                ),
            ])
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
    State("sig-store",  "data"),
    State("data-fp",    "data"),
    State("cache-bm",   "data"),
    prevent_initial_call=True,
)
def cb_biomarker(active_tab, sig_rec, fp, cache_fp):
    if active_tab != "tab-bm":
        return (dash.no_update,) * 4
    if cache_fp and cache_fp == fp:
        return (dash.no_update,) * 4
    try:
        bm_src = _s2df(sig_rec)
        bm_df  = compute_biomarker_score(bm_src)
        fig    = create_biomarker_score_chart(bm_df)

        # Cancer gene table — significant genes only
        all_genes = bm_src["symbol"].astype(str).str.upper().tolist()
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
    Input("sig-store",   "data"),
    Input("enr-store",   "data"),
    Input("tabs",        "active_tab"),
    State("lfc-thresh",  "value"),
    State("p-thresh",    "value"),
    prevent_initial_call=True,
)
def cb_insights(sig_rec, enr_rec, active_tab, lfc_t, p_t):
    if active_tab != "tab-insights":
        return dash.no_update
    lfc_t = lfc_t or 1.0
    p_t   = p_t   or 0.05
    try:
        sig    = _s2df(sig_rec)
        enr_df = pd.DataFrame(enr_rec) if enr_rec else pd.DataFrame()
        insights = generate_insights(sig, enr_df, lfc_t, p_t)

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


# ── PDF download (publishable quality — v3.8) ────────────────────────────────
@app.callback(
    Output("dl-pdf",       "data"),
    Input("btn-pdf",       "n_clicks"),
    State("volcano",       "figure"),
    State("pca",           "figure"),
    State("bubble",        "figure"),
    State("path-bar",      "figure"),
    State("crosstalk",     "figure"),
    State("heatmap",       "figure"),
    # Additional graphs for full deep-scan PDF
    State("ma-plot",       "figure"),
    State("pval-hist",     "figure"),
    State("lfc-dist",      "figure"),
    State("rank-plot",     "figure"),
    State("gsea-bar",      "figure"),
    State("network",       "figure"),
    State("drug-scatter",  "figure"),
    State("drug-donut",    "figure"),
    State("bm-chart",      "figure"),
    State("meta-bar",      "figure"),
    State("enr-store",     "data"),
    State("bm-store",      "data"),
    State("store",         "data"),
    State("lfc-thresh",    "value"),
    State("p-thresh",      "value"),
    State("sig-store",     "data"),
    # ── Raw session-state extras for Deep Scan data tables ────────────────────
    State("pca-info",           "children"),  # PCA variance % text
    State("gene-input",         "value"),     # Gene Bio-Context query string
    State("chat-thread-store",  "data"),      # AI investigation thread
    prevent_initial_call=True,
)
def cb_pdf(n, vol, pca_f, bubble_f, bar_f, crosstalk_f, heatmap_f,
           ma_f, pval_f, lfc_dist_f, rank_f, gsea_f, net_f,
           drug_scatter_f, drug_donut_f, bm_f, meta_bar_f,
           enr_recs, bm_recs, data_recs, lfc_t, p_t, sig_recs,
           pca_info, gene_query, audit_thread):
    if not n:
        return dash.no_update
    try:
        df     = _s2df(data_recs)
        enr_df = pd.DataFrame(enr_recs or [])
        drug_df = get_drug_targets(
            df.loc[(df["log2FC"].abs() >= lfc_t) & (df["padj"] < p_t),
                   "symbol"].str.upper().tolist()
        )

        up    = int(((df["log2FC"] >  lfc_t) & (df["padj"] < p_t)).sum())
        dn    = int(((df["log2FC"] < -lfc_t) & (df["padj"] < p_t)).sum())
        stats = {"total": len(df), "up": up, "down": dn, "sig": up + dn}

        insights = generate_insights(df, enr_df, lfc_t, p_t)

        # ── AI Discussion (4-layer Deep Analysis) for PDF section 15 ────────────
        ai_discussion = None
        ai_powered_by = ""
        try:
            from modules.ai_summary import get_biological_story
            _pdf_ctx = _build_audit_context(sig_recs, enr_recs, lfc_t, p_t)
            if _pdf_ctx:
                _pdf_ctx["extra"] = (
                    f"Report generated {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC. "
                    + _pdf_ctx.get("extra", "")
                )
                ai_discussion, ai_powered_by = get_biological_story(_pdf_ctx)
        except Exception as ai_exc:
            log.warning(f"pdf ai discussion: {ai_exc}")

        def _fig(f):
            return go.Figure(f) if f else None

        bm_df = pd.DataFrame(bm_recs or [])

        pdf = generate_pdf_report(
            _fig(vol),
            enr_df,
            _fig(pca_f),
            drug_df=drug_df,
            insights=insights,
            summary_stats=stats,
            bubble_fig=_fig(bubble_f),
            bar_fig=_fig(bar_f),
            crosstalk_fig=_fig(crosstalk_f),
            heatmap_fig=_fig(heatmap_f),
            ma_fig=_fig(ma_f),
            pval_fig=_fig(pval_f),
            lfc_dist_fig=_fig(lfc_dist_f),
            rank_fig=_fig(rank_f),
            gsea_fig=_fig(gsea_f),
            network_fig=_fig(net_f),
            drug_scatter_fig=_fig(drug_scatter_f),
            drug_donut_fig=_fig(drug_donut_f),
            bm_fig=_fig(bm_f),
            meta_bar_fig=_fig(meta_bar_f),
            bm_df=bm_df,
            pca_variance_text=pca_info if isinstance(pca_info, str) else None,
            gene_query=gene_query or None,
            ai_discussion=ai_discussion,
            powered_by=ai_powered_by,
            ai_audit_thread=audit_thread or [],
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
# Deep-Analysis Research Consultant callbacks
# ─────────────────────────────────────────────────────────────────────────────

def _build_audit_context(
    sig_rec,
    enr_rec,
    lfc_thresh,
    p_thresh,
    active_tab: str = "",
    pca_variance: str = "",
) -> dict:
    """
    Assemble the rich context_data dict fed to the 4-layer AI prompt.
    active_tab and pca_variance are forwarded so the AI can give
    tab-specific data-grounded interpretation.
    """
    sig_df = _s2df(sig_rec)
    enr_df = pd.DataFrame(enr_rec) if enr_rec else pd.DataFrame()

    if sig_df.empty:
        return {}

    sig_df = sig_df.copy()
    sig_df["_score"] = (
        sig_df["log2FC"].abs()
        * -np.log10(sig_df["padj"].clip(lower=1e-300))
    )

    n_up   = int((sig_df["log2FC"] > 0).sum())
    n_down = int((sig_df["log2FC"] < 0).sum())

    # Top 15 genes by meta-score (AI gets direction via sign)
    top_genes = (
        sig_df.nlargest(15, "_score")[["symbol", "log2FC"]].to_dict("records")
    )

    # Pathway list
    top_paths = (
        enr_df.head(10)["pathway"].tolist()
        if not enr_df.empty and "pathway" in enr_df.columns
        else []
    )

    # Enrichment z-score detail
    enr_details = []
    if not enr_df.empty:
        for _, row in enr_df.head(8).iterrows():
            enr_details.append({
                "pathway":       str(row.get("pathway", ""))[:50],
                "z_score":       row.get("activation_z_score",
                                         row.get("z_score", None)),
                "overlap_count": row.get("overlap_count", "?"),
            })

    # Jaccard gene-overlap pairs
    jaccard_pairs = []
    if not enr_df.empty and "genes" in enr_df.columns and len(enr_df) >= 2:
        gene_sets = []
        for _, row in enr_df.head(6).iterrows():
            gs = {g.strip().upper()
                  for g in str(row.get("genes", "")).split(",") if g.strip()}
            if gs:
                gene_sets.append((str(row.get("pathway", ""))[:35], gs))
        for i, (pa, sa) in enumerate(gene_sets):
            for pb, sb in gene_sets[i + 1:]:
                union = sa | sb
                if union:
                    inter = sa & sb
                    jaccard_pairs.append({
                        "a": pa, "b": pb,
                        "j": len(inter) / len(union),
                        "shared": len(inter),
                    })
        jaccard_pairs.sort(key=lambda x: -x["j"])

    # Drug-target count context
    extra_ctx = f"{n_up} up-regulated, {n_down} down-regulated significant genes."
    try:
        from data.gene_annotations import DRUG_GENE_INTERACTIONS
        sig_syms  = sig_df["symbol"].str.upper().tolist()
        druggable = [g for g in sig_syms if g in DRUG_GENE_INTERACTIONS]
        if druggable:
            extra_ctx += (
                f" {len(druggable)} genes have FDA-approved drug overlays: "
                f"{', '.join(druggable[:5])}."
            )
    except Exception:
        pass

    return {
        "top_genes":     top_genes,
        "top_pathways":  top_paths,
        "jaccard_pairs": jaccard_pairs,
        "enr_details":   enr_details,
        "n_up":          n_up,
        "n_down":        n_down,
        "extra":         extra_ctx,
        "active_tab":    active_tab,
        "pca_variance":  pca_variance,
    }


# ── Open Offcanvas + run deep analysis ────────────────────────────────────────
@app.callback(
    Output("audit-offcanvas",    "is_open"),
    Output("audit-analysis-div", "children"),
    Output("audit-model-badge",  "children"),
    Output("audit-ctx-store",    "data"),
    Output("chat-thread-store",  "data"),   # reset thread on new audit
    Input("btn-audit",           "n_clicks"),
    State("sig-store",   "data"),
    State("enr-store",   "data"),
    State("lfc-thresh",  "value"),
    State("p-thresh",    "value"),
    State("pathway-db",  "value"),
    State("tabs",        "active_tab"),
    State("pca-info",    "children"),   # PCA variance text (populated after PCA tab visit)
    prevent_initial_call=True,
)
def cb_open_audit(n_clicks, sig_rec, enr_rec, lfc_thresh, p_thresh, pathway_db,
                  active_tab, pca_info):
    from modules.ai_summary import get_biological_story_cached, check_available_model

    # Normalise pca_info — it may be a string or a Dash component dict
    pca_variance = pca_info if isinstance(pca_info, str) else ""

    context_data = _build_audit_context(
        sig_rec, enr_rec, lfc_thresh, p_thresh,
        active_tab=active_tab or "",
        pca_variance=pca_variance,
    )

    # Badge
    badge_label, badge_color = check_available_model()
    badge = dbc.Badge(badge_label, color=badge_color, className="me-2",
                      style={"fontSize": 10})

    if not context_data:
        body = dbc.Alert(
            "No significant genes at current thresholds. "
            "Adjust |Log2FC| or p-value cutoffs and try again.",
            color="warning", className="small",
        )
        return True, body, badge, {}, []

    try:
        summary, powered_by, from_cache = get_biological_story_cached(
            context_data,
            lfc_thresh    = float(lfc_thresh  or 1.0),
            p_thresh      = float(p_thresh    or 0.05),
            pathway_db    = str(pathway_db    or "all"),
            force_refresh = False,
        )
    except Exception as exc:
        log.error(f"cb_open_audit: {exc}")
        summary    = f"Analysis error: {exc}"
        powered_by = ""
        from_cache = False

    # Attribution line
    attr_parts = []
    if powered_by and "Rule" not in powered_by and "Offline" not in powered_by:
        attr_parts.append(html.Span(["⚡ ", html.B(powered_by)],
                                    className="me-2 text-success small"))
    elif powered_by:
        attr_parts.append(html.Span(powered_by,
                                    className="me-2 text-muted small"))
    if from_cache:
        attr_parts.append(html.Span(
            "📦 cached (24 h TTL)",
            style={"fontSize": 10, "color": "#888"},
        ))
    badge_row = html.Div([badge] + attr_parts, className="d-flex align-items-center")

    analysis_div = dcc.Markdown(
        summary,
        style={"fontSize": 13, "lineHeight": "1.75"},
    )

    return True, analysis_div, badge_row, context_data, []


# ── Follow-up chat ─────────────────────────────────────────────────────────────
@app.callback(
    Output("chat-thread-store",  "data",  allow_duplicate=True),
    Output("audit-chat-history", "children"),
    Output("chat-input",         "value"),
    Input("chat-submit",         "n_clicks"),
    Input("chat-input",          "n_submit"),
    State("chat-input",          "value"),
    State("chat-thread-store",   "data"),
    State("audit-ctx-store",     "data"),
    prevent_initial_call=True,
)
def cb_chat_followup(n_clicks, n_submit, question, thread, context_data):
    from modules.ai_summary import get_consultation_response

    thread  = thread or []
    question = (question or "").strip()

    if not question:
        return thread, _render_chat_history(thread), ""

    context_data = context_data or {}

    # Append user turn
    thread = thread + [{"role": "user", "content": question}]

    try:
        response, powered_by = get_consultation_response(
            question, thread[:-1], context_data,
        )
    except Exception as exc:
        log.error(f"cb_chat_followup: {exc}")
        response   = f"Error generating response: {exc}"
        powered_by = ""

    # Append assistant turn
    thread = thread + [{"role": "assistant", "content": response}]

    return thread, _render_chat_history(thread), ""


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=7860)
