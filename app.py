import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx, callback, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import io, base64, logging, json, hashlib, time
from datetime import datetime
from collections import Counter
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.io as pio

# Configure Plotly to serve assets locally and avoid external CDN issues
pio.templates.default = "plotly"
plotly_config = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
    'responsive': True,
    'config': {
        'mathjax': None,  # Disable MathJax
        'katex': None     # Disable KaTeX
    }
}

# Configure Plotly to not use external CDN and disable math rendering
import plotly
plotly.io.renderers.default = 'browser'
plotly.offline.init_notebook_mode(connected=False)

# Disable KaTeX math rendering to avoid external CDN calls
plotly_config['staticPlot'] = True

from modules.plots import (
    create_volcano_plot,
    create_heatmap,
    create_pathway_enrichment,
    create_3d_pca,
    create_gsea_bar,
    create_network_graph,
    create_meta_score_bar
)
from modules.analysis import (
    run_gsea_preranked,
    run_pca_3d,
    run_wgcna_lite,
    compute_meta_score
)
from modules.bio_context import fetch_gene_info
from modules.export import generate_pdf_report, compute_data_integrity_score
from modules.session_manager import get_session_manager
from modules.advanced_export import get_exporter
from modules.data_validator import DataValidator, validate_deg_data
from modules.progress_tracker import get_progress_tracker, create_upload_progress_tracker
from modules.batch_analysis import get_batch_analyzer

# ── CONFIGURATION ────────────────────────────────────────────────────────────

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MIN_GENES = 3
MAX_GENES = 50000
REQUIRED_COLUMNS = ['gene_symbol', 'log2_fold_change', 'adjusted_p_value']

# ── ENHANCED DEMO DATA WITH REALISTIC DISTRIBUTIONS ─────────────────────────

np.random.seed(42)

# Generate more realistic DEG data with proper statistical distributions
def generate_demo_data(n_genes=5000):
    """Generate realistic RNA-seq DEG data with proper statistical properties"""

    # Base expression levels (log-normal distribution)
    base_expr = np.random.lognormal(0, 1.5, n_genes)

    # True differentially expressed genes (20% of total)
    n_de = int(0.2 * n_genes)
    de_indices = np.random.choice(n_genes, n_de, replace=False)

    # Generate realistic fold changes
    lfc_true = np.zeros(n_genes)
    lfc_true[de_indices] = np.random.normal(0, 1.5, n_de)
    lfc_true[de_indices] = np.clip(lfc_true[de_indices], -6, 6)

    # Add measurement noise
    lfc_observed = lfc_true + np.random.normal(0, 0.3, n_genes)

    # Generate p-values with proper null distribution
    # For truly DE genes, p-values follow beta distribution
    # For null genes, p-values are uniform
    pvals = np.random.uniform(0, 1, n_genes)

    # Make truly DE genes have lower p-values
    effect_sizes = np.abs(lfc_true)
    for i, effect in enumerate(effect_sizes):
        if effect > 0.5:  # Significant effect
            # Beta distribution for significant p-values
            pvals[i] = np.random.beta(0.5, effect * 2)

    # Apply multiple testing correction (BH method approximation)
    sorted_indices = np.argsort(pvals)
    padj = pvals.copy()
    for i, idx in enumerate(sorted_indices):
        padj[idx] = min(1.0, pvals[idx] * n_genes / (i + 1))

    # Generate gene symbols (mix of real and synthetic)
    real_genes = [
        'TNF', 'IL6', 'IL1B', 'CXCL8', 'CCL2', 'MYD88', 'NFKB1', 'RELA', 'STAT1', 'STAT3',
        'IRF1', 'IRF3', 'IRF7', 'TLR4', 'IRAK1', 'TRAF6', 'MAPK14', 'JUN', 'FOS', 'EGR1',
        'PTGS2', 'MMP9', 'VEGFA', 'HIF1A', 'TP53', 'CDKN1A', 'BCL2', 'MCL1', 'BAX', 'CASP3',
        'IFNG', 'IFNB1', 'MX1', 'OAS1', 'ISG15', 'IFIT1', 'DDX58', 'MAVS', 'STING1', 'CGAS',
        'GAPDH', 'ACTB', 'B2M', 'HLA-A', 'HLA-DRA', 'HLA-DRB1', 'CD74', 'CIITA', 'TAP1', 'TAPBP'
    ]

    synthetic_genes = [f'GENE_{i:04d}' for i in range(len(real_genes), n_genes)]
    gene_symbols = real_genes + synthetic_genes

    return pd.DataFrame({
        'symbol': gene_symbols,
        'log2FC': lfc_observed,
        'padj': padj,
        'pvalue': pvals,
        'baseMean': base_expr
    })

DEMO_DF = generate_demo_data(2000)
DEMO_DF['pvalue'] = DEMO_DF['padj']
DEMO_DF['baseMean'] = np.random.lognormal(6, 2, len(DEMO_DF))

# ── LOGGING & MONITORING ────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# ── UTILITY FUNCTIONS ───────────────────────────────────────────────────────

def create_session_id():
    """Generate unique session identifier"""
    return hashlib.md5(f"{time.time()}_{np.random.randint(1000000)}".encode()).hexdigest()[:8]

# ── APP INIT ─────────────────────────────────────────────────────────────────

app = dash.Dash(__name__,
                external_stylesheets=[],
                external_scripts=[],
                suppress_callback_exceptions=True,
                serve_locally=True,  # Serve assets locally instead of CDN
                meta_tags=[
                    {"name": "viewport", "content": "width=device-width, initial-scale=1"},
                    {"name": "description", "content": "Advanced RNA-seq Pathway Analysis Tool"},
                    {"property": "og:title", "content": "RNA-seq Pathway Visualizer"},
                    {"property": "og:description", "content": "Interactive differential expression and pathway enrichment analysis"},
                    {"property": "og:type", "content": "website"}
                ])

server = app.server

@server.after_request
def set_security_headers(response):
    # Allow cross-origin access for Hugging Face Spaces
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    
    # Only list currently recognised Permissions-Policy features to silence
    # "Unrecognized feature" browser warnings caused by deprecated names.
    response.headers['Permissions-Policy'] = (
        'camera=(), microphone=(), geolocation=(), payment=(), '
        'accelerometer=(), gyroscope=(), magnetometer=(), '
        'usb=(), bluetooth=(), ambient-light-sensor=(), '
        'document-domain=(), layout-animations=(), legacy-image-formats=(), '
        'oversized-images=(), vr=(), wake-lock=()'
    )
    # Allow cross-origin storage so Edge Tracking Prevention doesn't block
    # the Dash session cookie when the app is embedded in an iframe.
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin-allow-popups'
    response.headers['Cross-Origin-Embedder-Policy'] = 'credentialless'
    
    # Prevent external script injection
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: https:; "
        "connect-src 'self' https://plotly.com;"
    )
    return response

# Enhanced CSS with accessibility and modern styling
app.index_string = app.index_string.replace(
    '</head>',
    '<style>'
    '.js-plotly-plot .plotly .modebar{z-index:1001!important;pointer-events:all!important;}'
    '.upload-zone:hover{background:#f0f7ff!important;border-color:#4dabf7!important;}'
    '.upload-zone.dragover{background:#e3f2fd!important;border-color:#2196f3!important;}'
    '::-webkit-scrollbar {width: 6px;height: 6px;}'
    '::-webkit-scrollbar-track {background: #f1f1f1;border-radius: 3px;}'
    '::-webkit-scrollbar-thumb {background: #b0bec5;border-radius: 3px;}'
    '::-webkit-scrollbar-thumb:hover {background: #78909c;}'
    '.quality-indicator {font-size: 11px; padding: 2px 6px; border-radius: 3px;}'
    '.quality-good {background: #e8f5e8; color: #2e7d32;}'
    '.quality-warning {background: #fff3e0; color: #ef6c00;}'
    '.quality-error {background: #ffebee; color: #c62828;}'
    '.help-tooltip {cursor: help; border-bottom: 1px dotted #666;}'
    '.export-btn {transition: all 0.2s ease;}'
    '.export-btn:hover {transform: translateY(-1px); box-shadow: 0 2px 4px rgba(0,0,0,0.2);}'
    '@media (max-width: 768px) {.card-body {padding: 1rem;}}'
    '@media (prefers-reduced-motion: reduce) {* {animation-duration: 0.01ms !important;}}'
    '</style></head>'
)

# ── LAYOUT ───────────────────────────────────────────────────────────────────

app.layout = dbc.Container([

    # ── HEADER ──────────────────────────────────────────
    dbc.Row([dbc.Col([
        html.Div([
            html.H2("🧬 Advanced RNA-Seq Pathway Visualizer",
                    style={'color':'white','marginBottom':'4px',
                           'fontSize':'20px','fontWeight':'700'}),
            html.P("Interactive Differential Expression ↔ Pathway Analysis  |  "
                   "Industry-Grade Analysis Suite",
                   style={'color':'rgba(255,255,255,0.82)',
                          'marginBottom':0,'fontSize':'11px'}),
            html.Div([
                html.Span("v3.0.0", style={'fontSize':'10px','color':'rgba(255,255,255,0.6)'}),
                html.Span(" • ", style={'margin':'0 8px','color':'rgba(255,255,255,0.4)'}),
                html.Span("Real-time statistical validation", style={'fontSize':'10px','color':'rgba(255,255,255,0.6)'})
            ], style={'marginTop':'4px'})
        ], style={
            'background':'linear-gradient(135deg,#1a237e,#1565c0,#0277bd)',
            'padding':'16px 24px',
            'borderRadius':'8px',
            'marginBottom':'12px',
            'position':'relative',
            'overflow':'hidden'
        })
    ], width=12)]),

    # ── ALERT BANNER ────────────────────────────────────
    dbc.Row([dbc.Col([
        dbc.Alert(id='main-alert', is_open=False, dismissable=True,
                  duration=8000, style={'fontSize': '13px'})
    ], width=12)]),

    # ── UPLOAD VALIDATION ALERT ─────────────────────────
    dbc.Row([dbc.Col([
        dbc.Alert(id='upload-validation-alert', is_open=False, dismissable=True,
                  duration=10000, style={'fontSize': '12px'})
    ], width=12)]),

    dbc.Row([dbc.Col([
        html.Div(id='data-quality-panel', style={'marginBottom':'12px'})
    ], width=12)]),

    # ── PROGRESS INDICATOR ──────────────────────────────
    dbc.Row([dbc.Col([
        dcc.Loading(
            type='default',
            color='#1565c0',
            children=html.Div(id='progress-indicator-panel', style={
                'marginBottom': '12px',
                'padding': '8px',
                'backgroundColor': '#f5f5f5',
                'borderRadius': '4px',
                'minHeight': '0px'
            })
        )
    ], width=12)]),

    # ── ENHANCED CONTROLS ROW ───────────────────────────
    dbc.Row([dbc.Col([
        dbc.Card([dbc.CardBody([
            html.Div([

                # Upload Section
                html.Div([
                    html.Label([
                        "Upload DEG CSV ",
                        html.I(className="fas fa-info-circle help-tooltip",
                               title="Upload CSV with columns: gene_symbol, log2_fold_change, adjusted_p_value")
                    ], style={'fontWeight': '600', 'fontSize': '12px', 'marginBottom': '4px'}),
                    dcc.Upload(
                        id='main-upload-comp',
                        children=html.Div([
                            html.I(className="fas fa-cloud-upload-alt",
                                   style={'fontSize':'24px','color':'#6c757d','display':'block','marginBottom':'8px'}),
                            html.Span('Drag & Drop or ', style={'fontWeight':'500'}),
                            html.A('Select File', style={'color':'#007bff','textDecoration':'underline'})
                        ]),
                        style={
                            'width': '280px', 'height': '60px', 'lineHeight': '60px',
                            'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '8px',
                            'textAlign': 'center', 'borderColor': '#adb5bd', 'fontSize': '12px',
                            'cursor': 'pointer', 'backgroundColor': '#f8f9fa',
                            'transition': 'all 0.2s ease'
                        },
                        max_size=MAX_FILE_SIZE,
                        multiple=False
                    ),
                    html.Div([
                        html.Small("Max 10MB • CSV format required",
                                   style={'color':'#6c757d','fontSize':'10px'})
                    ], style={'marginTop':'4px'})
                ], style={'flexShrink': '1', 'minWidth': '120px'}),

                # Demo Button
                html.Div([
                    html.Label("Quick Start", style={'display': 'block', 'marginBottom': '4px'}),
                    html.Div([
                        dbc.Button([
                            html.I(className="fas fa-play", style={'marginRight':'6px'}),
                            "Load Demo Data"
                        ], id='load-demo-btn', color="outline-primary",
                           size="sm", style={'height': '44px', 'fontSize': '12px', 'marginRight': '4px'}),
                        dbc.Button([
                            html.I(className="fas fa-layer-group", style={'marginRight':'6px'}),
                            "Batch Mode"
                        ], id='toggle-batch-mode-btn', color="outline-secondary",
                           size="sm", style={'height': '44px', 'fontSize': '12px'})
                    ], style={'display':'flex','gap':'4px'})
                ], style={'flexShrink': '1', 'minWidth': '200px'}),

                # Statistical Parameters
                html.Div([
                    html.Label([
                        "Log2FC Threshold ",
                        html.I(className="fas fa-info-circle help-tooltip",
                               title="Minimum absolute fold change for significance")
                    ], style={'fontWeight': '600', 'fontSize': '12px', 'marginBottom': '4px'}),
                    dcc.Slider(id='lfc-thresh-slider', min=0.0, max=3.0, step=0.1,
                               value=1.0, marks={
                                   0.0: '0.0', 0.5: '0.5', 1.0: '1.0', 1.5: '1.5',
                                   2.0: '2.0', 2.5: '2.5', 3.0: '3.0'
                               },
                               tooltip={'placement': 'bottom', 'always_visible': False}),
                ], style={'width': '220px', 'flexShrink': '1', 'minWidth': '120px'}),

                # Multiple Testing Correction
                html.Div([
                    html.Label([
                        "Significance Level ",
                        html.I(className="fas fa-info-circle help-tooltip",
                               title="Adjusted p-value threshold (BH correction applied)")
                    ], style={'fontWeight': '600', 'fontSize': '12px', 'marginBottom': '4px'}),
                    dcc.Dropdown(id='padj-thresh-dd',
                                 options=[
                                     {'label': '0.001 (Stringent)', 'value': 0.001},
                                     {'label': '0.01 (Moderate)', 'value': 0.01},
                                     {'label': '0.05 (Standard)', 'value': 0.05},
                                     {'label': '0.1 (Relaxed)', 'value': 0.1}
                                 ],
                                 value=0.05, clearable=False,
                                 style={'width': '160px', 'fontSize': '12px'})
                ], style={'flexShrink': '1', 'minWidth': '120px'}),

                # Analysis Options
                html.Div([
                    html.Label("Analysis Options", style={'fontWeight': '600', 'fontSize': '12px', 'marginBottom': '4px'}),
                    dbc.Checklist(
                        id='analysis-options',
                        options=[
                            {"label": "Show gene labels", "value": "show_labels"},
                            {"label": "Auto-scale plots", "value": "auto_scale"},
                            {"label": "Statistical validation", "value": "stat_validation"}
                        ],
                        value=["show_labels", "stat_validation"],
                        inline=True,
                        style={'fontSize': '11px'}
                    )
                ], style={'flexShrink': '1', 'minWidth': '120px'}),

                # Action Buttons
                html.Div([
                    html.Label("Actions", style={'display': 'block', 'marginBottom': '4px'}),
                    html.Div([
                        dbc.Button([
                            html.I(className="fas fa-redo", style={'marginRight':'4px'}),
                            "Reset"
                        ], id='reset-btn', color="outline-secondary", size="sm",
                           style={'height': '32px', 'fontSize': '11px', 'marginRight':'6px'}),
                        dbc.Button([
                            html.I(className="fas fa-download", style={'marginRight':'4px'}),
                            "Export"
                        ], id='export-btn', color="outline-success", size="sm",
                           className="export-btn", style={'height': '32px', 'fontSize': '11px', 'cursor': 'pointer'})
                    ], style={'display':'flex','gap':'4px'})
                ], style={'flexShrink': '1', 'minWidth': '120px'}),

                # Session Management Buttons
                html.Div([
                    html.Label("Session", style={'display': 'block', 'marginBottom': '4px'}),
                    html.Div([
                        dbc.Button([
                            html.I(className="fas fa-save", style={'marginRight':'4px'}),
                            "Save"
                        ], id='save-session-btn', color="outline-info", size="sm",
                           style={'height': '32px', 'fontSize': '11px', 'marginRight':'6px'}),
                        dbc.Button([
                            html.I(className="fas fa-folder-open", style={'marginRight':'4px'}),
                            "Load"
                        ], id='load-session-btn', color="outline-info", size="sm",
                           style={'height': '32px', 'fontSize': '11px'})
                    ], style={'display':'flex','gap':'4px'})
                ], style={'flexShrink': '1', 'minWidth': '120px'}),

            ], style={
                'display': 'flex',
                'gap': '16px',
                'alignItems': 'flex-end',
                'flexWrap': 'wrap',
                'width': '100%'
            })
        ])], className="shadow-sm mb-3")
    ], width=12)]),

    # ── ONBOARDING & HELP ───────────────────────────────
    dbc.Row([dbc.Col([
        dbc.Collapse([
            dbc.Alert([
                html.Div([
                    html.H6("🚀 Quick Start Guide", style={'marginBottom':'8px','color':'#2e7d32'}),
                    html.Div([
                        html.Div([
                            html.Span("1️⃣", style={'fontSize':'16px','marginRight':'8px'}),
                            html.Strong("Load Data: "),
                            "Upload your DEG CSV or click 'Load Demo Data'"
                        ], style={'marginBottom':'4px'}),
                        html.Div([
                            html.Span("2️⃣", style={'fontSize':'16px','marginRight':'8px'}),
                            html.Strong("Explore: "),
                            "Use lasso selection on volcano plot to select genes"
                        ], style={'marginBottom':'4px'}),
                        html.Div([
                            html.Span("3️⃣", style={'fontSize':'16px','marginRight':'8px'}),
                            html.Strong("Analyze: "),
                            "Pathway enrichment runs automatically"
                        ], style={'marginBottom':'4px'}),
                        html.Div([
                            html.Span("4️⃣", style={'fontSize':'16px','marginRight':'8px'}),
                            html.Strong("Navigate: "),
                            "Click pathway bubbles to highlight genes"
                        ], style={'marginBottom':'4px'}),
                    ], style={'fontSize':'12px','lineHeight':'1.4'})
                ], style={'display':'flex','gap':'16px'})
            ], color="success", style={'fontSize': '12px', 'padding': '12px 16px'})
        ], id='help-panel', is_open=True)
    ], width=12)]),

    # ── BATCH MODE SECTION ──────────────────────────────
    dbc.Row([dbc.Col([
        dbc.Collapse([
            dbc.Card([
                dbc.CardBody([
                    html.H6("📦 Batch Processing Mode", style={'marginBottom': '12px', 'color': '#1a237e'}),
                    html.P("Upload and analyze multiple DEG files simultaneously. Compare gene signatures across samples.",
                          style={'fontSize': '12px', 'marginBottom': '12px', 'color': '#666'}),
                    html.Div([
                        html.Div([
                            html.Label("Upload Multiple Files", style={'fontWeight': '600', 'fontSize': '12px', 'marginBottom': '6px', 'display': 'block'}),
                            dcc.Upload(
                                id='batch-file-upload',
                                children=html.Div([
                                    html.I(className="fas fa-folder-open", style={'fontSize': '20px', 'color': '#666', 'display': 'block', 'marginBottom': '6px'}),
                                    'Drag & Drop or Select Multiple CSV Files'
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '80px',
                                    'lineHeight': '80px',
                                    'borderWidth': '2px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '6px',
                                    'textAlign': 'center',
                                    'borderColor': '#adb5bd',
                                    'fontSize': '12px',
                                    'cursor': 'pointer',
                                    'backgroundColor': '#f8f9fa'
                                },
                                max_size=50*1024*1024,  # 50MB total for batch
                                multiple=True
                            ),
                        ], style={'flexGrow': '1', 'marginRight': '12px'}),
                        html.Div([
                            dbc.Button([
                                html.I(className="fas fa-cog", style={'marginRight': '6px'}),
                                "Process Batch"
                            ], id='process-batch-btn', color="primary", className='mt-4', style={'width': '100%'}),
                            html.Div(id='batch-process-status', style={'marginTop': '8px', 'fontSize': '11px'})
                        ], style={'min Width': '150px'})
                    ], style={'display': 'flex', 'gap': '12px'})
                ])
            ], className="shadow-sm")
        ], id='batch-mode-section', is_open=False)
    ], width=12)]),

    # ── MAIN ANALYSIS PANELS ────────────────────────────
    dbc.Row([

        # LEFT: Volcano + Heatmap (vertical stack)
        dbc.Col([
            dbc.Card([dbc.CardBody([
                html.Div([
                    html.Div([
                        html.Span("🌋 Volcano Plot", style={'fontWeight':'600','fontSize':'14px',
                                  'color':'#1a237e','marginBottom':'4px','display':'block'}),
                        html.Small("Interactive differential expression analysis • Lasso select genes",
                                   style={'color':'#666','fontSize':'11px'}),
                        html.Div(id='volcano-stats', style={'marginTop':'4px'})
                    ], style={'marginBottom':'8px'}),

                    dcc.Graph(
                        id='volcano-graph-obj',
                        config={
                            'displayModeBar': True,
                            'scrollZoom': True,
                            'responsive': True,
                            'modeBarButtonsToAdd': ['lasso2d', 'select2d', 'pan2d', 'zoom2d', 'resetScale2d'],
                            'toImageButtonOptions': {
                                'format': 'png',
                                'filename': f'volcano_plot_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                                'height': 800,
                                'width': 1200,
                                'scale': 2
                            }
                        },
                        style={'height': '450px', 'width': '100%'}
                    ),

                    html.Hr(style={'margin':'16px 0','borderColor':'#e9ecef'}),

                    html.Div([
                        html.Span("🟥 Expression Heatmap", style={'fontWeight':'600','fontSize':'14px',
                                  'color':'#1a237e','marginBottom':'4px','display':'block'}),
                        html.Small("Fold-change profile of selected genes • Red=upregulated, Blue=downregulated",
                                   style={'color':'#666','fontSize':'11px'}),
                        html.Div(id='heatmap-stats', style={'marginTop':'4px'})
                    ], style={'marginBottom':'8px'}),

                    dcc.Graph(
                        id='gene-heatmap-obj',
                        config={
                            'displayModeBar': True,
                            'responsive': True,
                            'toImageButtonOptions': {
                                'format': 'png',
                                'filename': f'expression_heatmap_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                                'height': 600,
                                'width': 800,
                                'scale': 2
                            }
                        },
                        style={'height': '360px', 'width': '100%'}
                    ),
                ])
            ])], className="shadow-sm")
        ], xs=12, sm=12, md=12, lg=7, xl=7),

        # RIGHT: Pathway enrichment
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.Div([
                    html.Span("🧪 Pathway Enrichment Analysis", style={'fontWeight': '700', 'fontSize': '15px'}),
                    html.Small(" • KEGG + Reactome • Fisher exact + BH correction • Real-time validation",
                               style={'color': '#6c757d', 'fontSize': '11px'})
                ])),
                dbc.CardBody([
                    dcc.Loading(
                        type='circle',
                        color='#1565c0',
                        children=[
                            dcc.Graph(
                                id='pathway-bubble-obj',
                                config={
                                    "displayModeBar": True,
                                    "responsive": True,
                                    "toImageButtonOptions": {
                                        'format': 'png',
                                        'filename': f'pathway_enrichment_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                                        'height': 700,
                                        'width': 900,
                                        'scale': 2
                                    }
                                },
                                style={'height': '450px', 'width': '100%'}
                            ),
                            html.Div(id='enrichment-table-container', className="mt-3")
                        ]
                    )
                ])
            ], className="shadow-sm")
        ], xs=12, sm=12, md=12, lg=5, xl=5)
    ], className="mb-3"),

    # ── ADVANCED ANALYSIS SUMMARY ───────────────────────
    dbc.Row([dbc.Col([
        dbc.Card([dbc.CardBody([
            html.Div(id='summary-panel',
                     children=html.P("Load data to see comprehensive analysis summary.",
                                     style={'color': '#aaa', 'fontStyle': 'italic', 'margin': 0}))
        ])], className="shadow-sm")
    ], width=12)], className="mb-4"),

    # ── ADVANCED MULTI-OMICS ANALYTICS ─────────────────
    dbc.Row([dbc.Col([
        dbc.Card([
            dbc.CardHeader(html.Div([
                html.Span("🧬 Advanced Multi-Omics Analytics", style={'fontWeight': '700', 'fontSize': '15px'}),
                html.Small(" • 3D PCA • WGCNA-lite • Preranked GSEA • Gene metadata • Reporting",
                           style={'color': '#6c757d', 'fontSize': '11px'})
            ])),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Span("3D PCA Projection", style={'fontWeight': '600', 'fontSize': '13px'}),
                            html.Small("Clusters inferred from expression signature similarity.",
                                       style={'color': '#666', 'fontSize': '11px'})
                        ], style={'marginBottom':'6px'}),
                        dcc.Loading(
                            type='circle',
                            color='#1565c0',
                            children=dcc.Graph(
                                id='pca-3d-graph',
                                config={'displayModeBar': True, 'responsive': True},
                                style={'height': '360px', 'width': '100%'}
                            )
                        )
                    ], xs=12, sm=12, md=12, lg=4, xl=4),
                    dbc.Col([
                        html.Div([
                            html.Span("Preranked GSEA", style={'fontWeight': '600', 'fontSize': '13px'}),
                            html.Small("Rank genes by effect size and p-value for pathway discovery.",
                                       style={'color': '#666', 'fontSize': '11px'})
                        ], style={'marginBottom':'6px'}),
                        dcc.Loading(
                            type='circle',
                            color='#1565c0',
                            children=[
                                dcc.Graph(
                                    id='gsea-bar-graph',
                                    config={'displayModeBar': True, 'responsive': True},
                                    style={'height': '360px', 'width': '100%'}
                                ),
                                html.Div(id='gsea-summary', style={'marginTop': '8px', 'fontSize': '12px', 'color': '#444'})
                            ]
                        )
                    ], xs=12, sm=12, md=12, lg=4, xl=4),
                    dbc.Col([
                        html.Div([
                            html.Span("Co-expression Network", style={'fontWeight': '600', 'fontSize': '13px'}),
                            html.Small("WGCNA-lite module detection from differential expression signal.",
                                       style={'color': '#666', 'fontSize': '11px'})
                        ], style={'marginBottom':'6px'}),
                        dcc.Loading(
                            type='circle',
                            color='#1565c0',
                            children=dcc.Graph(
                                id='network-graph',
                                config={'displayModeBar': True, 'responsive': True},
                                style={'height': '360px', 'width': '100%'}
                            )
                        )
                    ], xs=12, sm=12, md=12, lg=4, xl=4)
                ]),
                html.Hr(style={'margin':'16px 0','borderColor':'#e9ecef'}),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Span("Meta-Score Prioritisation", style={'fontWeight': '600', 'fontSize': '13px'}),
                            html.Small("Composite gene ranking combining effect size and significance.",
                                       style={'color': '#666', 'fontSize': '11px'})
                        ], style={'marginBottom':'6px'}),
                        dcc.Graph(
                            id='meta-score-bar',
                            config={'displayModeBar': True, 'responsive': True},
                            style={'height': '320px', 'width': '100%'}
                        )
                    ], xs=12, sm=12, md=12, lg=8, xl=8),
                    dbc.Col([
                        html.Div([
                            html.Span("Gene Bio-Context Lookup", style={'fontWeight': '600', 'fontSize': '13px'}),
                            html.Small("Fetch NCBI/UniProt metadata for selected gene symbols.",
                                       style={'color': '#666', 'fontSize': '11px'})
                        ], style={'marginBottom':'6px'}),
                        dcc.Input(id='gene-search-input', type='text', placeholder='Enter gene symbol', style={'width': '100%', 'marginBottom': '6px'}),
                        dbc.Button('Lookup Gene Info', id='lookup-gene-btn', color='primary', size='sm', className='mb-2'),
                        html.Div(id='gene-info-panel', style={'fontSize': '12px', 'color': '#333', 'lineHeight': '1.5'})
                    ], xs=12, sm=12, md=12, lg=4, xl=4)
                ]),
                html.Div([
                    dbc.Button('Download Analysis Report', id='download-pdf-btn', color='secondary', className='export-btn', n_clicks=0),
                    html.Span(id='integrity-display', style={'marginLeft': '12px', 'fontSize': '12px', 'color': '#444'})
                ], style={'marginTop': '12px'})
            ])
        ], className='shadow-sm')
    ], width=12)
], className='mb-4'),

    # ── SESSION INFO & EXPORT ───────────────────────────
    dbc.Row([dbc.Col([
        html.Div(id='session-info', style={'fontSize':'11px','color':'#666','textAlign':'center','marginBottom':'8px'})
    ], width=12)]),

    # ── ENHANCED STORES ────────────────────────────────
    dcc.Store(id='dge-data-store'),
    dcc.Store(id='gsea-results-store', data={}),
    dcc.Store(id='volcano-filter-genes', data=[]),
    dcc.Store(id='session-store', data={'session_id': create_session_id(), 'start_time': datetime.now().isoformat()}),
    dcc.Store(id='analysis-metadata', data={}),
    dcc.Store(id='gsea-store', data={}),
    dcc.Store(id='pca-store', data={}),
    dcc.Store(id='network-store', data={}),
    dcc.Store(id='integrity-store', data={}),
    dcc.Store(id='ora-params-store', data={}),
    dcc.Download(id='download-results'),
    
    # ── SESSION MANAGEMENT COMPONENTS ────────────────────
    dcc.Store(id='persistent-session-store', storage_type='local'),
    dcc.Download(id='download-session'),
    html.Div(id='session-modal-container'),
    
    # Save Session Modal
    dbc.Modal([
        dbc.ModalHeader("Save Analysis Session"),
        dbc.ModalBody([
            dbc.Form([
                html.Div([
                    dbc.Label("Session Name", html_for="session-name-input"),
                    dbc.Input(id='session-name-input', type='text', placeholder='e.g., PBMC_LPS_Analysis_v1')
                ], className='mb-3'),
                html.Div([
                    dbc.Label("Description (optional)", html_for="session-desc-input"),
                    dbc.Textarea(id='session-desc-input', placeholder='Add notes about this analysis...', rows=3)
                ], className='mb-3')
            ])
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="save-session-cancel", className="me-2"),
            dbc.Button("Save Session", id="save-session-confirm", color="primary")
        ])
    ], id="save-session-modal", is_open=False),
    
    # Load Session Modal
    dbc.Modal([
        dbc.ModalHeader("Load Analysis Session"),
        dbc.ModalBody([
            html.Div(id='sessions-list-container')
        ]),
        dbc.ModalFooter([
            dbc.Button("Close", id="load-session-cancel", className="me-2")
        ])
    ], id="load-session-modal", is_open=False, size="lg"),
    
    # Export Options Modal
    dbc.Modal([
        dbc.ModalHeader("Export Analysis Results"),
        dbc.ModalBody([
            html.Div([
                html.H6("Choose Export Format:", style={'marginBottom': '12px'}),
                dbc.ButtonGroup([
                    dbc.Button(
                        [html.I(className="fas fa-file-excel", style={'marginRight':'6px'}), "Excel"],
                        id='export-excel-btn', color='info', outline=True
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-code", style={'marginRight':'6px'}), "JSON"],
                        id='export-json-btn', color='info', outline=True
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-file-csv", style={'marginRight':'6px'}), "CSV"],
                        id='export-csv-btn', color='info', outline=True
                    ),
                ], style={'width': '100%', 'marginBottom': '12px'}, vertical=False),
                html.Hr(),
                html.P("Excel: Multi-sheet workbook with data, enrichment, summary, parameters", 
                       style={'fontSize': '11px', 'color': '#666', 'marginBottom': '4px'}),
                html.P("JSON: Complete analysis export with metadata for programmatic access", 
                       style={'fontSize': '11px', 'color': '#666', 'marginBottom': '4px'}),
                html.P("CSV: Separate files for gene data and enrichment results", 
                       style={'fontSize': '11px', 'color': '#666', 'marginBottom': '12px'}),
                html.Div(id='export-status-message', style={'fontSize': '11px', 'marginTop': '12px'})
            ])
        ]),
        dbc.ModalFooter([
            dbc.Button("Close", id="export-modal-close", className="me-2"),
        ], style={'justifyContent': 'flex-start'})
    ], id="export-options-modal", is_open=False),
    
    # Batch Comparison Results Modal
    dbc.Modal([
        dbc.ModalHeader("Batch Analysis Results"),
        dbc.ModalBody([
            dbc.Tabs([
                dbc.Tab([
                    html.Div(id='batch-comparison-table', style={'marginTop': '12px'})
                ], label="Comparison Summary"),
                dbc.Tab([
                    html.Div(id='batch-overlap-analysis', style={'marginTop': '12px'})
                ], label="Gene Overlaps"),
                dbc.Tab([
                    html.Div(id='batch-errors-panel', style={'marginTop': '12px'})
                ], label="Processing Status")
            ])
        ], style={'maxHeight': '70vh', 'overflowY': 'auto'}),
        dbc.ModalFooter([
            dbc.Button("Export Results", id="batch-export-btn", color="success", className="me-2"),
            dbc.Button("Close", id="batch-results-close")
        ])
    ], id="batch-results-modal", is_open=False, size="xl"),
    
    # Download components
    dcc.Download(id='download-excel'),
    dcc.Download(id='download-json'),
    dcc.Download(id='download-enrich-csv'),
    
    # Progress tracking
    dcc.Interval(id='progress-update-interval', interval=1000, n_intervals=0, disabled=True),  # Disabled by default, update every second
    dcc.Store(id='progress-active-store', data={}),

], fluid=True)

# ── CALLBACK 1: Load data (upload OR demo button) ────────────────────────────

@app.callback(
    Output('dge-data-store', 'data'),
    Output('volcano-graph-obj', 'figure'),
    Output('summary-panel', 'children'),
    Output('main-alert', 'children'),
    Output('main-alert', 'color'),
    Output('main-alert', 'is_open'),
    Output('help-panel', 'is_open'),
    Output('upload-validation-alert', 'children'),
    Output('upload-validation-alert', 'color'),
    Output('upload-validation-alert', 'is_open'),
    Input('main-upload-comp', 'contents'),
    Input('load-demo-btn', 'n_clicks'),
    Input('lfc-thresh-slider', 'value'),
    Input('padj-thresh-dd', 'value'),
    Input('volcano-filter-genes', 'data'),
    State('main-upload-comp', 'filename'),
    State('dge-data-store', 'data'),
    prevent_initial_call=False
)
def update_data_and_volcano(contents, n_clicks, lfc_thresh, p_thresh,
                             filtered_genes, filename, existing_data):

    triggered = ctx.triggered_id

    df = None
    validation_msg = dash.no_update
    validation_color = dash.no_update
    validation_open = False

    if triggered == 'load-demo-btn' and n_clicks:
        df = DEMO_DF.copy()

    elif triggered == 'main-upload-comp' and contents is not None:
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            temp_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            # Use enhanced validation
            df, errors, warnings, html_summary = validate_deg_data(temp_df)

            if errors:
                empty_fig = create_volcano_plot(pd.DataFrame(columns=['log2_fold_change', 'adjusted_p_value', 'gene_symbol']),
                                                'log2_fold_change', 'adjusted_p_value', 'gene_symbol')
                return (None, empty_fig, dash.no_update, 
                        dash.no_update, dash.no_update, False, True,
                        html_summary, 'danger', True)

            if warnings or html_summary:
                logger.warning(f"Data validation: {len(warnings)} warnings")
                # Show validation details as warning alert, but continue loading data
                validation_msg = html_summary
                validation_color = 'warning'
                validation_open = True
            else:
                validation_msg = dash.no_update
                validation_color = dash.no_update
                validation_open = False

            # Rename columns to match expected format
            df = df.rename(columns={
                'gene_symbol': 'symbol',
                'log2_fold_change': 'log2FC',
                'adjusted_p_value': 'padj'
            })

        except Exception as e:
            logger.error(f"Upload error: {e}")
            msg = f"Could not parse file: {str(e)}. Ensure it is a comma-separated CSV with required columns."
            return (None, dash.no_update, dash.no_update, 
                    msg, 'danger', True, True,
                    dash.no_update, dash.no_update, False)

    elif existing_data and triggered in ('lfc-thresh-slider', 'padj-thresh-dd', 'volcano-filter-genes'):
        df = pd.DataFrame(existing_data)

    else:
        empty_fig = create_volcano_plot(pd.DataFrame(columns=['log2FC', 'padj', 'symbol']),
                                        'log2FC', 'padj', 'symbol')
        return (None, empty_fig, dash.no_update, 
                dash.no_update, dash.no_update, False, True,
                dash.no_update, dash.no_update, False)

    fig = create_volcano_plot(df, 'log2FC', 'padj', 'symbol',
                               lfc_thresh=lfc_thresh, p_thresh=p_thresh,
                               filtered_genes=filtered_genes if filtered_genes else None)

    n_total = len(df)
    n_up   = int(((df['log2FC'] > lfc_thresh) & (df['padj'] < p_thresh)).sum())
    n_down = int(((df['log2FC'] < -lfc_thresh) & (df['padj'] < p_thresh)).sum())
    n_ns   = n_total - n_up - n_down

    top_up   = df[df['log2FC'] > 0].nsmallest(1, 'padj')
    top_down = df[df['log2FC'] < 0].nsmallest(1, 'padj')
    up_str   = f"{top_up.iloc[0]['symbol']} (log2FC={top_up.iloc[0]['log2FC']:.2f})"     if len(top_up)   else "N/A"
    down_str = f"{top_down.iloc[0]['symbol']} (log2FC={top_down.iloc[0]['log2FC']:.2f})" if len(top_down) else "N/A"

    summary = html.Div([
        html.H6("📊 Comprehensive Analysis Summary", style={'marginBottom': '12px', 'fontWeight': '700', 'color': '#1a237e'}),
        html.Div([
            html.Div([
                html.Strong("Dataset Overview", style={'fontSize': '13px', 'display': 'block', 'marginBottom': '6px'}),
                html.P(f"Total genes analyzed: {n_total:,}", style={'margin': '2px 0', 'fontSize': '12px'}),
                html.P(f"Significantly upregulated: {n_up:,} genes", style={'margin': '2px 0', 'fontSize': '12px', 'color': '#d32f2f'}),
                html.P(f"Significantly downregulated: {n_down:,} genes", style={'margin': '2px 0', 'fontSize': '12px', 'color': '#1976d2'}),
                html.P(f"Not significant: {n_ns:,} genes", style={'margin': '2px 0', 'fontSize': '12px', 'color': '#666'}),
            ], style={'flex': '1', 'minWidth': '200px'}),
            html.Div([
                html.Strong("Top Genes", style={'fontSize': '13px', 'display': 'block', 'marginBottom': '6px'}),
                html.P(f"Most upregulated: {up_str}", style={'margin': '2px 0', 'fontSize': '12px'}),
                html.P(f"Most downregulated: {down_str}", style={'margin': '2px 0', 'fontSize': '12px'}),
                html.P(f"Current thresholds: |log2FC| > {lfc_thresh}, p-adj < {p_thresh}",
                       style={'margin': '2px 0', 'fontSize': '11px', 'color': '#666'}),
            ], style={'flex': '1', 'minWidth': '200px'}),
        ], style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap', 'marginBottom': '12px'}),
        html.Hr(style={'margin': '8px 0', 'borderColor': '#e9ecef'}),
        html.Div([
            html.Small("💡 Tip: Use lasso selection on the volcano plot to identify pathway-enriched gene sets",
                       style={'color': '#666', 'fontSize': '11px'})
        ])
    ])

    success_msg = (f"\u2713 Loaded {n_total} genes \u2014 {n_up} up, {n_down} down at "
                   f"log2FC>{lfc_thresh}, padj<{p_thresh}. "
                   f"Use lasso selection on the volcano plot to explore pathways.")

    return (df.to_dict('records'), fig, summary,
            success_msg, 'success', True, False,
            validation_msg, validation_color, validation_open)


# ── CALLBACK 2: Lasso selection → pathway enrichment + heatmap ───────────────

@app.callback(
    Output('pathway-bubble-obj', 'figure'),
    Output('gene-heatmap-obj', 'figure'),
    Output('enrichment-table-container', 'children'),
    Output('gsea-results-store', 'data'),
    Input('volcano-graph-obj', 'selectedData'),
    State('dge-data-store', 'data'),
    prevent_initial_call=True
)
def run_enrichment_on_selection(selectedData, stored_data):
    empty_pathway = go.Figure().update_layout(
        title="\u2196 Use lasso selection on the volcano plot to run pathway enrichment",
        template='simple_white'
    )
    empty_heatmap = create_heatmap(pd.DataFrame(), [])
    empty_msg = html.P("Select genes to see enrichment results.",
                       style={'color': '#888', 'fontStyle': 'italic', 'padding': '8px'})

    if not selectedData or not selectedData.get('points') or not stored_data:
        return empty_pathway, empty_heatmap, empty_msg, {}

    selected_genes = list(dict.fromkeys(
        [p['customdata'] for p in selectedData['points'] if 'customdata' in p]
    ))

    pathway_fig, table, gene_index_dict = create_pathway_enrichment(selected_genes)
    heatmap_fig = create_heatmap(pd.DataFrame(stored_data), selected_genes)

    return pathway_fig, heatmap_fig, table, gene_index_dict


# ── CALLBACK 3: Pathway bubble click → highlight genes on volcano ─────────────

@app.callback(
    Output('volcano-filter-genes', 'data'),
    Input('pathway-bubble-obj', 'clickData'),
    State('gsea-results-store', 'data'),
    prevent_initial_call=True
)
def filter_volcano_by_pathway(clickData, gene_index_dict):
    if not clickData or not gene_index_dict:
        return []
    try:
        clicked_term = clickData['points'][0]['y']
        return gene_index_dict.get(clicked_term, [])
    except Exception:
        return []


# ── CALLBACK 4: Reset button clears pathway filter ───────────────────────────

@app.callback(
    Output('volcano-filter-genes', 'data', allow_duplicate=True),
    Input('reset-btn', 'n_clicks'),
    prevent_initial_call=True
)
def reset_selection(n_clicks):
    return []


# ── CALLBACK 5: Data Quality Indicators ──────────────────────────────────────

@app.callback(
    Output('data-quality-panel', 'children'),
    Output('volcano-stats', 'children'),
    Output('heatmap-stats', 'children'),
    Input('dge-data-store', 'data'),
    Input('lfc-thresh-slider', 'value'),
    Input('padj-thresh-dd', 'value'),
    Input('analysis-options', 'value'),
    prevent_initial_call=True
)
def update_data_quality_indicators(stored_data, lfc_thresh, p_thresh, analysis_options):
    if not stored_data:
        return dash.no_update, dash.no_update, dash.no_update

    df = pd.DataFrame(stored_data)

    # Data quality checks
    quality_indicators = []

    # Statistical validation
    if 'stat_validation' in analysis_options:
        n_sig = ((df['log2FC'].abs() > lfc_thresh) & (df['padj'] < p_thresh)).sum()
        n_total = len(df)
        sig_ratio = n_sig / n_total

        if sig_ratio < 0.01:
            quality_indicators.append(
                html.Span("⚠️ Low significance rate", className="quality-indicator quality-warning")
            )
        elif sig_ratio > 0.3:
            quality_indicators.append(
                html.Span("⚠️ High significance rate", className="quality-indicator quality-warning")
            )
        else:
            quality_indicators.append(
                html.Span("✓ Statistical validation passed", className="quality-indicator quality-good")
            )

    # Data range validation
    lfc_range = df['log2FC'].max() - df['log2FC'].min()
    if lfc_range > 10:
        quality_indicators.append(
            html.Span("⚠️ Wide fold-change range", className="quality-indicator quality-warning")
        )

    # Volcano plot stats
    n_up = ((df['log2FC'] > lfc_thresh) & (df['padj'] < p_thresh)).sum()
    n_down = ((df['log2FC'] < -lfc_thresh) & (df['padj'] < p_thresh)).sum()
    volcano_stats = html.Div([
        html.Small(f"Up: {n_up} • Down: {n_down} • Total: {len(df)} genes",
                   style={'color': '#666', 'fontSize': '10px'})
    ])

    # Heatmap stats (when genes are selected)
    heatmap_stats = html.Div([
        html.Small("Select genes on volcano plot to see heatmap",
                   style={'color': '#999', 'fontSize': '10px'})
    ])

    quality_panel = html.Div(quality_indicators, style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap'})

    return quality_panel, volcano_stats, heatmap_stats


# ── CALLBACK 6: Advanced Analytics Computation ───────────────────────────

@app.callback(
    Output('pca-3d-graph', 'figure'),
    Output('gsea-bar-graph', 'figure'),
    Output('network-graph', 'figure'),
    Output('meta-score-bar', 'figure'),
    Output('gsea-summary', 'children'),
    Output('gsea-store', 'data'),
    Output('pca-store', 'data'),
    Output('network-store', 'data'),
    Output('integrity-store', 'data'),
    Input('dge-data-store', 'data'),
    Input('lfc-thresh-slider', 'value'),
    Input('padj-thresh-dd', 'value'),
    prevent_initial_call=True
)
def compute_advanced_analytics(stored_data, lfc_thresh, p_thresh):
    default_fig = go.Figure().update_layout(template='simple_white')
    if not stored_data:
        return default_fig, default_fig, default_fig, default_fig, '', {}, {}, {}, {}

    try:
        df = pd.DataFrame(stored_data)
        if df.empty:
            return default_fig, default_fig, default_fig, default_fig, 'No data available for advanced analytics.', {}, {}, {}, {}

        meta_df = compute_meta_score(df.copy())
        pca_result = run_pca_3d(df)
        gsea_result = run_gsea_preranked(df)
        network_result = run_wgcna_lite(df)
        integrity_score = compute_data_integrity_score(df)

        pca_fig = create_3d_pca(pca_result)
        gsea_fig = create_gsea_bar(gsea_result)
        network_fig = create_network_graph(network_result)
        meta_fig = create_meta_score_bar(meta_df)

        if gsea_result.get('error'):
            gsea_summary_text = html.P(
                f"Preranked GSEA unavailable: {gsea_result['error']}",
                style={'color': '#c0392b', 'fontSize': '12px'}
            )
        else:
            pathway_count = len(gsea_result.get('results', pd.DataFrame()))
            gsea_summary_text = html.P(
                f"Preranked GSEA computed {pathway_count} significant pathways.",
                style={'color': '#444', 'fontSize': '12px'}
            )

        gsea_store_payload = gsea_result.copy()
        if isinstance(gsea_store_payload.get('results'), pd.DataFrame):
            gsea_store_payload['results'] = gsea_store_payload['results'].to_dict('records')

        network_store_payload = {k: v for k, v in network_result.items() if k != 'graph'}

        return (
            pca_fig,
            gsea_fig,
            network_fig,
            meta_fig,
            gsea_summary_text,
            gsea_store_payload,
            pca_result,
            network_store_payload,
            integrity_score
        )
    except Exception as e:
        logger.error(f"Advanced analytics error: {e}", exc_info=True)
        err_msg = html.P(f"Analytics error: {str(e)[:200]}",
                         style={'color': '#c0392b', 'fontSize': '12px'})
        return default_fig, default_fig, default_fig, default_fig, err_msg, {}, {}, {}, {}


# ── CALLBACK 7: Session Info Display ─────────────────────────────────────────

@app.callback(
    Output('session-info', 'children'),
    Input('session-store', 'data'),
    Input('dge-data-store', 'data'),
    prevent_initial_call=True
)
def update_session_info(session_data, data_store):
    if not session_data:
        return ""

    session_id = session_data.get('session_id', 'Unknown')[:8]
    start_time = session_data.get('start_time', '')
    if start_time:
        try:
            start_dt = datetime.fromisoformat(start_time)
            duration = datetime.now() - start_dt
            duration_str = f"{duration.seconds // 3600}h {(duration.seconds % 3600) // 60}m"
        except:
            duration_str = "Unknown"
    else:
        duration_str = "Unknown"

    data_status = "No data loaded" if not data_store else f"{len(data_store)} genes loaded"

    return html.Small(f"Session {session_id} • {duration_str} • {data_status}",
                      style={'color': '#888', 'fontSize': '10px'})


# ── CALLBACK 7: Export Functionality ────────────────────────────────────────

@app.callback(
    Output('download-results', 'data'),
    Input('export-btn', 'n_clicks'),
    State('dge-data-store', 'data'),
    State('gsea-results-store', 'data'),
    State('volcano-filter-genes', 'data'),
    State('session-store', 'data'),
    prevent_initial_call=True
)
def export_analysis_results(n_clicks, data_store, gsea_results, selected_genes, session_data):
    if not n_clicks or not data_store:
        return dash.no_update

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create comprehensive export data
    export_data = {
        'session_info': session_data,
        'dataset_summary': {
            'total_genes': len(data_store),
            'selected_genes': len(selected_genes) if selected_genes else 0,
            'export_timestamp': timestamp
        },
        'differential_expression': data_store,
        'pathway_enrichment': gsea_results if gsea_results else {},
        'selected_gene_list': selected_genes if selected_genes else []
    }

    # Convert to JSON string
    json_str = json.dumps(export_data, indent=2, default=str)

    return dict(
        content=json_str,
        filename=f'rna_seq_analysis_{timestamp}.json',
        type='application/json'
    )


# ── CALLBACK 8: Enhanced Upload Validation ──────────────────────────────────

@app.callback(
    Output('main-upload-comp', 'style'),
    Input('main-upload-comp', 'contents'),
    State('main-upload-comp', 'style'),
    prevent_initial_call=True
)
def update_upload_style(contents, current_style):
    if contents:
        # Add success styling when file is uploaded
        return {**current_style,
                'borderColor': '#28a745',
                'backgroundColor': '#f8fff8'}
    return current_style


# ── CALLBACK 9: Advanced Analytics & PDF Report ───────────────────────────

@app.callback(
    Output('download-results', 'data'),
    Output('integrity-display', 'children'),
    Input('download-pdf-btn', 'n_clicks'),
    State('dge-data-store', 'data'),
    State('gsea-store', 'data'),
    State('gsea-results-store', 'data'),
    prevent_initial_call=True
)
def generate_analysis_report(n_clicks, data_store, gsea_store, gsea_results):
    if not n_clicks or not data_store:
        return dash.no_update, dash.no_update

    try:
        df = pd.DataFrame(data_store)
        if df.empty:
            return dash.no_update, 'No data loaded to generate report.'

        integrity = compute_data_integrity_score(df)
        analysis_params = {
            'lfc_thresh': 1.0,
            'padj_thresh': 0.05,
            'gene_sets': ['KEGG_2021_Human', 'Reactome_2022']
        }

        pdf_bytes = generate_pdf_report(
            df,
            integrity,
            gsea_results if gsea_results else {},
            gsea_store if gsea_store else {},
            analysis_params
        )

        integrity_text = (
            f"Integrity: {integrity['total']}/100 ({integrity['grade']}) "
            f"| {integrity['n_significant']} significant genes"
        )

        return (
            dcc.send_bytes(pdf_bytes,
                           filename=f'analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'),
            integrity_text
        )

    except Exception as e:
        logger.error(f"PDF report error: {e}", exc_info=True)
        return dash.no_update, f"Report generation failed: {str(e)[:200]}"


# ── CALLBACK 10: Gene Metadata Lookup ─────────────────────────────────────

@app.callback(
    Output('gene-info-panel', 'children'),
    Input('lookup-gene-btn', 'n_clicks'),
    State('gene-search-input', 'value'),
    prevent_initial_call=True
)
def lookup_gene_metadata(n_clicks, gene_symbol):
    if not n_clicks or not gene_symbol:
        return ''

    info = fetch_gene_info(gene_symbol.strip())
    if not info or 'error' in info:
        message = info.get('error', 'Gene lookup failed.') if isinstance(info, dict) else 'Gene lookup failed.'
        return html.Div(message, style={'color': '#c0392b', 'fontSize': '12px'})

    details = [
        html.P(f"Symbol: {gene_symbol.upper()}", style={'margin': '0 0 4px 0', 'fontWeight': '600'}),
        html.P(f"Full Name: {info.get('full_name', 'N/A')}", style={'margin': '0 0 4px 0'}),
        html.P(f"Location: {info.get('location', 'N/A')}", style={'margin': '0 0 4px 0'}),
        html.P(f"Subcellular Location: {info.get('subcellular_location', 'N/A')}", style={'margin': '0 0 4px 0'}),
        html.P(f"Function: {info.get('function', 'N/A')}", style={'margin': '0 0 4px 0'}),
        html.P(f"NCBI ID: {info.get('ncbi_id', 'N/A')} | UniProt ID: {info.get('uniprot_id', 'N/A')}", style={'margin': '0 0 4px 0'})
    ]

    if info.get('domains'):
        details.append(html.P(f"Domains: {', '.join(info.get('domains', []))}", style={'margin': '0 0 4px 0'}))
    if info.get('disease_associations'):
        details.append(html.P(f"Disease associations: {', '.join(info.get('disease_associations', []))}", style={'margin': '0 0 4px 0'}))

    return html.Div(details, style={'fontSize': '12px', 'lineHeight': '1.4'})


# ── CALLBACK 11: Session Management - Save Session Modal ──────────────────

@app.callback(
    Output("save-session-modal", "is_open"),
    Input("save-session-btn", "n_clicks"),
    Input("save-session-cancel", "n_clicks"),
    Input("save-session-confirm", "n_clicks"),
    State("save-session-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_save_session_modal(save_clicks, cancel_clicks, confirm_clicks, is_open):
    if not ctx.triggered:
        return is_open
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "save-session-confirm":
        return False
    return not is_open if trigger_id == "save-session-btn" else is_open


# ── CALLBACK 12: Session Management - Perform Save ──────────────────────────

@app.callback(
    Output('main-alert', 'children'),
    Output('main-alert', 'color'),
    Output('main-alert', 'is_open'),
    Output('session-name-input', 'value'),
    Output('session-desc-input', 'value'),
    Input("save-session-confirm", "n_clicks"),
    State('session-name-input', 'value'),
    State('session-desc-input', 'value'),
    State('dge-data-store', 'data'),
    State('gsea-results-store', 'data'),
    State('volcano-filter-genes', 'data'),
    State('lfc-thresh-slider', 'value'),
    State('padj-thresh-dd', 'value'),
    prevent_initial_call=True
)
def save_session(n_clicks, session_name, description, dge_data, gsea_results, selected_genes, lfc_thresh, padj_thresh):
    if not n_clicks or not session_name or session_name.strip() == '':
        return 'Session name is required', 'danger', True, session_name or '', description or ''
    
    try:
        session_mgr = get_session_manager()
        
        # Prepare session data
        session_data = {
            'data': pd.DataFrame(dge_data) if dge_data else pd.DataFrame(),
            'parameters': {
                'log2_fold_change': lfc_thresh,
                'adjusted_p_value': padj_thresh
            },
            'enrichment_results': pd.DataFrame(gsea_results) if isinstance(gsea_results, list) else {},
            'selected_genes': selected_genes or [],
            'metadata': {
                'description': description or '',
                'data_rows': len(dge_data) if dge_data else 0,
                'selected_genes_count': len(selected_genes) if selected_genes else 0
            }
        }
        
        result = session_mgr.save_session(session_name.strip(), session_data)
        
        if result['success']:
            return result['message'], 'success', True, '', ''
        else:
            return result['message'], 'danger', True, session_name or '', description or ''
    
    except Exception as e:
        logger.error(f"Error saving session: {str(e)}")
        return f'Error saving session: {str(e)}', 'danger', True, session_name or '', description or ''


# ── CALLBACK 13: Session Management - Load Session Modal ──────────────────

@app.callback(
    Output("load-session-modal", "is_open"),
    Output('sessions-list-container', 'children'),
    Input("load-session-btn", "n_clicks"),
    Input("load-session-cancel", "n_clicks"),
    State("load-session-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_load_session_modal(load_clicks, cancel_clicks, is_open):
    if not ctx.triggered:
        return is_open, []
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "load-session-btn":
        # Build sessions list
        session_mgr = get_session_manager()
        sessions = session_mgr.list_sessions()
        
        if not sessions:
            sessions_ui = dbc.Alert("No saved sessions found", color="info")
        else:
            sessions_ui = dbc.ListGroup([
                dbc.ListGroupItem([
                    html.Div([
                        html.H6(s['name'], style={'marginBottom': '4px', 'fontWeight': '600'}),
                        html.Small(f"Saved: {s['timestamp']} | Genes: {s['gene_count']}", style={'color': '#666'}),
                        html.Small(s['description']) if s['description'] else None,
                        dbc.Button('Load', id={'type': 'load-session-btn-item', 'index': s['name']}, 
                                  color='primary', size='sm', className='mt-2')
                    ])
                ]) for s in sessions
            ])
        
        return True, sessions_ui
    
    return not is_open if trigger_id == "load-session-btn" else is_open, []


# ── CALLBACK 14: Session Management - Perform Load ──────────────────────────

@app.callback(
    Output('dge-data-store', 'data'),
    Output('lfc-thresh-slider', 'value'),
    Output('padj-thresh-dd', 'value'),
    Output('volcano-filter-genes', 'data'),
    Output('main-alert', 'children'),
    Output('main-alert', 'color'),
    Output('main-alert', 'is_open'),
    Input({'type': 'load-session-btn-item', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def load_selected_session(n_clicks_list):
    if not ctx.triggered or not any(n_clicks_list or []):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, '', 'info', False
    
    try:
        session_name = ctx.triggered[0]["prop_id"].split('"index":"')[1].split('"')[0]
        session_mgr = get_session_manager()
        
        session_data, status = session_mgr.load_session(session_name)
        
        if not status['success']:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, status['message'], 'danger', True
        
        # Extract data
        dge_data = session_data['data'].to_dict('records') if not session_data['data'].empty else []
        params = session_data.get('parameters', {})
        selected_genes = session_data.get('selected_genes', [])
        
        msg = f"Session '{session_name}' loaded successfully ({len(selected_genes)} selected genes)"
        return dge_data, params.get('log2_fold_change', 1.0), params.get('adjusted_p_value', 0.05), selected_genes, msg, 'success', True
    
    except Exception as e:
        logger.error(f"Error loading session: {str(e)}")
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, f'Error loading session: {str(e)}', 'danger', True


# ── CALLBACK 15: Export - Show Export Options Modal ────────────────────────

@app.callback(
    Output("export-options-modal", "is_open"),
    Input("export-btn", "n_clicks"),
    Input("export-modal-close", "n_clicks"),
    State("export-options-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_export_modal(export_clicks, close_clicks, is_open):
    if not ctx.triggered:
        return is_open
    
    return not is_open


# ── CALLBACK 16: Export - Excel Format ──────────────────────────────────────

@app.callback(
    Output('download-excel', 'data'),
    Output('export-status-message', 'children'),
    Input('export-excel-btn', 'n_clicks'),
    State('dge-data-store', 'data'),
    State('gsea-results-store', 'data'),
    State('volcano-filter-genes', 'data'),
    State('lfc-thresh-slider', 'value'),
    State('padj-thresh-dd', 'value'),
    State('integrity-store', 'data'),
    prevent_initial_call=True
)
def export_excel(n_clicks, gene_data, enrichment_results, selected_genes, lfc_thresh, padj_thresh, integrity_data):
    if not n_clicks:
        return dash.no_update, ''
    
    try:
        exporter = get_exporter()
        gene_df = pd.DataFrame(gene_data) if gene_data else pd.DataFrame()
        
        excel_bytes = exporter.export_to_excel(
            gene_df,
            pd.DataFrame(enrichment_results) if isinstance(enrichment_results, list) else enrichment_results,
            selected_genes or [],
            {'log2_fold_change': lfc_thresh, 'adjusted_p_value': padj_thresh},
            integrity_data or {}
        )
        
        filename = f'gene_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        status_msg = dbc.Alert(
            [html.I(className="fas fa-check-circle", style={'marginRight': '8px'}),
             f"✅ Excel exported successfully as {filename}"],
            color="success", dismissable=True
        )
        return dcc.send_bytes(excel_bytes, filename), status_msg
    
    except ImportError as e:
        status_msg = dbc.Alert(
            [html.I(className="fas fa-exclamation-triangle", style={'marginRight': '8px'}),
             "Note: Excel export requires openpyxl. Try JSON or CSV format instead."],
            color="warning", dismissable=True
        )
        return dash.no_update, status_msg
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        status_msg = dbc.Alert(
            f"Export failed: {str(e)[:100]}",
            color="danger", dismissable=True
        )
        return dash.no_update, status_msg


# ── CALLBACK 17: Export - JSON Format ───────────────────────────────────────

@app.callback(
    Output('download-json', 'data'),
    Input('export-json-btn', 'n_clicks'),
    State('dge-data-store', 'data'),
    State('gsea-results-store', 'data'),
    State('volcano-filter-genes', 'data'),
    State('lfc-thresh-slider', 'value'),
    State('padj-thresh-dd', 'value'),
    State('integrity-store', 'data'),
    prevent_initial_call=True
)
def export_json(n_clicks, gene_data, enrichment_results, selected_genes, lfc_thresh, padj_thresh, integrity_data):
    if not n_clicks:
        return dash.no_update
    
    try:
        exporter = get_exporter()
        gene_df = pd.DataFrame(gene_data) if gene_data else pd.DataFrame()
        
        json_bytes = exporter.export_to_json(
            gene_df,
            pd.DataFrame(enrichment_results) if isinstance(enrichment_results, list) else enrichment_results,
            selected_genes or [],
            {'log2_fold_change': lfc_thresh, 'adjusted_p_value': padj_thresh},
            integrity_data or {}
        )
        
        filename = f'gene_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        return dcc.send_bytes(json_bytes, filename)
    
    except Exception as e:
        logger.error(f"JSON export error: {str(e)}")
        return dash.no_update


# ── CALLBACK 18: Export - CSV Format ────────────────────────────────────────

@app.callback(
    Output('download-enrich-csv', 'data'),
    Input('export-csv-btn', 'n_clicks'),
    State('gsea-results-store', 'data'),
    prevent_initial_call=True
)
def export_csv(n_clicks, enrichment_results):
    if not n_clicks:
        return dash.no_update
    
    try:
        exporter = get_exporter()
        csv_bytes = exporter.export_enrichment_csv(enrichment_results)
        filename = f'enrichment_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        return dcc.send_bytes(csv_bytes, filename)
    
    except Exception as e:
        logger.error(f"CSV export error: {str(e)}")
        return dash.no_update


# ── CALLBACK 19: Progress Indicator Update ──────────────────────────────────

@app.callback(
    Output('progress-indicator-panel', 'children'),
    Input('progress-update-interval', 'n_intervals'),
    State('progress-active-store', 'data'),
    prevent_initial_call=True
)
def update_progress_indicator(n_intervals, progress_data):
    """Update progress indicator every 500ms"""
    if not progress_data:
        return html.Div()  # Return empty div instead of string
    
    active_op = progress_data.get('active_operation')
    last_op = progress_data.get('last_operation')
    
    if not active_op or active_op != last_op:
        # No active operation or it changed
        return html.Div()
    
    try:
        tracker = get_progress_tracker(active_op)
        if not tracker:
            return html.Div()
            
        summary = tracker.get_summary()
        if not summary or summary.get('status') == 'completed':
            return html.Div()
        
        # Generate progress HTML
        progress_html = html.Div([
            html.Div([
                html.Span(f"⏳ {operation}", style={'fontWeight': '600', 'fontSize': '12px'}),
                html.Span(f"{summary['progress_percentage']}%", 
                         style={'float': 'right', 'fontSize': '11px', 'color': '#666'})
            ], style={'marginBottom': '6px', 'display': 'flex', 'justifyContent': 'space-between'}),
            dbc.Progress(value=summary['progress_percentage'], style={'height': '6px', 'marginBottom': '6px'}),
            html.Div([
                html.Span(
                    f"• {step['name']}" + 
                    (" ✓" if step['status'] == 'completed' else 
                     " ⊙" if step['status'] == 'running' else 
                     " ⊗" if step['status'] == 'error' else " ◯"),
                    style={
                        'fontSize': '10px',
                        'color': {
                            'completed': '#4CAF50',
                            'running': '#FF9800',
                            'error': '#f44336',
                            'pending': '#999'
                        }.get(step['status'], '#999'),
                        'display': 'block',
                        'marginBottom': '2px'
                    }
                ) for step in summary['steps'][-3:]  # Show last 3 steps
            ], style={'fontSize': '10px'})
        ], style={'padding': '8px'})
        
        return progress_html
    
    except Exception as e:
        logger.debug(f"Progress update error: {str(e)}")
        return ''


# ── CALLBACK 20: Batch Mode - Toggle Section ───────────────────────────────

@app.callback(
    Output("batch-mode-section", "is_open"),
    Input("toggle-batch-mode-btn", "n_clicks"),
    State("batch-mode-section", "is_open"),
    prevent_initial_call=True
)
def toggle_batch_mode(n_clicks, is_open):
    if not n_clicks:
        return is_open
    return not is_open


# ── CALLBACK 21: Batch Mode - Process Files ─────────────────────────────────

@app.callback(
    Output("batch-results-modal", "is_open"),
    Output("batch-comparison-table", "children"),
    Output("batch-overlap-analysis", "children"),
    Output("batch-errors-panel", "children"),
    Output("batch-process-status", "children"),
    Input("process-batch-btn", "n_clicks"),
    State("batch-file-upload", "contents"),
    State("batch-file-upload", "filename"),
    State("lfc-thresh-slider", "value"),
    State("padj-thresh-dd", "value"),
    prevent_initial_call=True
)
def process_batch_files(n_clicks, contents, filenames, lfc_thresh, padj_thresh):
    if not n_clicks or not contents:
        return False, '', '', '', 'No files uploaded'
    
    try:
        import hashlib
        job_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
        batch_analyzer = get_batch_analyzer()
        
        # Parse uploaded files
        files_metadata = []
        for content, filename in zip(contents, filenames or []):
            try:
                content_type, content_string = content.split(',')
                decoded = base64.b64decode(content_string)
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                files_metadata.append({'filename': filename, 'data': df})
            except Exception as e:
                logger.error(f"Error parsing {filename}: {str(e)}")
        
        if not files_metadata:
            return False, '', '', 'No valid files parsed', dbc.Alert('Error: No valid CSV files found', color='danger')
        
        # Create batch job
        job = batch_analyzer.create_batch_job(job_id, files_metadata)
        
        # Process batch
        params = {
            'log2_fold_change': lfc_thresh,
            'adjusted_p_value': padj_thresh
        }
        job_status = batch_analyzer.process_files(job_id, validate_deg_data, params)
        
        # Get comparison analysis
        comparison_data = batch_analyzer.get_comparison_analysis(job_id)
        
        # Build result displays
        if comparison_data.get('error'):
            return False, dbc.Alert(comparison_data['error'], color='danger'), '', '', dbc.Alert('Processing completed with errors', color='warning')
        
        # Comparison table
        comparison_df = pd.DataFrame(comparison_data.get('comparison_table', []))
        comparison_table = dbc.Table.from_dataframe(comparison_df, striped=True, bordered=True, hover=True)
        
        # Overlap analysis
        overlaps = comparison_data.get('overlap_analysis', {})
        overlap_elements = []
        for pair, overlap_data in overlaps.items():
            overlap_elements.append(
                html.Div([
                    html.H6(f"⊆ {pair}", style={'marginBottom': '6px'}),
                    html.P(f"Intersection: {overlap_data['intersection']} genes | "
                           f"Jaccard Similarity: {overlap_data['jaccard']:.2%}",
                          style={'fontSize': '12px', 'marginBottom': '6px'}),
                    html.Small(f"Sample genes: {', '.join(overlap_data['genes'][:5])}...")
                ], style={'padding': '8px', 'borderLeft': '3px solid #1565c0', 'marginBottom': '8px'})
            )
        
        # Error summary
        errors_summary = []
        for filename, result in job.results.items():
            if result.get('status') in ['failed', 'error']:
                errors_summary.append(
                    dbc.Alert([
                        html.Strong(filename),
                        html.Br(),
                        html.Small(result.get('error', 'Unknown error'))
                    ], color='danger')
                )
            elif result.get('warnings'):
                errors_summary.append(
                    dbc.Alert([
                        html.Strong(filename),
                        html.Br(),
                        html.Small('; '.join(result['warnings']))
                    ], color='warning')
                )
        
        status_msg = dbc.Alert([
            html.I(className="fas fa-check-circle", style={'marginRight': '8px'}),
            f"✅ Batch processed: {job_status['completed_files']}/{job_status['files_count']} files completed"
        ], color='success')
        
        return True, comparison_table, html.Div(overlap_elements) if overlap_elements else html.P('No significant overlaps'), html.Div(errors_summary) if errors_summary else html.P('All files processed successfully'), status_msg
    
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        return False, '', '', '', dbc.Alert(f'Error: {str(e)[:100]}', color='danger')


# ── CALLBACK 22: Batch Mode - Close Modal ──────────────────────────────────

@app.callback(
    Output("batch-results-modal", "is_open"),
    Input("batch-results-close", "n_clicks"),
    State("batch-results-modal", "is_open"),
    prevent_initial_call=True
)
def close_batch_modal(n_clicks, is_open):
    if n_clicks:
        return False
    return is_open


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7860)
