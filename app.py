import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import io, base64, logging, json, hashlib, time
from datetime import datetime
from collections import Counter
import plotly.graph_objects as go
from modules.plots import create_volcano_plot, create_heatmap, create_pathway_enrichment

# ── CONFIGURATION ────────────────────────────────────────────────────────────

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MIN_GENES = 50
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

def validate_deg_data(df):
    """
    Comprehensive validation of DEG data with detailed error reporting
    """
    errors = []
    warnings = []

    # Check for required columns (case-insensitive matching)
    col_lower_map = {c.lower(): c for c in df.columns}
    missing_cols = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            # Try common alternatives (case-insensitive)
            alternatives = {
                'gene_symbol': ['symbol', 'gene', 'gene_symbol', 'id', 'name', 'gene_name', 'genename', 'gene_id'],
                'log2_fold_change': ['log2foldchange', 'log2fc', 'logfc', 'lfc', 'log2_fold_change', 'fold_change', 'foldchange', 'log2_fc'],
                'adjusted_p_value': ['padj', 'fdr', 'qvalue', 'adj_p', 'adjusted_p_value', 'adj.p.val', 'adj_pval', 'p_adj', 'p.adj']
            }
            found_alt = None
            for alt in alternatives.get(col, []):
                actual = col_lower_map.get(alt.lower())
                if actual is not None:
                    found_alt = actual
                    break
            if found_alt:
                df = df.rename(columns={found_alt: col})
                col_lower_map = {c.lower(): c for c in df.columns}
            else:
                missing_cols.append(col)

    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")

    if errors:
        return None, errors, warnings

    # Data type validation
    try:
        df['log2_fold_change'] = pd.to_numeric(df['log2_fold_change'], errors='coerce')
        df['adjusted_p_value'] = pd.to_numeric(df['adjusted_p_value'], errors='coerce')
    except Exception as e:
        errors.append(f"Data type conversion failed: {str(e)}")
        return None, errors, warnings

    # Remove rows with missing values
    original_len = len(df)
    df = df.dropna(subset=['gene_symbol', 'log2_fold_change', 'adjusted_p_value'])
    if len(df) < original_len:
        warnings.append(f"Removed {original_len - len(df)} rows with missing values")

    # Validate data ranges
    if (df['adjusted_p_value'] < 0).any() or (df['adjusted_p_value'] > 1).any():
        errors.append("Adjusted p-values must be between 0 and 1")

    if df['log2_fold_change'].abs().max() > 20:
        warnings.append("Extremely large fold changes detected (>20). Please verify data scaling.")

    # Check for duplicates
    dup_genes = df['gene_symbol'].duplicated().sum()
    if dup_genes > 0:
        warnings.append(f"Found {dup_genes} duplicate gene symbols. Keeping first occurrence.")
        df = df.drop_duplicates(subset=['gene_symbol'], keep='first')

    # Check data size
    if len(df) < MIN_GENES:
        errors.append(f"Dataset too small: {len(df)} genes. Minimum required: {MIN_GENES}")

    if len(df) > MAX_GENES:
        warnings.append(f"Large dataset: {len(df)} genes. Performance may be affected.")

    # Statistical quality checks
    n_sig = ((df['log2_fold_change'].abs() > 1) & (df['adjusted_p_value'] < 0.05)).sum()
    if n_sig < 10:
        warnings.append("Very few significantly differentially expressed genes detected. Consider relaxing thresholds.")

    return df, errors, warnings

def create_session_id():
    """Generate unique session identifier"""
    return hashlib.md5(f"{time.time()}_{np.random.randint(1000000)}".encode()).hexdigest()[:8]

# ── APP INIT ─────────────────────────────────────────────────────────────────

app = dash.Dash(__name__,
                external_stylesheets=[],
                suppress_callback_exceptions=True,
                meta_tags=[
                    {"name": "viewport", "content": "width=device-width, initial-scale=1"},
                    {"name": "description", "content": "Advanced RNA-seq Pathway Analysis Tool"},
                    {"property": "og:title", "content": "RNA-seq Pathway Visualizer"},
                    {"property": "og:description", "content": "Interactive differential expression and pathway enrichment analysis"},
                    {"property": "og:type", "content": "website"}
                ])

server = app.server

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
                html.Span("v2.1.0", style={'fontSize':'10px','color':'rgba(255,255,255,0.6)'}),
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

    # ── DATA QUALITY INDICATORS ─────────────────────────
    dbc.Row([dbc.Col([
        html.Div(id='data-quality-panel', style={'marginBottom':'12px'})
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
                    dbc.Button([
                        html.I(className="fas fa-play", style={'marginRight':'6px'}),
                        "Load Demo Data"
                    ], id='load-demo-btn', color="outline-primary",
                       size="sm", style={'height': '44px', 'fontSize': '12px'})
                ], style={'flexShrink': '1', 'minWidth': '120px'}),

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
                           className="export-btn", style={'height': '32px', 'fontSize': '11px'})
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
    dcc.Download(id='download-results'),

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

    if triggered == 'load-demo-btn' and n_clicks:
        df = DEMO_DF.copy()

    elif triggered == 'main-upload-comp' and contents is not None:
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            temp_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            # Use enhanced validation
            df, errors, warnings = validate_deg_data(temp_df)

            if errors:
                msg = "Validation failed: " + "; ".join(errors)
                empty_fig = create_volcano_plot(pd.DataFrame(columns=['log2_fold_change', 'adjusted_p_value', 'gene_symbol']),
                                                'log2_fold_change', 'adjusted_p_value', 'gene_symbol')
                return None, empty_fig, dash.no_update, msg, 'danger', True, True

            if warnings:
                logger.warning("Data validation warnings: " + "; ".join(warnings))

            # Rename columns to match expected format
            df = df.rename(columns={
                'gene_symbol': 'symbol',
                'log2_fold_change': 'log2FC',
                'adjusted_p_value': 'padj'
            })

        except Exception as e:
            logger.error(f"Upload error: {e}")
            msg = f"Could not parse file: {str(e)}. Ensure it is a comma-separated CSV with required columns."
            return None, dash.no_update, dash.no_update, msg, 'danger', True, True

    elif existing_data and triggered in ('lfc-thresh-slider', 'padj-thresh-dd', 'volcano-filter-genes'):
        df = pd.DataFrame(existing_data)

    else:
        empty_fig = create_volcano_plot(pd.DataFrame(columns=['log2FC', 'padj', 'symbol']),
                                        'log2FC', 'padj', 'symbol')
        return None, empty_fig, dash.no_update, dash.no_update, dash.no_update, False, True

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
            success_msg, 'success', True, False)


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


# ── CALLBACK 6: Session Info Display ─────────────────────────────────────────

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


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7860)
