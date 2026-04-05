import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import io, base64, logging
import plotly.graph_objects as go
from modules.plots import create_volcano_plot, create_heatmap, create_pathway_enrichment

# ── DEMO DATA ───────────────────────────────────────────────────────────────

np.random.seed(42)
DEMO_GENES = [
    'TNF','IL6','IL1B','CXCL8','CCL2','MYD88','NFKB1','RELA','STAT1','STAT3',
    'IRF1','IRF3','IRF7','TLR4','IRAK1','TRAF6','MAPK14','JUN','FOS','EGR1',
    'PTGS2','MMP9','VEGFA','HIF1A','TP53','CDKN1A','BCL2','MCL1','BAX','CASP3',
    'IFNG','IFNB1','MX1','OAS1','ISG15','IFIT1','DDX58','MAVS','STING1','CGAS',
    'CD68','CD14','CD163','FCGR1A','ITGAM','CSF1R','TYROBP','SPI1','KLF4','MRC1',
    'LYZ','S100A8','S100A9','CXCL10','CCL5','CCL3','IL10','TGFB1','FOXP3','IL2',
    'CD3D','CD4','CD8A','GZMB','PRF1','NKG7','GNLY','KLRB1','NCR1','TIGIT',
    'GAPDH','ACTB','B2M','HLA-A','HLA-DRA','HLA-DRB1','CD74','CIITA','TAP1','TAPBP'
]
_n = len(DEMO_GENES)
_lfc = np.random.normal(0, 2, _n)
_pval = 10 ** (-np.abs(_lfc) * np.random.uniform(1, 3, _n))
_padj = np.minimum(_pval * _n, 1.0)
DEMO_DF = pd.DataFrame({'symbol': DEMO_GENES, 'log2FC': _lfc, 'padj': _padj})

# ── LOGGING ─────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── APP INIT ─────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
                suppress_callback_exceptions=True)
server = app.server

app.index_string = app.index_string.replace(
    '</head>',
    '<style>'
    '.js-plotly-plot .plotly .modebar{z-index:1001!important;pointer-events:all!important;}'
    '.upload-zone:hover{background:#f0f7ff!important;}'
    '::-webkit-scrollbar {width: 6px;height: 6px;}'
    '::-webkit-scrollbar-track {background: #f1f1f1;border-radius: 3px;}'
    '::-webkit-scrollbar-thumb {background: #b0bec5;border-radius: 3px;}'
    '::-webkit-scrollbar-thumb:hover {background: #78909c;}'
    '</style></head>'
)

# ── LAYOUT ───────────────────────────────────────────────────────────────────

app.layout = dbc.Container([



    # ── HEADER ──────────────────────────────────────────
    dbc.Row([dbc.Col([
        html.Div([
            html.H2("🧬 Real-Time RNA-Seq Pathway Visualizer",
                    style={'color':'white','marginBottom':'4px',
                           'fontSize':'20px','fontWeight':'700'}),
            html.P("Interactive Differential Expression ↔ Pathway Analysis  |"
                   "  GSEApy · Plotly Dash · KEGG · Reactome",
                   style={'color':'rgba(255,255,255,0.82)',
                          'marginBottom':0,'fontSize':'11px'})
        ], style={
            'background':'linear-gradient(90deg,#1a237e,#1565c0)',
            'padding':'16px 24px',
            'borderRadius':'8px',
            'marginBottom':'12px'
        })
    ], width=12)]),

    # ── ALERT BANNER ────────────────────────────────────
    dbc.Row([dbc.Col([
        dbc.Alert(id='main-alert', is_open=False, dismissable=True,
                  duration=7000, style={'fontSize': '13px'})
    ], width=12)]),

    # ── CONTROLS ROW ────────────────────────────────────
    dbc.Row([dbc.Col([
        dbc.Card([dbc.CardBody([
            html.Div([

                # Upload
                html.Div([
                    html.Label("Upload DEG CSV", style={'fontWeight': '600', 'fontSize': '12px', 'marginBottom': '4px'}),
                    dcc.Upload(
                        id='main-upload-comp',
                        children=html.Div(['\U0001f4c2 Drag & Drop or ', html.A('Select File')]),
                        style={
                            'width': '280px', 'height': '44px', 'lineHeight': '44px',
                            'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '8px',
                            'textAlign': 'center', 'borderColor': '#adb5bd', 'fontSize': '12px',
                            'cursor': 'pointer', 'backgroundColor': '#f8f9fa'
                        },
                        max_size=50 * 1024 * 1024,
                        multiple=False
                    )
                ], style={'flexShrink': '1', 'minWidth': '120px'}),

                # Demo button
                html.Div([
                    html.Label(" ", style={'display': 'block', 'marginBottom': '4px'}),
                    dbc.Button("\u25b6 Load Demo (PBMC LPS vs Control)",
                               id='load-demo-btn', color="outline-primary",
                               size="sm", style={'height': '44px', 'fontSize': '12px'})
                ], style={'flexShrink': '1', 'minWidth': '120px'}),

                # Log2FC threshold
                html.Div([
                    html.Label("Log2FC Threshold",
                               style={'fontWeight': '600', 'fontSize': '12px', 'marginBottom': '4px'}),
                    dcc.Slider(id='lfc-thresh-slider', min=0.5, max=3.0, step=0.25,
                               value=1.0, marks={0.5: '0.5', 1.0: '1.0', 1.5: '1.5',
                                                 2.0: '2.0', 3.0: '3.0'},
                               tooltip={'placement': 'bottom', 'always_visible': False}),
                ], style={'width': '220px', 'flexShrink': '1', 'minWidth': '120px'}),

                # padj threshold
                html.Div([
                    html.Label("Adj. P-value Threshold",
                               style={'fontWeight': '600', 'fontSize': '12px', 'marginBottom': '4px'}),
                    dcc.Dropdown(id='padj-thresh-dd',
                                 options=[{'label': str(v), 'value': v}
                                          for v in [0.001, 0.01, 0.05, 0.1]],
                                 value=0.05, clearable=False,
                                 style={'width': '140px', 'fontSize': '12px'})
                ], style={'flexShrink': '1', 'minWidth': '120px'}),

                # Reset button
                html.Div([
                    html.Label(" ", style={'display': 'block', 'marginBottom': '4px'}),
                    dbc.Button("\u21ba Reset Selection", id='reset-btn',
                               color="outline-secondary", size="sm",
                               style={'height': '44px', 'fontSize': '12px'})
                ], style={'flexShrink': '1', 'minWidth': '120px'}),

            ], style={'display': 'flex', 'gap': '16px', 'alignItems': 'flex-end', 'flexWrap': 'wrap', 'width': '100%'})
        ])], className="shadow-sm mb-3")
    ], width=12)]),

    # ── ONBOARDING HINT (hidden after data loads) ───────
    dbc.Row([dbc.Col([
        dbc.Alert([
            html.Strong("How to use: "),
            "\u2460 Load demo data or upload your own DEG CSV  \u2192  "
            "\u2461 Use lasso selection on the volcano plot to select genes  \u2192  "
            "\u2462 Pathway enrichment runs automatically  \u2192  "
            "\u2463 Click any pathway bubble to highlight its genes on the volcano plot"
        ], color="info", id='onboarding-hint',
           style={'fontSize': '12px', 'padding': '10px 16px'})
    ], width=12)]),

    # ── MAIN PANELS ─────────────────────────────────────
    dbc.Row([

        # LEFT: Volcano + Heatmap (vertical stack)
        dbc.Col([
            dbc.Card([dbc.CardBody([
                html.Div([
                    html.Div([
                        html.Span("🌋 Volcano Plot",
                                  style={'fontWeight':'600','fontSize':'13px',
                                         'color':'#1a237e','marginBottom':'4px',
                                         'display':'block'}),
                        html.Small("Use lasso tool to select genes → triggers pathway enrichment",
                                   style={'color':'#888','fontSize':'11px'}),
                    ], style={'marginBottom':'6px'}),

                    dcc.Graph(
                        id='volcano-graph-obj',
                        config={
                            'displayModeBar': True,
                            'scrollZoom': True,
                            'responsive': True,
                            'modeBarButtonsToAdd': ['lasso2d', 'select2d'],
                            'toImageButtonOptions': {
                                'format': 'png',
                                'filename': 'volcano_plot',
                                'height': 600,
                                'width': 900,
                                'scale': 2
                            }
                        },
                        style={'height': '420px', 'width': '100%'}
                    ),

                    html.Hr(style={'margin':'12px 0','borderColor':'#e9ecef'}),

                    html.Div([
                        html.Span("🟥 Fold-Change Heatmap",
                                  style={'fontWeight':'600','fontSize':'13px',
                                         'color':'#1a237e','marginBottom':'4px',
                                         'display':'block'}),
                        html.Small("Log2 fold-change of selected genes — red=up, blue=down",
                                   style={'color':'#888','fontSize':'11px'}),
                    ], style={'marginBottom':'6px'}),

                    dcc.Graph(
                        id='gene-heatmap-obj',
                        config={'displayModeBar': True, 'responsive': True},
                        style={'height': '340px', 'width': '100%'}
                    ),
                ])
            ])], className="shadow-sm")
        ], xs=12, sm=12, md=12, lg=7, xl=7),

        # RIGHT: Pathway enrichment
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.Div([
                    html.Span("Pathway Enrichment", style={'fontWeight': '700', 'fontSize': '14px'}),
                    html.Small(" \u2014 KEGG + Reactome | Fisher exact + BH correction",
                               style={'color': '#6c757d', 'fontSize': '11px'})
                ])),
                dbc.CardBody([
                    dcc.Loading(
                        type='circle',
                        color='#1565c0',
                        children=[
                            dcc.Graph(
                                id='pathway-bubble-obj',
                                config={"displayModeBar": True, "responsive": True},
                                style={'height': '430px', 'width': '100%'}
                            ),
                            html.Div(id='enrichment-table-container',
                                     className="mt-2")
                        ]
                    )
                ])
            ], className="shadow-sm")
        ], xs=12, sm=12, md=12, lg=5, xl=5)
    ], className="mb-3"),

    # ── ANALYSIS SUMMARY ────────────────────────────────
    dbc.Row([dbc.Col([
        dbc.Card([dbc.CardBody([
            html.Div(id='summary-panel',
                     children=html.P("Load data to see analysis summary.",
                                     style={'color': '#aaa', 'fontStyle': 'italic', 'margin': 0}))
        ])], className="shadow-sm")
    ], width=12)], className="mb-4"),

    # ── STORES ──────────────────────────────────────────
    dcc.Store(id='dge-data-store'),
    dcc.Store(id='gsea-results-store', data={}),
    dcc.Store(id='volcano-filter-genes', data=[]),

], fluid=True)

# ── CALLBACK 1: Load data (upload OR demo button) ────────────────────────────

@app.callback(
    Output('dge-data-store', 'data'),
    Output('volcano-graph-obj', 'figure'),
    Output('summary-panel', 'children'),
    Output('main-alert', 'children'),
    Output('main-alert', 'color'),
    Output('main-alert', 'is_open'),
    Output('onboarding-hint', 'is_open'),
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
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            c_lfc  = next((c for c in df.columns if c.lower() in
                           ['log2fc', 'logfc', 'lfc', 'log2_fold_change', 'log2foldchange']), None)
            c_padj = next((c for c in df.columns if c.lower() in
                           ['padj', 'fdr', 'qvalue', 'p.adj', 'adjusted_pvalue', 'adj.p.val']), None)
            c_gene = next((c for c in df.columns if c.lower() in
                           ['symbol', 'gene', 'gene_symbol', 'id', 'name', 'gene_name']), None)

            if not all([c_lfc, c_padj, c_gene]):
                msg = (f"Column mapping failed. Need: gene symbol, fold change, adjusted p-value. "
                       f"Found columns: {list(df.columns)}. "
                       f"Rename to: symbol, log2FC, padj.")
                empty_fig = create_volcano_plot(pd.DataFrame(columns=['log2FC', 'padj', 'symbol']),
                                                'log2FC', 'padj', 'symbol')
                return None, empty_fig, dash.no_update, msg, 'danger', True, True

            df = df.rename(columns={c_lfc: 'log2FC', c_padj: 'padj', c_gene: 'symbol'})
            df['log2FC'] = pd.to_numeric(df['log2FC'], errors='coerce')
            df['padj']   = pd.to_numeric(df['padj'],   errors='coerce').replace(0, 1e-300)
            df = df.dropna(subset=['log2FC', 'padj', 'symbol'])

            if len(df) < 10:
                msg = f"Only {len(df)} valid genes found after parsing. Check your CSV format."
                return None, dash.no_update, dash.no_update, msg, 'warning', True, True

        except Exception as e:
            logger.error(f"Upload error: {e}")
            msg = "Could not parse file. Ensure it is a comma-separated CSV with a header row."
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
        html.H6("Analysis Summary", style={'marginBottom': '8px', 'fontWeight': '700'}),
        html.P(f"Dataset contains {n_total} genes. "
               f"Most significant upregulated: {up_str}. "
               f"Most significant downregulated: {down_str}.",
               style={'fontSize': '13px', 'marginBottom': '10px'}),
        html.Div([
            dbc.Badge(f"Total Genes: {n_total}", color="secondary", className="me-2"),
            dbc.Badge(f"Upregulated: {n_up}", color="danger", className="me-2"),
            dbc.Badge(f"Downregulated: {n_down}", color="primary", className="me-2"),
            dbc.Badge(f"Not Significant: {n_ns}", color="light",
                      text_color="dark", className="me-2"),
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


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7860)
