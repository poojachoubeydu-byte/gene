import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import io
import base64
import logging
import gseapy as gp
import plotly.graph_objects as go
import random

# Import our custom modules
from modules.plots import create_volcano_plot, create_heatmap, create_pathway_enrichment # Import create_pathway_enrichment

# Initialize logging for troubleshooting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server # Add server = app.server

# Fix for Plotly modebar being unclickable
app.index_string = app.index_string.replace('</head>', '<style>.js-plotly-plot .plotly .modebar { z-index: 1001 !important; pointer-events: all !important; }</style></head>')

# --- DEMO DATA ---
random.seed(42)
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
n = len(DEMO_GENES)
log2fc = np.random.normal(0, 2, n)
pvalue = 10 ** (-np.abs(log2fc) * np.random.uniform(1, 3, n))
padj = np.minimum(pvalue * n, 1.0)
DEMO_DF = pd.DataFrame({'symbol': DEMO_GENES, 'log2FC': log2fc, 'padj': padj})
# --- END DEMO DATA ---

app.layout = dbc.Container([
    html.Div(
        style={'backgroundColor': 'linear-gradient(to right, #007bff, #17a2b8)', 'padding': '20px', 'color': 'white'},
        children=[
            html.H2("Real-Time RNA-Seq Pathway Visualizer", className="text-center", style={'color': 'white'}),
            html.P("Interactive RNA-Seq ↔ Pathway Analysis Dashboard", className="text-center", style={'color': 'white'}),
            html.Small("Powered by GSEApy + Plotly Dash", className="text-right", style={'float': 'right', 'color': 'white'})
        ]
    ),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Button(
                        id='load-demo-btn',
                        children="▶ Load Demo Dataset (PBMC LPS vs Control)",
                        color="outline-secondary", size="sm",
                        className="mb-3"
                    ),
                    dbc.Alert(id='upload-alert', is_open=False, dismissable=True, duration=6000),
                    html.Div(
                        dcc.Upload(
                            id='main-upload-comp',
                            children=html.Div(['Drag and Drop or ', html.A('Select CSV File')]),
                            style={
                                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                'borderWidth': '2px', 'borderStyle': 'dashed',
                                'borderRadius': '10px', 'textAlign': 'center', 'margin': '10px 0 40px 0' # Increased margin-bottom
                            },
                            multiple=False
                        ),
                    ),
                    html.Div([
                        html.Label("Log2FC Threshold"),
                        dcc.Slider(
                            id='lfc-thresh-slider',
                            min=0.5, max=3.0, step=0.25, value=1.0,
                            marks={0.5:'0.5', 1.0:'1.0', 1.5:'1.5', 2.0:'2.0', 3.0:'3.0'}
                        ),
                    ], style={'marginBottom': '20px'}),

                    html.Div([
                        html.Label("Adjusted P-value Threshold"),
                        dcc.Dropdown(
                            id='padj-thresh-dropdown',
                            options=[{'label': str(x), 'value': x} for x in [0.001, 0.01, 0.05, 0.1]],
                            value=0.05,
                            clearable=False
                        ),
                    ], style={'marginBottom': '30px'}),
                    dcc.Tabs(id='tabs', value='volcano', children=[
                        dcc.Tab(label='Volcano Plot', value='volcano', children=[
                            dcc.Graph(id='volcano-graph-obj', config={"displayModeBar": True, "scrollZoom": True}, style={'height': '600px', 'width': '100%'})
                        ]),
                        dcc.Tab(label='Heatmap', value='heatmap', children=[
                            dcc.Graph(id='gene-heatmap-obj', config={"displayModeBar": True, "scrollZoom": True}, style={'height': '600px', 'width': '100%'})
                        ])
                    ])
                ])
            ], className="shadow-sm")
        ], width=7),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Pathway Enrichment", className="mb-0")),
                dbc.CardBody([
                    dbc.Spinner(
                        id='enrichment-loading',
                        children=[
                            dcc.Graph(id='pathway-bubble-obj', config={"displayModeBar": True, "scrollZoom": True}, style={'height': '400px'}),
                            html.Div(id='enrichment-table-container', className="mt-3")
                        ],
                        type="border",
                        color="primary"
                    )
                ])
            ], className="shadow-sm")
        ], width=5)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card(
                id='summary-card',
                style={'display': 'none'},
                children=[
                    dbc.CardBody(
                        [
                            html.H4("Analysis Summary", className="card-title"),
                            html.Div(id='summary-content')
                        ]
                    )
                ]
            )
        ], width=12)
    ]),

    dcc.Store(id='dge-data-store'),
    dcc.Store(id='volcano-filter-genes', data=[]), # Store for pathway-filtered genes
    dcc.Store(id='gsea-results-store', data={}), # Store for GSEApy enrichment results
    dcc.Store(id='gene-pathway-index-store') # Store for gene-pathway reverse index
], fluid=True)

@app.callback(
    Output('dge-data-store', 'data'),
    Output('volcano-graph-obj', 'figure'),
    Output('upload-alert', 'children'),
    Output('upload-alert', 'color'),
    Output('upload-alert', 'is_open'),
    Input('main-upload-comp', 'contents'),
    State('main-upload-comp', 'filename'),
    Input('tabs', 'value'),
    Input('volcano-filter-genes', 'data'), # Pathway-filtered genes
    Input('lfc-thresh-slider', 'value'),
    Input('padj-thresh-dropdown', 'value'),
    Input('load-demo-btn', 'n_clicks')
)
def update_output(contents, filename, tab, filtered_genes, lfc_thresh, p_thresh, n_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return None, create_volcano_plot(pd.DataFrame(columns=['log2FC', 'padj', 'symbol']), 'log2FC', 'padj', 'symbol', filtered_genes=filtered_genes, lfc_thresh=lfc_thresh, p_thresh=p_thresh), "", "secondary", False
    else:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'load-demo-btn':
        df = DEMO_DF.copy()
        alert_message = f"Loaded demo dataset ({len(df)} genes). {len(df[(df['log2FC'] > lfc_thresh) | (df['log2FC'] < -lfc_thresh)])} DEGs at current thresholds."
        alert_color = "success"
        alert_open = True
    elif contents is not None:
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
            # 1. Fuzzy Column Mapping (DeepSeek-V3 Optimized Logic)
            c_lfc = next((c for c in df.columns if c.lower() in ['log2fc', 'logfc', 'lfc', 'log2_fold_change', 'log2foldchange']), None)
            c_padj = next((c for c in df.columns if c.lower() in ['padj', 'fdr', 'qvalue', 'p.adj', 'adjusted_pvalue', 'adj.p.val']), None)
            c_gene = next((c for c in df.columns if c.lower() in ['symbol', 'gene', 'gene_symbol', 'id', 'name', 'gene_name']), None)

            if not all([c_lfc, c_padj, c_gene]):
                missing_cols = [col for col, c in zip(['log2FC', 'padj', 'symbol'], [c_lfc, c_padj, c_gene]) if c is None]
                found_cols = {col: c for col, c in zip(['log2FC', 'padj', 'symbol'], [c_lfc, c_padj, c_gene]) if c is not None}
                alert_message = f"Missing columns: {', '.join(missing_cols)}. Found: {found_cols}."
                alert_color = "danger"
                alert_open = True
                logger.error(alert_message)
                return None, dash.no_update, alert_message, alert_color, alert_open

            # 2. Rename and Coerce (Claude-Style Cleanliness)
            df = df.rename(columns={c_lfc: 'log2FC', c_padj: 'padj', c_gene: 'symbol'})
            df['log2FC'] = pd.to_numeric(df['log2FC'], errors='coerce')
            
            # 3. Prevent log10(0) math errors (Staff-level stability)
            df['padj'] = pd.to_numeric(df['padj'], errors='coerce').replace(0, 1e-300)
            
            # 4. Clean NaN values
            df = df.dropna(subset=['log2FC', 'padj', 'symbol'])

            n = len(df)
            n_sig = len(df[(df['log2FC'] > lfc_thresh) | (df['log2FC'] < -lfc_thresh)])
            alert_message = f"Loaded {n} genes. {n_sig} significant DEGs at current thresholds. Use lasso selection on the volcano to explore pathways."
            alert_color = "success"
            alert_open = True

        except Exception as e:
            alert_message = f"Error loading data. Please ensure the file is a valid CSV."
            alert_color = "danger"
            alert_open = True
            logger.error(f"CRITICAL UPLOAD ERROR: {e}")
            return None, dash.no_update, alert_message, alert_color, alert_open
    else:
        return None, create_volcano_plot(pd.DataFrame(columns=['log2FC', 'padj', 'symbol']), 'log2FC', 'padj', 'symbol', filtered_genes=filtered_genes, lfc_thresh=lfc_thresh, p_thresh=p_thresh), "", "secondary", False

    fig = create_volcano_plot(df, 'log2FC', 'padj', 'symbol', filtered_genes=filtered_genes, lfc_thresh=lfc_thresh, p_thresh=p_thresh)
    return df.to_dict('records'), fig, alert_message, alert_color, alert_open

@app.callback(
    Output('pathway-bubble-obj', 'figure'),
    Output('gene-heatmap-obj', 'figure'),
    Output('enrichment-table-container', 'children'),
    Output('gsea-results-store', 'data'),
    Output('gene-pathway-index-store', 'data'),
    Input('volcano-graph-obj', 'selectedData'),
    State('dge-data-store', 'data')
)
def update_pathway_and_heatmap(selectedData, stored_data):
    if selectedData is None or not selectedData['points'] or not stored_data:
        empty_fig = go.Figure().update_layout(title="Instruction: Please select genes on the volcano plot to see enrichment")
        return empty_fig, create_heatmap(pd.DataFrame(), []), html.P("Instruction: Please select genes on the volcano plot to see enrichment"), {}, None

    # Extract gene symbols from selected data
    selected_genes = [point['customdata'] for point in selectedData['points'] if 'customdata' in point]
    selected_genes = list(set(selected_genes))

    # Create pathway enrichment bubble chart and heatmap
    pathway_chart, enrichment_table, gene_pathway_index = create_pathway_enrichment(selected_genes)
    heatmap = create_heatmap(pd.DataFrame.from_dict(stored_data), selected_genes)

    return pathway_chart, heatmap, enrichment_table, gene_pathway_index, gene_pathway_index

# Reverse Link: Pathway Selection Filters Volcano Plot
@app.callback(
    Output('volcano-filter-genes', 'data'),
    Input('pathway-bubble-obj', 'clickData'),
    State('gsea-results-store', 'data')
)
def filter_volcano_by_pathway(clickData, gsea_results):
    if clickData is None:
        return []

    if 'points' not in clickData or not clickData['points']:
        return []

    clicked_pathway = clickData['points'][0]['y']  # Get the clicked pathway name

    # Extract gene lists from the stored gsea_results
    filtered_genes = gsea_results.get(clicked_pathway, [])

    return filtered_genes

@app.callback(
    Output('summary-card', 'style'),
    Output('summary-card', 'children'),
    Input('dge-data-store', 'data'),
    State('lfc-thresh-slider', 'value'),
    State('padj-thresh-dropdown', 'value')
)
def update_summary_card(data, lfc_thresh, p_thresh):
    if data is None:
        return {'display': 'none'}, []

    df = pd.DataFrame.from_dict(data)
    total_genes = len(df)
    upregulated = len(df[(df['log2FC'] > lfc_thresh) & (df['padj'] < p_thresh)])
    downregulated = len(df[(df['log2FC'] < -lfc_thresh) & (df['padj'] < p_thresh)])
    not_significant = total_genes - upregulated - downregulated

    most_upregulated = df.sort_values(by='log2FC', ascending=False).iloc[0]
    most_downregulated = df.sort_values(by='log2FC', ascending=True).iloc[0]

    summary_sentence = f"This dataset shows {total_genes} genes significantly altered. The most upregulated gene is {most_upregulated['symbol']} (log2FC={most_upregulated['log2FC']:.2f}) and the most downregulated is {most_downregulated['symbol']} (log2FC={most_downregulated['log2FC']:.2f})."

    summary_content = dbc.CardBody([
        html.H4("Analysis Summary", className="card-title"),
        html.P(summary_sentence),
        dbc.Row([
            dbc.Col(dbc.Badge(f"Total Genes: {total_genes}", color="secondary", className="me-1"), width=4),
            dbc.Col(dbc.Badge(f"Upregulated: {upregulated}", color="danger", className="me-1"), width=4),
            dbc.Col(dbc.Badge(f"Downregulated: {downregulated}", color="primary", className="me-1"), width=4),
            dbc.Col(dbc.Badge(f"Not Significant: {not_significant}", color="light", text_color="dark", className="me-1"), width=4)
        ])
    ])

    return {'display': 'block'}, summary_content

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7860)
