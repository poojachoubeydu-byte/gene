import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import io
import base64
import logging
import gseapy as gp
import plotly.graph_objects as go

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

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Real-Time RNA-Seq Pathway Visualizer", className="text-primary mt-4"),
            html.P("Industry-Grade Differential Expression Analytics", className="text-muted"),
            html.Hr()
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div(['Drag and Drop or ', html.A('Select CSV File')]),
                        style={
                            'width': '100%', 'height': '60px', 'lineHeight': '60px',
                            'borderWidth': '2px', 'borderStyle': 'dashed',
                            'borderRadius': '10px', 'textAlign': 'center', 'margin': '10px 0 40px 0' # Increased margin-bottom
                        },
                        multiple=False
                    ),
                    dcc.Tabs(id='tabs', value='volcano', children=[
                        dcc.Tab(label='Volcano Plot', value='volcano', children=[
                            dcc.Graph(id='volcano-plot', config={"displayModeBar": True, "scrollZoom": True}, style={'height': '600px', 'width': '100%'})
                        ]),
                        dcc.Tab(label='Heatmap', value='heatmap', children=[
                            dcc.Graph(id='heatmap', config={"displayModeBar": True, "scrollZoom": True}, style={'height': '600px', 'width': '100%'})
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
                        children=[
                            dcc.Graph(id='pathway-chart', config={"displayModeBar": True, "scrollZoom": True}, style={'height': '400px'}),
                            html.Div(id='enrichment-table-container', className="mt-3")
                        ],
                        type="border",
                        color="primary"
                    ),
                    dbc.Spinner(
                        children=[
                            dcc.Graph(id='heatmap', config={"displayModeBar": True, "scrollZoom": True}, style={'height': '400px'})
                        ],
                        type="border",
                        color="primary"
                    )
                ])
            ], className="shadow-sm")
        ], width=5)
    ]),

    dcc.Store(id='stored-data'),
    dcc.Store(id='volcano-filter-genes', data=[]) # Store for pathway-filtered genes
], fluid=True)

@app.callback(
    Output('stored-data', 'data'),
    Output('volcano-plot', 'figure'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    Input('tabs', 'value'),
    Input('volcano-filter-genes', 'data') # Pathway-filtered genes
)
def update_output(contents, filename, tab, filtered_genes):
    if contents is None:
        return None, create_volcano_plot(pd.DataFrame(columns=['log2FC', 'padj', 'symbol']), 'log2FC', 'padj', 'symbol', filtered_genes=filtered_genes)
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        # 1. Fuzzy Column Mapping (DeepSeek-V3 Optimized Logic)
        c_lfc = next((c for c in df.columns if c.lower() in ['log2fc', 'logfc', 'lfc', 'log2_fold_change', 'log2foldchange']), None)
        c_padj = next((c for c in df.columns if c.lower() in ['padj', 'fdr', 'qvalue', 'p.adj', 'adjusted_pvalue', 'adj.p.val']), None)
        c_gene = next((c for c in df.columns if c.lower() in ['symbol', 'gene', 'gene_symbol', 'id', 'name', 'gene_name']), None)

        if not all([c_lfc, c_padj, c_gene]):
            logger.error(f"Missing Columns. Found: LFC={c_lfc}, P={c_padj}, Gene={c_gene}")
            return None, dash.no_update

        # 2. Rename and Coerce (Claude-Style Cleanliness)
        df = df.rename(columns={c_lfc: 'log2FC', c_padj: 'padj', c_gene: 'symbol'})
        df['log2FC'] = pd.to_numeric(df['log2FC'], errors='coerce')
        
        # 3. Prevent log10(0) math errors (Staff-level stability)
        df['padj'] = pd.to_numeric(df['padj'], errors='coerce').replace(0, 1e-300)
        
        # 4. Clean NaN values
        df = df.dropna(subset=['log2FC', 'padj', 'symbol'])

        fig = create_volcano_plot(df, 'log2FC', 'padj', 'symbol', filtered_genes=filtered_genes)
        return df.to_dict('records'), fig

    except Exception as e:
        logger.error(f"CRITICAL UPLOAD ERROR: {e}")
        return None, dash.no_update

@app.callback(
    Output('pathway-chart', 'figure'),
    Output('heatmap', 'figure'),
    Output('enrichment-table-container', 'children'),
    Input('volcano-plot', 'selectedData'),
    State('stored-data', 'data')
)
def update_pathway_and_heatmap(selectedData, stored_data):
    if selectedData is None or not selectedData['points'] or not stored_data:
        return go.Figure().update_layout(title="Instruction: Please select genes on the volcano plot to see enrichment"), create_heatmap(pd.DataFrame(), []), html.P("Instruction: Please select genes on the volcano plot to see enrichment")

    # Extract gene symbols from selected data
    selected_genes = [point['customdata'] for point in selectedData['points'] if 'customdata' in point]
    selected_genes = list(set(selected_genes))

    # Create pathway enrichment bubble chart and heatmap
    pathway_chart, enrichment_table = create_pathway_enrichment(selected_genes)
    heatmap = create_heatmap(pd.DataFrame(stored_data), selected_genes)

    return pathway_chart, heatmap, enrichment_table

# Reverse Link: Pathway Selection Filters Volcano Plot
@app.callback(
    Output('volcano-filter-genes', 'data'),
    Input('pathway-chart', 'clickData'),
    State('pathway-chart', 'figure')
)
def filter_volcano_by_pathway(clickData, figure):
    if clickData is None:
        return []

    if 'points' not in clickData or not clickData['points']:
        return []

    clicked_pathway = clickData['points'][0]['y']  # Get the clicked pathway name

    # Extract gene lists from the gseapy results
    try:
        enr = gp.enrichr(gene_list=["placeholder"], gene_sets=['KEGG_2021_Human'], organism='Human')
        pathway_to_genes = {}
        for term in enr.results['Term']:
            pathway_to_genes[term] = enr.results[enr.results['Term'] == term]['Genes'].iloc[0].split(";")
    except Exception as e:
        logger.error(f"Error extracting gene lists from gseapy results: {e}")
        return []

    filtered_genes = pathway_to_genes.get(clicked_pathway, [])

    return filtered_genes

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7860)
