import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import io
import base64
import logging

# Import our custom modules
from modules.plots import create_volcano_plot, create_pathway_bar_chart, create_heatmap
from modules.enrichment import EnrichmentEngine

# Initialize logging for troubleshooting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server
engine = EnrichmentEngine()

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
                            'borderRadius': '10px', 'textAlign': 'center', 'margin': '10px 0 30px 0'
                        },
                        multiple=False
                    ),
                    dcc.Tabs(id='tabs', value='volcano', children=[
                        dcc.Tab(label='Volcano Plot', value='volcano', children=[
                            dcc.Graph(id='volcano-plot', config={"displayModeBar": True, "scrollZoom": True}, style={'height': '600px'})
                        ]),
                        dcc.Tab(label='Heatmap', value='heatmap', children=[
                            dcc.Graph(id='heatmap', config={"displayModeBar": True, "scrollZoom": True}, style={'height': '600px'})
                        ])
                    ])
                ])
            ], className="shadow-sm")
        ], width=7),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Pathway Enrichment", className="mb-0")),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-enrichment",
                        type="default",
                        children=[
                            dcc.Graph(id='pathway-chart', style={'height': '400px'}),
                            html.Div(id='enrichment-table-container', className="mt-3")
                        ]
                    )
                ])
            ], className="shadow-sm")
        ], width=5)
    ]),

    dcc.Store(id='stored-data')
], fluid=True)

@app.callback(
    Output('stored-data', 'data'),
    Output('volcano-plot', 'figure'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    Input('tabs', 'value')
)
def update_output(contents, filename, tab):
    if contents is None:
        return None, create_volcano_plot(pd.DataFrame(columns=['log2FC', 'padj', 'symbol']), 'log2FC', 'padj', 'symbol')
    
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

        fig = create_volcano_plot(df, 'log2FC', 'padj', 'symbol')
        return df.to_dict('records'), fig

    except Exception as e:
        logger.error(f"CRITICAL UPLOAD ERROR: {e}")
        return None, dash.no_update

@app.callback(
    Output('pathway-chart', 'figure'),
    Output('enrichment-table-container', 'children'),
    Output('heatmap', 'figure'),
    Input('volcano-plot', 'selectedData'),
    State('stored-data', 'data'),
    Input('tabs', 'value')
)
def run_enrichment_logic(selectedData, stored_data, tab):
    if selectedData is None or not selectedData['points'] or not stored_data:
        return create_pathway_bar_chart(pd.DataFrame()), html.P("Lasso-select genes on the plot to start."), create_heatmap(pd.DataFrame())
    
    # Extract unique gene symbols from selection
    genes = [p['customdata'] for p in selectedData['points'] if 'customdata' in p]
    selected_genes = list(set(genes))
    
    # Run Backend
    results_df = engine.run_enrichment(selected_genes)
    
    if results_df.empty:
        return create_pathway_bar_chart(pd.DataFrame()), html.P("No pathways enriched for these genes."), create_heatmap(pd.DataFrame())

    bar_fig = create_pathway_bar_chart(results_df)
    
    table = dash_table.DataTable(
        data=results_df.head(10).to_dict('records'),
        columns=[
            {"name": "Pathway", "id": "pathway"},
            {"name": "Adj. P-val", "id": "adj_p_value", "format": {"specifier": ".2e"}, "type": "numeric"}
        ],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '10px', 'fontSize': '12px'},
        style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}
    )
    
    df = pd.DataFrame(stored_data)
    heatmap_fig = create_heatmap(df)
    
    return bar_fig, table, heatmap_fig

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7860)
