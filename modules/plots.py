import plotly.graph_objects as go
from plotly.graph_objects import Heatmap
import numpy as np
import pandas as pd
from scipy import stats
import gseapy as gp
from dash import dash_table

def create_volcano_plot(df, logfc_col, padj_col, gene_col, lfc_thresh=1.0, p_thresh=0.05, filtered_genes=None):
    """
    Generates a production-grade interactive volcano plot.
    Optimized for Lasso selection and high-speed WebGL rendering.
    """
    # Defensive copy to avoid mutating original data
    d = df.copy()
    
    # Calculate -log10(p-value) for the Y-axis
    # Use a small epsilon to avoid log(0) errors
    d['minus_log10_p'] = -np.log10(d[padj_col].replace(0, 1e-300))
    
    # Categorize genes for coloring
    d['status'] = 'Not Significant'
    d.loc[(d[logfc_col] > lfc_thresh) & (d[padj_col] < p_thresh), 'status'] = 'Up-regulated'
    d.loc[(d[logfc_col] < -lfc_thresh) & (d[padj_col] < p_thresh), 'status'] = 'Down-regulated'

    color_map = {
        'Up-regulated': '#E31A1C',   # Vivid Red
        'Down-regulated': '#1F78B4', # Deep Blue
        'Not Significant': '#D3D3D3' # Light Grey
    }

    fig = go.Figure()

    for status, color in color_map.items():
        sub = d[d['status'] == status]
        
        # Apply filtering if filtered_genes is provided
        if filtered_genes:
            sub = sub[sub[gene_col].isin(filtered_genes)]
        
        fig.add_trace(go.Scattergl(
            x=sub[logfc_col], 
            y=sub['minus_log10_p'], 
            mode='markers', 
            marker=dict(color=color), 
            customdata=sub[gene_col], # Add gene symbols to customdata
            hovertemplate=f"Gene: %{{customdata}}<br>Log2FC: %{{x}}<br>-log10(Adj. P-value): %{{y}}<extra></extra>"
        ))

    # Add threshold lines
    fig.add_hline(y=-np.log10(p_thresh), line_dash="dash", line_color="black", opacity=0.5)
    fig.add_vline(x=lfc_thresh, line_dash="dash", line_color="black", opacity=0.5)
    fig.add_vline(x=-lfc_thresh, line_dash="dash", line_color="black", opacity=0.5)

    fig.update_layout(
        template='simple_white',
        title="Differential Expression (Volcano Plot)",
        xaxis_title="Log2 Fold Change",
        yaxis_title="-log10(Adjusted P-value)",
        hovermode='closest',
        dragmode='lasso', # Default to Lasso for pathway selection
        clickmode='event+select',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    fig.update_layout(dragmode='lasso', clickmode='event+select')

    # Annotate top 15 significant genes
    significant_genes = d[d['status'] != 'Not Significant'].sort_values(by='minus_log10_p', ascending=False).head(15)

    if len(significant_genes) > 0:
        for _, row in significant_genes.iterrows():
            fig.add_annotation(
                x=row[logfc_col],
                y=row['minus_log10_p'],
                text=row[gene_col],
                showarrow=False,
                font=dict(size=9, color='black'),
                xanchor='left',
                yanchor='bottom'
            )

    return fig

def create_pathway_enrichment(selected_genes):
    """
    Performs pathway enrichment analysis using gseapy and returns a Plotly bubble chart and the enrichment results.
    """
    try:
        enr = gp.enrichr(gene_list=selected_genes, gene_sets=['KEGG_2021_Human'], organism='Human')
        enrichment_results = enr.results
        enrichment_results = enrichment_results[enrichment_results['Adjusted P-value'] < 0.05].copy() # Filter for significance and create a copy

        if enrichment_results.empty:
            return go.Figure().update_layout(title="No significant pathways found"), dash_table.DataTable(data=[], columns=[]), {}

        # Fix overlap string and calculate bubble size
        enrichment_results['overlap_count'] = enrichment_results['Overlap'].str.split('/').str[0].astype(int)
        min_o = enrichment_results['overlap_count'].min()
        max_o = enrichment_results['overlap_count'].max()
        enrichment_results['bubble_size'] = 8 + 32 * (enrichment_results['overlap_count'] - min_o) / max(max_o - min_o, 1)

        # Sort by Combined Score and take top 20
        enrichment_results = enrichment_results.sort_values(by='Combined Score', ascending=False).head(20)

        # Build gene-to-pathway index
        gene_pathway_index = {}
        for _, row in enrichment_results.iterrows():
            pathway = row['Term']
            genes = row['Genes'].split(';')
            gene_pathway_index[pathway] = genes


    except Exception as e:
        print(f"Enrichment Error: {e}")
        return go.Figure().update_layout(title="Enrichment Failed"), dash_table.DataTable(data=[], columns=[]), {}


    # Create Bubble Chart
    fig = go.Figure(data=[go.Scatter(
        x=enrichment_results['Combined Score'],
        y=enrichment_results['Term'],
        mode='markers',
        marker=dict(
            size=enrichment_results['bubble_size'],
            sizemode='diameter',
            color=enrichment_results['Adjusted P-value'],
            colorscale='Viridis',
            reversescale=True,
            colorbar=dict(title='Adjusted P-value')
        ),
        text=[f"Pathway: {term}<br>Combined Score: {score:.2f}<br>Overlap: {overlap}" for term, score, overlap in zip(enrichment_results['Term'], enrichment_results['Combined Score'], enrichment_results['Overlap'])],
        hoverinfo='text'
    )])

    fig.update_layout(
        title='Pathway Enrichment Analysis (KEGG)',
        xaxis_title='Combined Score',
        yaxis_title='Pathway',
        template='simple_white'
    )

    table = dash_table.DataTable(
        data=enrichment_results.head(10).to_dict('records'),
        columns=[
            {"name": "Pathway", "id": "Term"},
            {"name": "Adj. P-val", "id": "Adjusted P-value", "format": {"specifier": ".2e"}, "type": "numeric"},
            {"name": "Combined Score", "id": "Combined Score", "format": {"specifier": ".2f"}, "type": "numeric"},
            {"name": "Overlap", "id": "Overlap", "type": "numeric"}
        ],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '10px', 'fontSize': '12px'},
        style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}
    )

    return fig, table, gene_pathway_index

def create_heatmap(df, selected_genes):
    """
    Displays a DEG Signature Heatmap of the selected genes, showing Log2FC values.
    Per-sample expression matrix is not available from a DEG summary table. 
    Fold-change heatmap correctly represents the directionality and magnitude of differential expression for the selected gene set.
    """
    if df.empty or not selected_genes:
        return go.Figure().update_layout(title="No data or genes selected")
    
    # Filter the DataFrame for selected genes
    filtered_df = df[df['symbol'].isin(selected_genes)].copy()
    
    if filtered_df.empty:
        return go.Figure().update_layout(title="No selected genes found in the data")

    # Sort by Log2FC descending
    filtered_df = filtered_df.sort_values(by='log2FC', ascending=False)

    # Create the heatmap data (Nx1 matrix of Log2FC values)
    z = [[row['log2FC']] for _, row in filtered_df.iterrows()]
    y = filtered_df['symbol'].tolist()
    x = ['Log2 Fold Change']

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z,
        y=y,
        x=x,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title='Log2FC'),
        hovertemplate="Gene: %{y}<br>Log2FC: %{z}<br>Adj.P: %{customdata}<extra></extra>",
        customdata=filtered_df['padj']
    ))
    
    fig.update_layout(
        title='Fold-Change Profile of Selected Genes',
        xaxis_title='',
        yaxis_title='Gene',
        template='simple_white',
        margin=dict(l=100, r=20, t=80, b=100)
    )
    
    return fig
