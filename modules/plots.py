import plotly.graph_objects as go
from plotly.graph_objects import Heatmap
import numpy as np
import pandas as pd
from scipy import stats

def create_volcano_plot(df, logfc_col, padj_col, gene_col, lfc_thresh=1.0, p_thresh=0.05):
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
        fig.add_trace(go.Scattergl(x=sub[logfc_col], y=sub['minus_log10_p'], mode='markers', marker=dict(color=color), customdata=sub[gene_col]))

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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return fig

def create_pathway_bar_chart(enrichment_df):
    """
    Displays the top enriched pathways by significance.
    """
    if enrichment_df.empty:
        return go.Figure().update_layout(title="No pathways found. Try selecting more genes.")

    # Select top 10 for clarity
    top_df = enrichment_df.head(10).iloc[::-1] # Reverse for horizontal bar chart
    
    fig = go.Figure(go.Bar(
        x=-np.log10(top_df['adj_p_value']),
        y=top_df['pathway'],
        orientation='h',
        marker=dict(color='#33a02c'),
        hovertemplate="<b>Pathway:</b> %{y}<br><b>-log10(padj):</b> %{x:.2f}<extra></extra>"
    ))

    fig.update_layout(
        template='simple_white',
        title="Top Enriched Pathways",
        xaxis_title="-log10(Adjusted P-value)",
        yaxis_title="",
        margin=dict(l=200, r=20, t=40, b=40)
    )

    return fig
import numpy as np
from plotly.graph_objects import Heatmap
from scipy import stats

def create_heatmap(df):
    """
    Displays a Z-score normalized heatmap of the input data.
    """
    z_scores = stats.zscore(df, axis=1)
    fig = Heatmap(
        z=z_scores,
        colorscale='RdBu_r',
        reversescale=False,
        showscale=False
    )
    return fig
