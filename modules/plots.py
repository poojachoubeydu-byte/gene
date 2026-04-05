import plotly.graph_objects as go
import numpy as np
import pandas as pd
import gseapy as gp
from dash import dash_table, html
import requests


def create_volcano_plot(df, logfc_col, padj_col, gene_col,
                        lfc_thresh=1.0, p_thresh=0.05,
                        filtered_genes=None):
    """
    Volcano plot of differential gene expression.
    X-axis: log2 fold change. Y-axis: -log10(adjusted p-value).
    Supports lasso selection and pathway-driven gene highlighting.
    """
    d = df.copy()

    if d.empty:
        return go.Figure().update_layout(
            title="Load data or click 'Load Demo' to begin",
            template='simple_white',
            xaxis_title="Log2 Fold Change",
            yaxis_title="-log10(Adjusted P-value)"
        )

    d['minus_log10_p'] = -np.log10(d[padj_col].replace(0, 1e-300))

    d['status'] = 'Not Significant'
    d.loc[(d[logfc_col] > lfc_thresh) & (d[padj_col] < p_thresh), 'status'] = 'Up-regulated'
    d.loc[(d[logfc_col] < -lfc_thresh) & (d[padj_col] < p_thresh), 'status'] = 'Down-regulated'

    color_map = {
        'Up-regulated': '#E31A1C',
        'Down-regulated': '#1F78B4',
        'Not Significant': '#CCCCCC'
    }

    fig = go.Figure()

    for status, color in color_map.items():
        sub = d[d['status'] == status].copy()
        if filtered_genes and len(filtered_genes) > 0:
            sub = sub[sub[gene_col].isin(filtered_genes)]
        fig.add_trace(go.Scattergl(
            x=sub[logfc_col],
            y=sub['minus_log10_p'],
            mode='markers',
            name=status,
            marker=dict(color=color, size=6, opacity=0.75),
            customdata=sub[gene_col].values,
            hovertemplate=(
                "<b>%{customdata}</b><br>"
                "Log2FC: %{x:.3f}<br>"
                "-log10(padj): %{y:.3f}<extra></extra>"
            )
        ))

    fig.add_hline(y=-np.log10(p_thresh), line_dash="dash", line_color="grey", opacity=0.6)
    fig.add_vline(x=lfc_thresh, line_dash="dash", line_color="grey", opacity=0.6)
    fig.add_vline(x=-lfc_thresh, line_dash="dash", line_color="grey", opacity=0.6)

    sig = d[d['status'] != 'Not Significant'].nlargest(15, 'minus_log10_p')
    for row in sig.itertuples():
        fig.add_annotation(
            x=getattr(row, logfc_col),
            y=row.minus_log10_p,
            text=getattr(row, gene_col),
            showarrow=False,
            font=dict(size=9, color='#333333'),
            xanchor='left',
            yanchor='bottom'
        )

    fig.update_layout(
        template='simple_white',
        title=dict(text="Differential Expression — Volcano Plot", font=dict(size=14)),
        xaxis_title="Log2 Fold Change",
        yaxis_title="-log10(Adjusted P-value)",
        hovermode='closest',
        dragmode='lasso',
        clickmode='event+select',
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        margin=dict(l=50, r=30, t=80, b=50)
    )

    return fig


def create_pathway_enrichment(selected_genes):
    """
    Runs Enrichr ORA via GSEApy against KEGG_2021_Human and Reactome_2022.
    Returns: (fig, table_component, gene_index_dict)
    gene_index_dict = {pathway_term: [gene1, gene2, ...]}
    Scientific note: Uses Fisher exact test with BH correction (Enrichr default).
    Minimum 5 genes required for meaningful ORA.
    """
    if selected_genes is None or len(selected_genes) < 5:
        return (
            go.Figure().update_layout(
                title="Select \u22655 genes on the volcano plot to run enrichment",
                template='simple_white'
            ),
            html.P("Select at least 5 genes using lasso selection.",
                   style={'color': '#666', 'fontStyle': 'italic', 'padding': '10px'}),
            {}
        )

    try:
        enr = gp.enrichr(
            gene_list=selected_genes,
            gene_sets=['KEGG_2021_Human', 'Reactome_2022'],
            organism='Human',
            outdir=None,
            cutoff=0.05
        )
        results = enr.results.copy()

        results = results[results['Adjusted P-value'] < 0.05]

        if results.empty:
            return (
                go.Figure().update_layout(
                    title="No significant pathways found at padj < 0.05",
                    template='simple_white'
                ),
                html.P("No significant KEGG or Reactome pathways at adjusted p-value < 0.05."),
                {}
            )

        results['overlap_count'] = results['Overlap'].str.split('/').str[0].astype(int)
        min_o = results['overlap_count'].min()
        max_o = results['overlap_count'].max()
        results['bubble_size'] = 8 + 32 * (results['overlap_count'] - min_o) / max(max_o - min_o, 1)

        results = results.sort_values('Combined Score', ascending=False).head(20)

        gene_index_dict = {}
        for _, row in results.iterrows():
            genes = [g.strip() for g in str(row['Genes']).split(';') if g.strip()]
            gene_index_dict[row['Term']] = genes

        fig = go.Figure(data=[go.Scatter(
            x=results['Combined Score'],
            y=results['Term'],
            mode='markers',
            marker=dict(
                size=results['bubble_size'],
                sizemode='diameter',
                color=-np.log10(results['Adjusted P-value'] + 1e-300),
                colorscale='RdYlBu',
                reversescale=True,
                showscale=True,
                colorbar=dict(title='-log10<br>(Adj.P)', thickness=12)
            ),
            customdata=results['Term'].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Combined Score: %{x:.1f}<br>"
                "Overlap: " + results['Overlap'].astype(str) + "<br>"
                "Adj.P: " + results['Adjusted P-value'].map('{:.2e}'.format) +
                "<extra></extra>"
            ),
            text=results['Gene_set'] if 'Gene_set' in results.columns else results['Term']
        )])

        fig.update_layout(
            title=dict(text='Pathway Enrichment (KEGG + Reactome)', font=dict(size=13)),
            xaxis_title='Combined Score',
            yaxis_title='',
            template='simple_white',
            margin=dict(l=220, r=20, t=60, b=40),
            height=500,
            yaxis=dict(tickfont=dict(size=10))
        )

        display_cols = ['Term', 'Gene_set', 'Overlap', 'Adjusted P-value', 'Combined Score']
        display_cols = [c for c in display_cols if c in results.columns]
        table = dash_table.DataTable(
            data=results[display_cols].head(10).to_dict('records'),
            columns=[
                {'name': 'Pathway', 'id': 'Term'},
                {'name': 'Database', 'id': 'Gene_set'},
                {'name': 'Overlap', 'id': 'Overlap'},
                {'name': 'Adj. P-val', 'id': 'Adjusted P-value',
                 'type': 'numeric', 'format': {'specifier': '.2e'}},
                {'name': 'Score', 'id': 'Combined Score',
                 'type': 'numeric', 'format': {'specifier': '.1f'}},
            ],
            style_table={'overflowX': 'auto', 'maxHeight': '250px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '6px 10px',
                        'fontSize': '11px', 'whiteSpace': 'normal',
                        'height': 'auto', 'maxWidth': '200px'},
            style_header={'backgroundColor': '#f0f4f8', 'fontWeight': 'bold',
                          'fontSize': '11px'},
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'}
            ],
            export_format='csv',
            export_headers='display',
            page_size=10
        )

        return (fig, table, gene_index_dict)

    except Exception as e:
        return (
            go.Figure().update_layout(
                title="Enrichment failed \u2014 Enrichr API may be unavailable. Retry in a moment.",
                template='simple_white'
            ),
            html.P("Enrichr API error. Please retry."),
            {}
        )


def create_heatmap(df, selected_genes):
    """
    Fold-change profile heatmap of selected genes.
    SCIENTIFIC NOTE: Input is a DEG summary table (one row per gene),
    NOT a per-sample expression matrix. Therefore this heatmap correctly
    shows log2 fold-change magnitude and direction per gene.
    A per-sample expression matrix would be required for a true
    expression heatmap. Z-score normalization is intentionally NOT used
    here as it would be mathematically invalid on a single numeric column.
    """
    if df.empty or not selected_genes:
        return go.Figure().update_layout(
            title="Select genes on the volcano plot to populate this view",
            template='simple_white'
        )

    filtered = df[df['symbol'].isin(selected_genes)].copy()

    if len(filtered) < 2:
        return go.Figure().update_layout(
            title="Select at least 2 genes to display",
            template='simple_white'
        )

    filtered = filtered.sort_values('log2FC', ascending=True)

    z = [[v] for v in filtered['log2FC'].tolist()]
    y_labels = filtered['symbol'].tolist()

    customdata = filtered[['symbol', 'log2FC', 'padj']].values.tolist()

    fig = go.Figure(data=go.Heatmap(
        z=z,
        y=y_labels,
        x=['Log2 Fold Change'],
        colorscale='RdBu',
        zmid=0,
        zmin=max(-6, filtered['log2FC'].min()),
        zmax=min(6, filtered['log2FC'].max()),
        colorbar=dict(title='Log2FC', thickness=14),
        customdata=customdata,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Log2FC: %{customdata[1]:.3f}<br>"
            "Adj.P: %{customdata[2]:.2e}<extra></extra>"
        )
    ))

    fig.update_layout(
        title=dict(text='Fold-Change Profile of Selected Genes', font=dict(size=13)),
        template='simple_white',
        margin=dict(l=130, r=30, t=60, b=40),
        yaxis=dict(tickfont=dict(size=10)),
        xaxis=dict(tickfont=dict(size=11))
    )

    return fig
