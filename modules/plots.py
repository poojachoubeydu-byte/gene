import plotly.graph_objects as go
import numpy as np
import pandas as pd
import gseapy as gp
from dash import dash_table, html


def create_volcano_plot(df, logfc_col, padj_col, gene_col,
                        lfc_thresh=1.0, p_thresh=0.05,
                        filtered_genes=None):

    d = df.copy()

    if d.empty or logfc_col not in d.columns:
        return go.Figure().update_layout(
            title="Load data or click Load Demo to begin",
            template='simple_white',
            xaxis_title="Log2 Fold Change",
            yaxis_title="-log10(Adjusted P-value)"
        )

    d['minus_log10_p'] = -np.log10(d[padj_col].replace(0, 1e-300))

    d['status'] = 'Not Significant'
    d.loc[(d[logfc_col] > lfc_thresh) & (d[padj_col] < p_thresh), 'status'] = 'Up-regulated'
    d.loc[(d[logfc_col] < -lfc_thresh) & (d[padj_col] < p_thresh), 'status'] = 'Down-regulated'

    color_map = {
        'Up-regulated':    '#E31A1C',
        'Down-regulated':  '#1F78B4',
        'Not Significant': '#CCCCCC'
    }

    fig = go.Figure()

    for status, color in color_map.items():
        sub = d[d['status'] == status].copy()
        if filtered_genes is not None and len(filtered_genes) > 0:
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
                "-log10(padj): %{y:.3f}"
                "<extra></extra>"
            )
        ))

    fig.add_hline(y=-np.log10(p_thresh), line_dash="dash",
                  line_color="grey", opacity=0.6)
    fig.add_vline(x=lfc_thresh, line_dash="dash",
                  line_color="grey", opacity=0.6)
    fig.add_vline(x=-lfc_thresh, line_dash="dash",
                  line_color="grey", opacity=0.6)

    sig = d[d['status'] != 'Not Significant']
    if not sig.empty:
        top15 = sig.nlargest(15, 'minus_log10_p')
        for row in top15.itertuples():
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
        title=dict(text="Differential Expression \u2014 Volcano Plot",
                   font=dict(size=14)),
        xaxis_title="Log2 Fold Change",
        yaxis_title="-log10(Adjusted P-value)",
        hovermode='closest',
        dragmode='lasso',
        clickmode='event+select',
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=80, b=50)
    )

    return fig


def create_pathway_enrichment(selected_genes):
    """
    ORA via GSEApy Enrichr against KEGG_2021_Human and Reactome_2022.
    Statistical method: Fisher exact test, Benjamini-Hochberg correction.
    ALWAYS returns (fig, table_component, gene_index_dict) — exactly 3 values.
    """

    # ── Guard: too few genes ─────────────────────────────
    if not selected_genes or len(selected_genes) < 5:
        fig = go.Figure().update_layout(
            title="Select \u22655 genes on the volcano plot to run enrichment",
            template='simple_white'
        )
        msg = html.P(
            "Select at least 5 genes using lasso selection on the volcano plot.",
            style={'color': '#666', 'fontStyle': 'italic', 'padding': '10px'}
        )
        return fig, msg, {}

    # ── Run Enrichr ──────────────────────────────────────
    try:
        enr = gp.enrichr(
            gene_list=selected_genes,
            gene_sets=['KEGG_2021_Human', 'Reactome_2022'],
            outdir=None,
            cutoff=0.05
        )
        results = enr.results.copy()
    except Exception as e:
        fig = go.Figure().update_layout(
            title="Enrichr API unavailable \u2014 please retry in a moment",
            template='simple_white'
        )
        msg = html.P(
            f"Enrichr API error: {str(e)[:120]}. Please retry.",
            style={'color': '#c0392b', 'padding': '10px', 'fontSize': '12px'}
        )
        return fig, msg, {}

    # ── Filter significant results ───────────────────────
    results = results[results['Adjusted P-value'] < 0.05].copy()

    if results.empty:
        fig = go.Figure().update_layout(
            title="No significant pathways at padj < 0.05",
            template='simple_white'
        )
        msg = html.P(
            "No significant KEGG or Reactome pathways found. "
            "Try selecting more genes or relaxing thresholds.",
            style={'color': '#666', 'padding': '10px', 'fontSize': '12px'}
        )
        return fig, msg, {}

    # ── Fix Overlap: "3/120" → integer 3 ────────────────
    results['overlap_count'] = (
        results['Overlap'].str.split('/').str[0].astype(int)
    )
    min_o = results['overlap_count'].min()
    max_o = results['overlap_count'].max()
    results['bubble_size'] = (
        8 + 32 * (results['overlap_count'] - min_o) / max(max_o - min_o, 1)
    )

    # ── Sort and limit ───────────────────────────────────
    results = results.sort_values('Combined Score', ascending=False).head(20)

    # ── Build gene index dict ────────────────────────────
    gene_index_dict = {}
    for _, row in results.iterrows():
        genes = [g.strip() for g in str(row['Genes']).split(';') if g.strip()]
        gene_index_dict[row['Term']] = genes

    # ── Bubble chart ─────────────────────────────────────
    neg_log_p = -np.log10(results['Adjusted P-value'].clip(lower=1e-300))

    fig = go.Figure(data=[go.Scatter(
        x=results['Combined Score'],
        y=results['Term'],
        mode='markers',
        marker=dict(
            size=results['bubble_size'].tolist(),
            sizemode='diameter',
            color=neg_log_p.tolist(),
            colorscale='RdYlBu',
            reversescale=True,
            showscale=True,
            colorbar=dict(title='-log10<br>(Adj.P)', thickness=12)
        ),
        customdata=results['Term'].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Combined Score: %{x:.1f}<br>"
            "Overlap: %{text}<br>"
            "-log10(Adj.P): %{marker.color:.2f}"
            "<extra></extra>"
        ),
        text=results['Overlap'].tolist()
    )])

    fig.update_layout(
        title=dict(text='Pathway Enrichment (KEGG + Reactome)',
                   font=dict(size=13)),
        xaxis_title='Combined Score',
        yaxis_title='',
        template='simple_white',
        margin=dict(l=220, r=20, t=60, b=40),
        height=480,
        yaxis=dict(tickfont=dict(size=10))
    )

    # ── Table ─────────────────────────────────────────────
    table_cols = ['Term', 'Overlap', 'Adjusted P-value', 'Combined Score']
    if 'Gene_set' in results.columns:
        table_cols = ['Term', 'Gene_set', 'Overlap',
                      'Adjusted P-value', 'Combined Score']

    table = dash_table.DataTable(
        data=results[table_cols].head(10).to_dict('records'),
        columns=[
            {'name': 'Pathway',    'id': 'Term'},
            {'name': 'Database',   'id': 'Gene_set'},
            {'name': 'Overlap',    'id': 'Overlap'},
            {'name': 'Adj. P-val', 'id': 'Adjusted P-value',
             'type': 'numeric', 'format': {'specifier': '.2e'}},
            {'name': 'Score',      'id': 'Combined Score',
             'type': 'numeric', 'format': {'specifier': '.1f'}},
        ],
        style_table={'overflowX': 'auto', 'maxHeight': '220px',
                     'overflowY': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px 8px',
                    'fontSize': '11px', 'whiteSpace': 'normal',
                    'height': 'auto', 'maxWidth': '180px'},
        style_header={'backgroundColor': '#f0f4f8', 'fontWeight': 'bold',
                      'fontSize': '11px'},
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'}
        ],
        export_format='csv',
        export_headers='display',
        page_size=10
    )

    # ── CRITICAL: always return exactly 3 values ─────────
    return fig, table, gene_index_dict


def create_heatmap(df, selected_genes):
    """
    Fold-change profile heatmap.
    INPUT: DEG summary table (one row per gene, columns include
    'symbol', 'log2FC', 'padj'). NOT a per-sample expression matrix.
    Z-score normalization is intentionally omitted — it is mathematically
    invalid on a single numeric column (log2FC). This heatmap correctly
    shows fold-change direction and magnitude per gene.
    """

    if df is None or df.empty or not selected_genes:
        return go.Figure().update_layout(
            title="Select genes on the volcano plot to populate this view",
            template='simple_white'
        )

    filtered = df[df['symbol'].isin(selected_genes)].copy()

    if len(filtered) < 2:
        return go.Figure().update_layout(
            title="Select at least 2 genes to display the heatmap",
            template='simple_white'
        )

    filtered = filtered.sort_values('log2FC', ascending=True)

    z = [[float(v)] for v in filtered['log2FC'].tolist()]
    y_labels = filtered['symbol'].tolist()
    customdata = filtered[['symbol', 'log2FC', 'padj']].values.tolist()

    lfc_abs_max = min(6, filtered['log2FC'].abs().max())

    fig = go.Figure(data=go.Heatmap(
        z=z,
        y=y_labels,
        x=['Log2 Fold Change'],
        colorscale='RdBu',
        zmid=0,
        zmin=-lfc_abs_max,
        zmax=lfc_abs_max,
        colorbar=dict(title='Log2FC', thickness=14),
        customdata=customdata,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Log2FC: %{customdata[1]:.3f}<br>"
            "Adj.P: %{customdata[2]:.2e}"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        title=dict(text='Fold-Change Profile of Selected Genes',
                   font=dict(size=13)),
        template='simple_white',
        margin=dict(l=120, r=30, t=60, b=40),
        yaxis=dict(tickfont=dict(size=10)),
        xaxis=dict(tickfont=dict(size=11))
    )

    return fig
