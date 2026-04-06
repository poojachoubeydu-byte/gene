"""
visualizer_pro.py — Professional GSEA Running Enrichment Score Curve
Computes the classic running ES from DEG data + a gene set, requires no
external GSEA run — it derives the metric from the uploaded CSV.
"""

import numpy as np
import plotly.graph_objects as go


def create_gsea_plot(term, gene_set_genes, ranked_df,
                     gene_col='symbol', score_col='rank_metric'):
    """
    Render the classic GSEA running enrichment score (ES) curve.

    Parameters
    ----------
    term : str
        Pathway / gene-set name shown in the plot title.
    gene_set_genes : list[str]
        Genes that belong to this pathway (from Enrichr 'Genes' column).
    ranked_df : pd.DataFrame
        DEG table with at minimum 'symbol', 'log2FC', 'padj' columns.
        Will be ranked internally by the signed -log10(padj) metric.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if ranked_df is None or ranked_df.empty or not gene_set_genes:
        fig = go.Figure()
        fig.update_layout(
            title="Click a pathway bubble to view its GSEA enrichment curve",
            template='simple_white',
            height=280
        )
        return fig

    df = ranked_df.copy()

    # Build ranking metric: sign(log2FC) × -log10(padj)  (standard GSEA input)
    df['_padj_safe'] = df['padj'].clip(lower=1e-300)
    df['_rank'] = np.sign(df['log2FC']) * (-np.log10(df['_padj_safe']))
    df = df.sort_values('_rank', ascending=False).reset_index(drop=True)

    gene_list   = df['symbol'].tolist()
    rank_scores = df['_rank'].values
    hit_set     = set(g.strip().upper() for g in gene_set_genes if g.strip())

    N   = len(gene_list)
    hits = [i for i, g in enumerate(gene_list) if g.upper() in hit_set]
    Nh  = len(hits)

    if Nh == 0:
        fig = go.Figure()
        fig.update_layout(title=f"No genes from '{term}' found in ranked list",
                          template='simple_white', height=280)
        return fig

    # Compute running ES (weighted Kolmogorov–Smirnov statistic)
    hit_mask = np.array([1 if g.upper() in hit_set else 0 for g in gene_list])
    Nr = np.sum(np.abs(rank_scores) * hit_mask)
    Nr = Nr if Nr > 0 else 1.0

    running_es = np.zeros(N)
    es = 0.0
    for i in range(N):
        if hit_mask[i]:
            es += abs(rank_scores[i]) / Nr
        else:
            es -= 1.0 / max(N - Nh, 1)
        running_es[i] = es

    peak_idx = int(np.argmax(np.abs(running_es)))
    es_score = running_es[peak_idx]
    color    = '#E31A1C' if es_score >= 0 else '#1F78B4'

    # Running ES line
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(N)),
        y=running_es.tolist(),
        mode='lines',
        name='Running ES',
        line=dict(color=color, width=2),
        hovertemplate='Rank %{x}<br>ES: %{y:.3f}<extra></extra>'
    ))

    # Peak annotation
    fig.add_vline(x=peak_idx, line_dash='dot', line_color='grey', opacity=0.7)
    fig.add_annotation(
        x=peak_idx, y=es_score,
        text=f"ES = {es_score:.3f}",
        showarrow=True, arrowhead=2, arrowcolor=color,
        font=dict(size=10, color=color),
        bgcolor='rgba(255,255,255,0.8)'
    )

    # Hit rug at the bottom
    hit_x = [i for i in range(N) if hit_mask[i]]
    fig.add_trace(go.Scatter(
        x=hit_x,
        y=[-abs(running_es.min()) * 0.1 - 0.02] * len(hit_x),
        mode='markers',
        marker=dict(symbol='line-ns', size=8, color='#555', line=dict(width=1)),
        name='Gene hits',
        hovertemplate='%{customdata}<extra>Gene hit</extra>',
        customdata=[gene_list[i] for i in hit_x]
    ))

    trunc = (term[:55] + '…') if len(term) > 55 else term
    fig.update_layout(
        title=dict(text=f'GSEA Curve — {trunc}', font=dict(size=12)),
        xaxis_title='Gene rank',
        yaxis_title='Running ES',
        template='simple_white',
        height=280,
        margin=dict(l=50, r=20, t=46, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1, font=dict(size=10)),
        showlegend=True
    )
    return fig
