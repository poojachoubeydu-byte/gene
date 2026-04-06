import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import networkx as nx
import gseapy as gp
from dash import dash_table, html


def create_volcano_plot(df, logfc_col, padj_col, gene_col,
                        lfc_thresh=1.0, p_thresh=0.05,
                        filtered_genes=None):

    d = df.copy()

    if d.empty or logfc_col not in d.columns:
        fig = go.Figure().update_layout(
            title="Load data or click Load Demo to begin",
            template='simple_white',
            xaxis_title="Log2 Fold Change",
            yaxis_title="-log10(Adjusted P-value)"
        )
        # ADDED: FEATURE-D — empty state annotation when no data at all
        if len(df) == 0:
            fig.add_annotation(
                text="⬆ Upload a DESeq2 / edgeR CSV or click Load Demo Data",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=15, color="#aaaaaa"),
                align="center"
            )
        return fig

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
            ),
            selected=dict(
                marker=dict(
                    opacity=1.0,
                    size=9,
                    color=color
                )
            ),
            unselected=dict(
                marker=dict(
                    opacity=0.2,
                    size=5,
                    color=color
                )
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


def create_pathway_enrichment(selected_genes, gene_sets='KEGG_2021_Human'):  # ADDED: FEATURE-B
    """
    ORA via GSEApy Enrichr against selected gene set database.
    Statistical method: Fisher exact test, Benjamini-Hochberg correction.
    ALWAYS returns (fig, table_component, gene_index_dict, enrichment_json) — exactly 4 values.
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
        return fig, msg, {}, None  # FIXED: BUG-1

    # ── Run Enrichr ──────────────────────────────────────
    try:
        enr = gp.enrichr(
            gene_list=selected_genes,
            gene_sets=gene_sets if gene_sets else 'KEGG_2021_Human',  # ADDED: FEATURE-B
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
        return fig, msg, {}, None  # FIXED: BUG-1

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
        return fig, msg, {}, None  # FIXED: BUG-1

    # FIXED: BUG-2 — robust Overlap parsing ("5/100" or plain int); proper sizeref
    results = results.copy()
    results['overlap_count'] = results['Overlap'].apply(
        lambda x: int(str(x).split('/')[0]) if '/' in str(x) else int(x)
    )
    max_overlap = results['overlap_count'].max()
    sizeref = 2.0 * max_overlap / (30 ** 2) if max_overlap > 0 else 1

    # ── Sort and limit ───────────────────────────────────
    results = results.sort_values('Combined Score', ascending=False).head(20)

    # ── Normalise Combined Score to 0-100 range for display ──
    # Raw Enrichr Combined Score scale varies by gseapy version.
    cs_max = results['Combined Score'].max()
    if cs_max > 1000:
        results['Combined Score Display'] = (
            results['Combined Score'] / cs_max * 100
        ).round(1)
        score_col = 'Combined Score Display'
        xaxis_label = 'Normalised Combined Score (0–100)'
    else:
        score_col = 'Combined Score'
        xaxis_label = 'Combined Score'

    # ── Truncate long pathway names ──────────────────────
    results['Term_display'] = results['Term'].apply(
        lambda x: x[:50] + '…' if len(str(x)) > 50 else str(x)
    )

    # ── Build gene index dict ────────────────────────────
    gene_index_dict = {}
    for _, row in results.iterrows():
        genes = [g.strip() for g in str(row['Genes']).split(';') if g.strip()]
        gene_index_dict[row['Term']] = genes

    # ── Bubble chart ─────────────────────────────────────
    neg_log_p = -np.log10(results['Adjusted P-value'].clip(lower=1e-300))

    fig = go.Figure(data=[go.Scatter(
        x=results[score_col],
        y=results['Term_display'],
        mode='markers',
        marker=dict(
            size=results['overlap_count'].tolist(),  # FIXED: BUG-2
            sizemode='diameter',
            sizeref=sizeref,                         # FIXED: BUG-2
            color=neg_log_p.tolist(),
            colorscale='RdYlBu',
            reversescale=True,
            showscale=True,
            colorbar=dict(
                title=dict(text='-log10<br>Adj.P', side='right'),
                thickness=14,
                len=0.75,
                x=1.02,
                xanchor='left'
            )
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
        xaxis_title=xaxis_label,
        yaxis_title='',
        template='simple_white',
        margin=dict(l=320, r=80, t=50, b=50),
        height=520,
        xaxis=dict(
            title=dict(
                text='Normalised Combined Score (0–100)',
                font=dict(size=11)
            )
        ),
        yaxis=dict(
            tickfont=dict(size=9),
            automargin=True
        )
    )

    # ── Table ─────────────────────────────────────────────
    table_cols_base = ['Term', 'overlap_count', 'Adjusted P-value']  # FIXED: FEATURE-E
    if 'Gene_set' in results.columns:
        table_cols_base = ['Term', 'Gene_set', 'overlap_count', 'Adjusted P-value']  # FIXED: FEATURE-E
    table_cols_present = table_cols_base + [score_col]

    table = dash_table.DataTable(
        data=results[table_cols_present].head(10).to_dict('records'),
        columns=[
            {'name': 'Pathway',    'id': 'Term'},
            {'name': 'Database',   'id': 'Gene_set'},
            {'name': 'Genes in Overlap', 'id': 'overlap_count', 'type': 'numeric'},  # FIXED: FEATURE-E
            {'name': 'Adj. P-val', 'id': 'Adjusted P-value',
             'type': 'numeric', 'format': {'specifier': '.2e'}},
            {'name': 'Score',      'id': score_col,
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

    # FIXED: BUG-1 — serialize enrichment results for enrichment-results-store
    enrichment_json = results[['Term', 'Genes']].to_json(orient='records')

    # ── CRITICAL: always return exactly 4 values ─────────
    return fig, table, gene_index_dict, enrichment_json


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

    # FIXED: BUG-5 — fall back to bar chart when no per-sample expression columns exist
    numeric_df = filtered.select_dtypes(include=[np.number])
    sample_cols = [c for c in numeric_df.columns
                   if c not in ['log2FC', 'padj', 'baseMean', 'lfcSE', 'stat', 'pvalue']]
    if len(sample_cols) < 2:
        bar_df = filtered[['symbol', 'log2FC']].dropna()
        fig = go.Figure(go.Bar(
            x=bar_df['symbol'], y=bar_df['log2FC'],
            marker_color=['#E31A1C' if v > 0 else '#1F78B4' for v in bar_df['log2FC']]
        ))
        fig.update_layout(
            title='Log2 Fold Change of Selected Genes (upload expression matrix for heatmap)',
            xaxis_title='Gene', yaxis_title='Log2FC',
            template='simple_white'
        )
        return fig

    z = [[float(v)] for v in filtered['log2FC'].tolist()]
    y_labels = filtered['symbol'].tolist()
    customdata = filtered[['symbol', 'log2FC', 'padj']].values.tolist()

    # Use actual data range — do not clamp artificially.
    # Symmetric scale centered at 0 preserves relative differences.
    actual_max = float(filtered['log2FC'].abs().max())
    lfc_abs_max = actual_max if actual_max > 0 else 1.0

    fig = go.Figure(data=go.Heatmap(
        z=z,
        y=filtered['symbol'].str.strip().tolist(),
        x=[''],
        colorscale='RdBu_r',
        zmid=0,
        zmin=-lfc_abs_max,
        zmax=lfc_abs_max,
        colorbar=dict(
            title=dict(text='Log2FC', side='right'),
            thickness=16,
            len=0.8
        ),
        customdata=customdata,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Log2FC: %{customdata[1]:.3f}<br>"
            "Adj.P: %{customdata[2]:.2e}"
            "<extra></extra>"
        )
    ))

    n_genes = len(filtered)
    fig.update_layout(
        title=dict(
            text=f'Fold-Change Profile — {n_genes} Selected Genes',
            font=dict(size=12)
        ),
        template='simple_white',
        margin=dict(l=180, r=40, t=50, b=30),
        # FIXED: BUG-3 — explicit tick positioning so labels render on correct rows
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(filtered))),
            ticktext=filtered['symbol'].tolist(),
            tickfont=dict(size=10),
            automargin=True,
            ticklabelposition='outside left'
        ),
        xaxis=dict(showticklabels=False)
    )

    fig.add_annotation(
        text="Red = upregulated  |  Blue = downregulated  "
             "|  Sorted by Log2FC",
        xref='paper', yref='paper',
        x=0.5, y=-0.08,
        showarrow=False,
        font=dict(size=9, color='#666666'),
        xanchor='center'
    )

    return fig


def create_3d_pca(pca_result):
    if not pca_result or 'genes' not in pca_result:
        return go.Figure().update_layout(
            title='Load data to generate the 3D PCA projection',
            template='simple_white'
        )

    df = pd.DataFrame({
        'gene': pca_result['genes'],
        'pc1': pca_result['pc1'],
        'pc2': pca_result['pc2'],
        'pc3': pca_result['pc3'],
        'cluster': pca_result['clusters']
    })

    fig = px.scatter_3d(
        df,
        x='pc1', y='pc2', z='pc3',
        color='cluster',
        hover_name='gene',
        opacity=0.8,
        color_continuous_scale='Spectral',
        labels={'pc1': 'PC1', 'pc2': 'PC2', 'pc3': 'PC3'}
    )

    fig.update_traces(marker=dict(size=5), selector=dict(type='scatter3d'))
    fig.update_layout(
        template='simple_white',
        title='3D PCA Projection',
        margin=dict(l=0, r=0, t=40, b=0),
        legend_title_text='Cluster'
    )
    return fig


def create_gsea_bar(gsea_result):
    if not gsea_result or gsea_result.get('results', pd.DataFrame()).empty:
        return go.Figure().update_layout(
            title='Preranked GSEA results unavailable',
            template='simple_white'
        )

    results = gsea_result['results'].copy()
    if results.empty:
        return go.Figure().update_layout(
            title='No significant pathways available',
            template='simple_white'
        )

    results = results.sort_values('NES', ascending=False).head(12)
    results['Term_trunc'] = results['Term'].apply(
        lambda x: x if len(str(x)) <= 45 else str(x)[:42] + '...'
    )

    fig = go.Figure(data=go.Bar(
        x=results['NES'],
        y=results['Term_trunc'],
        orientation='h',
        marker=dict(
            color=results['FDR q-val'],
            colorscale='RdYlGn',
            reversescale=True,
            colorbar=dict(title='FDR q-val')
        ),
        text=results['Term'].tolist(),
        hovertemplate='Pathway: %{text}<br>NES: %{x:.2f}<br>FDR q-val: %{marker.color:.2e}<extra></extra>'
    ))

    fig.update_layout(
        template='simple_white',
        title='Preranked GSEA Pathway Enrichment',
        xaxis_title='Normalized Enrichment Score (NES)',
        yaxis_title='',
        margin=dict(l=240, r=40, t=50, b=40),
        height=420
    )
    return fig


def create_network_graph(network_data):
    if not network_data or 'graph' not in network_data:
        return go.Figure().update_layout(
            title='No co-expression network available',
            template='simple_white'
        )

    G = network_data['graph']
    if G.number_of_nodes() == 0:
        return go.Figure().update_layout(
            title='No network edges were generated',
            template='simple_white'
        )

    pos = nx.spring_layout(G, seed=42)
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=0.7, color='#888'),
            hoverinfo='none'
        ))

    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers+text',
        marker=dict(
            size=6,
            color=[network_data.get('labels', {}).get(node, 0) for node in G.nodes()],
            colorscale='Viridis',
            showscale=False
        ),
        text=[node for node in G.nodes()],
        hovertemplate='<b>%{text}</b><extra></extra>'
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        template='simple_white',
        title='WGCNA-lite Co-expression Network',
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig


def create_meta_score_bar(df):
    if df is None or df.empty or 'meta_score' not in df.columns:
        return go.Figure().update_layout(
            title='Meta-Score prioritisation will appear here',
            template='simple_white'
        )

    top = df.sort_values('meta_score', ascending=False).head(20)
    fig = go.Figure(data=go.Bar(
        x=top['meta_score'][::-1],
        y=top['symbol'][::-1],
        orientation='h',
        marker=dict(
            color=top['meta_score'][::-1],
            colorscale='RdYlGn',
            cmin=0,
            cmax=100
        ),
        hovertemplate='Gene: %{y}<br>Meta-Score: %{x:.1f}<extra></extra>'
    ))

    fig.update_layout(
        template='simple_white',
        title='Top Genes by Meta-Score',
        xaxis_title='Meta-Score',
        yaxis_title='',
        margin=dict(l=160, r=40, t=50, b=40),
        height=380
    )
    return fig
