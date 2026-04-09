"""
Plots Module — Apex Bioinformatics Platform
============================================
All Plotly figure builders. Every function is stateless and safe for
concurrent workers.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

CLUSTER_PAL = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
               "#1abc9c", "#e67e22", "#16a085"]

_BLANK_LAYOUT = dict(
    template="simple_white",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    margin=dict(l=20, r=20, t=40, b=20),
)


def blank(msg: str = "") -> go.Figure:
    fig = go.Figure()
    if msg:
        fig.add_annotation(text=msg, xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=13, color="#666"))
    fig.update_layout(**_BLANK_LAYOUT)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Volcano Plot  (with cancer-gene overlay)
# ─────────────────────────────────────────────────────────────────────────────

def create_volcano_plot(df, logfc_col, padj_col, gene_col,
                        lfc_thresh=1.0, p_thresh=0.05, filtered_genes=None):
    """
    Volcano plot — shows ONLY significant genes (those passing both lfc_thresh
    and p_thresh).  Cancer-gene and FDA drug-target overlays are retained.
    Non-significant gray cloud is removed to keep focus on actionable hits.
    Threshold dashed lines remain as reference for the significance boundary.
    """
    from data.gene_annotations import CANCER_GENE_CENSUS, DRUG_GENE_INTERACTIONS

    if isinstance(df, list):
        df = pd.DataFrame(df)
    if df.empty:
        return go.Figure()

    df = df.copy()
    df["_nlp"] = -np.log10(df[padj_col].clip(lower=1e-300))

    # Significance masks on the FULL dataset (used for overlay lookups)
    up_mask = (df[logfc_col] >  lfc_thresh) & (df[padj_col] < p_thresh)
    dn_mask = (df[logfc_col] < -lfc_thresh) & (df[padj_col] < p_thresh)
    df["_cat"] = "Non-significant"
    df.loc[up_mask, "_cat"] = "Up-regulated"
    df.loc[dn_mask, "_cat"] = "Down-regulated"

    n_up  = int(up_mask.sum())
    n_dn  = int(dn_mask.sum())
    n_sig = n_up + n_dn

    # ── Keep ONLY significant genes in the plot ───────────────────────────────
    df = df[df["_cat"] != "Non-significant"].copy()
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No significant genes at current thresholds.",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=13, color="#888"))
        fig.update_layout(template="simple_white",
                          xaxis_title="Log2 Fold Change",
                          yaxis_title="−log10(Adj. p-value)")
        return fig

    df[gene_col] = df[gene_col].astype(str)
    has_base = "baseMean" in df.columns

    color_map = {
        "Up-regulated":   "#e74c3c",
        "Down-regulated": "#3498db",
    }

    fig = go.Figure()

    for cat, color in color_map.items():
        sub = df[df["_cat"] == cat]
        if sub.empty:
            continue
        syms      = sub[gene_col].astype(str).str.strip().str.upper().values
        base_vals = sub["baseMean"].values if has_base else np.zeros(len(sub))
        customdata = np.column_stack([syms, base_vals])
        ht = ("<b>%{customdata[0]}</b><br>"
              "Log2FC: %{x:.3f}<br>"
              "−log10(adj.p): %{y:.2f}<br>")
        if has_base:
            ht += "BaseMean: %{customdata[1]:.1f}"
        ht += "<extra></extra>"
        fig.add_trace(go.Scattergl(
            x=sub[logfc_col], y=sub["_nlp"],
            mode="markers", name=cat,
            marker=dict(color=color, size=7, opacity=0.82,
                        line=dict(width=0.4, color="white")),
            customdata=customdata,
            hovertemplate=ht,
        ))

    # ── Oncogene overlay (star, gold border) — only significant ──────────────
    onco_up = df[
        (df["_cat"] == "Up-regulated") &
        df[gene_col].str.upper().map(
            lambda g: CANCER_GENE_CENSUS.get(g, {}).get("role") == "Oncogene"
        )
    ]
    if not onco_up.empty:
        cd = np.column_stack([
            onco_up[gene_col].values,
            onco_up[gene_col].str.upper().map(
                lambda g: CANCER_GENE_CENSUS.get(g, {}).get("cancer_types", ["?"])[0]
            ).values,
        ])
        fig.add_trace(go.Scattergl(
            x=onco_up[logfc_col], y=onco_up["_nlp"],
            mode="markers+text", name="Oncogene ↑",
            text=onco_up[gene_col], textposition="top center",
            textfont=dict(size=9, color="#c0392b"),
            marker=dict(symbol="star", size=14, color="#e74c3c",
                        line=dict(color="#f39c12", width=2)),
            customdata=cd,
            hovertemplate="<b>%{customdata[0]}</b> (Oncogene)<br>"
                          "Cancer: %{customdata[1]}<extra></extra>",
        ))

    # ── TSG overlay (diamond, blue border) — only significant ────────────────
    tsg_dn = df[
        (df["_cat"] == "Down-regulated") &
        df[gene_col].str.upper().map(
            lambda g: CANCER_GENE_CENSUS.get(g, {}).get("role") == "TSG"
        )
    ]
    if not tsg_dn.empty:
        cd2 = np.column_stack([
            tsg_dn[gene_col].values,
            tsg_dn[gene_col].str.upper().map(
                lambda g: CANCER_GENE_CENSUS.get(g, {}).get("cancer_types", ["?"])[0]
            ).values,
        ])
        fig.add_trace(go.Scattergl(
            x=tsg_dn[logfc_col], y=tsg_dn["_nlp"],
            mode="markers+text", name="TSG ↓",
            text=tsg_dn[gene_col], textposition="bottom center",
            textfont=dict(size=9, color="#2980b9"),
            marker=dict(symbol="diamond", size=12, color="#3498db",
                        line=dict(color="#1a5276", width=2)),
            customdata=cd2,
            hovertemplate="<b>%{customdata[0]}</b> (Tumor Suppressor)<br>"
                          "Cancer: %{customdata[1]}<extra></extra>",
        ))

    # ── FDA drug-target open-circle overlay ──────────────────────────────────
    annotated = set()
    if not onco_up.empty:
        annotated |= set(onco_up[gene_col].str.upper())
    if not tsg_dn.empty:
        annotated |= set(tsg_dn[gene_col].str.upper())

    drug_sig = df[df[gene_col].str.upper().isin(DRUG_GENE_INTERACTIONS.keys())]
    drug_sig = drug_sig[~drug_sig[gene_col].str.upper().isin(annotated)]
    if not drug_sig.empty:
        _drug_cd = np.column_stack([
            drug_sig[gene_col].values,
            np.zeros(len(drug_sig)),
        ])
        fig.add_trace(go.Scattergl(
            x=drug_sig[logfc_col], y=drug_sig["_nlp"],
            mode="markers", name="Drug target",
            marker=dict(symbol="circle-open", size=15, color="#27ae60",
                        line=dict(color="#27ae60", width=2)),
            customdata=_drug_cd,
            hovertemplate="<b>%{customdata[0]}</b> — FDA drug target<extra></extra>",
        ))

    # ── Threshold reference lines ─────────────────────────────────────────────
    fig.add_hline(y=-np.log10(p_thresh), line_dash="dash",
                  line_color="gray", opacity=0.4,
                  annotation_text=f"p={p_thresh}", annotation_position="right",
                  annotation_font_size=9)
    fig.add_vline(x= lfc_thresh, line_dash="dash", line_color="gray", opacity=0.4)
    fig.add_vline(x=-lfc_thresh, line_dash="dash", line_color="gray", opacity=0.4)

    # ── Label top non-annotated significant genes ─────────────────────────────
    already_labelled = annotated.copy()
    if not drug_sig.empty:
        already_labelled |= set(drug_sig[gene_col].str.upper())

    sig_rest = df[
        ~df[gene_col].str.upper().isin(already_labelled)
    ].nlargest(15, "_nlp")
    for rec in sig_rest[[logfc_col, "_nlp", gene_col]].to_dict("records"):
        fig.add_annotation(
            x=rec[logfc_col], y=rec["_nlp"],
            text=str(rec[gene_col]),
            showarrow=False, font=dict(size=9), yshift=9,
        )

    fig.update_layout(
        template="simple_white",
        title=dict(
            text=(
                f"Significant Genes  "
                f"<span style='color:#e74c3c'>▲ {n_up} up</span>  "
                f"<span style='color:#3498db'>▼ {n_dn} down</span>  "
                f"<span style='font-size:11px;color:#888;'>"
                f"★ Oncogene  ◆ TSG  ○ Drug target  "
                f"— lasso or box-select to run enrichment</span>"
            ),
            font=dict(size=13),
        ),
        xaxis_title="Log2 Fold Change",
        yaxis_title="−log10(Adj. p-value)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        dragmode="select",
        clickmode="event+select",
        hovermode="closest",
        margin=dict(t=65),
        selections=[],
        newselection=dict(line=dict(color="#3498db", width=1, dash="dot")),
        activeselection=dict(fillcolor="#3498db", opacity=0.07),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Meta-Score Bar
# ─────────────────────────────────────────────────────────────────────────────

def create_meta_score_bar(df: pd.DataFrame) -> go.Figure:
    if isinstance(df, list):
        df = pd.DataFrame(df)
    if df is None or df.empty or "meta_score" not in df.columns:
        return blank("No meta-score data.")

    HOUSE = {"ACTB", "GAPDH", "B2M", "RPLP0", "HPRT1", "GUSB", "TBP"}
    top = df[~df["symbol"].isin(HOUSE)].nlargest(20, "meta_score")

    fig = go.Figure(go.Bar(
        x=top["meta_score"][::-1],
        y=top["symbol"][::-1],
        orientation="h",
        marker=dict(
            color=top["meta_score"][::-1],
            colorscale="RdYlGn", cmin=0, cmax=100,
            showscale=True,
            colorbar=dict(title="Meta-Score", thickness=12),
        ),
        hovertemplate="<b>%{y}</b>: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(
        template="simple_white",
        title="Top Genes by Meta-Score",
        xaxis_title="Meta-Score (0–100)",
        margin=dict(l=80, r=20, t=50, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Pathway Bubble + Bar
# ─────────────────────────────────────────────────────────────────────────────

def create_pathway_bubble(enr: pd.DataFrame) -> go.Figure:
    """
    Industry-standard enrichment bubble chart (ClusterProfiler / enrichplot style).

    Axes  : X = GeneRatio (overlap / query size)  — biological coverage
    Size  : √(overlap_count) × 4 — absolute hit count, sqrt-scaled
    Color : −log10(adjusted_p_value) — statistical significance
    Shape : direction (▲ Activated, ▼ Inhibited, ● Inconclusive)
    """
    if enr is None or enr.empty:
        return blank("No significant pathways found.")

    top = enr.head(20).copy()
    top["nlp"] = -np.log10(top["adjusted_p_value"].clip(lower=1e-300))

    # X-axis: GeneRatio if available, else fall back to enrichment_score
    x_col   = "gene_ratio"   if "gene_ratio"   in top.columns else "enrichment_score"
    x_title = "Gene Ratio (overlap / query size)" if x_col == "gene_ratio" \
              else "−log10(Adj. p-value)"

    # Pathway labels with database prefix
    top["label"] = top["pathway"].str[:50]
    if "database" in top.columns:
        top["label"] = top["database"].str.upper().str[:3] + " · " + top["label"]

    # Sort by GeneRatio descending so highest-coverage pathways appear at top
    top = top.sort_values(x_col, ascending=False).reset_index(drop=True)
    # Reverse for horizontal layout (highest at top of y-axis)
    top = top[::-1].reset_index(drop=True)

    # Bubble sizes: √(overlap) — avoids extreme spread on large pathways
    sizes = np.sqrt(top["overlap_count"].clip(lower=1)) * 4.5
    sizes = np.clip(sizes, 8, 45)

    # Direction → marker symbol (Plotly supports per-point symbol arrays)
    has_dir = "direction" in top.columns
    if has_dir:
        sym_map = {"Activated": "triangle-up",
                   "Inhibited": "triangle-down",
                   "Inconclusive": "circle"}
        symbols = top["direction"].map(sym_map).fillna("circle").tolist()
    else:
        symbols = "circle"

    # Optional extra hover columns
    has_or  = "odds_ratio"            in top.columns
    has_az  = "activation_zscore"     in top.columns
    has_ews = "effect_weighted_score" in top.columns
    has_gr  = "gene_ratio"            in top.columns

    # Build customdata array — up to 6 columns
    cd_cols  = ["overlap_count", "pathway_size", "adjusted_p_value"]
    cd_extra = []
    if has_or:  cd_extra.append("odds_ratio")
    if has_az:  cd_extra.append("activation_zscore")
    if has_dir: cd_extra.append("direction")
    all_cd = cd_cols + cd_extra
    customdata = top[all_cd].values

    # Build hover template
    ht_lines = [
        "<b>%{y}</b>",
        "Gene Ratio: %{x:.3f}" if x_col == "gene_ratio" else "Score: %{x:.2f}",
        "Overlap: %{customdata[0]:.0f} / %{customdata[1]:.0f}",
        "Adj. p: %{customdata[2]:.2e}",
    ]
    idx = 3
    if has_or:
        ht_lines.append(f"Odds Ratio: %{{customdata[{idx}]:.2f}}")
        idx += 1
    if has_az:
        ht_lines.append(f"Activation z: %{{customdata[{idx}]:.2f}}")
        idx += 1
    if has_dir:
        ht_lines.append(f"Direction: %{{customdata[{idx}]}}")
    hover_template = "<br>".join(ht_lines) + "<extra></extra>"

    fig = go.Figure(go.Scatter(
        x=top[x_col],
        y=top["label"],
        mode="markers",
        marker=dict(
            size=sizes,
            color=top["nlp"],
            colorscale="Viridis",
            reversescale=False,
            showscale=True,
            colorbar=dict(
                title=dict(text="−log10<br>adj. p", side="right"),
                thickness=13, len=0.7,
            ),
            symbol=symbols,
            line=dict(width=1, color="white"),
        ),
        customdata=customdata,
        hovertemplate=hover_template,
    ))

    # Direction legend annotations (top-right corner)
    if has_dir:
        dir_legend = (
            "▲ Activated  ▼ Inhibited  ● Inconclusive"
        )
        fig.add_annotation(
            text=dir_legend, xref="paper", yref="paper",
            x=1.0, y=-0.06, showarrow=False,
            font=dict(size=9, color="#555"), xanchor="right",
        )

    fig.update_layout(
        template="simple_white",
        title=dict(
            text="Pathway Enrichment  "
                 "<span style='font-size:11px;color:#888;'>"
                 "X=GeneRatio · Size=Overlap · Color=Significance · "
                 "Shape=Direction</span>",
            font=dict(size=13),
        ),
        xaxis=dict(title=x_title, rangemode="tozero"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        height=max(360, 28 * len(top) + 80),
        margin=dict(l=10, r=20, t=50, b=50),
    )
    return fig


def create_pathway_bar(enr: pd.DataFrame) -> go.Figure:
    """
    IPA-style directional bar chart.

    Color : Activated = #c0392b (red), Inhibited = #2980b9 (blue),
            Inconclusive = #7f8c8d (gray)
    X-axis: −log10(Adj. p-value)
    Text  : overlap count + gene ratio %
    """
    if enr is None or enr.empty:
        return blank("No pathways enriched.")

    top = enr.head(15).copy()
    top["nlp"]   = -np.log10(top["adjusted_p_value"].clip(lower=1e-300))
    top["label"] = top["pathway"].str[:48]

    # Direction-based bar colors
    dir_color_map = {
        "Activated":    "#c0392b",
        "Inhibited":    "#2980b9",
        "Inconclusive": "#7f8c8d",
    }
    has_dir = "direction" in top.columns
    if has_dir:
        colors = top["direction"].map(dir_color_map).fillna("#7f8c8d").tolist()
    else:
        colors = "#2c3e50"

    # Text: "N genes (GR: X.X%)"
    has_gr = "gene_ratio" in top.columns
    if has_gr:
        bar_text = (
            top["overlap_count"].astype(str)
            + " (" + (top["gene_ratio"] * 100).round(1).astype(str) + "%)"
        )
    else:
        bar_text = top["overlap_count"].astype(str) + " genes"

    # Odds ratio for hover
    has_or = "odds_ratio" in top.columns
    has_az = "activation_zscore" in top.columns

    # Reverse for horizontal bars: want highest significance at top
    top_r  = top.iloc[::-1].reset_index(drop=True)
    c_rev  = colors[::-1] if isinstance(colors, list) else colors
    bt_rev = bar_text.iloc[::-1].reset_index(drop=True)

    cd_cols = ["adjusted_p_value", "overlap_count"]
    if has_or:  cd_cols.append("odds_ratio")
    if has_az:  cd_cols.append("activation_zscore")
    if has_dir: cd_cols.append("direction")
    customdata = top_r[cd_cols].values

    ht_lines = [
        "<b>%{y}</b>",
        "−log10 adj.p: %{x:.2f}",
        "Adj. p: %{customdata[0]:.2e}",
        "Overlap: %{customdata[1]:.0f} genes",
    ]
    idx = 2
    if has_or:
        ht_lines.append(f"Odds Ratio: %{{customdata[{idx}]:.2f}}")
        idx += 1
    if has_az:
        ht_lines.append(f"Activation z: %{{customdata[{idx}]:.2f}}")
        idx += 1
    if has_dir:
        ht_lines.append(f"Direction: %{{customdata[{idx}]}}")
    hover_template = "<br>".join(ht_lines) + "<extra></extra>"

    fig = go.Figure(go.Bar(
        x=top_r["nlp"],
        y=top_r["label"],
        orientation="h",
        marker=dict(
            color=c_rev,
            line=dict(color="white", width=0.5),
        ),
        text=bt_rev,
        textposition="inside",
        insidetextanchor="start",
        textfont=dict(size=10, color="white"),
        customdata=customdata,
        hovertemplate=hover_template,
    ))

    # Direction color legend as annotation
    if has_dir:
        fig.add_annotation(
            text=(
                "<span style='color:#c0392b'>■ Activated</span>  "
                "<span style='color:#2980b9'>■ Inhibited</span>  "
                "<span style='color:#7f8c8d'>■ Inconclusive</span>"
            ),
            xref="paper", yref="paper",
            x=0.5, y=-0.08,
            showarrow=False, font=dict(size=9), xanchor="center",
        )

    fig.update_layout(
        template="simple_white",
        title=dict(
            text="Top Enriched Pathways  "
                 "<span style='font-size:11px;color:#888;'>"
                 "color = activation direction</span>",
            font=dict(size=13),
        ),
        xaxis_title="−log10(Adj. p-value)",
        height=max(340, 26 * len(top) + 80),
        margin=dict(l=10, r=10, t=50, b=55),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Pathway Crosstalk Heatmap  (Jaccard similarity)
# ─────────────────────────────────────────────────────────────────────────────

def create_pathway_crosstalk(crosstalk_df: pd.DataFrame) -> go.Figure:
    if crosstalk_df is None or crosstalk_df.empty:
        return blank("Insufficient enriched pathways for crosstalk analysis.")

    fig = go.Figure(go.Heatmap(
        z=crosstalk_df.values,
        x=crosstalk_df.columns.tolist(),
        y=crosstalk_df.index.tolist(),
        colorscale="YlOrRd",
        zmin=0, zmax=1,
        zsmooth=False,
        hovertemplate=(
            "<b>%{y}</b><br>vs <b>%{x}</b><br>"
            "Jaccard similarity: %{z:.2f}<extra></extra>"
        ),
        colorbar=dict(title="Jaccard", thickness=12),
    ))
    fig.update_layout(
        template="simple_white",
        title="Pathway Crosstalk — Jaccard Gene Overlap",
        xaxis=dict(tickangle=-40, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
        height=440,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Drug Target Chart
# ─────────────────────────────────────────────────────────────────────────────

def create_drug_target_chart(drug_df: pd.DataFrame, deg_df: pd.DataFrame) -> go.Figure:
    """
    Scatter: x = Log2FC of gene, y = number of FDA drugs targeting it.
    Bubble size = number of total drugs (FDA + clinical).
    Color = FDA-approved count.
    """
    if drug_df is None or drug_df.empty:
        return blank("No FDA-approved drug targets among significant genes.")

    # Aggregate per gene
    g = drug_df.groupby("gene").agg(
        n_drugs=("drug", "count"),
        n_fda=("fda", "sum"),
        drug_types=("type", lambda x: ", ".join(x.unique()[:3])),
        indications=("indication", lambda x: " | ".join(x.unique()[:2])),
    ).reset_index()

    # Merge with DEG data for fold change
    if deg_df is not None and not deg_df.empty:
        lfc_map = dict(zip(deg_df["symbol"].str.upper(), deg_df["log2FC"]))
        g["log2FC"] = g["gene"].map(lfc_map).fillna(0)
    else:
        g["log2FC"] = 0.0

    fig = go.Figure(go.Scatter(
        x=g["log2FC"],
        y=g["n_fda"],
        mode="markers+text",
        text=g["gene"],
        textposition="top center",
        textfont=dict(size=10),
        marker=dict(
            size=np.clip(g["n_drugs"] * 8, 12, 50),
            color=g["n_fda"],
            colorscale="Greens",
            showscale=True,
            colorbar=dict(title="FDA drugs", thickness=12),
            line=dict(color="#2c3e50", width=1),
            opacity=0.85,
        ),
        customdata=np.column_stack([g["drug_types"], g["indications"], g["n_drugs"]]),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Log2FC: %{x:.2f}<br>"
            "FDA-approved drugs: %{y}<br>"
            "Types: %{customdata[0]}<br>"
            "Indications: %{customdata[1]}<br>"
            "Total drugs: %{customdata[2]}<extra></extra>"
        ),
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="#aaa", opacity=0.5)
    fig.add_annotation(
        x=max(g["log2FC"].max() * 0.6, 0.5), y=g["n_fda"].max() * 0.9,
        text="⬆ Upregulated targets<br>(inhibitor candidates)",
        showarrow=False, font=dict(size=9, color="#c0392b"),
        bgcolor="rgba(255,220,210,0.7)",
    )
    fig.add_annotation(
        x=min(g["log2FC"].min() * 0.6, -0.5), y=g["n_fda"].max() * 0.9,
        text="⬇ Downregulated targets<br>(activator candidates)",
        showarrow=False, font=dict(size=9, color="#2980b9"),
        bgcolor="rgba(210,230,255,0.7)",
    )

    fig.update_layout(
        template="simple_white",
        title="Drug Target Landscape — Significant DE Genes",
        xaxis_title="Log2 Fold Change",
        yaxis_title="# FDA-Approved Drugs Targeting Gene",
        height=440,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Drug type breakdown donut
# ─────────────────────────────────────────────────────────────────────────────

def create_drug_type_donut(drug_df: pd.DataFrame) -> go.Figure:
    if drug_df is None or drug_df.empty:
        return blank("No drug data.")

    counts = drug_df[drug_df["fda"] == True].groupby("type").size().sort_values(ascending=False)
    if counts.empty:
        return blank("No FDA-approved drugs found.")

    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.45,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value} drugs (%{percent})<extra></extra>",
        marker=dict(colors=["#27ae60", "#3498db", "#e74c3c", "#f39c12",
                             "#9b59b6", "#1abc9c", "#e67e22", "#16a085"]),
    ))
    fig.update_layout(
        template="simple_white",
        title="FDA Drug Mechanism Classes",
        height=340,
        showlegend=True,
        legend=dict(font=dict(size=10)),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Biomarker Score Chart
# ─────────────────────────────────────────────────────────────────────────────

def create_biomarker_score_chart(bm_df: pd.DataFrame) -> go.Figure:
    if bm_df is None or bm_df.empty or "biomarker_score" not in bm_df.columns:
        return blank("Run analysis to see biomarker scores.")

    HOUSE = {"ACTB", "GAPDH", "B2M", "RPLP0", "HPRT1", "GUSB", "TBP"}
    top = bm_df[~bm_df["symbol"].isin(HOUSE)].nlargest(25, "biomarker_score")

    colors = []
    for _, row in top.iterrows():
        role = str(row.get("cancer_role", ""))
        if role == "Oncogene":
            colors.append("#e74c3c")
        elif role == "TSG":
            colors.append("#3498db")
        elif role == "Both":
            colors.append("#9b59b6")
        elif row.get("is_drug_target"):
            colors.append("#27ae60")
        else:
            colors.append("#95a5a6")

    fig = go.Figure(go.Bar(
        x=top["biomarker_score"][::-1],
        y=top["symbol"][::-1],
        orientation="h",
        marker=dict(color=colors[::-1]),
        customdata=np.column_stack([
            top["cancer_role"][::-1].fillna(""),
            top["is_drug_target"][::-1].fillna(""),
            top["biomarker_type"][::-1].fillna(""),
            top["log2FC"][::-1],
        ]),
        hovertemplate=(
            "<b>%{y}</b> — Score: %{x:.1f}<br>"
            "Cancer role: %{customdata[0]}<br>"
            "Drug target: %{customdata[1]}<br>"
            "Biomarker: %{customdata[2]}<br>"
            "Log2FC: %{customdata[3]:.2f}<extra></extra>"
        ),
    ))

    # Legend annotations
    for label, color in [("Oncogene","#e74c3c"),("TSG","#3498db"),
                          ("Both","#9b59b6"),("Drug target","#27ae60"),("Other","#95a5a6")]:
        fig.add_annotation(
            text=f"<span style='color:{color}'>■</span> {label}",
            xref="paper", yref="paper",
            x=1.01, y={"Oncogene":0.98,"TSG":0.90,"Both":0.82,"Drug target":0.74,"Other":0.66}[label],
            showarrow=False, font=dict(size=10), align="left",
        )

    fig.update_layout(
        template="simple_white",
        title="Biomarker Priority Score (Meta + Clinical Bonus)",
        xaxis_title="Biomarker Score (0–100)",
        margin=dict(l=80, r=130, t=50, b=40),
        height=480,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MA Plot, P-value histogram, LFC distribution, Rank metric
# ─────────────────────────────────────────────────────────────────────────────

def create_ma_plot(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    has_base = "baseMean" in df.columns
    x = (np.log10(df["baseMean"].clip(lower=0.1))
         if has_base else np.arange(len(df), dtype=float))
    df["_x"] = x
    df["_nlp"] = -np.log10(df["padj"])
    sig = df["_nlp"] > -np.log10(0.05)

    fig = go.Figure()
    for mask, col, name in [(~sig, "#c8c8c8", "NS"),
                             (sig & (df["log2FC"] > 0), "#e74c3c", "Up"),
                             (sig & (df["log2FC"] < 0), "#3498db", "Down")]:
        sub = df[mask]
        fig.add_trace(go.Scattergl(
            x=sub["_x"], y=sub["log2FC"],
            mode="markers", name=name,
            marker=dict(color=col, size=5, opacity=0.7),
            text=sub["symbol"],
            hovertemplate="<b>%{text}</b>  Log2FC %{y:.3f}<extra></extra>",
        ))
    fig.add_hline(y=0, line_color="gray", line_dash="dash", opacity=0.5)
    fig.update_layout(
        template="simple_white",
        title="MA Plot (Mean vs Fold Change)",
        xaxis_title="log10(Base Mean)" if has_base else "Gene Rank",
        yaxis_title="Log2 Fold Change",
        height=340, legend=dict(orientation="h", y=1.05),
    )
    return fig


def create_pval_hist(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=df["padj"], nbinsx=40,
        marker_color="#3498db", opacity=0.75,
    ))
    fig.add_vline(x=0.05, line_dash="dash", line_color="red",
                  annotation_text="p=0.05", annotation_position="top right")
    fig.update_layout(
        template="simple_white", title="Adj. P-value Distribution",
        xaxis_title="Adj. p-value", yaxis_title="Count",
        height=300, showlegend=False,
    )
    return fig


def create_lfc_dist(df: pd.DataFrame) -> go.Figure:
    up = (df["log2FC"] > 1) & (df["padj"] < 0.05)
    dn = (df["log2FC"] < -1) & (df["padj"] < 0.05)
    ns = ~(up | dn)
    fig = go.Figure()
    for mask, col, name in [(ns, "#c8c8c8", "Non-sig"),
                             (up, "#e74c3c", "Up"),
                             (dn, "#3498db", "Down")]:
        fig.add_trace(go.Histogram(
            x=df.loc[mask, "log2FC"], nbinsx=30, name=name,
            marker_color=col, opacity=0.7,
        ))
    fig.update_layout(
        template="simple_white", barmode="overlay",
        title="Log2FC Distribution",
        xaxis_title="Log2FC", yaxis_title="Count",
        height=300, legend=dict(orientation="h", y=1.05),
    )
    return fig


def create_rank_metric(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["rank"] = np.sign(df["log2FC"]) * -np.log10(df["padj"])
    df = df.sort_values("rank", ascending=False).reset_index(drop=True)
    up = df["rank"] > 0
    dn = df["rank"] < 0
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index[up], y=df.loc[up, "rank"],
                         marker_color="#e74c3c", name="Up", showlegend=False))
    fig.add_trace(go.Bar(x=df.index[dn], y=df.loc[dn, "rank"],
                         marker_color="#3498db", name="Down", showlegend=False))
    for _, row in df.head(5).iterrows():
        fig.add_annotation(x=row.name, y=row["rank"], text=row["symbol"],
                           showarrow=False, font=dict(size=8), yshift=8)
    for _, row in df.tail(5).iterrows():
        fig.add_annotation(x=row.name, y=row["rank"], text=row["symbol"],
                           showarrow=False, font=dict(size=8), yshift=-14)
    fig.update_layout(
        template="simple_white",
        title="GSEA Pre-rank Metric  (sign × −log10 p)",
        xaxis_title="Gene Rank", yaxis_title="Rank Metric",
        height=320, bargap=0,
    )
    return fig


def create_top_heatmap(df: pd.DataFrame, lfc_t: float = 1.0,
                       p_t: float = 0.05, top_n: int = 100) -> go.Figure:
    """
    Sample Expression Heatmap — Top 100 genes by meta-score.

    Since aggregate-only CSVs carry no per-sample counts, six pseudo-samples
    are generated from log2FC so the heatmap conveys fold-change direction and
    magnitude across a simulated two-group design:

        Group A (control, 3 replicates) : expression ≈ −log2FC/2 + N(0, 0.25)
        Group B (treatment, 3 replicates): expression ≈ +log2FC/2 + N(0, 0.25)

    Values are clipped to [−6, +6] matching the RdBu colorscale limits.
    When real sample columns are present in df they are used directly.
    """
    HOUSE = {"ACTB", "GAPDH", "B2M", "RPLP0", "HPRT1", "GUSB", "TBP"}

    df = df.copy()

    # ── Rank by composite meta-score: |LFC| × −log10(padj) ──────────────────
    df["_score"] = (df["log2FC"].abs()
                    * -np.log10(df["padj"].clip(lower=1e-300)))
    sig = df[(df["log2FC"].abs() >= lfc_t) & (df["padj"] < p_t)]
    pool = sig if not sig.empty else df
    pool = pool[~pool["symbol"].isin(HOUSE)]
    top  = pool.nlargest(min(top_n, len(pool)), "_score").reset_index(drop=True)

    if top.empty:
        return blank("No genes pass the significance threshold for the heatmap.")

    # ── Detect real sample columns vs aggregate-only input ───────────────────
    reserved = {"symbol", "log2FC", "padj", "pvalue", "baseMean",
                "_score", "_cat", "_nlp"}
    sample_cols = [c for c in df.columns if c not in reserved
                   and pd.api.types.is_numeric_dtype(df[c])]

    rng = np.random.default_rng(42)   # reproducible noise

    if sample_cols:
        # Real per-sample expression data present — use it directly
        z = top[sample_cols].values.clip(-6, 6)
        col_labels = sample_cols
        n_a = len(sample_cols) // 2
    else:
        # Generate 6 pseudo-samples: 3 Group A + 3 Group B
        lfc_arr = top["log2FC"].values
        noise   = rng.normal(0, 0.25, (len(top), 6))
        half    = lfc_arr[:, None] / 2
        z = np.hstack([-half + noise[:, :3],
                        half + noise[:, 3:]]).clip(-6, 6)
        col_labels = ["A_rep1", "A_rep2", "A_rep3",
                      "B_rep1", "B_rep2", "B_rep3"]
        n_a = 3

    gene_labels = top["symbol"].tolist()

    # ── Row-wise Z-score normalization ───────────────────────────────────────
    # Normalise each gene across its samples so colour contrast reflects
    # relative within-gene expression change, not absolute LFC magnitude.
    from scipy.stats import zscore as _zscore
    from scipy.cluster.hierarchy import linkage as _hclust, leaves_list as _leaves
    from scipy.spatial.distance import pdist as _pdist

    with np.errstate(invalid="ignore"):
        z = _zscore(z, axis=1)
    z = np.nan_to_num(z, nan=0.0)   # zero-variance genes → flat row of 0s

    # ── Hierarchical clustering: reorder rows to group co-regulated genes ────
    if len(gene_labels) > 2:
        dist  = _pdist(z, metric="euclidean")
        link  = _hclust(dist, method="ward")
        order = _leaves(link)
        z          = z[order]
        gene_labels = [gene_labels[i] for i in order]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=col_labels,
        y=gene_labels,
        colorscale="RdBu",
        reversescale=True,    # RdBu is Red-low/Blue-high; reverse → Red=High, Blue=Low
        zmid=0, zmin=-2.5, zmax=2.5,
        zsmooth=False,        # sharp tiles — no interpolation between cells
        hovertemplate=(
            "Gene: <b>%{y}</b><br>Sample: %{x}<br>"
            "Z-score: %{z:.2f}<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="Z-score<br>(Row-normalized)", side="right", font=dict(size=11)),
            thickness=14,
            tickvals=[-2, -1, 0, 1, 2],
        ),
    ))

    # ── Colored group-header blocks (shapes + annotations in paper coords) ───
    n_cols = len(col_labels)
    # x positions in "x" axis domain: -0.5 … n_cols-0.5
    x_a_start, x_a_end = -0.5, n_a - 0.5
    x_b_start, x_b_end = n_a - 0.5, n_cols - 0.5

    for x0, x1, fill, border, font_color, label in [
        (x_a_start, x_a_end, "rgba(39,174,96,0.30)",  "#1e8449", "#14602e", "Group A"),
        (x_b_start, x_b_end, "rgba(142,68,173,0.30)", "#6c3483", "#4a2060", "Group B"),
    ]:
        fig.add_shape(
            type="rect",
            xref="x", yref="paper",
            x0=x0, x1=x1, y0=1.0, y1=1.07,
            fillcolor=fill,
            line=dict(color=border, width=1.5),
        )
        fig.add_annotation(
            xref="x", yref="paper",
            x=(x0 + x1) / 2, y=1.035,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=11, color=font_color),
        )

    fig.update_layout(
        template="simple_white",
        title=dict(
            text=(f"Sample Expression Heatmap — Top {len(top)} Genes "
                  f"<span style='font-size:10px;color:#888;'>"
                  f"(row-wise Z-score · hierarchical clustering)"
                  + (" · pseudo-samples" if not sample_cols else "")
                  + "</span>"),
            font=dict(size=13),
        ),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9), side="bottom"),
        yaxis=dict(
            tickfont=dict(size=8),
            autorange="reversed",
            dtick=max(1, len(top) // 20),   # avoid label overlap for 100 genes
        ),
        height=max(340, min(900, len(top) * 8 + 100)),
        margin=dict(l=130, r=20, t=80, b=60),  # l=130 keeps gene labels clear of tiles
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3-D PCA
# ─────────────────────────────────────────────────────────────────────────────

def create_pca_3d(res: dict) -> go.Figure:
    """Sample-centric PCA: each dot is a sample, coloured by experimental group."""
    var     = res["explained_variance"]
    samples = res["samples"]
    groups  = res["groups"]

    GROUP_COLORS = {"Group A": "#27ae60", "Group B": "#8e44ad"}

    fig = go.Figure()
    for grp, color in GROUP_COLORS.items():
        m = [g == grp for g in groups]
        fig.add_trace(go.Scatter3d(
            x=[x for x, b in zip(res["pc1"], m) if b],
            y=[y for y, b in zip(res["pc2"], m) if b],
            z=[z for z, b in zip(res["pc3"], m) if b],
            mode="markers+text",
            name=grp,
            text=[s for s, b in zip(samples, m) if b],
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(size=11, color=color, opacity=0.92,
                        line=dict(width=1, color="white")),
            hovertemplate="<b>%{text}</b><br>PC1 %{x:.3f}  PC2 %{y:.3f}  PC3 %{z:.3f}<extra></extra>",
        ))
    fig.update_layout(
        template="simple_white", height=520,
        title=dict(
            text=(f"Sample-Centric 3D PCA"
                  f"<br><span style='font-size:11px;color:#888;'>"
                  f"PC1 {var[0]:.1f}%  PC2 {var[1]:.1f}%  PC3 {var[2]:.1f}%"
                  f"  |  {res['n_genes']} significant genes as features</span>"),
            font=dict(size=14),
        ),
        scene=dict(
            xaxis_title=f"PC1 ({var[0]:.1f}%)",
            yaxis_title=f"PC2 ({var[1]:.1f}%)",
            zaxis_title=f"PC3 ({var[2]:.1f}%)",
        ),
        legend=dict(title="Sample Group"),
        margin=dict(l=0, r=0, t=80, b=0),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Gene Network
# ─────────────────────────────────────────────────────────────────────────────

def create_network_graph(network_data: dict) -> go.Figure:
    G = network_data.get("graph")
    if G is None or G.number_of_nodes() < 3:
        return blank("Not enough correlated genes at current threshold.")

    # Force-directed layout: k scales with node count so modules spread apart
    _k  = 2.0 / max(1, G.number_of_nodes() ** 0.5)
    pos = nx.spring_layout(G, seed=42, k=_k, iterations=80)
    lfc_map = network_data.get("lfc_values", {})
    modules = network_data.get("labels", {})

    # Color edges by correlation sign
    edge_pos = []
    edge_neg = []
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        seg = [pos[u][0], pos[v][0], None], [pos[u][1], pos[v][1], None]
        if w >= 0:
            edge_pos.extend(zip(*seg))
        else:
            edge_neg.extend(zip(*seg))

    traces = []
    if edge_pos:
        ex, ey = [x for x, _ in edge_pos], [y for _, y in edge_pos]
        traces.append(go.Scattergl(x=ex, y=ey, mode="lines",
                                   line=dict(width=0.8, color="#e74c3c"),
                                   hoverinfo="none", showlegend=True, name="Corr +"))
    if edge_neg:
        ex, ey = [x for x, _ in edge_neg], [y for _, y in edge_neg]
        traces.append(go.Scattergl(x=ex, y=ey, mode="lines",
                                   line=dict(width=0.8, color="#3498db"),
                                   hoverinfo="none", showlegend=True, name="Corr −"))

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_lfc   = [lfc_map.get(n, 0) for n in G.nodes()]
    node_mod   = [str(modules.get(n, 0)) for n in G.nodes()]
    node_deg   = [G.degree(n) for n in G.nodes()]
    traces.append(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=list(G.nodes()),
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(
            size=[max(8, 6 + d) for d in node_deg],
            color=node_lfc,
            colorscale="RdBu", reversescale=True,
            showscale=True,
            colorbar=dict(title="Log2FC", thickness=12),
            line=dict(width=1, color="#fff"),
        ),
        customdata=np.column_stack([node_mod, node_deg]),
        hovertemplate=(
            "<b>%{text}</b><br>Module: %{customdata[0]}<br>"
            "Degree: %{customdata[1]}<extra></extra>"
        ),
        showlegend=False,
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="simple_white",
        title=f"Co-expression Network  ({G.number_of_nodes()} nodes · {G.number_of_edges()} edges)",
        showlegend=True,
        legend=dict(font=dict(size=10)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig
