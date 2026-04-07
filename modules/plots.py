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
    from data.gene_annotations import CANCER_GENE_CENSUS, DRUG_GENE_INTERACTIONS

    if isinstance(df, list):
        df = pd.DataFrame(df)
    if df.empty:
        return go.Figure()

    df = df.copy()
    df["_nlp"] = -np.log10(df[padj_col].clip(lower=1e-300))

    up = (df[logfc_col] >  lfc_thresh) & (df[padj_col] < p_thresh)
    dn = (df[logfc_col] < -lfc_thresh) & (df[padj_col] < p_thresh)
    df["_cat"] = "Non-significant"
    df.loc[up, "_cat"] = "Up-regulated"
    df.loc[dn, "_cat"] = "Down-regulated"

    color_map = {
        "Up-regulated":    "#e74c3c",
        "Down-regulated":  "#3498db",
        "Non-significant": "#c8c8c8",
    }

    has_base = "baseMean" in df.columns
    # ── Decimation: keep all significant + top non-sig by p-value (max 5000 total) ──
    PLOT_CAP = 5000
    if len(df) > PLOT_CAP:
        sig_mask = df["_cat"] != "Non-significant"
        sig_df   = df[sig_mask]
        ns_df    = df[~sig_mask]
        ns_slots = max(0, PLOT_CAP - len(sig_df))
        ns_df    = ns_df.nsmallest(ns_slots, padj_col) if ns_slots > 0 else ns_df.iloc[0:0]
        df = pd.concat([sig_df, ns_df], ignore_index=True)

    fig = go.Figure()

    for cat, color in color_map.items():
        sub = df[df["_cat"] == cat]
        if sub.empty:
            continue
        base_vals = sub["baseMean"].values if has_base else np.zeros(len(sub))
        customdata = np.column_stack([sub[gene_col].values, base_vals])
        ht = ("<b>%{customdata[0]}</b><br>"
              "Log2FC: %{x:.3f}<br>"
              "−log10(p): %{y:.2f}<br>")
        if has_base:
            ht += "BaseMean: %{customdata[1]:.1f}"
        ht += "<extra></extra>"
        fig.add_trace(go.Scatter(            # SVG — required for customdata in selectedData
            x=sub[logfc_col], y=sub["_nlp"],
            mode="markers", name=cat,
            marker=dict(color=color, size=6, opacity=0.75),
            customdata=customdata,
            hovertemplate=ht,
        ))

    # Ensure gene column is string before any .str operations
    df[gene_col] = df[gene_col].astype(str)

    # ── Oncogene overlay (star marker, gold border) ──────────────────────────
    onco_up = df[
        up &
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
        fig.add_trace(go.Scatter(
            x=onco_up[logfc_col], y=onco_up["_nlp"],
            mode="markers+text",
            name="Oncogene ↑",
            text=onco_up[gene_col],
            textposition="top center",
            textfont=dict(size=9, color="#c0392b"),
            marker=dict(symbol="star", size=13, color="#e74c3c",
                        line=dict(color="#f39c12", width=2)),
            customdata=cd,
            hovertemplate="<b>%{customdata[0]}</b> (Oncogene)<br>"
                          "Cancer: %{customdata[1]}<extra></extra>",
        ))

    # ── TSG overlay (diamond marker, blue border) ────────────────────────────
    tsg_dn = df[
        dn &
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
        fig.add_trace(go.Scatter(
            x=tsg_dn[logfc_col], y=tsg_dn["_nlp"],
            mode="markers+text",
            name="TSG ↓",
            text=tsg_dn[gene_col],
            textposition="bottom center",
            textfont=dict(size=9, color="#2980b9"),
            marker=dict(symbol="diamond", size=11, color="#3498db",
                        line=dict(color="#1a5276", width=2)),
            customdata=cd2,
            hovertemplate="<b>%{customdata[0]}</b> (Tumor Suppressor)<br>"
                          "Cancer: %{customdata[1]}<extra></extra>",
        ))

    # ── Drug-target stars ────────────────────────────────────────────────────
    drug_sig = df[
        (df["_cat"] != "Non-significant") &
        df[gene_col].str.upper().isin(DRUG_GENE_INTERACTIONS.keys())
    ]
    # Exclude already-labelled oncogene/TSG to avoid clutter
    already = set(onco_up[gene_col].str.upper().tolist() + tsg_dn[gene_col].str.upper().tolist())
    drug_sig = drug_sig[~drug_sig[gene_col].str.upper().isin(already)]
    if not drug_sig.empty:
        # customdata col-0 = gene name so lasso selection captures these genes
        _drug_cd = np.column_stack([
            drug_sig[gene_col].values,
            np.zeros(len(drug_sig)),
        ])
        fig.add_trace(go.Scatter(
            x=drug_sig[logfc_col], y=drug_sig["_nlp"],
            mode="markers",
            name="Drug target",
            marker=dict(symbol="circle-open", size=14, color="#27ae60",
                        line=dict(color="#27ae60", width=2)),
            customdata=_drug_cd,
            hovertemplate="<b>%{customdata[0]}</b> — FDA drug target<extra></extra>",
        ))

    # ── Threshold lines ──────────────────────────────────────────────────────
    fig.add_hline(y=-np.log10(p_thresh), line_dash="dash", line_color="gray", opacity=0.45)
    fig.add_vline(x= lfc_thresh,         line_dash="dash", line_color="gray", opacity=0.45)
    fig.add_vline(x=-lfc_thresh,         line_dash="dash", line_color="gray", opacity=0.45)

    # ── Label top significant (non-annotated) ────────────────────────────────
    sig_rest = df[
        (df["_cat"] != "Non-significant") &
        ~df[gene_col].str.upper().isin(
            set(onco_up[gene_col].str.upper().tolist()) | set(tsg_dn[gene_col].str.upper().tolist())
        )
    ].nlargest(8, "_nlp")
    for rec in sig_rest[["_nlp", logfc_col, gene_col]].to_dict("records"):
        fig.add_annotation(
            x=rec[logfc_col], y=rec["_nlp"],
            text=str(rec[gene_col]),
            showarrow=False, font=dict(size=9), yshift=9,
        )

    fig.update_layout(
        template="simple_white",
        title=dict(text="Volcano Plot  <span style='font-size:11px;color:#888;'>"
                        "★ Oncogene  ◆ TSG  ○ Drug target  "
                        "— lasso or box-select to run enrichment</span>",
                   font=dict(size=14)),
        xaxis_title="Log2 Fold Change",
        yaxis_title="−log10(Adj. p-value)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        dragmode="select",          # default tool = box-select; user can switch to lasso
        clickmode="event+select",   # enable both click and select events
        margin=dict(t=60),
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
    if enr is None or enr.empty:
        return blank("No significant pathways found.")

    top = enr.head(15).copy()
    top["nlp"]   = -np.log10(top["adjusted_p_value"].clip(lower=1e-300))
    top["label"] = top["pathway"].str[:52]
    if "database" in top.columns:
        db_short = top["database"].str.upper().str[:3]
        top["label"] = db_short + " · " + top["label"]

    fig = go.Figure(go.Scattergl(
        x=top["enrichment_score"], y=top["label"],
        mode="markers",
        marker=dict(
            size=np.clip(top["overlap_count"] * 6, 8, 55),
            color=top["nlp"],
            colorscale="RdYlBu", reversescale=True,
            showscale=True,
            colorbar=dict(title="−log10 p", thickness=12),
        ),
        text=top.get("genes", top["pathway"]),
        customdata=top[["overlap_count", "pathway_size", "adjusted_p_value"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>Enrich. Score: %{x:.1f}<br>"
            "Overlap: %{customdata[0]}/%{customdata[1]}<br>"
            "Adj. p: %{customdata[2]:.2e}<extra></extra>"
        ),
    ))
    fig.update_layout(
        template="simple_white",
        title="Pathway Enrichment Bubble",
        xaxis_title="Enrichment Score (−log10 p)",
        yaxis=dict(autorange="reversed"),
        height=440, margin=dict(l=10, r=10, t=45, b=30),
    )
    return fig


def create_pathway_bar(enr: pd.DataFrame) -> go.Figure:
    if enr is None or enr.empty:
        return blank("No pathways enriched.")

    top = enr.head(12).copy()
    top["nlp"]   = -np.log10(top["adjusted_p_value"].clip(lower=1e-300))
    top["label"] = top["pathway"].str[:50]

    fig = go.Figure(go.Bar(
        x=top["nlp"][::-1], y=top["label"][::-1],
        orientation="h",
        marker=dict(color=top["enrichment_score"][::-1],
                    colorscale="Blues", showscale=False),
        text=top["overlap_count"][::-1].astype(str) + " genes",
        textposition="inside",
        hovertemplate="<b>%{y}</b><br>−log10 p: %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        template="simple_white",
        title="Top Enriched Pathways",
        xaxis_title="−log10(Adj. p-value)",
        height=380, margin=dict(l=10, r=10, t=45, b=30),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Pathway Crosstalk Heatmap  (Jaccard similarity)
# ─────────────────────────────────────────────────────────────────────────────

def create_pathway_crosstalk(crosstalk_df: pd.DataFrame) -> go.Figure:
    if crosstalk_df is None or crosstalk_df.empty:
        return blank("Insufficient enriched pathways for crosstalk analysis.")

    fig = go.Figure(go.Heatmapgl(
        z=crosstalk_df.values,
        x=crosstalk_df.columns.tolist(),
        y=crosstalk_df.index.tolist(),
        colorscale="YlOrRd",
        zmin=0, zmax=1,
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
                       p_t: float = 0.05, top_n: int = 30) -> go.Figure:
    HOUSE = {"ACTB", "GAPDH", "B2M", "RPLP0", "HPRT1", "GUSB", "TBP"}
    sig = df[(df["log2FC"].abs() >= lfc_t) & (df["padj"] < p_t)]
    if sig.empty:
        sig = df
    sig = sig[~sig["symbol"].isin(HOUSE)]
    top = pd.concat([
        sig.nlargest(top_n, "log2FC"),
        sig.nsmallest(top_n // 2, "log2FC"),
    ]).drop_duplicates("symbol")
    z = np.column_stack([-np.log10(top["padj"]), top["log2FC"]])
    fig = go.Figure(go.Heatmapgl(
        z=z.T,
        x=top["symbol"].tolist(),
        y=["-log10(p)", "Log2FC"],
        colorscale="RdBu", reversescale=True, zmid=0,
        hovertemplate="Gene: %{x}<br>Metric: %{y}<br>Value: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Value", thickness=12),
    ))
    fig.update_layout(
        template="simple_white",
        title=f"Top {len(top)} DE Genes — Heatmap",
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        height=320, margin=dict(l=80, r=10, t=45, b=80),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3-D PCA
# ─────────────────────────────────────────────────────────────────────────────

def create_pca_3d(res: dict) -> go.Figure:
    clusters = np.array(res["clusters"])
    genes    = res["genes"]
    var      = res["explained_variance"]
    lfc      = res.get("lfc", [0] * len(genes))

    fig = go.Figure()
    for k in range(res["n_clusters"]):
        m = clusters == k
        fig.add_trace(go.Scatter3d(
            x=np.array(res["pc1"])[m],
            y=np.array(res["pc2"])[m],
            z=np.array(res["pc3"])[m],
            mode="markers",
            name=f"Cluster {k + 1}",
            marker=dict(size=6, color=CLUSTER_PAL[k % len(CLUSTER_PAL)], opacity=0.87),
            text=[g for g, b in zip(genes, m) if b],
            hovertemplate="<b>%{text}</b><br>PC1 %{x:.3f}  PC2 %{y:.3f}  PC3 %{z:.3f}<extra></extra>",
        ))
    fig.update_layout(
        template="simple_white", height=520,
        title=dict(
            text=(f"3D PCA Projection"
                  f"<br><span style='font-size:11px;color:#888;'>"
                  f"PC1 {var[0]:.1f}%  PC2 {var[1]:.1f}%  PC3 {var[2]:.1f}%"
                  f"  |  {res['n_clusters']} clusters  {len(genes)} genes</span>"),
            font=dict(size=14),
        ),
        scene=dict(
            xaxis_title=f"PC1 ({var[0]:.1f}%)",
            yaxis_title=f"PC2 ({var[1]:.1f}%)",
            zaxis_title=f"PC3 ({var[2]:.1f}%)",
        ),
        legend=dict(title="Cluster"),
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

    pos     = nx.spring_layout(G, seed=42, k=2.5)
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
