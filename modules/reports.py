"""
PDF Report Generator — Apex Bioinformatics Platform v4.0 (Deep Scan Edition)
=============================================================================
Full deep-scan PDF including:
  • Summary statistics
  • Key Biological Insights (Auto-Insights)
  • 1.  Volcano Plot                   (high-res 3× kaleido)
  • 2.  Pathway Enrichment Bubble & Bar (high-res)
  • 3.  Full Enrichment Score Table    (ALL rows, no head() cap)
  • 4.  Pathway Crosstalk Heatmap      (high-res)
  • 5.  FDA Drug Targets Table         (raw session data)
  • 6.  3D PCA Projection              (high-res) + variance text
  • 7.  Top Gene Expression Heatmap    (high-res)
  • 8.  Diagnostic Charts             (MA plot, p-val hist, LFC dist, rank metric)
  • 9.  GSEA Pre-Ranked Results
  • 10. WGCNA-Lite Co-expression Network
  • 11. Drug Target Analysis Charts
  • 12. Biomarker Priority Score Chart + raw data table
  • 13. Meta-Score Ranking
  • 14. Gene Bio-Context Query
  • 15. AI Discussion (Deep Scan — Gemini 2.5 Flash / Groq / Rule-Based)
"""

import io
import logging
import re
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    HRFlowable, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle,
)

log = logging.getLogger(__name__)

_BLUE  = colors.HexColor("#1a5276")
_GREEN = colors.HexColor("#1e8449")
_LIGHT = colors.HexColor("#f4f6f7")
_RED   = colors.HexColor("#c0392b")
_LGRAY = colors.HexColor("#bdc3c7")

# Page content width (letter − margins)
_PAGE_W = 539   # 612 − 36 − 36 − 1 fudge


def _fig_to_image(fig, width: int = 800, height: int = 480) -> io.BytesIO | None:
    """Render Plotly figure to high-res PNG bytes via kaleido.

    scale=3 yields effectively 2400×1440 px source → crisp at any print DPI.
    Returns None on failure so callers can substitute a placeholder string.
    """
    if fig is None:
        return None
    try:
        import plotly.io as pio
        img_bytes = pio.to_image(
            fig, format="png",
            width=width, height=height,
            scale=3,          # 3× → publishable print quality
            engine="kaleido",
        )
        buf = io.BytesIO(img_bytes)
        buf.seek(0)
        return buf
    except Exception as e:
        log.warning(f"kaleido render failed: {e}")
        return None


def _table_style(n_cols: int) -> TableStyle:
    return TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  _BLUE),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 0), (-1, -1), 7),
        ("ALIGN",          (0, 0), (-1, -1), "LEFT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _LIGHT]),
        ("GRID",           (0, 0), (-1, -1), 0.3, colors.grey),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 3),
        ("TOPPADDING",     (0, 0), (-1, -1), 3),
    ])


def generate_pdf_report(
    volcano_fig,
    pathway_df: pd.DataFrame,
    pca_fig,
    drug_df: pd.DataFrame | None = None,
    insights: list | None = None,
    summary_stats: dict | None = None,
    bubble_fig=None,
    bar_fig=None,
    crosstalk_fig=None,
    heatmap_fig=None,
    # ── Deep-scan graph panels ────────────────────────────────────────────────
    ma_fig=None,
    pval_fig=None,
    lfc_dist_fig=None,
    rank_fig=None,
    gsea_fig=None,
    network_fig=None,
    drug_scatter_fig=None,
    drug_donut_fig=None,
    bm_fig=None,
    meta_bar_fig=None,
    # ── Deep-scan raw session-state data ─────────────────────────────────────
    bm_df: pd.DataFrame | None = None,           # biomarker scores table
    pca_variance_text: str | None = None,         # PCA variance % from pca-info
    gene_query: str | None = None,                # Gene Bio-Context search query
    ai_discussion: str | None = None,
    powered_by: str = "",
) -> bytes:
    """Return PDF bytes — every graph rendered at 3× scale, raw tables included."""

    # ── Normalise inputs ──────────────────────────────────────────────────────
    if isinstance(pathway_df, list):
        pathway_df = pd.DataFrame(pathway_df)
    if isinstance(drug_df, list):
        drug_df = pd.DataFrame(drug_df)
    if isinstance(bm_df, list):
        bm_df = pd.DataFrame(bm_df)

    def _to_fig(f):
        if f is None:
            return None
        if isinstance(f, go.Figure):
            return f
        try:
            return go.Figure(f)
        except Exception:
            return None

    volcano_fig      = _to_fig(volcano_fig)
    pca_fig          = _to_fig(pca_fig)
    bubble_fig       = _to_fig(bubble_fig)
    bar_fig          = _to_fig(bar_fig)
    crosstalk_fig    = _to_fig(crosstalk_fig)
    heatmap_fig      = _to_fig(heatmap_fig)
    ma_fig           = _to_fig(ma_fig)
    pval_fig         = _to_fig(pval_fig)
    lfc_dist_fig     = _to_fig(lfc_dist_fig)
    rank_fig         = _to_fig(rank_fig)
    gsea_fig         = _to_fig(gsea_fig)
    network_fig      = _to_fig(network_fig)
    drug_scatter_fig = _to_fig(drug_scatter_fig)
    drug_donut_fig   = _to_fig(drug_donut_fig)
    bm_fig           = _to_fig(bm_fig)
    meta_bar_fig     = _to_fig(meta_bar_fig)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=36, rightMargin=36,
        topMargin=36, bottomMargin=36,
    )
    styles = getSampleStyleSheet()
    body   = styles["Normal"]
    h2     = styles["Heading2"]
    h3     = styles["Heading3"]
    elems  = []

    def hr():
        elems.append(HRFlowable(width="100%", thickness=0.5, color=_LGRAY))
        elems.append(Spacer(1, 8))

    def section_img(fig, w=520, h=320, placeholder="Chart not rendered — visit the tab first."):
        """Render fig at 3× scale and embed at display size w×h pts."""
        img = _fig_to_image(fig, width=w, height=h)
        if img:
            # Display at 80 % of source to leave page margins; kaleido 3× keeps it crisp
            elems.append(Image(img, width=w * 0.80, height=h * 0.80))
        else:
            elems.append(Paragraph(f"({placeholder})", body))

    def note_style():
        return ParagraphStyle("note", parent=styles["Italic"],
                              textColor=colors.HexColor("#555555"), fontSize=8)

    # ── Title header ──────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "title", parent=styles["Title"],
        textColor=_BLUE, spaceAfter=4,
    )
    elems.append(Paragraph("Apex Bioinformatics Platform", title_style))
    elems.append(Paragraph("Differential Gene Expression Analysis Report", h2))
    elems.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC  |  "
        f"AI Engine: {powered_by or 'Rule-Based · Offline'}",
        styles["Italic"],
    ))
    hr()

    # ── Summary statistics ────────────────────────────────────────────────────
    if summary_stats:
        elems.append(Paragraph("Analysis Summary", h2))
        stat_data = [[
            "Total Genes", "Up-regulated", "Down-regulated", "Significant",
        ], [
            str(summary_stats.get("total", "—")),
            str(summary_stats.get("up",    "—")),
            str(summary_stats.get("down",  "—")),
            str(summary_stats.get("sig",   "—")),
        ]]
        st = Table(stat_data, colWidths=[_PAGE_W // 4] * 4)
        st.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  _BLUE),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTNAME",    (0, 1), (-1, 1),  "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 10),
            ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
            ("GRID",        (0, 0), (-1, -1), 0.4, colors.grey),
        ]))
        elems.append(st)
        elems.append(Spacer(1, 12))

    # ── PCA Variance (raw session data) ───────────────────────────────────────
    if pca_variance_text:
        elems.append(Paragraph(
            f"<b>PCA Variance:</b> {pca_variance_text}", body,
        ))
        elems.append(Spacer(1, 6))

    # ── Gene Bio-Context Query ────────────────────────────────────────────────
    if gene_query and gene_query.strip():
        elems.append(Paragraph(
            f"<b>Gene Bio-Context Query:</b> {gene_query.strip()}", body,
        ))
        elems.append(Spacer(1, 6))

    hr()

    # ── Auto-Insights ─────────────────────────────────────────────────────────
    if insights:
        elems.append(Paragraph("Key Biological Insights", h2))
        for ins in insights[:6]:
            elems.append(Paragraph(
                f"{ins.get('icon', '')} <b>{ins.get('title', '')}</b>", h3,
            ))
            body_text = re.sub(r"<[^>]+>", "", ins.get("body", ""))
            elems.append(Paragraph(body_text, body))
            elems.append(Spacer(1, 6))
        hr()

    sec = 1   # running section counter

    # ── 1. Volcano Plot ───────────────────────────────────────────────────────
    elems.append(Paragraph(f"{sec}. Volcano Plot", h2))
    section_img(volcano_fig, 620, 380,
                "Volcano chart not rendered — visit the Volcano tab first.")
    elems.append(Spacer(1, 12))
    hr()
    sec += 1

    # ── 2. Pathway Enrichment Charts ──────────────────────────────────────────
    elems.append(Paragraph(f"{sec}. Pathway Enrichment Analysis", h2))
    if bubble_fig or bar_fig:
        if bubble_fig:
            elems.append(Paragraph("Enrichment Bubble Chart", h3))
            section_img(bubble_fig, 580, 360)
            elems.append(Spacer(1, 8))
        if bar_fig:
            elems.append(Paragraph("Pathway Bar Chart", h3))
            section_img(bar_fig, 580, 340)
            elems.append(Spacer(1, 8))
    else:
        elems.append(Paragraph(
            "(Enrichment charts not captured — run enrichment on the Volcano tab first.)",
            body,
        ))
    elems.append(Spacer(1, 6))
    hr()
    sec += 1

    # ── 3. Full Enrichment Score Table (ALL rows, no cap) ─────────────────────
    elems.append(Paragraph(f"{sec}. Enrichment Score Table — Full Results", h2))
    if pathway_df is not None and not pathway_df.empty:
        show = [c for c in ["pathway", "database", "overlap_count",
                             "adjusted_p_value", "enrichment_score", "genes"]
                if c in pathway_df.columns]
        disp = pathway_df[show].copy()
        if "adjusted_p_value" in disp.columns:
            disp["adjusted_p_value"] = disp["adjusted_p_value"].map("{:.2e}".format)
        if "enrichment_score" in disp.columns:
            disp["enrichment_score"] = disp["enrichment_score"].map(
                lambda v: f"{float(v):.3f}" if pd.notna(v) else "—"
            )
        if "genes" in disp.columns:
            disp["genes"] = disp["genes"].str[:60]
        header = [[c.replace("_", " ").title() for c in show]]
        rows_  = [[str(v) for v in row] for row in disp.values]
        col_w  = [_PAGE_W // len(show)] * len(show)
        t = Table(header + rows_, colWidths=col_w, repeatRows=1)
        t.setStyle(_table_style(len(show)))
        elems.append(t)
        elems.append(Paragraph(
            f"({len(disp)} pathways — all rows shown)", note_style(),
        ))
    else:
        elems.append(Paragraph(
            "No enrichment results (run Volcano tab first).", body,
        ))
    elems.append(Spacer(1, 12))
    hr()
    sec += 1

    # ── 4. Pathway Crosstalk Heatmap ──────────────────────────────────────────
    if crosstalk_fig is not None:
        elems.append(Paragraph(f"{sec}. Pathway Crosstalk (Jaccard Gene Overlap)", h2))
        section_img(crosstalk_fig, 600, 400)
        elems.append(Spacer(1, 12))
        hr()
        sec += 1

    # ── 5. FDA Drug Targets Table ─────────────────────────────────────────────
    if drug_df is not None and not drug_df.empty:
        elems.append(Paragraph(f"{sec}. FDA-Approved Drug Targets", h2))
        fda = drug_df[drug_df["fda"] == True].copy() if "fda" in drug_df.columns else drug_df
        show_d = [c for c in ["gene", "drug", "type", "indication"] if c in fda.columns]
        if show_d:
            header_d = [[c.title() for c in show_d]]
            rows_d   = [[str(row[c])[:45] for c in show_d]
                        for _, row in fda.head(40).iterrows()]
            col_w_d  = [_PAGE_W // len(show_d)] * len(show_d)
            td = Table(header_d + rows_d, colWidths=col_w_d, repeatRows=1)
            td.setStyle(_table_style(len(show_d)))
            elems.append(td)
        elems.append(Spacer(1, 12))
        hr()
        sec += 1

    # ── 6. 3D PCA Projection ──────────────────────────────────────────────────
    elems.append(Paragraph(f"{sec}. 3D PCA Projection", h2))
    if pca_variance_text:
        elems.append(Paragraph(pca_variance_text, note_style()))
        elems.append(Spacer(1, 4))
    section_img(pca_fig, 580, 380,
                "PCA chart not rendered — visit the 3D PCA tab first.")
    elems.append(Spacer(1, 12))
    hr()
    sec += 1

    # ── 7. Top Gene Expression Heatmap ────────────────────────────────────────
    if heatmap_fig is not None:
        elems.append(Paragraph(f"{sec}. Top Gene Expression Heatmap", h2))
        section_img(heatmap_fig, 580, 440)
        elems.append(Spacer(1, 12))
        hr()
        sec += 1

    # ── 8. Diagnostic Charts ──────────────────────────────────────────────────
    diag_figs = [
        (ma_fig,       "MA Plot (Mean Expression vs. Log2FC)"),
        (pval_fig,     "P-value Distribution Histogram"),
        (lfc_dist_fig, "Log2FC Distribution"),
        (rank_fig,     "GSEA Rank Metric"),
    ]
    rendered_diag = [(f, t) for f, t in diag_figs if f is not None]
    if rendered_diag:
        elems.append(PageBreak())
        elems.append(Paragraph(f"{sec}. Diagnostic Charts", h2))
        for fig, title in rendered_diag:
            elems.append(Paragraph(title, h3))
            section_img(fig, 540, 320)
            elems.append(Spacer(1, 10))
        hr()
        sec += 1

    # ── 9. GSEA Pre-Ranked Results ────────────────────────────────────────────
    if gsea_fig is not None:
        elems.append(Paragraph(f"{sec}. GSEA Pre-Ranked Results", h2))
        section_img(gsea_fig, 560, 380)
        elems.append(Spacer(1, 12))
        hr()
        sec += 1

    # ── 10. WGCNA-Lite Co-expression Network ──────────────────────────────────
    if network_fig is not None:
        elems.append(Paragraph(f"{sec}. WGCNA-Lite Co-expression Network", h2))
        section_img(network_fig, 560, 420)
        elems.append(Spacer(1, 12))
        hr()
        sec += 1

    # ── 11. Drug Target Analysis Charts ───────────────────────────────────────
    drug_charts = [
        (drug_scatter_fig, "Drug Target Landscape (FDA Overlay)"),
        (drug_donut_fig,   "Drug Type Distribution"),
    ]
    rendered_drug = [(f, t) for f, t in drug_charts if f is not None]
    if rendered_drug:
        elems.append(Paragraph(f"{sec}. Drug Target Analysis", h2))
        for fig, title in rendered_drug:
            elems.append(Paragraph(title, h3))
            section_img(fig, 540, 340)
            elems.append(Spacer(1, 10))
        hr()
        sec += 1

    # ── 12. Biomarker Priority Score (chart + raw data table) ─────────────────
    has_bm_chart = bm_fig is not None
    has_bm_table = bm_df is not None and not bm_df.empty
    if has_bm_chart or has_bm_table:
        elems.append(Paragraph(f"{sec}. Biomarker Priority Score", h2))
        if has_bm_chart:
            section_img(bm_fig, 560, 420)
            elems.append(Spacer(1, 10))
        if has_bm_table:
            elems.append(Paragraph("Raw Biomarker Score Data", h3))
            bm_show = [c for c in
                       ["symbol", "log2FC", "padj", "baseMean",
                        "biomarker_score", "clinical_score", "stat_score"]
                       if c in bm_df.columns]
            if bm_show:
                bm_disp = bm_df[bm_show].copy()
                for col in ["log2FC", "padj", "baseMean",
                            "biomarker_score", "clinical_score", "stat_score"]:
                    if col in bm_disp.columns:
                        bm_disp[col] = bm_disp[col].map(
                            lambda v: f"{float(v):.4f}" if pd.notna(v) else "—"
                        )
                bm_header = [[c.replace("_", " ").title() for c in bm_show]]
                bm_rows   = [[str(v) for v in row] for row in bm_disp.values]
                col_w_bm  = [_PAGE_W // len(bm_show)] * len(bm_show)
                tb = Table(bm_header + bm_rows, colWidths=col_w_bm, repeatRows=1)
                tb.setStyle(_table_style(len(bm_show)))
                elems.append(tb)
                elems.append(Paragraph(
                    f"({len(bm_disp)} genes — all rows shown)", note_style(),
                ))
        elems.append(Spacer(1, 12))
        hr()
        sec += 1

    # ── 13. Meta-Score Ranking ────────────────────────────────────────────────
    if meta_bar_fig is not None:
        elems.append(Paragraph(f"{sec}. Meta-Score Ranking", h2))
        section_img(meta_bar_fig, 560, 400)
        elems.append(Spacer(1, 12))
        hr()
        sec += 1

    # ── 14. Gene Bio-Context ──────────────────────────────────────────────────
    if gene_query and gene_query.strip():
        elems.append(Paragraph(f"{sec}. Gene Bio-Context Query", h2))
        elems.append(Paragraph(
            f"Genes queried: <b>{gene_query.strip()}</b>", body,
        ))
        elems.append(Paragraph(
            "(Full annotations visible in the sidebar of the running app.)", note_style(),
        ))
        elems.append(Spacer(1, 12))
        hr()
        sec += 1

    # ── 15. AI Discussion (Deep Scan) ─────────────────────────────────────────
    if ai_discussion:
        elems.append(PageBreak())
        elems.append(Paragraph(f"{sec}. Discussion — AI Deep Scan", h2))

        if powered_by:
            elems.append(Paragraph(
                f"<i>Generated by: {powered_by}</i>",
                ParagraphStyle("attr", parent=styles["Italic"],
                               textColor=colors.HexColor("#666666"), fontSize=8),
            ))
            elems.append(Spacer(1, 6))

        clean_text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", ai_discussion)
        clean_text = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", clean_text)
        paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]
        for para in paragraphs:
            elems.append(Paragraph(para, ParagraphStyle(
                "disc", parent=styles["Normal"],
                fontSize=10, leading=14, spaceAfter=6,
            )))

        elems.append(Spacer(1, 8))
        hr()

    # ── Footer ────────────────────────────────────────────────────────────────
    elems.append(Paragraph(
        "Apex Bioinformatics Platform — For research use only. "
        "Results should be validated experimentally before clinical application.",
        ParagraphStyle("footer", parent=styles["Italic"],
                       textColor=colors.grey, fontSize=8),
    ))

    doc.build(elems)
    buf.seek(0)
    return buf.getvalue()
