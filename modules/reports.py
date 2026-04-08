"""
PDF Report Generator — Apex Bioinformatics Platform v3.8
=========================================================
Generates a polished multi-section, publishable-quality PDF including:
  • Summary statistics
  • Volcano plot image
  • Pathway Enrichment Bubble & Bar charts
  • Full Enrichment Score table (all rows)
  • Pathway Crosstalk heatmap
  • 3D PCA image
  • Top Gene Expression Heatmap
  • Drug targets table
  • Auto-Insights
  • AI Discussion (Deep Scanning analysis)
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


def _fig_to_image(fig, width: int = 600, height: int = 360) -> io.BytesIO | None:
    """Render Plotly figure to high-res PNG bytes via kaleido. Returns None on failure."""
    if fig is None:
        return None
    try:
        import plotly.io as pio
        img_bytes = pio.to_image(
            fig, format="png",
            width=width, height=height,
            scale=2,            # 2× for print quality
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
    # ── New parameters (v3.8) ─────────────────────────────────────────────
    bubble_fig=None,
    bar_fig=None,
    crosstalk_fig=None,
    heatmap_fig=None,
    ai_discussion: str | None = None,
    powered_by: str = "",
) -> bytes:
    """Return PDF as bytes.

    Parameters
    ----------
    volcano_fig     : Plotly Figure or dict — Volcano plot
    pathway_df      : DataFrame — full enrichment results table
    pca_fig         : Plotly Figure or dict — 3D PCA
    drug_df         : DataFrame — drug target table (optional)
    insights        : list of insight dicts from generate_insights()
    summary_stats   : dict with keys total/up/down/sig
    bubble_fig      : Plotly Figure — enrichment bubble chart (optional)
    bar_fig         : Plotly Figure — pathway bar chart (optional)
    crosstalk_fig   : Plotly Figure — pathway crosstalk heatmap (optional)
    heatmap_fig     : Plotly Figure — top gene expression heatmap (optional)
    ai_discussion   : str — LLM-generated biological discussion text (optional)
    powered_by      : str — model label for the Discussion footer
    """
    # ── Normalise inputs ──────────────────────────────────────────────────────
    if isinstance(pathway_df, list):
        pathway_df = pd.DataFrame(pathway_df)
    if isinstance(drug_df, list):
        drug_df = pd.DataFrame(drug_df)

    def _to_fig(f):
        if f is None:
            return None
        if isinstance(f, go.Figure):
            return f
        try:
            return go.Figure(f)
        except Exception:
            return None

    volcano_fig  = _to_fig(volcano_fig)
    pca_fig      = _to_fig(pca_fig)
    bubble_fig   = _to_fig(bubble_fig)
    bar_fig      = _to_fig(bar_fig)
    crosstalk_fig = _to_fig(crosstalk_fig)
    heatmap_fig  = _to_fig(heatmap_fig)

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

    def section_img(fig, w=460, h=290, placeholder="Chart not rendered — visit the tab first."):
        img = _fig_to_image(fig, width=w, height=h)
        if img:
            elems.append(Image(img, width=w * 0.77, height=h * 0.77))
        else:
            elems.append(Paragraph(f"({placeholder})", body))

    # ── Title header ──────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "title", parent=styles["Title"],
        textColor=_BLUE, spaceAfter=4,
    )
    elems.append(Paragraph("Apex Bioinformatics Platform", title_style))
    elems.append(Paragraph("Differential Gene Expression Analysis Report", h2))
    elems.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC",
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
        st = Table(stat_data, colWidths=[120] * 4)
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

    # ── 1. Volcano Plot ───────────────────────────────────────────────────────
    elems.append(Paragraph("1. Volcano Plot", h2))
    section_img(volcano_fig, 600, 360,
                "Volcano chart not rendered — visit the Volcano tab first.")
    elems.append(Spacer(1, 12))
    hr()

    # ── 2. Pathway Enrichment Charts ──────────────────────────────────────────
    elems.append(Paragraph("2. Pathway Enrichment Analysis", h2))

    if bubble_fig or bar_fig:
        if bubble_fig:
            elems.append(Paragraph("Enrichment Bubble Chart", h3))
            section_img(bubble_fig, 560, 340)
            elems.append(Spacer(1, 8))
        if bar_fig:
            elems.append(Paragraph("Pathway Bar Chart", h3))
            section_img(bar_fig, 560, 320)
            elems.append(Spacer(1, 8))
    else:
        elems.append(Paragraph(
            "(Enrichment charts not captured — run enrichment on the Volcano tab first.)",
            body,
        ))

    elems.append(Spacer(1, 6))
    hr()

    # ── 3. Full Enrichment Score Table ────────────────────────────────────────
    elems.append(Paragraph("3. Enrichment Score Table (Full Results)", h2))
    if pathway_df is not None and not pathway_df.empty:
        show = [c for c in ["pathway", "database", "overlap_count",
                             "adjusted_p_value", "enrichment_score", "genes"]
                if c in pathway_df.columns]
        disp = pathway_df[show].copy()                     # ALL rows — no head() limit
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
        # Dynamic column widths based on number of columns
        avail_w = 520
        col_w = [avail_w // len(show)] * len(show)
        t = Table(header + rows_, colWidths=col_w, repeatRows=1)
        t.setStyle(_table_style(len(show)))
        elems.append(t)
    else:
        elems.append(Paragraph(
            "No enrichment results (run Volcano tab first).", body,
        ))
    elems.append(Spacer(1, 12))
    hr()

    # ── 4. Pathway Crosstalk Heatmap ──────────────────────────────────────────
    if crosstalk_fig is not None:
        elems.append(Paragraph("4. Pathway Crosstalk (Jaccard Gene Overlap)", h2))
        section_img(crosstalk_fig, 580, 380)
        elems.append(Spacer(1, 12))
        hr()

    # ── 5. Drug Targets ───────────────────────────────────────────────────────
    sec = 5 if crosstalk_fig is not None else 4
    if drug_df is not None and not drug_df.empty:
        elems.append(Paragraph(f"{sec}. FDA-Approved Drug Targets", h2))
        fda = drug_df[drug_df["fda"] == True].copy() if "fda" in drug_df.columns else drug_df
        show_d = [c for c in ["gene", "drug", "type", "indication"] if c in fda.columns]
        if show_d:
            header_d = [[c.title() for c in show_d]]
            rows_d   = [[str(row[c])[:40] for c in show_d]
                        for _, row in fda.head(30).iterrows()]
            td = Table(header_d + rows_d, repeatRows=1)
            td.setStyle(_table_style(len(show_d)))
            elems.append(td)
        elems.append(Spacer(1, 12))
        hr()
        sec += 1

    # ── 6. 3D PCA ─────────────────────────────────────────────────────────────
    elems.append(Paragraph(f"{sec}. 3D PCA Projection", h2))
    section_img(pca_fig, 560, 360,
                "PCA chart not rendered — visit the 3D PCA tab first.")
    elems.append(Spacer(1, 12))
    hr()
    sec += 1

    # ── 7. Top Gene Expression Heatmap ───────────────────────────────────────
    if heatmap_fig is not None:
        elems.append(Paragraph(f"{sec}. Top Gene Expression Heatmap", h2))
        section_img(heatmap_fig, 560, 420)
        elems.append(Spacer(1, 12))
        hr()
        sec += 1

    # ── 8. AI Discussion (Deep Scanning) ─────────────────────────────────────
    if ai_discussion:
        elems.append(PageBreak())
        elems.append(Paragraph(f"{sec}. Discussion — AI Deep Scan", h2))

        # Source attribution
        if powered_by:
            elems.append(Paragraph(
                f"<i>Generated by: {powered_by}</i>",
                ParagraphStyle("attr", parent=styles["Italic"],
                               textColor=colors.HexColor("#666666"), fontSize=8),
            ))
            elems.append(Spacer(1, 6))

        # Strip markdown bold/italic for clean PDF text
        clean_text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", ai_discussion)
        clean_text = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", clean_text)
        # Split on double newlines to preserve paragraphs
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
