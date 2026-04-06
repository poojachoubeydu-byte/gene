"""
PDF Report Generator — Apex Bioinformatics Platform
=====================================================
Generates a polished multi-section PDF including:
  • Summary statistics
  • Volcano plot image
  • Pathway enrichment table
  • Drug targets table
  • 3D PCA image
  • Auto-insights
"""

import io
import logging
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

log = logging.getLogger(__name__)

_BLUE  = colors.HexColor("#1a5276")
_GREEN = colors.HexColor("#1e8449")
_LIGHT = colors.HexColor("#f4f6f7")
_RED   = colors.HexColor("#c0392b")


def _fig_to_image(fig, width: int = 600, height: int = 360) -> io.BytesIO | None:
    """Render Plotly figure to PNG bytes via kaleido. Returns None on failure."""
    if fig is None:
        return None
    try:
        import plotly.io as pio
        img_bytes = pio.to_image(fig, format="png", width=width, height=height, engine="kaleido")
        buf = io.BytesIO(img_bytes)
        buf.seek(0)
        return buf
    except Exception as e:
        log.warning(f"kaleido: {e}")
        return None


def _table_style(n_cols: int) -> TableStyle:
    return TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  _BLUE),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 7),
        ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, _LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
    ])


def generate_pdf_report(
    volcano_fig,
    pathway_df: pd.DataFrame,
    pca_fig,
    drug_df: pd.DataFrame | None = None,
    insights: list | None = None,
    summary_stats: dict | None = None,
) -> bytes:
    """Return PDF as bytes."""
    if isinstance(pathway_df, list):
        pathway_df = pd.DataFrame(pathway_df)
    if isinstance(drug_df, list):
        drug_df = pd.DataFrame(drug_df)

    # Ensure Figure objects
    if not isinstance(volcano_fig, go.Figure):
        try:
            volcano_fig = go.Figure(volcano_fig)
        except Exception:
            volcano_fig = None
    if not isinstance(pca_fig, go.Figure):
        try:
            pca_fig = go.Figure(pca_fig)
        except Exception:
            pca_fig = None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=36, rightMargin=36,
                            topMargin=36, bottomMargin=36)
    styles  = getSampleStyleSheet()
    body    = styles["Normal"]
    h1      = styles["Heading1"]
    h2      = styles["Heading2"]
    elems   = []

    def hr():
        elems.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#bdc3c7")))
        elems.append(Spacer(1, 8))

    # ── Title page header ─────────────────────────────────────────────────────
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

    # ── Summary statistics banner ─────────────────────────────────────────────
    if summary_stats:
        elems.append(Paragraph("Analysis Summary", h2))
        stat_data = [[
            "Total Genes", "Up-regulated", "Down-regulated", "Significant",
        ], [
            str(summary_stats.get("total", "—")),
            str(summary_stats.get("up", "—")),
            str(summary_stats.get("down", "—")),
            str(summary_stats.get("sig", "—")),
        ]]
        st = Table(stat_data, colWidths=[120]*4)
        st.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0),  _BLUE),
            ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTNAME",     (0, 1), (-1, 1),  "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 10),
            ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
            ("GRID",         (0, 0), (-1, -1), 0.4, colors.grey),
        ]))
        elems.append(st)
        elems.append(Spacer(1, 12))

    # ── Auto-Insights ────────────────────────────────────────────────────────
    if insights:
        elems.append(Paragraph("Key Biological Insights", h2))
        for ins in insights[:6]:
            elems.append(Paragraph(
                f"{ins.get('icon','')} <b>{ins.get('title','')}</b>",
                styles["Heading3"],
            ))
            import re
            body_text = re.sub(r"<[^>]+>", "", ins.get("body", ""))
            elems.append(Paragraph(body_text, body))
            elems.append(Spacer(1, 6))
        hr()

    # ── 1. Volcano Plot ───────────────────────────────────────────────────────
    elems.append(Paragraph("1. Volcano Plot", h2))
    vol_img = _fig_to_image(volcano_fig)
    if vol_img:
        elems.append(Image(vol_img, width=460, height=290))
    else:
        elems.append(Paragraph(
            "(Chart not rendered — visit the Volcano tab before downloading.)", body,
        ))
    elems.append(Spacer(1, 12))
    hr()

    # ── 2. Pathway Enrichment ─────────────────────────────────────────────────
    elems.append(Paragraph("2. Significantly Enriched Pathways", h2))
    if pathway_df is not None and not pathway_df.empty:
        show = [c for c in ["pathway", "database", "overlap_count",
                             "adjusted_p_value", "genes"]
                if c in pathway_df.columns]
        disp = pathway_df[show].head(20).copy()
        if "adjusted_p_value" in disp.columns:
            disp["adjusted_p_value"] = disp["adjusted_p_value"].map("{:.2e}".format)
        if "genes" in disp.columns:
            disp["genes"] = disp["genes"].str[:50]
        header  = [[c.replace("_", " ").title() for c in show]]
        rows_   = [[str(v) for v in row] for row in disp.values]
        t = Table(header + rows_, repeatRows=1)
        t.setStyle(_table_style(len(show)))
        elems.append(t)
    else:
        elems.append(Paragraph("No enrichment results (run Volcano tab first).", body))
    elems.append(Spacer(1, 12))
    hr()

    # ── 3. Drug Targets ───────────────────────────────────────────────────────
    if drug_df is not None and not drug_df.empty:
        elems.append(Paragraph("3. FDA-Approved Drug Targets", h2))
        fda = drug_df[drug_df["fda"] == True].copy() if "fda" in drug_df.columns else drug_df
        show_d = [c for c in ["gene", "drug", "type", "indication"] if c in fda.columns]
        header_d = [[c.title() for c in show_d]]
        rows_d   = [[str(row[c])[:40] for c in show_d] for _, row in fda.head(25).iterrows()]
        td = Table(header_d + rows_d, repeatRows=1)
        td.setStyle(_table_style(len(show_d)))
        elems.append(td)
        elems.append(Spacer(1, 12))
        hr()

    # ── 4. 3D PCA ─────────────────────────────────────────────────────────────
    elems.append(Paragraph("4. 3D PCA Projection", h2))
    pca_img = _fig_to_image(pca_fig)
    if pca_img:
        elems.append(Image(pca_img, width=460, height=290))
    else:
        elems.append(Paragraph(
            "(Chart not rendered — visit the 3D PCA tab before downloading.)", body,
        ))
    elems.append(Spacer(1, 10))

    # ── Footer ────────────────────────────────────────────────────────────────
    hr()
    elems.append(Paragraph(
        "Apex Bioinformatics Platform — For research use only. "
        "Results should be validated experimentally before clinical application.",
        ParagraphStyle("footer", parent=styles["Italic"],
                       textColor=colors.grey, fontSize=8),
    ))

    doc.build(elems)
    buf.seek(0)
    return buf.getvalue()
