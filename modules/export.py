import io
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, PageBreak
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
import pandas as pd
import numpy as np


def compute_data_integrity_score(df: pd.DataFrame) -> dict:
    total_cells = df.shape[0] * df.shape[1]
    missing = df.isnull().sum().sum()
    completeness = round((1 - missing / total_cells) * 25, 1)

    import re
    pattern = re.compile(r'^[A-Z][A-Z0-9\-]{1,19}$')
    valid = df['symbol'].astype(str).apply(lambda x: bool(pattern.match(x))).mean()
    symbol_format = round(valid * 25, 1)

    sig_frac = (df['padj'] < 0.05).mean()
    if 0.01 <= sig_frac <= 0.40:
        p_score = 25.0
    elif sig_frac < 0.01 or sig_frac > 0.60:
        p_score = 5.0
    else:
        p_score = 15.0

    pos_frac = (df['log2FC'] > 0).mean()
    symmetry = 1 - abs(pos_frac - 0.5) * 2
    fc_symmetry = round(symmetry * 25, 1)

    total = round(completeness + symbol_format + p_score + fc_symmetry, 1)
    grade = 'A' if total >= 90 else 'B' if total >= 75 else 'C' if total >= 60 else 'D'

    return {
        'total': total,
        'grade': grade,
        'components': {
            'completeness': completeness,
            'symbol_format': symbol_format,
            'pvalue_distribution': p_score,
            'fc_symmetry': fc_symmetry
        },
        'n_genes': len(df),
        'n_significant': int((df['padj'] < 0.05).sum()),
        'sig_fraction': round(sig_frac * 100, 1)
    }


def generate_pdf_report(df: pd.DataFrame,
                        integrity: dict,
                        gsea_result: dict,
                        ora_result: dict,
                        analysis_params: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'Title', parent=styles['Title'],
        fontSize=18, spaceAfter=12,
        textColor=colors.HexColor('#1565c0'),
        alignment=TA_CENTER
    )
    h2_style = ParagraphStyle(
        'H2', parent=styles['Heading2'],
        fontSize=12, spaceBefore=14, spaceAfter=8,
        textColor=colors.HexColor('#1565c0')
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'],
        fontSize=10, spaceAfter=8, leading=14
    )
    caption_style = ParagraphStyle(
        'Caption', parent=styles['Normal'],
        fontSize=8, textColor=colors.grey, alignment=TA_CENTER
    )

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    story.append(Paragraph('Multi-Omics Precision Discovery Suite', title_style))
    story.append(Paragraph('Reproducibility Report', styles['Heading2']))
    story.append(Paragraph(f'Generated: {now}', caption_style))
    story.append(HRFlowable(width='100%', thickness=2, color=colors.HexColor('#1565c0')))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph('1. Data Integrity Assessment', h2_style))
    story.append(Paragraph(
        f"Dataset contains {integrity['n_genes']:,} genes, "
        f"with {integrity['n_significant']:,} significant genes (" 
        f"{integrity['sig_fraction']}%). "
        f"Data Integrity Score: {integrity['total']}/100 (Grade {integrity['grade']}).",
        body_style
    ))
    story.append(Spacer(1, 0.2*cm))

    score_table = Table([
        ['Component', 'Score'],
        ['Completeness', f"{integrity['components']['completeness']}/25"],
        ['Symbol Format', f"{integrity['components']['symbol_format']}/25"],
        ['P-value Distribution', f"{integrity['components']['pvalue_distribution']}/25"],
        ['FC Symmetry', f"{integrity['components']['fc_symmetry']}/25"]
    ], colWidths=[8*cm, 4*cm])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1565c0')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')])
    ]))
    story.append(score_table)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph('2. Methods', h2_style))
    methods_text = (
        f"Differential expression thresholds: |log2FC| > {analysis_params.get('lfc_thresh', 1.0)} "
        f"and adjusted p-value < {analysis_params.get('padj_thresh', 0.05)}. "
        f"GSEA gene sets: {', '.join(analysis_params.get('gene_sets', []))}. "
        f"Preranked GSEA used sign(log2FC) × -log10(p-value) as ranking metric. "
        f"Co-expression network analysis used hierarchical clustering and Pearson correlation. "
        f"Meta-Score combines effect size and significance into a composite prioritisation score."
    )
    story.append(Paragraph(methods_text, body_style))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph('3. Top 10 Differentially Expressed Genes', h2_style))
    top_degs = df.sort_values('padj', ascending=True).head(10)
    headers = ['Gene', 'Log2FC', 'Adj. P-value']
    data = [headers]
    for _, row in top_degs.iterrows():
        data.append([
            str(row['symbol']),
            f"{row['log2FC']:.3f}",
            f"{row['padj']:.2e}"
        ])
    table = Table(data, colWidths=[5*cm, 4*cm, 4*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1565c0')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')])
    ]))
    story.append(table)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph('4. Pathway Enrichment Results (ORA)', h2_style))
    ora_df = ora_result.get('results', pd.DataFrame())
    if ora_df.empty:
        story.append(Paragraph('No ORA results available.', body_style))
    else:
        headers = ['Pathway', 'Overlap', 'Adj. P-value', 'Score']
        data = [headers]
        for _, row in ora_df.head(10).iterrows():
            data.append([
                str(row.get('Term', ''))[:45],
                str(row.get('Overlap', '')),
                f"{row.get('Adjusted P-value', 1):.2e}",
                f"{row.get('Score', row.get('Combined Score', 0)):.1f}"
            ])
        ptable = Table(data, colWidths=[6*cm, 3*cm, 3*cm, 3*cm])
        ptable.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1565c0')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')])
        ]))
        story.append(ptable)
        story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph('5. Reproducibility Statement', h2_style))
    repro = (
        'This report was generated automatically by the Multi-Omics Precision Discovery Suite. '
        'All analysis parameters are recorded for reproducibility. Review the auto-generated methods before publication. '
        'Use the same input dataset and thresholds to reproduce the results. '
        'This document is intended as a computational reproducibility record and should be validated by the investigator.'
    )
    story.append(Paragraph(repro, body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
