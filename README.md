---
title: Gene Dashboard
emoji: 🧬
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🧬 Real-Time RNA-Seq Pathway Visualizer

An interactive, industry-grade bioinformatics dashboard for differential gene expression analysis and biological pathway mapping. Built in Python using Plotly Dash, GSEApy, and Pandas.

## Biological Context

When a cell responds to a drug, pathogen, or disease state, thousands of genes change their activity simultaneously. RNA sequencing (RNA-Seq) measures these changes across the entire transcriptome. However, a list of 500 differentially expressed genes (DEGs) conveys little biological meaning on its own — insight emerges only when we ask: *which coordinated biological programs (pathways) are being activated or suppressed?*

This tool answers that question interactively. A researcher uploads their DEG results, explores a volcano plot of all expressed genes, lassos a region of interest, and immediately sees which KEGG and Reactome pathways are enriched. Clicking any pathway bubble highlights exactly which of the user's genes belong to it — closing the loop between statistical output and biological interpretation.

## Features

- **Interactive Volcano Plot** — lasso-select any gene cluster; gene labels on top DEGs
- **Pathway Enrichment** — Over-Representation Analysis via Enrichr API (KEGG 2021 + Reactome 2022); Fisher exact test with Benjamini-Hochberg correction
- **Bidirectional Linking** — lasso genes → see pathways; click pathway → highlight genes on volcano
- **Fold-Change Heatmap** — ranked magnitude and direction view of selected genes
- **Adjustable Thresholds** — log2FC and adjusted p-value sliders
- **Demo Dataset** — pre-loaded PBMC LPS vs Control synthetic data; no upload needed
- **Export** — enrichment table downloadable as CSV directly from the table

## Input Format

Upload a CSV file with at minimum these three columns (column names are flexible — the tool detects common variants):

| Column | Accepted names | Description |
|--------|---------------|-------------|
| Gene symbol | symbol, gene, gene_name, id, name | HGNC gene symbol (e.g. TNF, IL6) |
| Fold change | log2FC, logFC, LFC, log2FoldChange | Log2 fold change between conditions |
| Adjusted p-value | padj, FDR, qvalue, p.adj | Multiple-testing corrected p-value |

Example row: `TNF, 3.45, 0.0001`

## Public Datasets for Testing

- NCBI GEO: https://www.ncbi.nlm.nih.gov/geo/ — search any GSE accession
- DESeq2 output from any bulk RNA-Seq pipeline works directly

## Known Limitations

- Enrichment uses Enrichr web API — requires internet connectivity; may be slow under high API load
- Input limited to 50 MB; datasets >50,000 genes may be slow to render
- Single-cell RNA-Seq mode is not yet implemented — bulk DEG tables only
- GSEA preranked mode (which does not require a significance threshold) is not yet implemented

## Tech Stack

`plotly-dash` · `gseapy` · `pandas` · `numpy` · `scipy` · `dash-bootstrap-components` · `gunicorn` · Docker

## Citation

If you use the Enrichr API for published results, please cite:
Kuleshov MV et al. Enrichr: a comprehensive gene set enrichment analysis web server. *Nucleic Acids Research*, 2016.

# Last Build: 04/05/2026 11:42:13
