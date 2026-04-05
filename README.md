# Real-Time RNA-Seq Pathway Visualizer (RRPV)

## Biological Context
When a cell is exposed to a drug, a pathogen, or undergoes disease progression, thousands of genes change their activity simultaneously. RNA sequencing (RNA-Seq) measures these changes across the entire genome. However, reading a list of 500 differentially expressed genes tells us almost nothing — the biological meaning emerges only when we ask: *which coordinated biological programs (pathways) are being activated or suppressed?*

This tool answers that question interactively. A researcher uploads their gene expression data, sees a volcano plot of all differentially expressed genes, and can click any gene to immediately see which biological pathways it participates in. Conversely, clicking a pathway highlights all its member genes on the volcano plot.

## Features
- **Interactive Volcano Plot**: Lasso-select genes to run real-time enrichment.
- **Pathway Enrichment**: Uses Enrichr (KEGG, GO, MSigDB) via GSEApy.
- **Bidirectional Linking**: Click a pathway to highlight genes in the volcano plot.

## Local Installation
1. Install requirements: pip install -r requirements.txt
2. Run the app: python app.py
3. Open http://127.0.0.1:8050/ in your browser.

## Deployment
This app is containerized with Docker and ready for deployment on AWS Amplify or Hugging Face Spaces.
