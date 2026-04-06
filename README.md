---
title: Apex Bioinformatics Platform
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# 🧬 Apex Bioinformatics Platform

**Advanced RNA-Seq Multi-Omics Dashboard** — built for non-clinical research labs.
Supports **5+ concurrent users** with full session isolation.

---

## What makes this different from every other free tool

| Feature | Apex | GEO2R | Enrichr | GSEA App | STRING |
|---|:---:|:---:|:---:|:---:|:---:|
| Interactive lasso → real-time enrichment | ✅ | ❌ | ❌ | ❌ | ❌ |
| FDA drug target overlay on DE results | ✅ | ❌ | ❌ | ❌ | ❌ |
| Cancer gene census overlay on volcano | ✅ | ❌ | ❌ | ❌ | ❌ |
| Pathway crosstalk heatmap | ✅ | ❌ | ❌ | ❌ | ❌ |
| Biomarker priority composite score | ✅ | ❌ | ❌ | ❌ | ❌ |
| Auto-generated biological insights | ✅ | ❌ | ❌ | ❌ | ❌ |
| GSEA + ORA + Network + PCA in one view | ✅ | ❌ | ❌ | ❌ | ❌ |
| 3D PCA with K-means clustering | ✅ | ❌ | ❌ | ❌ | ❌ |
| WGCNA-lite co-expression network | ✅ | ❌ | ❌ | ❌ | ❌ |
| PDF report with all panels | ✅ | ❌ | ❌ | ❌ | ❌ |

---

## Tabs & Features

### 🏆 Meta-Score
Composite gene priority score (0–100) combining statistical significance and effect size.
Genes scoring ≥ 70 are highlighted green. Housekeeping genes excluded automatically.

### 🌋 Volcano & Pathway
- **Volcano plot** with cancer gene overlays:
  - ★ Gold star = known oncogene (COSMIC)
  - ◆ Blue diamond = tumor suppressor gene
  - ○ Green circle = FDA-approved drug target
- **Draw a lasso / box selection** on the volcano → pathway enrichment updates instantly for that gene subset
- **Pathway bubble + bar charts** across KEGG, GO-BP, Reactome (150+ pathways each, offline)
- **Pathway Crosstalk heatmap** — Jaccard similarity between enriched pathways reveals convergent biology

### 🔵 3D PCA
Principal component analysis with K-means clustering. Interactive 3D rotation.
PC1/PC2/PC3 explained variance shown. Genes colored by cluster.

### 📊 Advanced Analytics
- MA plot (mean expression vs fold change)
- Top DE gene heatmap (significance × effect size)
- Adjusted p-value distribution
- Log2FC distribution
- GSEA pre-rank metric waterfall

### 🧬 GSEA
Pre-ranked GSEA against KEGG 2021 + Reactome 2022.
Each user runs in an isolated temp directory — no cross-user file collisions.

### 🔗 Gene Network
WGCNA-lite co-expression network. Nodes colored by Log2FC (red = up, blue = down).
Edge color: red = positive correlation, blue = negative correlation.

### 💊 Drug Targets ★ *Unique*
Maps your significant DE genes to **80+ FDA-approved drugs** from a built-in curated database.
- Scatter: Log2FC vs number of drugs → upregulated targets = inhibitor candidates
- Donut chart: drug mechanism class breakdown
- Full table with drug name, type, indication

### 🎯 Biomarker Score ★ *Unique*
Extended priority score with clinical bonuses:
- +15 points — FDA-approved drug target
- +10 points — COSMIC Tier-1 cancer gene
- +5 points — Established clinical biomarker

Color-coded bar chart: red = oncogene, blue = TSG, purple = both, green = drug target only.

### 📋 Auto-Insights ★ *Unique*
Rule-based biological narrative automatically generated from your results:
- Dataset overview (up/down balance)
- Top driver genes
- Drug target count + examples
- Oncogene activation / TSG suppression pattern detection
- Pathway theme summary
- Immune checkpoint gene detection
- QC warnings (e.g. too few significant genes)

No AI, no internet required — fully reproducible.

### 🔍 Gene Bio-Context (sidebar)
Click any gene symbol to fetch live annotations from **NCBI Entrez + UniProt**:
full name, molecular function, subcellular location, disease associations.
Results are cached per session to respect API rate limits.

---

## Input Format

Upload a CSV with columns (flexible naming):

| Required | Aliases accepted |
|---|---|
| `symbol` | gene_name, gene, hgnc_symbol, geneName |
| `log2FC` | log2FoldChange, lfc, foldChange |
| `padj` | adj.p.val, fdr, p_adj, adj_p_value |

Optional: `baseMean` / `avgExpr` — improves Meta-Score and MA plot.

**Example row:**
```
TP53, 2.5, 0.00001, 1240.5
```

A 50-gene demo dataset is loaded automatically if no file is uploaded.

---

## Multi-User Architecture

- **Gunicorn** with 4 sync workers — handles 5+ concurrent users
- **Per-session `dcc.Store`** — complete data isolation between users
- **GSEA temp dirs** — unique UUID-prefixed directory per call, auto-cleaned
- **Enrichment analyzers** — stateless objects, safely shared across workers
- **Bio-context cache** — thread-locked in-memory cache, shared read-only
- **Pure computation functions** — no global mutable state

---

## Running Locally

```bash
# Clone
git clone https://huggingface.co/spaces/Poojachoubeydubyte/gene-dashboard
cd gene-dashboard

# Install
pip install -r requirements.txt

# Run (development)
python app.py

# Run (production — matches HuggingFace deployment)
gunicorn -w 4 -b 0.0.0.0:7860 --timeout 120 app:server
```

---

## Tech Stack

| Component | Library |
|---|---|
| Web framework | Dash 2.x + Dash Bootstrap Components |
| Visualisation | Plotly |
| Statistics | SciPy, statsmodels |
| Machine learning | scikit-learn (PCA, K-means) |
| Network analysis | NetworkX |
| GSEA | GSEApy |
| PDF generation | ReportLab |
| Server | Gunicorn |

---

## Data Privacy

All data is processed **server-side in your session only**.
No data is stored, logged, or shared between users.
Suitable for unpublished research data in a lab environment.

---

## Disclaimer

For research use only. Results should be validated experimentally
before any clinical or diagnostic application.

---

*Built with ❤️ for non-clinical bioinformatics research labs.*
