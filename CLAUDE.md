# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

**Local development:**
```bash
pip install -r requirements.txt
python app.py
```

**Production (Gunicorn, mirrors Docker/HF deployment):**
```bash
gunicorn -w 4 -b 0.0.0.0:7860 --timeout 120 app:server
```

**Docker:**
```bash
docker build -t apex-bio .
docker run -p 7860:7860 --env-file .env apex-bio
```

## Environment Variables

Create a `.env` file in the project root (never commit it). The app gracefully degrades if keys are absent:

```
GEMINI_API_KEY=...   # Primary AI engine (Gemini 2.5 Flash)
GROQ_API_KEY=...     # Fallback AI engine (Llama 3 via Groq LPU)
```

`load_dotenv(override=False)` is called at startup so HuggingFace Space secrets always take precedence over `.env`.

## Architecture Overview

### Entry Point: `app.py`

Single-file Dash application (`app.server` is the Gunicorn entry point). Responsibilities:
- Initialises the Dash app with Bootstrap theme
- Detects available AI keys at startup and sets the copilot badge
- Defines all Dash callback functions (the interactive logic)
- `_normalise(df)` — two-pass column-name detection (exact then substring) that maps arbitrary DEG CSV column names to the internal schema: `symbol`, `log2FC`, `padj`, `pvalue`, `baseMean`
- `_s2df(data)` — reconstructs a DataFrame from `dcc.Store` records; returns an empty schema-valid DataFrame (never demo data) so callers can detect the no-data state
- `_get_analyzer(db_choice)` — per-worker cache of `EnrichmentAnalyzer` / `MultiDatabaseEnrichment` instances

### Key Callback: `cb_volcano_enr`

Global enrichment runs automatically whenever `sig-store` changes (all sig genes). Critical behaviour:
- **ORA cap**: `MAX_ORA_GENES = 500` — if sig gene count exceeds 500, the callback takes the top-500 by `padj` (`nsmallest`). This is required because ORA loses statistical power when query/background ratio > ~5%. The `capped` flag drives the `sel-counter` label.
- **`sel-counter`** shows "Top 500 … (concentrated signal from N total)" when capped, otherwise "N Significant Genes".
- **Background**: uses the full uploaded dataset size, not the 17k-decimated plot.
- `gene_lfc_map` / `gene_nlp_map` are built from the full df and passed to enrichment for activation Z-score calculation.

### Module Map (`modules/`)

| Module | Role |
|---|---|
| `analysis.py` | All heavy computation: PCA 3D, WGCNA-lite (cosine similarity + β=6 soft power), GSEA pre-ranked (1 000 perms, per-call temp dir for worker safety), meta-score, biomarker score, pathway crosstalk (Jaccard). Every function is pure/stateless for Gunicorn safety. |
| `pathway.py` | Offline enrichment via built-in KEGG/GO-BP/Reactome gene sets. Fisher exact test + BH FDR. Outputs odds ratio, 95% CI, gene ratio, activation Z-score, effect-weighted score. `EnrichmentAnalyzer` (single DB) and `MultiDatabaseEnrichment` (all three) are the public classes. |
| `plots.py` | All Plotly figure factories (`create_volcano_plot`, `create_pathway_bubble`, `create_pathway_bar`, `create_top_heatmap`, `create_pca_3d`, `create_ma_plot`, etc.). Returns `go.Figure`; never touches Dash state. |
| `reports.py` | PDF export via ReportLab. `_fig_to_image()` renders each Plotly figure with kaleido at `scale=3` (effective 2400 × 1440 px). Covers 15 panel sections including AI discussion. |
| `bio_context.py` | Gene annotations via NCBI Entrez + UniProt REST. Thread-safe in-process dict cache (TTL 1 h). `fetch_gene_info(symbol)` is the public API. |
| `ai_summary.py` | Tiered AI waterfall: Gemini 2.5 Flash → Groq Llama 3 → rule-based. In-process dict cache keyed by SHA-256 of (genes, thresholds, db) with 24 h TTL. `invalidate_cache_key()` is called on forced Refresh. |
| `data_validator.py` | Input CSV validation |
| `session_manager.py` | Per-user session isolation for concurrent Gunicorn workers |
| `id_mapper.py` | Gene ID / symbol mapping utilities |

### `data/gene_annotations.py`

Provides `get_drug_targets()` and `get_cancer_gene_info()` — static FDA drug overlay and Cancer Gene Census data used in the Volcano plot (oncogene ★ / TSG ◆ markers).

### `bio_context.py` (root)

Re-exports everything from `modules/bio_context.py` (`from modules.bio_context import *`). Exists for backwards-compatibility with older import paths.

## Key Design Decisions

**Gene symbol normalisation** is applied at every boundary (upload, lasso selection, DB lookups): strip whitespace/quotes, uppercase. The helper `_clean_sym()` is duplicated in both `app.py` and `modules/analysis.py` intentionally — keep them in sync.

**Plot directionality** in `create_pathway_bar` and `create_pathway_bubble` follows IPA convention: Activated = red (`#c0392b`), Inhibited = blue (`#2980b9`), Inconclusive = gray. The `direction` column is computed in `pathway.py` from activation Z-score. The bubble chart uses marker shape (▲/▼/●) in addition to color.

**Concurrency model**: 4 Gunicorn sync workers. All module-level state that is safe to share is read-only (gene-set dicts, analyzer instances). GSEA uses per-call `tempfile.mkdtemp()` to avoid worker collisions. The AI and bio-context caches use Python's GIL for dict safety (no explicit locks in `ai_summary.py`; `bio_context.py` uses `threading.Lock`).

**Large dataset handling**: The volcano plot decimates datasets > 17 k genes before rendering. Pan/zoom guard prevents lasso callbacks from firing during map interaction. The enrichment ORA cap (500 genes) is a separate concern — it applies to statistical power, not rendering.

**Enrichment is offline-first**: `pathway.py` ships built-in KEGG/GO-BP/Reactome gene sets; no Enrichr API call is required. The Enrichr web API (mentioned in README) is a legacy path.

**PDF scale**: `scale=3` in `_fig_to_image()` is intentional — do not lower it without understanding the print-DPI impact.

## Deployment

- **HuggingFace Spaces**: Docker SDK, port 7860, secrets set via Space settings UI
- **Local Docker**: `docker run -p 7860:7860 --env-file .env apex-bio`
- The non-root `labuser` in the Dockerfile is a security requirement — keep it.
