"""
AI Summary Module — Apex Bioinformatics Platform v5.0 (Deep-Analysis Consultant)
=================================================================================
Tiered waterfall: Gemini 2.5 Flash (primary) → Groq Llama 3 (secondary) → Rule-Based.

Public API
----------
get_biological_story_cached(context_data, ...)   → (summary, powered_by, from_cache)
    4-layer deep analysis with SHA-256 server-side cache (24 h TTL).

get_consultation_response(question, thread, context_data)  → (response, powered_by)
    Interactive follow-up — no caching, always fresh.

check_available_model()                          → (label, badge_color)
    Env-var check only — no network calls.

invalidate_cache_key(key)                        → None
    Called on forced Refresh to bypass cached result.
"""

import hashlib
import logging
import os
import time

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Server-side AI response cache — keyed by (genes, lfc, pval, db)
# TTL = 24 hours.  Thread-safe for single-process Dash / Gunicorn because
# Python's GIL protects dict reads/writes.
# ─────────────────────────────────────────────────────────────────────────────

_CACHE_TTL: int = 24 * 3600
_AI_CACHE:  dict[str, tuple[float, str, str]] = {}   # key → (ts, summary, powered_by)


def _make_cache_key(
    top_genes: list[dict],
    lfc_thresh: float,
    p_thresh: float,
    pathway_db: str,
    active_tab: str = "",
) -> str:
    gene_part = ",".join(sorted(g["symbol"].upper() for g in top_genes))
    raw       = (
        f"{gene_part}|{round(lfc_thresh, 3)}|{round(p_thresh, 5)}"
        f"|{pathway_db or 'all'}|{active_tab or 'none'}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(key: str) -> tuple[str, str] | None:
    entry = _AI_CACHE.get(key)
    if entry is None:
        return None
    ts, summary, powered_by = entry
    if time.time() - ts > _CACHE_TTL:
        del _AI_CACHE[key]
        return None
    return summary, powered_by


def _cache_set(key: str, summary: str, powered_by: str) -> None:
    _AI_CACHE[key] = (time.time(), summary, powered_by)


def invalidate_cache_key(key: str) -> None:
    """Remove a single entry — called when the user forces a Refresh."""
    _AI_CACHE.pop(key, None)


def cache_stats() -> dict:
    now  = time.time()
    live = {k: v for k, v in _AI_CACHE.items() if now - v[0] <= _CACHE_TTL}
    return {"live_entries": len(live), "total_entries": len(_AI_CACHE),
            "ttl_hours": _CACHE_TTL / 3600}


# ── Per-tab interpretation instructions ──────────────────────────────────────
# Maps tab ID → (human label, what the AI must specifically do).
# The AI is FORCED to interpret actual dataset values, not describe the tool.
_TAB_INSTRUCTIONS: dict[str, tuple[str, str]] = {
    "tab-meta": (
        "Meta-Score Prioritization",
        "Interpret the META-SCORE RANKING of the actual genes listed. "
        "Explain what the top-ranked genes (highest |LFC| × −log10p combined score) "
        "reveal about the dominant biological process. Identify functional clusters in "
        "the top 20 ranked genes and explain which genes achieve high rank through "
        "extreme significance vs. large fold-change.",
    ),
    "tab-volcano": (
        "Volcano & Pathway Analysis",
        "Interpret the ACTUAL VOLCANO PLOT DATA: comment on the up/down asymmetry "
        "(ratio of up vs. down genes), name the extreme outliers visible in the top-right "
        "and top-left quadrants (high |LFC| + very low p-value), and explain what the "
        "spatial clustering of significant genes implies about the biology. Then explain "
        "what the enriched pathways listed reveal about the coordinated gene program.",
    ),
    "tab-pca": (
        "3D PCA Projection",
        "Interpret THIS DATASET'S PCA RESULTS using the variance percentages provided. "
        "State which fraction of total variance PC1/PC2/PC3 capture. Based on the top "
        "up/down genes in the dataset, hypothesize what biological signal each PC likely "
        "represents (e.g., treatment response, cell identity, proliferation). Interpret "
        "what the sample distribution in 3D space implies about group separation and "
        "transcriptional coherence.",
    ),
    "tab-adv": (
        "Advanced Analytics — Heatmap & Diagnostics",
        "Interpret THIS DATASET'S HEATMAP AND DIAGNOSTIC PLOTS. For the heatmap: "
        "describe what the major co-expression clusters of up/down-regulated genes "
        "suggest as distinct biological programs. For the MA plot: comment on whether "
        "the LFC is expression-magnitude dependent (funnel shape = regression to mean). "
        "For the p-value histogram: is it anti-conservative (spike near 0) suggesting "
        "true signal, or uniform suggesting noise?",
    ),
    "tab-gsea": (
        "GSEA Pre-Ranked Results",
        "Interpret THESE SPECIFIC GSEA RESULTS. For each top enriched gene set: state "
        "whether it is positively or negatively enriched (NES direction), what the "
        "enrichment score magnitude implies about effect strength, and how the GSEA "
        "findings confirm or contradict the over-representation results from pathway "
        "enrichment. Identify any GSEA pathways that were NOT detected by ORA and explain why.",
    ),
    "tab-net": (
        "WGCNA-lite Co-expression Network",
        "Interpret THIS DATASET'S CO-EXPRESSION NETWORK topology. Identify the top hub "
        "genes (highest connectivity) from the gene list and explain their biological "
        "significance. Describe what the major network modules imply about co-regulated "
        "biological programs. Which gene-gene connections represent the most mechanistically "
        "important regulatory relationships based on known biology?",
    ),
    "tab-drugs": (
        "FDA Drug Target Landscape",
        "Interpret the DRUG TARGET OVERLAY for this specific dataset. Name which "
        "significantly DE genes have FDA-approved drugs, whether their up/down regulation "
        "makes them inhibitor or activator candidates, and which drug-target interactions "
        "are most clinically actionable. Identify any cases where a drug target is "
        "up-regulated (suggesting inhibitor use) or down-regulated (suggesting activator/replacement).",
    ),
    "tab-bm": (
        "Biomarker Priority Score",
        "Interpret the BIOMARKER PRIORITY SCORES for this dataset. Name the top-ranked "
        "biomarker candidates and explain whether their high score is driven by statistical "
        "strength (high LFC/padj), clinical bonuses (FDA drug target / COSMIC tier-1), or both. "
        "Distinguish novel candidates from already-validated biomarkers. Comment on what "
        "the score distribution implies about the proportion of clinically vs. statistically "
        "driven candidates.",
    ),
    "tab-insights": (
        "Auto-Generated Insights",
        "Provide a CROSS-TAB SYNTHESIS integrating the gene-level, pathway-level, network, "
        "and clinical (drug/biomarker) findings from this dataset into a single unified "
        "biological narrative.",
    ),
}

# ── Shared system prompt ──────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are a Senior Bioinformatician with 20 years of experience in oncology "
    "and systems biology. You interpret RNA-seq differential expression results "
    "for a research audience. Be rigorous, precise, and use correct HGNC gene "
    "nomenclature. Ground every statement strictly in the data provided."
)


# ─────────────────────────────────────────────────────────────────────────────
# Public status helper
# ─────────────────────────────────────────────────────────────────────────────

def check_available_model() -> tuple[str, str]:
    """
    Returns (label, badge_color) for the first detected API key.
    No network calls — purely an env-var check.
    Priority: Gemini → Groq → Offline.
    """
    if os.environ.get("GEMINI_API_KEY", "").strip():
        return "Online · Gemini 2.5 Flash", "success"
    if os.environ.get("GROQ_API_KEY", "").strip():
        return "Online · Groq Llama 3", "info"
    return "Offline · Rule-Based", "secondary"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_deep_analysis_prompt(context_data: dict) -> str:
    """
    4-layer Expert Investigator prompt.

    context_data keys used:
        top_genes     — list of {symbol, log2FC}
        top_pathways  — list of pathway name strings
        jaccard_pairs — list of {a, b, j, shared}
        enr_details   — list of {pathway, z_score, overlap_count} (optional)
        n_up, n_down  — significant gene counts (optional)
        active_tab    — tab ID string (optional, e.g. 'tab-pca')
        pca_variance  — PCA variance text (optional, for tab-pca)
        extra         — additional context string (optional)
    """
    top_genes     = context_data.get("top_genes",     [])
    top_pathways  = context_data.get("top_pathways",  [])
    jaccard_pairs = context_data.get("jaccard_pairs", [])
    enr_details   = context_data.get("enr_details",   [])
    n_up          = context_data.get("n_up",  0)
    n_down        = context_data.get("n_down", 0)
    extra         = context_data.get("extra", "")
    active_tab    = context_data.get("active_tab",   "")
    pca_variance  = context_data.get("pca_variance", "")

    # Split genes by direction
    up_genes = [g for g in top_genes if float(g.get("log2FC", 0)) > 0]
    dn_genes = [g for g in top_genes if float(g.get("log2FC", 0)) < 0]

    up_block = "\n".join(
        f"  ↑ {g['symbol']}: Log2FC = {float(g['log2FC']):+.3f}"
        for g in up_genes[:8]
    ) or "  (none in top selection)"

    dn_block = "\n".join(
        f"  ↓ {g['symbol']}: Log2FC = {float(g['log2FC']):+.3f}"
        for g in dn_genes[:8]
    ) or "  (none in top selection)"

    pathway_block = (
        "\n".join(f"  {i+1}. {p}" for i, p in enumerate(top_pathways[:10]))
        or "  No significantly enriched pathways detected."
    )

    # Enrichment z-score detail
    enr_detail_block = ""
    if enr_details:
        lines = ["Pathway Enrichment Details (activation z-score, overlap genes):"]
        for d in enr_details[:8]:
            z = d.get("z_score", None)
            z_str = f"{float(z):+.2f}" if isinstance(z, (int, float)) else "N/A"
            lines.append(
                f"  {str(d.get('pathway',''))[:45]}: z={z_str}, "
                f"overlap={d.get('overlap_count','?')}"
            )
        enr_detail_block = "\n" + "\n".join(lines) + "\n"

    # Jaccard crosstalk
    jaccard_block = ""
    if jaccard_pairs:
        lines = ["Pathway Gene Overlap (Jaccard Index):"]
        for pair in jaccard_pairs[:6]:
            lines.append(
                f"  {pair['a']} ↔ {pair['b']}: "
                f"J={pair['j']:.2f} ({pair['shared']} shared genes)"
            )
        jaccard_block = "\n" + "\n".join(lines) + "\n"

    counts_line = ""
    if n_up or n_down:
        counts_line = f"Total significant: {n_up} up-regulated, {n_down} down-regulated.\n"

    extra_block = f"\nExperiment context: {extra}\n" if extra else ""

    # ── Tab-specific current-view instruction ────────────────────────────────
    tab_label, tab_instruction = _TAB_INSTRUCTIONS.get(
        active_tab, ("", "")
    )
    current_view_block = ""
    if tab_label:
        pca_line = (
            f"\nPCA Variance (actual data): {pca_variance}\n"
            if pca_variance and active_tab == "tab-pca"
            else ""
        )
        current_view_block = (
            f"=== CURRENT VIEW: {tab_label} ===\n"
            f"MANDATORY INSTRUCTION: {tab_instruction}\n"
            f"{pca_line}"
            "RULE: Your response MUST reference specific gene names, numerical "
            "values, and observed patterns from the data below. Do NOT explain "
            "what the analysis type is in general — interpret what THIS DATASET shows.\n\n"
        )

    return (
        "Perform a DEEP MULTI-LAYER BIOLOGICAL ANALYSIS of these RNA-seq "
        "differential expression results.\n\n"
        f"{current_view_block}"
        "=== EXPRESSION DATA ===\n"
        f"{counts_line}"
        f"Top Up-Regulated Genes:\n{up_block}\n\n"
        f"Top Down-Regulated Genes:\n{dn_block}\n\n"
        f"Top Enriched Pathways:\n{pathway_block}\n"
        f"{enr_detail_block}"
        f"{jaccard_block}"
        f"{extra_block}\n"
        "=== FOUR-LAYER INVESTIGATION ===\n\n"
        "**LAYER 1 — Statistical Topology:**\n"
        "Analyze the expression landscape holistically. What biological 'story' do the "
        "top up- and down-regulated genes tell? Identify lineage-specific signatures "
        "(erythroid, immune, epithelial, stem cell), stress-response motifs, or oncogenic "
        "activation patterns. Note any asymmetry between the up/down arms and whether the "
        "magnitude distribution suggests a strong driver event or a diffuse regulatory shift.\n\n"
        "**LAYER 2 — Pathway Synergy:**\n"
        "Cross-reference enriched pathways with gene-level directional data. Identify where "
        "multiple pathways converge (co-activation or co-repression) and where enrichment "
        "conflicts with gene direction (e.g., a pathway is enriched but its key members are "
        "suppressed — indicating a compensatory or paradoxical response). If z-scores are "
        "provided, interpret each activation state explicitly. Cite specific gene–pathway "
        "relationships and flag any activation/inhibition conflicts.\n\n"
        "**LAYER 3 — Outlier & Cluster Diagnostics:**\n"
        "Identify genes that do not fit the dominant expression pattern and hypothesize their "
        "biological role (e.g., feedback inhibitors, bystander effects, alternative splicing "
        "artefacts). Comment on what the Jaccard pathway overlaps imply about co-regulation "
        "modularity — are these tightly linked responses or parallel independent cascades? "
        "What does the implied cluster structure suggest about transcriptional coherence and "
        "the number of distinct biological programs active simultaneously?\n\n"
        "**LAYER 4 — Literature Synthesis & Actionable Hypotheses:**\n"
        "Synthesize a specific disease or biological-context hypothesis. Name known co-activation "
        "or co-repression signatures that match this pattern (e.g., 'JAK2/VEGFA co-activation is "
        "a documented resistance mechanism in myeloproliferative neoplasms'). State the most "
        "likely disease context with supporting logic. List the top 2–3 most actionable drug "
        "target candidates, specifying whether they are direct DE targets or upstream regulators, "
        "and note any existing approved drugs or active clinical trials.\n\n"
        "Minimum 350 words. Each layer MUST be headed exactly as shown. Use HGNC nomenclature "
        "throughout. Cite gene symbols and pathway names in every layer."
    )


def _build_chat_prompt(question: str, thread: list[dict], context_data: dict) -> str:
    """Context-aware follow-up prompt preserving the investigation thread."""
    top_genes    = context_data.get("top_genes",    [])
    top_pathways = context_data.get("top_pathways", [])
    active_tab   = context_data.get("active_tab",   "")
    pca_variance = context_data.get("pca_variance", "")
    n_up         = context_data.get("n_up",  0)
    n_down       = context_data.get("n_down", 0)

    gene_summary = ", ".join(
        f"{g['symbol']} ({float(g['log2FC']):+.2f})"
        for g in top_genes[:12]
    ) or "N/A"
    pathway_summary = "; ".join(top_pathways[:6]) or "N/A"

    # Active view line
    tab_label = _TAB_INSTRUCTIONS.get(active_tab, ("", ""))[0]
    view_line = f"  Active view: {tab_label}\n" if tab_label else ""
    pca_line  = f"  PCA variance: {pca_variance}\n" if pca_variance and active_tab == "tab-pca" else ""
    counts_line = f"  Significant: {n_up} up, {n_down} down\n" if (n_up or n_down) else ""

    # Last 6 exchanges condensed
    thread_block = ""
    if thread:
        parts = []
        for msg in thread[-6:]:
            tag   = "Q" if msg.get("role") == "user" else "A"
            parts.append(f"[{tag}]: {str(msg.get('content',''))[:450]}")
        thread_block = "INVESTIGATION THREAD (recent):\n" + "\n".join(parts) + "\n\n"

    return (
        "You are the Senior Bioinformatician consultant in an ongoing investigation. "
        "Maintain investigative continuity and build on the established analysis.\n\n"
        "DATASET CONTEXT:\n"
        f"  Top genes: {gene_summary}\n"
        f"  Top pathways: {pathway_summary}\n"
        f"{counts_line}"
        f"{view_line}"
        f"{pca_line}"
        "\n"
        f"{thread_block}"
        f"FOLLOW-UP QUESTION: {question}\n\n"
        "RULE: Answer using specific data from this dataset — gene names, fold-changes, "
        "p-values, pathway names. Do NOT give generic textbook descriptions. "
        "If the question is about what a tab/graph 'shows', interpret the ACTUAL VALUES "
        "and patterns visible in this dataset's data."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — Google Gemini 2.5 Flash
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini(prompt: str, max_tokens: int = 2500) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=_SYSTEM_PROMPT,
    )
    resp = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": max_tokens, "temperature": 0.3},
    )
    return resp.text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — Groq Llama 3 70B
# ─────────────────────────────────────────────────────────────────────────────

def _call_groq(prompt: str, max_tokens: int = 2500) -> str:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    from groq import Groq
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — Rule-Based (always works)
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based_summary(context_data: dict) -> str:
    top_genes    = context_data.get("top_genes",    [])
    top_pathways = context_data.get("top_pathways", [])
    n_up         = context_data.get("n_up",  0)
    n_down       = context_data.get("n_down", 0)

    if not top_genes:
        return (
            "**Summary (Rule-Based):** No significant genes detected at current "
            "thresholds. Try relaxing the |Log2FC| or p-value cutoffs."
        )

    top       = top_genes[0]
    lfc_val   = float(top.get("log2FC", 0))
    direction = "up-regulated" if lfc_val > 0 else "down-regulated"
    up        = [g["symbol"] for g in top_genes if float(g.get("log2FC", 0)) > 0]
    down      = [g["symbol"] for g in top_genes if float(g.get("log2FC", 0)) < 0]

    lines = [
        f"**LAYER 1 — Statistical Topology (Rule-Based):** "
        f"The highest-ranked gene is **{top['symbol']}** "
        f"(Log2FC = {lfc_val:+.2f}, {direction})."
    ]
    if n_up or n_down:
        lines.append(f"Dataset contains {n_up} up-regulated and {n_down} down-regulated significant genes.")
    if up:
        lines.append(f"Top up-regulated genes include: {', '.join(up[:5])}.")
    if down:
        lines.append(f"Top down-regulated genes include: {', '.join(down[:5])}.")

    lines.append(
        "\n**LAYER 2 — Pathway Synergy (Rule-Based):** "
    )
    if top_pathways:
        extra = (f" and {len(top_pathways) - 1} additional pathway(s)"
                 if len(top_pathways) > 1 else "")
        lines.append(
            f"Statistical enrichment detected in **{top_pathways[0]}**{extra}. "
            "Configure an AI API key for activation z-score and synergy analysis."
        )
    else:
        lines.append("No enriched pathways detected at current thresholds.")

    lines.append(
        "\n**LAYER 3 — Cluster Diagnostics (Rule-Based):** "
        "Configure GEMINI_API_KEY or GROQ_API_KEY for outlier identification and "
        "cluster coherence analysis."
    )
    lines.append(
        "\n**LAYER 4 — Literature Synthesis (Rule-Based):** "
        "AI-powered literature synthesis unavailable offline. "
        "Add an API key in your .env file and click Refresh."
    )

    return " ".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public API — Deep Analysis
# ─────────────────────────────────────────────────────────────────────────────

def get_biological_story(context_data: dict, **kwargs) -> tuple[str, str]:
    """
    Waterfall: Gemini 2.5 Flash → Groq Llama 3 → Rule-Based.
    Uses the 4-layer Expert Investigator prompt.

    Returns (summary, powered_by).
    """
    _gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    _groq_key   = os.environ.get("GROQ_API_KEY",   "").strip()
    print(
        f"[AI Consultant] Keys — GEMINI: {'✅' if _gemini_key else '❌'}  "
        f"GROQ: {'✅' if _groq_key else '❌'}"
    )

    prompt = _build_deep_analysis_prompt(context_data)

    try:
        text = _call_gemini(prompt, max_tokens=2500)
        log.info("AI: Gemini 2.5 Flash succeeded")
        print("[AI Consultant] Gemini SUCCESS")
        return text, "Google · Gemini 2.5 Flash"
    except Exception as exc:
        log.warning(f"AI: Gemini failed — {exc}")
        print(f"[AI Consultant] Gemini FAILED — {type(exc).__name__}: {exc}")

    try:
        text = _call_groq(prompt, max_tokens=2500)
        log.info("AI: Groq (Llama 3 70B) succeeded")
        print("[AI Consultant] Groq SUCCESS")
        return text, "Groq · Llama 3 70B"
    except Exception as exc:
        log.warning(f"AI: Groq failed — {exc}")
        print(f"[AI Consultant] Groq FAILED — {type(exc).__name__}: {exc}")

    print("[AI Consultant] Falling back to Rule-Based")
    log.info("AI: using rule-based fallback")
    return _rule_based_summary(context_data), "Rule-Based · Offline"


def get_biological_story_cached(
    context_data: dict,
    lfc_thresh: float = 1.0,
    p_thresh: float = 0.05,
    pathway_db: str = "all",
    force_refresh: bool = False,
    **kwargs,
) -> tuple[str, str, bool]:
    """
    Cache-aware wrapper around get_biological_story.

    Returns (summary, powered_by, from_cache).
    """
    top_genes  = context_data.get("top_genes", [])
    active_tab = context_data.get("active_tab", "")
    key        = _make_cache_key(top_genes, lfc_thresh, p_thresh, pathway_db, active_tab)

    if force_refresh:
        invalidate_cache_key(key)
        log.info(f"AI cache: force-refresh — key {key[:12]}… evicted")
    else:
        cached = _cache_get(key)
        if cached is not None:
            log.info(f"AI cache: HIT — key {key[:12]}…")
            print(f"[AI Consultant] Cache HIT (key {key[:12]}…)")
            summary, powered_by = cached
            return summary, powered_by, True

    log.info(f"AI cache: MISS — calling live API (key {key[:12]}…)")
    print(f"[AI Consultant] Cache MISS — calling live API (key {key[:12]}…)")
    summary, powered_by = get_biological_story(context_data, **kwargs)
    _cache_set(key, summary, powered_by)
    return summary, powered_by, False


# ─────────────────────────────────────────────────────────────────────────────
# Public API — Interactive Consultation (no caching)
# ─────────────────────────────────────────────────────────────────────────────

def get_consultation_response(
    question: str,
    thread: list[dict],
    context_data: dict,
) -> tuple[str, str]:
    """
    Handle a follow-up question in the investigation thread.
    No caching — always a fresh call so the model sees the full thread.

    Parameters
    ----------
    question     : the user's follow-up question
    thread       : list of {role, content} dicts (prior exchanges)
    context_data : same dict accepted by get_biological_story

    Returns (response, powered_by).
    """
    if not question or not question.strip():
        return (
            "Please enter a question to continue the investigation.",
            "Rule-Based · Offline",
        )

    prompt = _build_chat_prompt(question.strip(), thread, context_data)

    # Tier 1 — Gemini
    try:
        text = _call_gemini(prompt, max_tokens=1500)
        log.info("Consultation: Gemini 2.5 Flash succeeded")
        return text, "Google · Gemini 2.5 Flash"
    except Exception as exc:
        log.warning(f"Consultation: Gemini failed — {exc}")

    # Tier 2 — Groq
    try:
        text = _call_groq(prompt, max_tokens=1500)
        log.info("Consultation: Groq (Llama 3 70B) succeeded")
        return text, "Groq · Llama 3 70B"
    except Exception as exc:
        log.warning(f"Consultation: Groq failed — {exc}")

    # Offline fallback
    return (
        f"No AI engine available. Your question was: *'{question}'*. "
        "Please configure GEMINI_API_KEY or GROQ_API_KEY in your .env file, "
        "then click Refresh.",
        "Rule-Based · Offline",
    )
