"""
AI Summary Module — Apex Bioinformatics Platform v4.0
======================================================
Tiered waterfall: Gemini 2.5 Flash (primary) → Groq Llama 3 (secondary) → Rule-Based.

get_biological_story(context_data) always returns a result — never an empty
box, even with no API keys or no internet connection.

API keys (set in .env file):
    GEMINI_API_KEY     — https://aistudio.google.com/app       (free, no card)
    GROQ_API_KEY       — https://console.groq.com              (free, no card)
"""

import hashlib
import logging
import os
import time

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Server-side AI response cache — keyed by (genes, lfc, pval, db)
# TTL = 24 hours.  Thread-safe enough for single-process Dash / Gunicorn
# because Python's GIL protects dict reads/writes.
# ─────────────────────────────────────────────────────────────────────────────

_CACHE_TTL: int = 24 * 3600                          # seconds
_AI_CACHE:  dict[str, tuple[float, str, str]] = {}   # key → (ts, summary, powered_by)


def _make_cache_key(
    top_genes: list[dict],
    lfc_thresh: float,
    p_thresh: float,
    pathway_db: str,
) -> str:
    """Return a stable SHA-256 hex key for the given query parameters."""
    gene_part  = ",".join(sorted(g["symbol"].upper() for g in top_genes))
    raw        = f"{gene_part}|{round(lfc_thresh, 3)}|{round(p_thresh, 5)}|{pathway_db or 'all'}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(key: str) -> tuple[str, str] | None:
    """Return (summary, powered_by) if cached and not expired, else None."""
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
    """Diagnostic helper — returns count and oldest/newest timestamps."""
    now = time.time()
    live = {k: v for k, v in _AI_CACHE.items() if now - v[0] <= _CACHE_TTL}
    return {
        "live_entries": len(live),
        "total_entries": len(_AI_CACHE),
        "ttl_hours": _CACHE_TTL / 3600,
    }

# ── Shared system prompt ──────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are a Senior Bioinformatician with 20 years of experience in oncology "
    "and systems biology. You interpret RNA-seq differential expression results "
    "for a research audience. Be precise, use correct HGNC gene nomenclature, "
    "and ground every statement strictly in the data provided."
)


# ─────────────────────────────────────────────────────────────────────────────
# Public status helper — called on page load for the startup badge
# ─────────────────────────────────────────────────────────────────────────────

def check_available_model() -> tuple[str, str]:
    """
    Returns (label, badge_color) for the first detected API key.
    Does NOT make any network calls — purely an env-var check.

    Priority: Gemini → Groq → Offline (matches get_biological_story waterfall).

    Returns
    -------
    (label, badge_color)
        label       — short model name for the badge
        badge_color — Bootstrap color ('success', 'info', 'secondary')
    """
    if os.environ.get("GEMINI_API_KEY", "").strip():
        return "Online · Gemini 2.5 Flash", "success"
    if os.environ.get("GROQ_API_KEY", "").strip():
        return "Online · Groq Llama 3", "info"
    return "Offline · Rule-Based", "secondary"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder — accepts optional jaccard_pairs for Deep Scanning
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(context_data: dict) -> str:
    top_genes     = context_data.get("top_genes",     [])
    top_pathways  = context_data.get("top_pathways",  [])
    extra         = context_data.get("extra",         "")
    jaccard_pairs = context_data.get("jaccard_pairs", [])

    gene_block = "\n".join(
        f"  {g['symbol']}: Log2FC = {float(g['log2FC']):+.3f}"
        for g in top_genes[:10]
    ) or "  No significant genes."

    pathway_block = (
        "\n".join(f"  - {p}" for p in top_pathways[:5])
        or "  No significantly enriched pathways detected."
    )

    # ── Jaccard overlap block (Deep Scanning) ─────────────────────────────────
    jaccard_block = ""
    if jaccard_pairs:
        lines = ["Jaccard Gene Overlap Between Top Pathways (shared gene burden):"]
        for pair in jaccard_pairs[:5]:
            lines.append(
                f"  {pair['a']} ↔ {pair['b']}: "
                f"J={pair['j']:.2f} ({pair['shared']} shared genes)"
            )
        jaccard_block = "\n" + "\n".join(lines) + "\n"

    extra_block = f"\nAdditional context: {extra}\n" if extra else ""

    return (
        "Interpret these RNA-seq differential expression results:\n\n"
        f"Top Significant Genes (ranked by |LFC| × −log10p meta-score):\n"
        f"{gene_block}\n\n"
        f"Top Enriched Pathways:\n"
        f"{pathway_block}\n"
        f"{jaccard_block}"
        f"{extra_block}\n"
        "Write a concise 3–5 sentence biological interpretation. Cover:\n"
        "1. The most likely biological process or disease context.\n"
        "2. Whether the pattern indicates pathway activation or inhibition.\n"
        "3. Any clinically relevant genes or potential drug targets.\n"
        + ("4. What the pathway gene overlap pattern implies about crosstalk "
           "or co-regulation.\n" if jaccard_pairs else "")
        + "\nUse precise scientific language. Stay grounded in the data."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — Google Gemini 2.5 Flash (primary)
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
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
        generation_config={"max_output_tokens": 600, "temperature": 0.3},
    )
    return resp.text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — Groq Llama 3 70B (secondary fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _call_groq(prompt: str) -> str:
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
        max_tokens=600,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — Rule-Based (always works, no internet required)
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based_summary(context_data: dict) -> str:
    top_genes    = context_data.get("top_genes",    [])
    top_pathways = context_data.get("top_pathways", [])

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
        f"**Summary (Rule-Based):** The highest-ranked gene is **{top['symbol']}** "
        f"(Log2FC = {lfc_val:+.2f}, {direction})."
    ]
    if up:
        lines.append(f"Up-regulated genes include: {', '.join(up[:5])}.")
    if down:
        lines.append(f"Down-regulated genes include: {', '.join(down[:5])}.")
    if top_pathways:
        extra = (f" and {len(top_pathways) - 1} other pathway(s)"
                 if len(top_pathways) > 1 else "")
        lines.append(
            f"Statistical enrichment detected in **{top_pathways[0]}**{extra}."
        )

    return " ".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_biological_story(context_data: dict, **kwargs) -> tuple[str, str]:
    """
    Waterfall: Gemini 2.5 Flash → Groq Llama 3 → Rule-Based.

    Parameters
    ----------
    context_data : dict with keys:
        'top_genes'     — list of {'symbol': str, 'log2FC': float}
        'top_pathways'  — list of pathway name strings
        'jaccard_pairs' — list of {'a', 'b', 'j', 'shared'} dicts (optional)
        'extra'         — additional context string (optional)

    Returns
    -------
    (summary, powered_by)
        summary    — markdown-formatted biological interpretation
        powered_by — short label for the status badge in the UI
    """
    _gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    _groq_key   = os.environ.get("GROQ_API_KEY",   "").strip()
    print(
        f"[AI Copilot] Keys — GEMINI: {'✅' if _gemini_key else '❌'}  "
        f"GROQ: {'✅' if _groq_key else '❌'}"
    )

    prompt = _build_prompt(context_data)

    # Tier 1 — Gemini 2.5 Flash
    try:
        text = _call_gemini(prompt)
        log.info("AI summary: Gemini 2.5 Flash succeeded")
        print("[AI Copilot] Gemini SUCCESS")
        return text, "Google · Gemini 2.5 Flash"
    except Exception as exc:
        log.warning(f"AI summary: Gemini failed — {exc}")
        print(f"[AI Copilot] Gemini FAILED — {type(exc).__name__}: {exc}")

    # Tier 2 — Groq Llama 3
    try:
        text = _call_groq(prompt)
        log.info("AI summary: Groq (Llama 3 70B) succeeded")
        print("[AI Copilot] Groq SUCCESS")
        return text, "Groq · Llama 3 70B"
    except Exception as exc:
        log.warning(f"AI summary: Groq failed — {exc}")
        print(f"[AI Copilot] Groq FAILED — {type(exc).__name__}: {exc}")

    # Tier 3 — Rule-Based (guaranteed)
    print("[AI Copilot] Falling back to Rule-Based")
    log.info("AI summary: using rule-based fallback")
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

    Parameters
    ----------
    context_data  : same dict accepted by get_biological_story
    lfc_thresh    : |Log2FC| cutoff used to build sig-store
    p_thresh      : adjusted p-value cutoff used to build sig-store
    pathway_db    : pathway database selection ('all', 'kegg', 'go', …)
    force_refresh : when True the cached entry is discarded and a fresh
                    API call is made regardless of TTL

    Returns
    -------
    (summary, powered_by, from_cache)
        from_cache — True when the result was served from the local cache
    """
    top_genes = context_data.get("top_genes", [])
    key       = _make_cache_key(top_genes, lfc_thresh, p_thresh, pathway_db)

    if force_refresh:
        invalidate_cache_key(key)
        log.info(f"AI cache: force-refresh — key {key[:12]}… evicted")
    else:
        cached = _cache_get(key)
        if cached is not None:
            log.info(f"AI cache: HIT — key {key[:12]}…")
            print(f"[AI Copilot] Cache HIT — serving from local cache (key {key[:12]}…)")
            summary, powered_by = cached
            return summary, powered_by, True

    log.info(f"AI cache: MISS — calling live API (key {key[:12]}…)")
    print(f"[AI Copilot] Cache MISS — calling live API (key {key[:12]}…)")
    summary, powered_by = get_biological_story(context_data, **kwargs)
    _cache_set(key, summary, powered_by)
    return summary, powered_by, False
