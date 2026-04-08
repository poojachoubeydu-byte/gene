"""
AI Summary Module — Apex Bioinformatics Platform v3.7 (Resilient AI Edition)
=============================================================================
Tiered waterfall: Groq (Llama 3 70B) → Gemini 1.5 Flash → Rule-Based.

get_biological_story(context_data) always returns a result — never an empty
box, even with no API keys or no internet connection.

API keys (set in HF Space → Settings → Repository secrets):
    GROQ_API_KEY   — https://console.groq.com        (free, no card)
    GEMINI_API_KEY — https://aistudio.google.com/app  (free, no card)
"""

import logging
import os

log = logging.getLogger(__name__)

# ── Shared system prompt: identical for both models so tone is consistent ─────
_SYSTEM_PROMPT = (
    "You are a Senior Bioinformatician with 20 years of experience in oncology "
    "and systems biology. You interpret RNA-seq differential expression results "
    "for a research audience. Be precise, use correct HGNC gene nomenclature, "
    "and ground every statement strictly in the data provided."
)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder — identical context_data format passed to every tier
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(context_data: dict) -> str:
    top_genes    = context_data.get("top_genes",    [])
    top_pathways = context_data.get("top_pathways", [])

    gene_block = "\n".join(
        f"  {g['symbol']}: Log2FC = {float(g['log2FC']):+.3f}"
        for g in top_genes[:10]
    ) or "  No significant genes."

    pathway_block = (
        "\n".join(f"  - {p}" for p in top_pathways[:5])
        or "  No significantly enriched pathways detected."
    )

    return (
        "Interpret these RNA-seq differential expression results:\n\n"
        f"Top Significant Genes (ranked by |LFC| × −log10p meta-score):\n"
        f"{gene_block}\n\n"
        f"Top Enriched Pathways:\n"
        f"{pathway_block}\n\n"
        "Write a concise 3–5 sentence biological interpretation. Cover:\n"
        "1. The most likely biological process or disease context.\n"
        "2. Whether the pattern indicates pathway activation or inhibition.\n"
        "3. Any clinically relevant genes or potential drug targets.\n\n"
        "Use precise scientific language. Stay grounded in the data."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — Groq (Llama 3 70B)
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
        max_tokens=400,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — Gemini 1.5 Flash
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=_SYSTEM_PROMPT,
    )
    resp = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": 400, "temperature": 0.3},
    )
    return resp.text.strip()


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

def get_biological_story(context_data: dict) -> tuple[str, str]:
    """
    Waterfall: Groq (Llama 3 70B) → Gemini 1.5 Flash → Rule-Based.

    Parameters
    ----------
    context_data : dict with keys:
        'top_genes'    — list of {'symbol': str, 'log2FC': float}
        'top_pathways' — list of pathway name strings

    Returns
    -------
    (summary, powered_by)
        summary    — markdown-formatted biological interpretation
        powered_by — short label for the status badge in the UI
    """
    prompt = _build_prompt(context_data)

    # Tier 1 — Groq
    try:
        text = _call_groq(prompt)
        log.info("AI summary: Groq (Llama 3 70B) succeeded")
        return text, "Groq · Llama 3 70B"
    except Exception as exc:
        log.warning(f"AI summary: Groq failed — {exc}")

    # Tier 2 — Gemini
    try:
        text = _call_gemini(prompt)
        log.info("AI summary: Gemini 1.5 Flash succeeded")
        return text, "Gemini · 1.5 Flash"
    except Exception as exc:
        log.warning(f"AI summary: Gemini failed — {exc}")

    # Tier 3 — Rule-Based (guaranteed)
    log.info("AI summary: using rule-based fallback")
    return _rule_based_summary(context_data), "Rule-Based · Offline"
