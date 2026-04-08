"""
AI Summary Module — Apex Bioinformatics Platform
=================================================
Calls the Groq free-tier API (Llama 3 8B) to generate a plain-English
biological interpretation of the current differential expression results.

Requires:  GROQ_API_KEY environment variable (set in HF Space secrets).
Get a free key at https://console.groq.com — no credit card required.

If the key is absent or the call fails the function returns a helpful
message rather than raising, so the rest of the app is unaffected.
"""

import logging
import os

import requests

log = logging.getLogger(__name__)

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_MODEL    = "llama3-8b-8192"


def get_ai_interpretation(top_genes: list[dict], top_pathways: list[str]) -> str:
    """
    Generate a biological summary via Groq (free tier).

    Parameters
    ----------
    top_genes : list of dicts with keys 'symbol' and 'log2FC'
                (top 10 genes ranked by meta-score)
    top_pathways : list of pathway name strings (top 5 enriched pathways)

    Returns
    -------
    str — markdown-formatted interpretation, or an informative fallback message.
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return (
            "**API key not configured.**\n\n"
            "To enable AI-powered summaries:\n"
            "1. Get a **free** Groq API key at [console.groq.com](https://console.groq.com) "
            "(no credit card required).\n"
            "2. In your Hugging Face Space, go to **Settings → Repository secrets** "
            "and add `GROQ_API_KEY` with your key.\n\n"
            "_The rest of the app works fully without this key._"
        )

    gene_block = "\n".join(
        f"  {g['symbol']}: Log2FC = {float(g['log2FC']):+.3f}"
        for g in top_genes[:10]
    )
    pathway_block = (
        "\n".join(f"  - {p}" for p in top_pathways[:5])
        or "  No significantly enriched pathways detected at current thresholds."
    )

    prompt = (
        "You are an expert bioinformatician interpreting RNA-seq differential "
        "expression results for a research audience.\n\n"
        f"Top 10 Significant Genes (ranked by statistical + effect-size score):\n"
        f"{gene_block}\n\n"
        f"Top Enriched Pathways:\n"
        f"{pathway_block}\n\n"
        "Write a concise 3–5 sentence biological interpretation. Address:\n"
        "1. The most likely biological process or disease context these genes suggest.\n"
        "2. Whether the overall expression pattern indicates pathway activation or inhibition.\n"
        "3. Any clinically relevant genes or potential drug targets worth highlighting.\n\n"
        "Use precise scientific language. Stay strictly grounded in the data above."
    )

    try:
        resp = requests.post(
            _GROQ_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       _MODEL,
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  400,
                "temperature": 0.3,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    except requests.exceptions.Timeout:
        log.warning("AI summary: Groq API timed out after 30 s")
        return (
            "⏱️ **AI summary timed out** (>30 s). "
            "Groq is usually sub-second — this may be a temporary outage. "
            "Try refreshing the tab."
        )
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        log.warning(f"AI summary: HTTP {status} from Groq")
        return (
            f"⚠️ **Groq API returned HTTP {status}.** "
            "Check that your `GROQ_API_KEY` is valid and has not been rate-limited."
        )
    except Exception as exc:
        log.warning(f"AI summary: unexpected error — {exc}")
        return f"⚠️ **AI summary unavailable:** {str(exc)[:200]}"
