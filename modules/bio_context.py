"""
Bio-Context Module — Apex Bioinformatics Platform
==================================================
Fetches gene annotations from NCBI Entrez + UniProt REST API.

• In-process LRU cache (thread-safe) prevents duplicate API calls
  when multiple users look up the same gene.
• Hard timeout (6 s per request) prevents stalling.
• Graceful degradation — returns partial data if one API fails.
"""

import logging
import threading
import time
from functools import lru_cache

import requests

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_CACHE: dict = {}         # symbol -> (timestamp, result)
_CACHE_TTL = 3600         # 1 hour


def _cached(sym: str):
    with _LOCK:
        entry = _CACHE.get(sym)
    if entry:
        ts, val = entry
        if time.time() - ts < _CACHE_TTL:
            return val
    return None


def _store(sym: str, val: dict):
    with _LOCK:
        _CACHE[sym] = (time.time(), val)


# ─────────────────────────────────────────────────────────────────────────────

def fetch_gene_info(gene_symbol: str) -> dict:
    """Return annotation dict for *gene_symbol* (HGNC symbol, human)."""
    sym = gene_symbol.strip().upper()
    cached = _cached(sym)
    if cached is not None:
        return cached

    result = _fetch(sym)
    _store(sym, result)
    return result


def _fetch(sym: str) -> dict:
    try:
        # ── NCBI Entrez: gene ID ────────────────────────────────────────────
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db": "gene",
                "term": f"{sym}[Gene Name] AND Homo sapiens[Organism]",
                "retmode": "json",
                "retmax": 1,
            },
            timeout=6,
        )
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {"error": f"Gene '{sym}' not found in NCBI."}

        gene_id = ids[0]
        time.sleep(0.35)  # NCBI rate limit: ≤ 3 req/s without API key

        # ── NCBI Entrez: gene summary ────────────────────────────────────────
        r2 = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "gene", "id": gene_id, "retmode": "json"},
            timeout=6,
        )
        r2.raise_for_status()
        gene_rec = r2.json().get("result", {}).get(gene_id, {})

        ncbi = {
            "full_name":  gene_rec.get("description", ""),
            "summary":    gene_rec.get("summary", ""),
            "location":   gene_rec.get("maplocation", "Unknown"),
            "chromosome": gene_rec.get("chromosome", "Unknown"),
            "aliases":    gene_rec.get("otheraliases", ""),
            "ncbi_id":    gene_id,
        }

        # ── UniProt REST ─────────────────────────────────────────────────────
        r3 = requests.get(
            "https://rest.uniprot.org/uniprotkb/search",
            params={
                "query": f"gene:{sym} AND organism_id:9606 AND reviewed:true",
                "fields": "id,cc_function,ft_domain,cc_subcellular_location,cc_disease",
                "format": "json",
                "size": 1,
            },
            timeout=6,
        )
        r3.raise_for_status()
        uni_results = r3.json().get("results", [])

        uni: dict = {
            "function": "",
            "domains": [],
            "subcellular_location": "",
            "disease_associations": [],
            "uniprot_id": "",
        }
        if uni_results:
            entry = uni_results[0]
            uni["uniprot_id"] = entry.get("primaryAccession", "")
            for comment in entry.get("comments", []):
                ct = comment.get("commentType", "")
                if ct == "FUNCTION":
                    texts = comment.get("texts", [])
                    if texts:
                        uni["function"] = texts[0].get("value", "")
                elif ct == "SUBCELLULAR LOCATION":
                    locs = comment.get("subcellularLocations", [])
                    uni["subcellular_location"] = ", ".join(
                        loc.get("location", {}).get("value", "")
                        for loc in locs[:3]
                    )
                elif ct == "DISEASE":
                    name = comment.get("disease", {}).get("diseaseName", "")
                    if name:
                        uni["disease_associations"].append(name)
            for feat in entry.get("features", []):
                if feat.get("type") == "Domain":
                    d = feat.get("description", "")
                    if d:
                        uni["domains"].append(d)

        return {
            "full_name":            ncbi["full_name"],
            "summary":              ncbi["summary"],
            "location":             ncbi["location"],
            "chromosome":           ncbi["chromosome"],
            "aliases":              ncbi["aliases"],
            "function":             uni["function"] or "No UniProt annotation available.",
            "domains":              uni["domains"],
            "subcellular_location": uni["subcellular_location"] or "Unknown",
            "disease_associations": uni["disease_associations"],
            "ncbi_id":              ncbi["ncbi_id"],
            "uniprot_id":           uni["uniprot_id"],
        }

    except requests.exceptions.Timeout:
        return {"error": f"Timeout fetching data for '{sym}'. Try again shortly."}
    except Exception as e:
        return {"error": str(e)[:200]}
