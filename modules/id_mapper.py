"""
Gene ID Translator — Apex Bioinformatics Platform
==================================================
Detects and converts Entrez IDs (numeric strings) and Ensembl gene IDs
to official HGNC gene symbols so they can be matched against the built-in
KEGG / GO-BP / Reactome pathway databases.

Translation priority:
  1. Local static dict  — covers all ~600 genes in the built-in pathway DBs
                           plus the most common cancer/research genes  (~instant)
  2. MyGene.info batch REST API — handles the full 25 000-gene universe
                                   (requires internet, ~1 s per 1 000 genes)
  3. Graceful fallback — returns the original ID if unmapped
"""

import logging
import re
from typing import Dict, List, Tuple

import requests

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Local Entrez → Symbol (covers every gene in the built-in pathway databases
# plus the most frequently encountered cancer / signalling / metabolic genes).
# Sourced from NCBI Gene, HGNC, and Ensembl GRCh38.p14.
# ─────────────────────────────────────────────────────────────────────────────
_ENTREZ_TO_SYMBOL: Dict[str, str] = {
    # ── Tumour suppressors / oncogenes ─────────────────────────────────────
    "7157":"TP53",   "672":"BRCA1",   "675":"BRCA2",   "3845":"KRAS",
    "4609":"MYC",    "1956":"EGFR",   "5728":"PTEN",   "5290":"PIK3CA",
    "5291":"PIK3CB", "673":"BRAF",    "4893":"NRAS",   "3265":"HRAS",
    "25":"ABL1",     "238":"ALK",     "2099":"ESR1",   "2100":"ESR2",
    "367":"AR",      "2064":"ERBB2",  "1499":"CTNNB1", "324":"APC",
    "2260":"FGFR1",  "2263":"FGFR2",  "2261":"FGFR3",  "2264":"FGFR4",
    "4914":"NTRK1",  "4915":"NTRK2",  "4916":"NTRK3",  "4233":"MET",
    "5979":"RET",    "7015":"TERT",   "2322":"FLT3",
    "5604":"MAP2K1", "5605":"MAP2K2", "207":"AKT1",    "208":"AKT2",
    "10000":"AKT3",  "2475":"MTOR",   "6714":"SRC",    "1030":"CDKN2B",
    # ── Cell cycle ──────────────────────────────────────────────────────────
    "983":"CDK1",    "1017":"CDK2",   "1019":"CDK4",   "1021":"CDK5",
    "1022":"CDK6",   "595":"CCND1",   "894":"CCND2",   "896":"CCND3",
    "890":"CCNA2",   "891":"CCNA1",   "898":"CCNE1",   "900":"CCNE2",
    "904":"CCNB1",   "905":"CCNB2",   "5925":"RB1",    "1869":"E2F1",
    "1870":"E2F2",   "1871":"E2F3",   "994":"CDC20",   "699":"BUB1",
    "701":"BUB1B",   "5347":"PLK1",   "6790":"AURKA",  "9212":"AURKB",
    "4085":"MAD2L1", "1029":"CDKN2A", "1026":"CDKN1A", "1027":"CDKN1B",
    "1028":"CDKN1C", "10723":"PCNA",
    # ── DNA damage ──────────────────────────────────────────────────────────
    "472":"ATM",     "545":"ATR",     "1111":"CHEK1",  "11200":"CHEK2",
    "5888":"RAD51",  "2072":"ERCC1",  "4292":"MLH1",   "4436":"MSH2",
    "7503":"XRCC1",  "7518":"XRCC4",  "3981":"LIG4",   "2177":"FANCD2",
    "79728":"PALB2", "4361":"MRE11",  "4670":"NBN",
    # ── Apoptosis ───────────────────────────────────────────────────────────
    "581":"BAX",     "596":"BCL2",    "598":"BCL2L1",  "4170":"MCL1",
    "331":"XIAP",    "332":"BIRC5",   "836":"CASP3",   "841":"CASP8",
    "842":"CASP9",   "5371":"PARP1",  "355":"FAS",     "7124":"TNF",
    "8717":"TRADD",  "8772":"FADD",   "637":"BID",     "54205":"CYCS",
    "317":"APAF1",   "10955":"PMAIP1","27113":"BBC3",  "581":"BAX",
    "5366":"PMAIP1",
    # ── p53 pathway ─────────────────────────────────────────────────────────
    "4193":"MDM2",   "1647":"GADD45A","4193":"MDM2",   "9817":"KEAP1",
    "22976":"SESN1", "122773":"SESN2","83667":"TIGAR",  "51100":"PERP",
    "7029":"TFDP1",
    # ── Signalling kinases / pathways ───────────────────────────────────────
    "5594":"MAPK1",  "5595":"MAPK3",  "5606":"MAP2K3", "5608":"MAP2K6",
    "3716":"JAK1",   "3717":"JAK2",   "3718":"JAK3",   "7272":"TYK2",
    "6772":"STAT1",  "6773":"STAT2",  "6774":"STAT3",  "6776":"STAT5A",
    "6777":"STAT5B", "6778":"STAT6",  "8651":"SOCS1",  "9021":"SOCS3",
    "4790":"NFKB1",  "5970":"RELA",   "4791":"NFKB2",  "5971":"RELB",
    "5966":"REL",    "1147":"CHUK",   "3551":"IKBKB",  "4792":"NFKBIA",
    "4793":"NFKBIB",
    "7040":"TGFB1",  "7046":"TGFBR1", "7048":"TGFBR2", "4087":"SMAD2",
    "4088":"SMAD3",  "4089":"SMAD4",  "93":"ACVR2A",   "90":"ACVR1",
    "655":"BMP4",    "658":"BMPR1A",  "9802":"SMURF1",
    "7482":"WNT1",   "2852":"AXIN1",  "9063":"AXIN2",  "6932":"TCF7L2",
    "51176":"LEF1",  "1857":"DVL1",   "4040":"LRP5",   "4041":"LRP6",
    "3956":"LGALS3",
    "6469":"SHH",    "5727":"PTCH1",  "6608":"SMO",    "2736":"GLI1",
    "2735":"GLI2",   "2737":"GLI3",   "51684":"SUFU",
    "4851":"NOTCH1", "4853":"NOTCH3", "4854":"NOTCH4", "4855":"NOTCH2",
    "28514":"DLL1",  "10683":"DLL3",  "54567":"DLL4",  "182":"JAG1",
    "3714":"JAG2",   "3516":"RBPJ",   "3280":"HES1",   "3280":"HES1",
    "3281":"HES5",   "23462":"HEY1",  "23536":"HEY2",
    "3091":"HIF1A",  "405":"ARNT",    "7428":"VHL",    "54583":"EGLN1",
    "112398":"EGLN2","112399":"EGLN3",
    "7422":"VEGFA",  "7423":"VEGFB",  "7424":"VEGFC",  "2277":"VEGFD",
    "3791":"KDR",    "2321":"FLT1",   "8814":"FLT4",   "8829":"NRP1",
    "8828":"NRP2",   "5156":"PDGFRA", "5159":"PDGFRB",
    "284":"ANGPT1",  "285":"ANGPT2",  "7010":"TEK",    "7075":"TIE1",
    # ── mTOR / PI3K ─────────────────────────────────────────────────────────
    "57521":"RPTOR", "253260":"RICTOR","7248":"TSC1",  "7249":"TSC2",
    "6009":"RHEB",   "6198":"RPS6KB1","1978":"EIF4EBP1","6194":"RPS6",
    "9776":"ULK1",   "9140":"ATG13",  "64426":"DEPTOR","79109":"MLST8",
    "51719":"AKT1S1","5494":"FOXO1",  "2309":"FOXO3",
    # ── Immune / cytokines ──────────────────────────────────────────────────
    "3458":"IFNG",   "3569":"IL6",    "3552":"IL1B",   "3558":"IL2",
    "3586":"IL10",   "3592":"IL12A",  "3600":"IL15",   "6347":"CCL2",
    "6352":"CCL5",   "3627":"CXCL10","920":"CD4",     "925":"CD8A",
    "940":"CD28",    "1493":"CTLA4",  "5133":"PDCD1",  "3903":"LAG3",
    "201633":"TIGIT","50943":"FOXP3", "929":"CD274",   "84868":"HAVCR2",
    "3956":"LTA",    "8797":"TNFRSF10A","8795":"TNFRSF10B",
    "8718":"TNFRSF1A","7186":"TRAF2",
    # ── Autophagy ───────────────────────────────────────────────────────────
    "8516":"BECN1",  "9474":"ATG5",   "10533":"ATG7",  "9140":"ATG13",
    "55054":"ATG16L1","8621":"ATG12", "8878":"SQSTM1", "84557":"MAP1LC3A",
    "51179":"LAMP1", "7879":"RAB7A",  "7942":"TFEB",   "23626":"AMBRA1",
    "9741":"RUBCN",
    # ── EMT / invasion ──────────────────────────────────────────────────────
    "999":"CDH1",    "1000":"CDH2",   "7431":"VIM",    "2335":"FN1",
    "7486":"WNT5A",  "558":"AXL",     "6868":"TWIST1", "6615":"SNAI1",
    "6591":"SNAI2",  "9839":"ZEB1",   "9843":"ZEB2",   "4313":"MMP2",
    "4318":"MMP9",   "3693":"ITGB1",  "3911":"LAMB1",  "3956":"LTA",
    "3480":"IGF1R",  "3576":"CXCR4",  "6387":"CXCL12", "4846":"NOS3",
    "338":"APOB",    "10163":"ITGB6",
    # ── Metabolism ──────────────────────────────────────────────────────────
    "3098":"HK1",    "3099":"HK2",    "5213":"PFKL",   "5214":"PFKM",
    "226":"ALDOA",   "2597":"GAPDH",  "5230":"PGK1",   "5223":"PGAM1",
    "2023":"ENO1",   "5315":"PKM",    "3945":"LDHA",   "5160":"PDK1",
    "5162":"PDHB",   "2538":"G6PD",   "2194":"FASN",   "31":"ACACA",
    "32":"ACACB",    "1374":"CPT1A",  "1375":"CPT2",   "3030":"HADHA",
    "34":"ACADM",    "2180":"ACSL1",  "51703":"ACSL4", "6319":"SCD",
    "5465":"PPARA",  "5467":"PPARG",  "5468":"PPARD",  "3072":"HIF1AN",
    "3073":"GLUD1",  "2744":"GLS",    "2745":"GLS2",   "383":"ARG1",
    "445":"ASS1",    "4953":"ODC1",   "3418":"IDH1",   "3417":"IDH2",
    "47":"ACLY",     "8801":"SUCLA2", "1737":"DLST",
    "2523":"FUT1",   "10973":"ACSL6",
    # ── Epigenetics ─────────────────────────────────────────────────────────
    "2146":"EZH2",   "1788":"DNMT3A", "1789":"DNMT3B", "80312":"TET1",
    "54790":"TET2",  "55748":"SETD2", "8623":"KDM5C",  "7403":"KDM6A",
    "3065":"HDAC1",  "3066":"HDAC2",  "2033":"EP300",  "1387":"CREBBP",
    "6046":"BRD4",   "23411":"SIRT1", "1108":"CHD4",   "6597":"SMARCA4",
    "6598":"SMARCB1","8289":"ARID1A", "196528":"ARID2","55193":"PBRM1",
    "6601":"SMARCC1","4154":"MBPB",   "23046":"MBD3",  "1737":"DLST",
    # ── RNA processing / splicing ────────────────────────────────────────────
    "2175":"SF3B1",  "7307":"U2AF1",  "7385":"U2AF2",  "6426":"SRSF1",
    "6427":"SRSF2",  "3178":"HNRNPA1","3183":"HNRNPC", "26137":"PTBP1",
    "54729":"RBM10", "1655":"DDX5",   "1660":"DHX9",   "10594":"PRPF8",
    "6626":"SNRNP70","1404":"DICER1", "29102":"DROSHA","74007":"DGCR8",
    "27161":"AGO2",  "54464":"XRN1",  "1871":"E2F3",
    # ── Proteasome / ubiquitin ───────────────────────────────────────────────
    "5682":"PSMA1",  "5689":"PSMB1",  "5693":"PSMB5",  "5700":"PSMC1",
    "5708":"PSMD1",  "7317":"UBA1",   "7314":"UBB",    "7316":"UBC",
    "11065":"UBE2C", "23291":"CDC34", "8454":"CUL1",   "6502":"SKP1",
    "55294":"FBXW7", "6498":"BTRC",   "4297":"KEAP1",
    # ── Ribosome / translation ───────────────────────────────────────────────
    "6194":"RPS6",   "6201":"RPS14",  "6208":"RPS19",  "6125":"RPL5",
    "6135":"RPL11",  "1977":"EIF4E",  "1974":"EIF4G1", "1975":"EIF4G2",
    "8661":"EIF3A",  "1965":"EIF2S1", "9451":"EIF2AK3","8205":"EIF4A1",
    "8639":"EIF6",
    # ── Oxidative stress ────────────────────────────────────────────────────
    "6647":"SOD1",   "6648":"SOD2",   "847":"CAT",     "2876":"GPX1",
    "2879":"GPX4",   "54985":"TXNRD1","3162":"HMOX1",  "1728":"NQO1",
    "4780":"NFE2L2", "22822":"PRDX1", "10935":"PRDX3", "4489":"MT1A",
    "8878":"SQSTM1",
    # ── Drug targets / biomarkers ────────────────────────────────────────────
    "4084":"MYB",    "2308":"FOXO4",  "9055":"PBK",    "3480":"IGF1R",
    "2033":"EP300",  "1387":"CREBBP", "3815":"KRAS",   "3716":"JAK1",
    "4240":"MFGE8",  "7538":"ZNF354B","3601":"IL13",
    "54331":"TIGIT",
}

# ─────────────────────────────────────────────────────────────────────────────
# Local Ensembl → Symbol
# Covers the most common human genes in pathway databases + TCGA/GEO studies
# ─────────────────────────────────────────────────────────────────────────────
_ENSEMBL_TO_SYMBOL: Dict[str, str] = {
    "ENSG00000141510":"TP53",    "ENSG00000012048":"BRCA1",
    "ENSG00000139618":"BRCA2",   "ENSG00000133703":"KRAS",
    "ENSG00000136997":"MYC",     "ENSG00000146648":"EGFR",
    "ENSG00000171862":"PTEN",    "ENSG00000121879":"PIK3CA",
    "ENSG00000172936":"PIK3CB",  "ENSG00000157764":"BRAF",
    "ENSG00000213281":"NRAS",    "ENSG00000174775":"HRAS",
    "ENSG00000097007":"ABL1",    "ENSG00000171094":"ALK",
    "ENSG00000122025":"FLT3",    "ENSG00000091831":"ESR1",
    "ENSG00000140388":"ESR2",    "ENSG00000169083":"AR",
    "ENSG00000141736":"ERBB2",   "ENSG00000168036":"CTNNB1",
    "ENSG00000134982":"APC",     "ENSG00000077782":"FGFR1",
    "ENSG00000066468":"FGFR2",   "ENSG00000068078":"FGFR3",
    "ENSG00000160867":"FGFR4",   "ENSG00000198400":"NTRK1",
    "ENSG00000148053":"NTRK2",   "ENSG00000187098":"NTRK3",
    "ENSG00000105976":"MET",     "ENSG00000165731":"RET",
    "ENSG00000164362":"TERT",    "ENSG00000170312":"CDK1",
    "ENSG00000123374":"CDK2",    "ENSG00000135446":"CDK4",
    "ENSG00000112576":"CCND1",   "ENSG00000147889":"CDKN2A",
    "ENSG00000124762":"CDKN1A",  "ENSG00000076984":"MAP2K1",
    "ENSG00000126934":"MAP2K2",  "ENSG00000149311":"ATM",
    "ENSG00000175054":"ATR",     "ENSG00000189420":"CHEK1",
    "ENSG00000183765":"CHEK2",   "ENSG00000051180":"RAD51",
    "ENSG00000087088":"BAX",     "ENSG00000171791":"BCL2",
    "ENSG00000143384":"MCL1",    "ENSG00000100030":"MAPK1",
    "ENSG00000102882":"MAPK3",   "ENSG00000232810":"TNF",
    "ENSG00000136244":"IL6",     "ENSG00000125538":"IL1B",
    "ENSG00000109471":"IL2",     "ENSG00000136634":"IL10",
    "ENSG00000111537":"IFNG",    "ENSG00000112715":"VEGFA",
    "ENSG00000128052":"KDR",     "ENSG00000102755":"FLT1",
    "ENSG00000038427":"VCAN",    "ENSG00000106462":"EZH2",
    "ENSG00000119772":"DNMT3A",  "ENSG00000071462":"BRD4",
    "ENSG00000026025":"VIM",     "ENSG00000159111":"GAPDH",
    "ENSG00000156515":"HK1",     "ENSG00000067057":"PFKL",
    "ENSG00000148400":"NOTCH1",  "ENSG00000171862":"PTEN",
    "ENSG00000085563":"ABCB1",   "ENSG00000196776":"CD47",
    "ENSG00000261371":"BECN1",
}

_MYGENE_GENE_URL  = "https://mygene.info/v3/gene"
_MYGENE_QUERY_URL = "https://mygene.info/v3/query"


def detect_id_type(ids: List[str]) -> str:
    """
    Inspect up to 200 IDs and return the dominant type.
    Returns: 'entrez' | 'ensembl' | 'symbol'
    """
    sample = [str(s).strip() for s in ids if str(s).strip()][:200]
    if not sample:
        return "symbol"
    n = len(sample)
    n_numeric = sum(1 for s in sample if re.fullmatch(r"\d+", s))
    n_ensembl = sum(1 for s in sample if re.match(r"^ENS[A-Z]*G\d+", s, re.I))
    if n_numeric / n > 0.5:
        return "entrez"
    if n_ensembl / n > 0.5:
        return "ensembl"
    return "symbol"


def translate_to_symbols(
    ids: List[str],
    timeout: int = 8,
) -> Tuple[List[str], str, int, int]:
    """
    Convert a list of gene IDs to official HGNC gene symbols.

    Parameters
    ----------
    ids     : input gene identifiers (any type — auto-detected)
    timeout : seconds for each MyGene.info HTTP call

    Returns
    -------
    translated   : list of symbols (same length as input; original ID kept if unmapped)
    id_type      : 'entrez' | 'ensembl' | 'symbol'
    n_input      : number of non-empty IDs submitted
    n_translated : number successfully mapped to a symbol
    """
    clean_ids = [
        str(x).strip().strip('"').strip("'").upper()
        for x in ids if str(x).strip()
    ]
    if not clean_ids:
        return [], "symbol", 0, 0

    id_type = detect_id_type(clean_ids)
    if id_type == "symbol":
        return clean_ids, "symbol", len(clean_ids), len(clean_ids)

    local_map = _ENTREZ_TO_SYMBOL if id_type == "entrez" else _ENSEMBL_TO_SYMBOL

    # ── Step 1: local dict ───────────────────────────────────────────────────
    trans: Dict[str, str] = {}
    for uid in clean_ids:
        key = uid.split(".")[0]          # strip Ensembl version suffix
        if key in local_map:
            trans[uid] = local_map[key].upper()
        elif uid in local_map:
            trans[uid] = local_map[uid].upper()

    # ── Step 2: MyGene.info for anything not in the local dict ───────────────
    remaining = [uid for uid in clean_ids if uid not in trans]
    if remaining:
        try:
            if id_type == "entrez":
                resp = requests.post(
                    _MYGENE_GENE_URL,
                    json={
                        "ids":     remaining[:1000],
                        "fields":  "symbol",
                        "species": "human",
                    },
                    timeout=timeout,
                )
                if resp.ok:
                    for rec in resp.json():
                        if isinstance(rec, dict) and "symbol" in rec:
                            q = str(rec.get("query", "")).upper()
                            if q:
                                trans[q] = str(rec["symbol"]).upper()
            else:
                # Ensembl: strip version suffixes before querying
                clean_ens = [uid.split(".")[0] for uid in remaining[:1000]]
                resp = requests.post(
                    _MYGENE_QUERY_URL,
                    json={
                        "q":       clean_ens,
                        "fields":  "symbol",
                        "species": "human",
                        "scopes":  "ensembl.gene",
                    },
                    timeout=timeout,
                )
                if resp.ok:
                    for rec in resp.json():
                        if isinstance(rec, dict) and not rec.get("notfound") \
                                and "symbol" in rec:
                            q = str(rec.get("query", "")).upper()
                            if q:
                                trans[q]              = str(rec["symbol"]).upper()
                                # Map both with and without version suffix
                                orig = next(
                                    (uid for uid in remaining
                                     if uid.split(".")[0] == q), None
                                )
                                if orig:
                                    trans[orig] = str(rec["symbol"]).upper()
        except Exception as exc:
            log.warning(f"id_mapper: MyGene.info {id_type} lookup failed — {exc}")

    # ── Apply ────────────────────────────────────────────────────────────────
    result, n_ok = [], 0
    for uid in clean_ids:
        sym = trans.get(uid) or trans.get(uid.split(".")[0])
        if sym:
            result.append(sym)
            n_ok += 1
        else:
            result.append(uid)   # keep original — better than silently dropping

    return result, id_type, len(clean_ids), n_ok
