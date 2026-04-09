"""
Pathway Enrichment Module — Apex Bioinformatics Platform
=========================================================
Provides offline-capable enrichment via built-in gene sets plus
Fisher-exact testing with FDR correction.

Statistical upgrades (v3.0):
  • Odds ratio + 95 % CI  (Wald method) for every pathway row
  • Gene ratio  (overlap / query size)  — like clusterProfiler output
  • Activation Z-score  (IPA-style directional scoring per pathway)
  • Effect-weighted enrichment score (Σ |LFC| × −log10p for overlap genes)
  • Exposed background size in public API (default 20 000)
  • Exposed padj threshold in public API (default 0.05 for strict mode)
  • Public helpers: list_databases(), get_pathway_genes()
"""

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Built-in gene-set database
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# KEGG — 40 pathways, 40–130 genes each (representative of actual KEGG 2024)
# ─────────────────────────────────────────────────────────────────────────────
_KEGG: Dict[str, List[str]] = {
    "Cell Cycle": [
        "CDK1","CDK2","CDK4","CDK6","CDK7","CCNA1","CCNA2","CCNB1","CCNB2",
        "CCND1","CCND2","CCND3","CCNE1","CCNE2","CCNH","RB1","RBL1","RBL2",
        "E2F1","E2F2","E2F3","E2F4","E2F5","E2F6","E2F7","TFDP1","TFDP2",
        "CDKN1A","CDKN1B","CDKN1C","CDKN2A","CDKN2B","CDKN2C","CDKN2D",
        "CDC20","CDC25A","CDC25B","CDC25C","BUB1","BUB1B","BUB3",
        "MAD1L1","MAD2L1","PTTG1","PTTG2","PCNA","WEE1","MYT1","PKMYT1",
        "PLK1","PLK2","PLK3","AURKA","AURKB","MKI67","TOP2A","TOP2B",
        "MCM2","MCM3","MCM4","MCM5","MCM6","MCM7","CDC6","CDC7","CDT1",
        "ORC1","ORC2","ORC3","ORC4","ORC5","ORC6","DBF4","CHEK1","CHEK2",
        "ATM","ATR","TP53","MDM2","GADD45A","SKP2","FBXW7","RBX1","SKP1",
    ],
    "Apoptosis": [
        "BAX","BAK1","BCL2","BCL2L1","BCL2L2","MCL1","BCL2L11","PMAIP1","BBC3",
        "TP53","FAS","FASLG","TNF","TNFRSF1A","TNFRSF10A","TNFRSF10B",
        "TRADD","FADD","RIPK1","CASP3","CASP6","CASP7","CASP8","CASP9","CASP10",
        "CYCS","APAF1","DIABLO","PARP1","PARP2","BID","BIRC2","BIRC3","XIAP",
        "BIRC5","CFLAR","AKT1","PIK3CA","MAP3K5","MAPK8","MAPK9","MAPK10",
        "NFKB1","RELA","IKBKA","IKBKB","NFKBIA","TRAF2","TRAF3","DAXX",
        "LMNA","DFFA","DFFB","ENDOG","HTRA2","SMAC","AIFM1","SPTAN1","ACIN1",
    ],
    "p53 Signaling": [
        "TP53","MDM2","MDM4","CDKN1A","CDK2","CDK4","CDK6","CCNE1","CCND1",
        "RB1","E2F1","ATM","ATR","CHEK1","CHEK2","GADD45A","GADD45B","GADD45G",
        "BAX","BBC3","PMAIP1","FAS","PERP","APAF1","CASP3","CASP9",
        "SESN1","SESN2","SESN3","TIGAR","SCO2","GLS2","DRAM1","DRAM2",
        "DDB2","XPC","POLK","POLH","POLI","RRM2B","ZMAT3","CYCLIN_G1",
        "TP63","TP73","HIPK2","SIAH1","BTG2","REPRIMO","PIDD1","REDD1",
        "PTEN","SFN","BRCA1","BRCA2","CABLES1",
    ],
    "MAPK Signaling": [
        "HRAS","KRAS","NRAS","ARAF","BRAF","RAF1","MAP3K1","MAP3K2","MAP3K3",
        "MAP3K4","MAP3K5","MAP3K7","MAP3K8","MAP2K1","MAP2K2","MAP2K3","MAP2K4",
        "MAP2K5","MAP2K6","MAP2K7","MAPK1","MAPK3","MAPK4","MAPK6","MAPK7",
        "MAPK8","MAPK9","MAPK10","MAPK11","MAPK12","MAPK13","MAPK14",
        "RPS6KA1","RPS6KA2","RPS6KA3","RPS6KA5","MNK1","MNK2",
        "EGF","EGFR","ERBB2","FGF1","FGFR1","PDGFA","PDGFRB","NGF","NTRK1",
        "MYC","JUN","FOS","ELK1","SRF","ATF2","SP1","MEF2C","DDIT3",
        "DUSP1","DUSP4","DUSP6","DUSP8","DUSP10","DUSP16",
        "TRAF2","TRAF6","TAB1","TAK1","GRB2","SOS1","SOS2","SHC1","GAB1",
        "PAK1","RAC1","CDC42","FLNB","STMN1","NF1","RASA1","RASSF1",
    ],
    "PI3K-AKT Signaling": [
        "PIK3CA","PIK3CB","PIK3CD","PIK3CG","PIK3R1","PIK3R2","PIK3R3",
        "PTEN","PDPK1","AKT1","AKT2","AKT3","MTOR","RPTOR","RICTOR",
        "TSC1","TSC2","RHEB","RPS6KB1","RPS6KB2","EIF4EBP1","EIF4B",
        "GSK3A","GSK3B","FOXO1","FOXO3","FOXO4","MDM2","TP53",
        "BCL2","BCL2L1","MCL1","BAD","CDKN1A","CDKN1B",
        "IRS1","IRS2","INSR","IGF1R","EGFR","ERBB2","ERBB3","PDGFRA","PDGFRB",
        "KIT","FLT3","RET","MET","FGFR1","FGFR2","FGFR3",
        "ITGA1","ITGA2","ITGB1","ITGAV","ITGB3","SRC","FAK1","PTK2",
        "GRB2","SOS1","KRAS","HRAS","NRAS","RAF1","MAP2K1","MAPK1",
        "PPP2CA","PPP2CB","PHLPP1","PHLPP2","THEM4","CTMP",
    ],
    "mTOR Signaling": [
        "MTOR","RPTOR","RICTOR","MLST8","DEPTOR","AKT1S1","PRAS40",
        "AKT1","AKT2","AKT3","TSC1","TSC2","RHEB","RRAGB","RRAGC",
        "RPS6KB1","RPS6KB2","EIF4EBP1","EIF4EBP2","EIF4EBP3",
        "ULK1","ULK2","ATG13","ATG101","RB1CC1","AMBRA1",
        "RPS6","EIF4E","EIF4G1","EIF4G2","EIF4G3","EIF4A1","EIF3A",
        "PTEN","PIK3CA","PIK3R1","PDPK1","SGKL","SGK1","SGK3",
        "LPIN1","LPIN2","SREBP1","TFEB","TFE3","TFEC","MITF",
        "PPARGC1A","PPARGC1B","BNIP3","BNIP3L","SQSTM1","SESN1","SESN2",
    ],
    "JAK-STAT Signaling": [
        "JAK1","JAK2","JAK3","TYK2","STAT1","STAT2","STAT3","STAT4","STAT5A",
        "STAT5B","STAT6","SOCS1","SOCS2","SOCS3","SOCS4","SOCS5","SOCS6",
        "PIAS1","PIAS3","IL2","IL3","IL4","IL5","IL6","IL7","IL9","IL10",
        "IL11","IL12A","IL13","IL15","IL21","IFNA1","IFNB1","IFNG",
        "EPOR","THPO","MPL","GHR","PRLR","CSF1R","CSF2RA","CSF3R",
        "LEPR","IL6ST","LIFR","OSMR","CNTFR","GRB2","SHC1","SHP1","SHP2",
        "PTPN6","PTPN11","PIM1","PIM2","BCL2","BCL2L1","CCND1","CDKN1A",
        "IRF1","IRF2","IRF3","IRF9","MX1","MX2","OAS1","PKR","ISG15",
    ],
    "Wnt Signaling": [
        "WNT1","WNT2","WNT3","WNT3A","WNT4","WNT5A","WNT5B","WNT6","WNT7A",
        "WNT7B","WNT8A","WNT8B","WNT9A","WNT10A","WNT10B","WNT11","WNT16",
        "FZD1","FZD2","FZD3","FZD4","FZD5","FZD6","FZD7","FZD8","FZD9","FZD10",
        "LRP5","LRP6","DVL1","DVL2","DVL3","AXIN1","AXIN2","APC","APC2",
        "GSK3A","GSK3B","CTNNB1","TCF7","TCF7L1","TCF7L2","LEF1",
        "MYC","CCND1","MMP7","CD44","WISP1","HHEX","SFRP1","SFRP2","SFRP3",
        "DKK1","DKK2","DKK3","DKK4","DACT1","DACT2","NKD1","NKD2",
        "NOTUM","CSNK1A1","CSNK1D","CSNK1E","CSNK2A1","CSNK2B",
        "RHOU","RHOV","RAC1","CDC42","JNK1","JNK2","NFAT5","NFATC1",
    ],
    "TGF-β Signaling": [
        "TGFB1","TGFB2","TGFB3","TGFBR1","TGFBR2","TGFBR3",
        "ACVR1","ACVR1B","ACVR1C","ACVR2A","ACVR2B","BMPR1A","BMPR1B","BMPR2",
        "SMAD1","SMAD2","SMAD3","SMAD4","SMAD5","SMAD6","SMAD7","SMAD9",
        "SMURF1","SMURF2","SKI","SKIL","TGIF1","TGIF2",
        "BMP2","BMP4","BMP5","BMP6","BMP7","BMP8A","BMP8B","GDF1","GDF5",
        "NODAL","LEFTY1","LEFTY2","INHBA","INHBB","AMH","MSTN","GDF11",
        "LTBP1","LTBP2","LTBP3","LTBP4","THBS1","THBS2","THBS3",
        "ID1","ID2","ID3","ID4","MYC","JUN","SP1","RUNX2","RUNX3",
        "BAMBI","FKBP1A","FKBP1B","NEDD4L","PPM1A","SARA","EVI1",
    ],
    "Notch Signaling": [
        "NOTCH1","NOTCH2","NOTCH3","NOTCH4","DLL1","DLL3","DLL4",
        "JAG1","JAG2","RBPJ","HES1","HES2","HES5","HES7",
        "HEY1","HEY2","HEYL","MAML1","MAML2","MAML3",
        "NUMB","NUMBL","LFNG","MFNG","RFNG","PSEN1","PSEN2","ADAM10",
        "APH1A","APH1B","PEN2","NICASTRIN","DVL1","DVL2","DVL3",
        "DTX1","DTX2","DTX3","DTX3L","DTX4","EP300","CREBBP",
        "KAT2B","SKIP","SHARP","RUNX1","RUNX2","MYC","CCND1","CDK8",
    ],
    "Hedgehog Signaling": [
        "SHH","IHH","DHH","PTCH1","PTCH2","SMO","GLI1","GLI2","GLI3",
        "SUFU","HHIP","BOC","GAS1","CDON","DISP1","DISP2",
        "CSNK1A1","CSNK1D","CSNK1E","PKA","DYRK2","GSK3B",
        "KIF7","KIF27","STK36","SPOP","CUL3","BTRC","GAS2L2",
        "IFT27","IFT46","IFT57","IFT88","IFT172","BBS1","BBS2",
        "HHAT","SCUBE2","FOXF1","FOXL1","FOXC2","SNAI1","SNAI2",
        "MYCN","MYC","CCND1","CCND2","CDK6","BCL2","VEGFA",
    ],
    "HIF-1 Signaling": [
        "HIF1A","HIF1B","ARNT","ARNT2","HIF3A","EPAS1",
        "VHL","EGLN1","EGLN2","EGLN3","HIF1AN","HSP90AA1","HSP90AB1",
        "VEGFA","VEGFB","VEGFC","VEGFD","EPO","SLC2A1","SLC2A3","SLC2A4",
        "LDHA","LDHB","ALDOA","ALDOC","PGK1","PGAM1","ENO1","PKM",
        "PDK1","PDK3","PFKFB3","PFKFB4","TIGAR","BNIP3","BNIP3L",
        "HMOX1","NOS2","CXCR4","CXCL12","MMP2","MMP9","TWIST1","SNAI1",
        "PLAUR","PAI1","SERPINE1","TF","TFRC","AMBRA1",
        "EGFR","ERBB2","IGF1R","INSR","PI3K","MAPK1","MAPK3",
        "EP300","CREBBP","RBBP5","HDAC1","HDAC2","CBP",
    ],
    "NF-κB Signaling": [
        "NFKB1","NFKB2","RELA","RELB","REL",
        "IKBKA","IKBKB","IKBKG","NFKBIA","NFKBIB","NFKBIE",
        "TRAF2","TRAF3","TRAF5","TRAF6","TRADD","RIPK1","RIPK3",
        "TNFRSF1A","TNFRSF1B","TNF","LTA","LTB","LTBR","IL1A","IL1B",
        "IL1R1","IL1R2","IL1RAP","MYD88","IRAK1","IRAK4","PELLINO1",
        "TLR1","TLR2","TLR3","TLR4","TLR5","TLR6","TLR7","TLR8","TLR9",
        "BCL2","BCL2L1","BIRC2","BIRC3","XIAP","CFLAR","CD40","CD40LG",
        "ICAM1","VCAM1","CCL2","CCL5","IL6","IL8","CXCL10","MMP9","COX2",
        "TAK1","TAB1","TAB2","NIK","MAP3K7","CARD11","MALT1","BCL10",
    ],
    "VEGF Signaling": [
        "VEGFA","VEGFB","VEGFC","VEGFD","PGF","KDR","FLT1","FLT4",
        "NRP1","NRP2","PDGFRA","PDGFRB","ANGPT1","ANGPT2","TIE1","TEK",
        "PIK3CA","PIK3R1","AKT1","MAPK1","MAPK3","SRC","FAK1","PTK2",
        "NOS3","PLCG1","PKC","DAG","IP3","PRKCB","PRKCA","PRKCD",
        "RAS","RAF1","MEK1","ERK1","ELK1","SRF","SP1",
        "MMP2","MMP9","MMP14","UPAR","PLAUR","PAI1","SERPINE1",
        "HIF1A","EPAS1","ANGPTL3","ANGPTL4","ANGPTL6",
        "CDC42","RAC1","RHOA","ROCK1","ROCK2","PAK1","PAK2","LIMK1",
    ],
    "DNA Damage Response": [
        "ATM","ATR","CHEK1","CHEK2","TP53","BRCA1","BRCA2","PALB2","NBN",
        "MRE11","RAD50","RAD51","RAD52","RAD54L","XRCC2","XRCC3","XRCC5","XRCC6",
        "MDC1","H2AFX","RNF8","RNF168","RNF4","53BP1","PTIP","JMJD2A",
        "PARP1","PARP2","XRCC1","LIG1","LIG3","LIG4","XRCC4","NHEJ1","DCLRE1C",
        "MSH2","MSH3","MSH6","MLH1","MLH3","PMS1","PMS2","EXO1","PCNA",
        "FANCA","FANCB","FANCC","FANCD1","FANCD2","FANCE","FANCF","FANCG",
        "FANCI","FANCJ","FANCL","FANCM","FANCN","FANCP","FANCQ","FANCR",
        "WRN","BLM","RECQL4","RECQL5","DNA2","FEN1","RPA1","RPA2","RPA3",
        "RFC1","RFC2","RFC3","RFC4","RFC5","MSH4","MSH5","SPO11",
    ],
    "Ribosome": [
        "RPSA","RPS2","RPS3","RPS3A","RPS4X","RPS4Y1","RPS5","RPS6","RPS7",
        "RPS8","RPS9","RPS10","RPS11","RPS12","RPS13","RPS14","RPS15","RPS15A",
        "RPS16","RPS17","RPS18","RPS19","RPS20","RPS21","RPS23","RPS24","RPS25",
        "RPS26","RPS27","RPS27A","RPS28","RPS29","RPS6KA1","RPS6KA3",
        "RPL3","RPL4","RPL5","RPL6","RPL7","RPL7A","RPL8","RPL9","RPL10",
        "RPL10A","RPL11","RPL12","RPL13","RPL13A","RPL14","RPL15","RPL17",
        "RPL18","RPL18A","RPL19","RPL21","RPL22","RPL22L1","RPL23","RPL23A",
        "RPL24","RPL26","RPL27","RPL27A","RPL28","RPL29","RPL30","RPL31",
        "RPL32","RPL34","RPL35","RPL35A","RPL36","RPL36A","RPL37","RPL37A",
        "RPL38","RPL39","RPL40","RPL41","RPLP0","RPLP1","RPLP2",
        "EIF4A1","EIF4E","EIF4G1","EIF4G2","EIF4G3","EIF2S1","EIF2S2","EIF2S3",
        "EIF3A","EIF3B","EIF3C","EIF3D","EIF3E","EIF3F","EIF3G","EIF3H",
        "EIF3I","EIF3J","EIF3K","EIF3L","EIF3M","EIF6","EIF1","EIF1AX",
    ],
    "Oxidative Phosphorylation": [
        "NDUFA1","NDUFA2","NDUFA3","NDUFA4","NDUFA5","NDUFA6","NDUFA7",
        "NDUFA8","NDUFA9","NDUFA10","NDUFA11","NDUFA12","NDUFA13",
        "NDUFB1","NDUFB2","NDUFB3","NDUFB4","NDUFB5","NDUFB6","NDUFB7",
        "NDUFB8","NDUFB9","NDUFB10","NDUFB11",
        "NDUFC1","NDUFC2","NDUFS1","NDUFS2","NDUFS3","NDUFS4","NDUFS5",
        "NDUFS6","NDUFS7","NDUFS8","NDUFV1","NDUFV2","NDUFV3",
        "SDHA","SDHB","SDHC","SDHD",
        "UQCRB","UQCRC1","UQCRC2","UQCRFS1","UQCRH","UQCRQ","UQCR10","UQCR11","CYC1",
        "COX4I1","COX4I2","COX5A","COX5B","COX6A1","COX6A2","COX6B1","COX6B2",
        "COX6C","COX7A1","COX7A2","COX7B","COX7C","COX8A",
        "ATP5F1A","ATP5F1B","ATP5F1C","ATP5F1D","ATP5F1E",
        "ATP5MC1","ATP5MC2","ATP5MC3","ATP5ME","ATP5MF","ATP5MG",
        "ATP5PB","ATP5PD","ATP5PF","ATP5PO","ATP5IF1",
        "MT-ND1","MT-ND2","MT-ND3","MT-ND4","MT-ND4L","MT-ND5","MT-ND6",
        "MT-CO1","MT-CO2","MT-CO3","MT-ATP6","MT-ATP8","MT-CYB",
        "CYCS","TFAM","NRF1","PPARGC1A","PPARGC1B",
    ],
    "Glycolysis / Gluconeogenesis": [
        "HK1","HK2","HK3","GCK","GPI","PFKL","PFKM","PFKP","ALDOA","ALDOB","ALDOC",
        "TPI1","GAPDH","GAPDHS","PGK1","PGK2","PGAM1","PGAM2","ENO1","ENO2","ENO3",
        "PKM","PKLR","LDHA","LDHB","LDHC","PDK1","PDK2","PDK3","PDK4",
        "PDHA1","PDHA2","PDHB","DLD","DLAT","PDHX",
        "PCK1","PCK2","G6PC","G6PC2","GPC1","FBP1","FBP2",
        "PGAM4","BPGM","MINPP1",
        "SLC2A1","SLC2A2","SLC2A3","SLC2A4","SLC2A5","SLC2A6",
        "PGLS","RPIA","RPE","TALDO1","TKT","TKTL1","G6PD","PGD","6PGL",
    ],
    "Fatty Acid Metabolism": [
        "FASN","ACACA","ACACB","ACLY","ACSS1","ACSS2",
        "SCD","SCD5","ELOVL1","ELOVL2","ELOVL3","ELOVL4","ELOVL5","ELOVL6","ELOVL7",
        "FADS1","FADS2","FADS3","FADS6","PTPLA",
        "CPT1A","CPT1B","CPT1C","CPT2","CRAT","SLCO2B1",
        "HADHA","HADHB","HADH","ACADM","ACADL","ACADS","ACADVL","ACAD9","ACAD10",
        "ECHS1","EHHADH","HSD17B10","ACAA2","ACAA1","ACAT1","ACAT2",
        "ACSL1","ACSL3","ACSL4","ACSL5","ACSL6","ACSBG1","ACSBG2",
        "PPARA","PPARG","PPARD","RXRA","RXRB","RXRG",
        "HMGCR","HMGCS1","HMGCS2","MVK","PMVK","MVD","FDPS","GGPS1",
        "SQLE","LSS","CYP51A1","SC4MOL","NSDHL","HSD17B7","DHCR7","DHCR24",
    ],
    "TCA Cycle / Citrate": [
        "CS","ACO1","ACO2","IDH1","IDH2","IDH3A","IDH3B","IDH3G",
        "OGDH","OGDHL","DLST","DLD","SUCLA2","SUCLG1","SUCLG2",
        "SDHA","SDHB","SDHC","SDHD","FH","MDH1","MDH2",
        "PC","ACLY","PDHA1","PDHA2","PDHB","PDHX","DLAT","LIPT1",
        "SLC25A1","SLC25A10","SLC25A11","SLC25A13","SLC25A12",
        "GOT1","GOT2","GPT","GLUD1","GLUD2","GLS","GLS2",
        "ACSS1","ACSS2","ACSS3",
    ],
    "Amino Acid Metabolism": [
        "SLC1A4","SLC1A5","SLC7A5","SLC7A8","SLC7A11","SLC3A2",
        "ASNS","ASNS2","ASPG","GOT1","GOT2","GPT","GPT2","GLUD1","GLUD2",
        "GLS","GLS2","GLUL","GDH1","ALDH18A1","PYCR1","PYCR2","PYCRL",
        "PHGDH","PSAT1","PSPH","SHMT1","SHMT2","MTHFD1","MTHFD2","MTHFD1L",
        "AMD1","ODC1","SMOX","SAT1","SAT2","AZIN1","AZIN2",
        "ARG1","ARG2","NOS1","NOS2","NOS3","ASS1","ASL","OTC","CPS1",
        "CTH","CBS","MPST","TST","AHCY","MAT1A","MAT2A","MAT2B",
        "TAT","HPD","HGD","FAH","PAH","TH","DDC","AADC","DBH","PNMT",
        "TPH1","TPH2","MAOA","MAOB","IDO1","IDO2","TDO2","KYAT3","KMO",
    ],
    "Proteasome": [
        "PSMA1","PSMA2","PSMA3","PSMA4","PSMA5","PSMA6","PSMA7",
        "PSMB1","PSMB2","PSMB3","PSMB4","PSMB5","PSMB6","PSMB7",
        "PSMB8","PSMB9","PSMB10","PSMC1","PSMC2","PSMC3","PSMC4","PSMC5","PSMC6",
        "PSMD1","PSMD2","PSMD3","PSMD4","PSMD6","PSMD7","PSMD8","PSMD9",
        "PSMD10","PSMD11","PSMD12","PSMD13","PSMD14",
        "PSMF1","SEM1","ADRM1","SHFM1",
        "UBA1","UBA2","UBA3","UBA6","UBA7",
        "UBB","UBC","UBD","UBE2A","UBE2B","UBE2C","UBE2D1","UBE2D2",
        "UBE2D3","UBE2E1","UBE2G1","UBE2G2","UBE2I","UBE2K","UBE2L3",
        "UBE2M","UBE2N","UBE2Q1","UBE2R1","UBE2S","UBE2T","UBE2W",
        "CUL1","CUL2","CUL3","CUL4A","CUL4B","CUL5","CUL7","CUL9",
        "RBX1","RBX2","SKP1","SKP2","FBXW7","FBXO6","FBXO11","BTRC",
        "VHL","SPOP","KEAP1","MDM2","BRCA1","HERC2","NEDD4","ITCH",
        "USP7","USP14","USP28","OTUB1","BAP1","BRCC3",
    ],
    "Spliceosome": [
        "SF1","U2AF1","U2AF2","SF3A1","SF3A2","SF3A3","SF3B1","SF3B2",
        "SF3B3","SF3B4","SF3B5","SF3B6","SNRNP70","SNRNP200","SNRNP25",
        "SNRNP27","SNRNPA","SNRNPB","SNRNPB2","SNRNPC","SNRNPD1","SNRNPD2",
        "SNRNPD3","SNRNPE","SNRNPF","SNRNPG","PRPF3","PRPF4","PRPF6","PRPF8",
        "PRPF18","PRPF19","PRPF31","PRPF38A","PRPF38B","PRPF40A","PRPF40B",
        "DDX23","DHX8","DHX15","DHX16","DHX38","HNRNPA1","HNRNPA2B1","HNRNPC",
        "HNRNPD","HNRNPF","HNRNPH1","HNRNPK","HNRNPL","HNRNPM","HNRNPR","HNRNPU",
        "SRSF1","SRSF2","SRSF3","SRSF4","SRSF5","SRSF6","SRSF7","SRSF9","SRSF10",
        "PTBP1","PTBP2","RBMX","RBM10","RBM17","RBM22","RBM25",
        "CELF1","MBNL1","TRA2A","TRA2B","CLK1","CLK2","CLK3","CLK4",
    ],
    "Autophagy": [
        "ULK1","ULK2","ATG1","ATG2A","ATG2B","ATG3","ATG4A","ATG4B","ATG4C","ATG4D",
        "ATG5","ATG7","ATG9A","ATG9B","ATG10","ATG12","ATG13","ATG14","ATG16L1","ATG16L2",
        "BECN1","BECN2","PIK3C3","PIK3R4","UVRAG","RUBCN","SH3GLB1","AMBRA1",
        "MAP1LC3A","MAP1LC3B","MAP1LC3C","GABARAP","GABARAPL1","GABARAPL2",
        "SQSTM1","NBR1","OPTN","NDP52","TAX1BP1","CALCOCO2",
        "RAB7A","RAB11A","RAB12","PLEKHM1","LAMP1","LAMP2","RAB9A",
        "TFEB","TFE3","MITF","ZKSCAN3","YWHA","MTOR","RPTOR","AKT1","PTEN",
        "TP53","DAPK1","DAPK2","BNIP3","BNIP3L","FUNDC1","FKBP51",
        "PINK1","PRKN","MFN1","MFN2","DNM1L","OPA1","CHCHD3","CHCHD6",
        "VPS4A","VPS4B","VPS34","RAB5A","EEA1","FYVE","WIPI1","WIPI2",
    ],
    "EMT / Invasion": [
        "CDH1","CDH2","CDH3","CDH4","CDH11","VIM","FN1","ACTA2",
        "TWIST1","TWIST2","SNAI1","SNAI2","SNAI3","ZEB1","ZEB2",
        "ZEB2","KLF8","SP1","ETS1","ETS2","SRF","FOXC1","FOXC2",
        "MMP1","MMP2","MMP3","MMP7","MMP9","MMP10","MMP11","MMP13","MMP14","MMP15",
        "TGFB1","TGFB2","TGFB3","SMAD2","SMAD3","SMAD4","WNT5A","WNT11",
        "NOTCH1","NOTCH3","DLL4","JAG1","HEY1","HES1",
        "EGF","EGFR","ERBB2","HGF","MET","AXL","GAS6","TYRO3","MERTK",
        "ITGB1","ITGB3","ITGB4","ITGB6","ITGA2","ITGA3","ITGA5","ITGA6",
        "FAK1","PTK2","SRC","PXN","BCAR1","ILK","PINCH1","PARVIN",
        "RAC1","RHOA","ROCK1","ROCK2","CDC42","PAK1","PAK2","LIMK1","CFL1",
        "CXCR4","CXCL12","CXCL1","CXCL8","IL6","IL11","JAK1","STAT3",
        "TNC","POSTN","SPARC","LGALS1","LGALS3","LGALS9","S100A4","S100A7",
    ],
    "ECM-Receptor Interaction": [
        "COL1A1","COL1A2","COL2A1","COL3A1","COL4A1","COL4A2","COL5A1","COL5A2",
        "COL6A1","COL6A2","COL6A3","COL7A1","COL11A1","COL12A1","COL14A1",
        "FN1","VTN","TNC","THBS1","THBS2","THBS3","THBS4","COMP","MATN1","MATN2",
        "LAMA1","LAMA2","LAMA3","LAMA4","LAMA5","LAMB1","LAMB2","LAMB3","LAMC1","LAMC2","LAMC3",
        "HSPG2","AGRN","NID1","NID2",
        "ITGA1","ITGA2","ITGA3","ITGA4","ITGA5","ITGA6","ITGA7","ITGA8","ITGA9","ITGA10","ITGA11",
        "ITGAV","ITGAX","ITGB1","ITGB2","ITGB3","ITGB4","ITGB5","ITGB6","ITGB7","ITGB8",
        "SPP1","IBSP","SDC1","SDC2","SDC3","SDC4","GPC1","GPC3","GPC4","GPC6",
        "CD44","RELN","HAPLN1","HAPLN3","ACAN","VCAN","BCAN","NCAN","CSPG4",
        "MMP1","MMP2","MMP9","MMP14","ADAM10","ADAM12","ADAM17","ADAMTS2",
    ],
    "Focal Adhesion": [
        "EGFR","ERBB2","ERBB3","PDGFRA","PDGFRB","INSR","IGF1R","VEGFR2","MET",
        "PIK3CA","PIK3CB","PIK3R1","PIP2","PIP3","PTEN","PDK1","AKT1","AKT2","AKT3",
        "MTOR","TSC1","TSC2","RPS6KB1","EIF4EBP1",
        "KRAS","HRAS","NRAS","RAF1","BRAF","MAP2K1","MAP2K2","MAPK1","MAPK3",
        "SRC","YES1","FYN","LCK","LYN","SHC1","GRB2","SOS1","SOS2","GAB1","GAB2",
        "FAK1","PTK2","PYK2","PTK2B","ILK","PINCH1","PINCH2","PARVA","PARVB","PARVG",
        "BCAR1","PAX","PXN","VASP","ENAH","TLN1","TLN2","VCL","FLNB","FLNC",
        "ITGA1","ITGA2","ITGA3","ITGA5","ITGAV","ITGB1","ITGB3","ITGB5","ITGB6",
        "RAC1","RAC2","CDC42","RHOA","RHOB","RHOC","RND1","RND3",
        "ROCK1","ROCK2","LIMK1","LIMK2","CFL1","CFL2","DIAPH1","VASP","ARPC1A",
        "BCL2","BCL2L1","BAD","CASP3","CASP9","XIAP","SURVIVIN",
        "CCND1","CCND2","CCNE1","CDK4","CDK6","CDK2","CDKN1A","CDKN1B",
        "MYC","JUN","ELK1","CREB1","ATF1","STAT3","NF2",
    ],
    "Cytokine-Cytokine Receptor Interaction": [
        "IL1A","IL1B","IL1R1","IL1R2","IL1RAP","IL1RL1","IL1RL2",
        "IL2","IL2RA","IL2RB","IL2RG","IL3","IL3RA","CSF2RB",
        "IL4","IL4R","IL13","IL13RA1","IL13RA2","IL6","IL6R","IL6ST",
        "IL7","IL7R","IL9","IL9R","IL10","IL10RA","IL10RB",
        "IL11","IL11RA","IL12A","IL12B","IL12RB1","IL12RB2",
        "IL15","IL15RA","IL17A","IL17B","IL17C","IL17D","IL17F","IL17RA","IL17RB",
        "IL18","IL18R1","IL18RAP","IL21","IL21R","IL22","IL22RA1","IL22RA2",
        "IL23A","IL23R","IL24","IL25","IL26","IL27","IL27RA","IL28A","IL28B",
        "IL33","IL33R","IL35","IL36A","IL36B","IL36G","IL36R","IL37",
        "TNF","TNFRSF1A","TNFRSF1B","LTA","LTB","LTBR","TNFRSF3","TNFRSF4",
        "TNFRSF5","TNFRSF6","TNFRSF6B","TNFRSF8","TNFRSF9","TNFRSF10A","TNFRSF10B",
        "TNFRSF11A","TNFRSF11B","TNFRSF12A","TNFRSF13B","TNFRSF13C","TNFRSF14",
        "IFNA1","IFNA2","IFNB1","IFNG","IFNGR1","IFNGR2","IFNAR1","IFNAR2",
        "CXCL1","CXCL2","CXCL3","CXCL5","CXCL6","CXCL8","CXCL9","CXCL10","CXCL11",
        "CXCL12","CXCL13","CXCL14","CXCL16","CXCR1","CXCR2","CXCR3","CXCR4","CXCR5",
        "CCL2","CCL3","CCL4","CCL5","CCL7","CCL8","CCL11","CCL13","CCL17","CCL19",
        "CCL20","CCL21","CCL22","CCL24","CCL25","CCL27","CCR1","CCR2","CCR3","CCR4",
        "CCR5","CCR6","CCR7","CCR8","CCR9","CCR10",
        "CSF1","CSF1R","CSF2","CSF2RA","CSF3","CSF3R",
        "EPO","EPOR","THPO","MPL","KIT","SCF","FLT3","FLT3LG",
    ],
    "Toll-Like Receptor Signaling": [
        "TLR1","TLR2","TLR3","TLR4","TLR5","TLR6","TLR7","TLR8","TLR9","TLR10",
        "MYD88","TIRAP","TRIF","TRAM","IRAK1","IRAK2","IRAK4","TRAF3","TRAF6",
        "TAK1","TAB1","TAB2","RIPK1","FADD","CASP8",
        "IRF1","IRF3","IRF5","IRF7","IFNA1","IFNB1",
        "NFKB1","RELA","IKBKA","IKBKB","IKBKG","MAP3K1","MAP3K7",
        "MAP2K1","MAP2K3","MAP2K4","MAP2K6","MAPK1","MAPK3","MAPK8","MAPK11","MAPK14",
        "IL1B","IL6","IL8","IL12A","TNF","CCL2","CCL5","CXCL10",
        "CD14","MD2","LPS","HMGB1","HSP70","HSP90AA1","CALR",
        "CTSL","CTSS","CTSB","UNC93B1","LRBA",
    ],
    "T Cell Receptor Signaling": [
        "CD3D","CD3E","CD3G","CD247","ZAP70","LCK","FYN","LAT","SLP76","GADS",
        "PLCγ1","PLCG1","IP3","DAG","PRKCQ","PRKCA","PKCA",
        "CALM1","CALM2","CALM3","CALCINEURIN","PPP3CA","PPP3CB","NFATC1","NFATC2","NFATC3",
        "PIK3CD","PIK3R1","AKT1","PDK1","TSC1","TSC2","MTOR","RPS6KB1",
        "GRB2","SOS1","HRAS","KRAS","RAF1","MAP2K1","MAPK3","ELK1",
        "NFKB1","RELA","IKBKA","IKBKB","IKBKG","CARD11","MALT1","BCL10",
        "CD4","CD8A","CD8B","MHC2","CD28","CD80","CD86","CTLA4","PDCD1",
        "LAG3","HAVCR2","TIGIT","CD226","CD96","BTLA","CD160",
        "IL2","IL2RA","IL2RB","IL2RG","JAK1","JAK3","STAT5A","STAT5B",
        "FOXP3","RORC","GATA3","TBX21","RUNX1","BCL6","IRF4",
    ],
    "Complement & Coagulation Cascades": [
        "C1QA","C1QB","C1QC","C1R","C1S","C2","C3","C4A","C4B","C5","C6","C7","C8A","C8B","C8G","C9",
        "C3AR1","C5AR1","C5AR2","CR1","CR2","CR3","CR4","ITGAM","ITGB2",
        "MASP1","MASP2","MBL2","FCN1","FCN2","FCN3","COLEC10","COLEC11",
        "CFB","CFD","CFP","CFH","CFHR1","CFI","CD55","CD46","CD59","C4BPA","C4BPB",
        "F2","F3","F4","F5","F7","F8","F9","F10","F11","F12","F13A1","F13B",
        "SERPINC1","SERPINE1","SERPIND1","THBD","TFPI","PROS1","PROC","PROZ",
        "VWF","ITGAV","ITGB3","GP1BA","GP1BB","GP9","SELP","SELE","SELL",
        "PLG","PLAT","PLAU","PLAUR","PAI1","A2M","KNG1",
    ],
    "PPAR Signaling": [
        "PPARA","PPARG","PPARD","RXRA","RXRB","RXRG",
        "FABP1","FABP2","FABP3","FABP4","FABP5","FABP6","FABP7",
        "ACSL1","ACSL3","ACSL4","ACSL5","ACSL6","SLC27A1","SLC27A2","SLC27A4",
        "CPT1A","CPT1B","CPT2","ACOX1","ACOX2","ACOX3","HADHA","HADHB",
        "ACADM","ACADL","ACADVL","HMGCS2","HMGCR",
        "FASN","ACACA","ACACB","SCD","SCD5","DGAT1","DGAT2",
        "LPL","APOC2","APOC3","APOAV","APOA1","APOA2","APOA5",
        "G0S2","ANGPTL4","ANGPTL3","ANGPTL8","PDK4","PCK1","G6PC",
        "ADIPOQ","ADIPOR1","ADIPOR2","RETN","LEP","LEPR","CFD",
        "CD36","SCARB1","NR1H3","NR1H4","ABCA1","ABCG1","CETP",
        "UCP1","UCP2","UCP3","IRS1","IRS2","INSR","PIK3CA","AKT1",
        "IL6","TNF","NFKB1","MMP9","MMP13","VEGFA",
    ],
    "ABC Transporters": [
        "ABCA1","ABCA2","ABCA3","ABCA4","ABCA7","ABCA8","ABCA12","ABCA13",
        "ABCB1","ABCB2","ABCB3","ABCB4","ABCB5","ABCB6","ABCB7","ABCB8","ABCB9","ABCB11",
        "ABCC1","ABCC2","ABCC3","ABCC4","ABCC5","ABCC6","ABCC8","ABCC9","ABCC10","ABCC11",
        "ABCD1","ABCD2","ABCD3","ABCD4","ABCE1","ABCF1","ABCF2","ABCF3",
        "ABCG1","ABCG2","ABCG4","ABCG5","ABCG8",
        "MDR1","MRP1","MRP2","BSEP","CFTR","TAP1","TAP2","SUR1","SUR2",
    ],
    "Calcium Signaling": [
        "EGFR","ERBB2","FGFR1","PDGFRA","IGF1R",
        "PLCB1","PLCB2","PLCB3","PLCB4","PLCG1","PLCG2","PLCD1","PLCΕ1",
        "ITPR1","ITPR2","ITPR3","RYR1","RYR2","RYR3",
        "SERCA1","SERCA2","SERCA3","ATP2A1","ATP2A2","ATP2A3",
        "CALM1","CALM2","CALM3","CAMK1","CAMK2A","CAMK2B","CAMK2D","CAMK2G",
        "CAMK4","CAMKK1","CAMKK2","PPP2B","PPP3CA","PPP3CB","NFATC1","NFATC2","NFATC3","NFATC4",
        "PHKB","PHKG1","PHKG2","GYS1","GYS2","PYGM","PYGL","PYGB",
        "NOS1","NOS2","NOS3","GUCY1A1","GUCY1B1","PDE1A","PDE1B","PDE1C",
        "PRKCG","PRKCA","PRKCB","PRKCD","PRKCE","PRKCΗ","PRKCH","PRKCI",
        "S100A1","S100A2","S100A4","S100A6","S100A8","S100A9","S100B","S100P",
        "TRPC1","TRPC3","TRPC4","TRPC5","TRPC6","TRPV1","TRPV4","ORAI1","STIM1",
        "GNA11","GNA14","GNA15","GNAQ","PTGER1","PTGER3","ADRB1","CHRM1","CHRM3",
    ],
    "Chemokine Signaling": [
        "CXCL1","CXCL2","CXCL3","CXCL5","CXCL6","CXCL8","CXCL9","CXCL10","CXCL11",
        "CXCL12","CXCL13","CXCL14","CXCL16","CXCL17",
        "CCL2","CCL3","CCL4","CCL5","CCL7","CCL8","CCL11","CCL13","CCL17","CCL19",
        "CCL20","CCL21","CCL22","CCL24","CCL25","CCL27","CCL28",
        "CXCR1","CXCR2","CXCR3","CXCR4","CXCR5","CXCR6","XCR1",
        "CCR1","CCR2","CCR3","CCR4","CCR5","CCR6","CCR7","CCR8","CCR9","CCR10",
        "GNB1","GNB2","GNG2","GNG3","GNG4","GNG5","GNG7","GNG10","GNG11",
        "PIK3CB","PIK3CG","PIK3R5","AKT1","PTEN","MTOR",
        "PLCB1","PLCB2","PLCB3","IP3","DAG","PRKCB","PRKCA",
        "ITPR1","CALM1","CAMK2A","CFL1","LIMK1","WASL","ARPC1A",
        "SRC","ZAP70","SHC1","GRB2","SOS1","KRAS","RAF1","MAP2K1","MAPK3",
        "STAT3","JAK2","JAK3","PIK3CA","PIK3R1","GNA11","GNAQ",
        "RAC1","CDC42","RHOA","PAK1","PAK2","PIX","DOCK2",
        "CXCL1","SELP","SELE","ICAM1","VCAM1","IL8","IL6","MMP2","MMP9",
    ],
    "Drug Metabolism — Cytochrome P450": [
        "CYP1A1","CYP1A2","CYP1B1","CYP2A6","CYP2B6","CYP2C8","CYP2C9","CYP2C18","CYP2C19",
        "CYP2D6","CYP2E1","CYP2J2","CYP3A4","CYP3A5","CYP3A7","CYP4A11","CYP4F2","CYP4F3",
        "CYP19A1","CYP21A2","CYP11A1","CYP11B1","CYP11B2","CYP17A1","CYP24A1","CYP27A1","CYP27B1",
        "GSTA1","GSTA2","GSTA4","GSTM1","GSTM2","GSTM3","GSTP1","GSTK1","GSTT1","GSTT2",
        "SULT1A1","SULT1A2","SULT1A3","SULT1B1","SULT1C2","SULT1E1","SULT2A1","SULT2B1",
        "UGT1A1","UGT1A3","UGT1A4","UGT1A6","UGT1A7","UGT1A8","UGT1A9","UGT1A10",
        "UGT2B4","UGT2B7","UGT2B10","UGT2B11","UGT2B15","UGT2B17",
        "NAT1","NAT2","TPMT","COMT","MAOA","MAOB","AOX1","XDH","ADH1A","ADH1B","ADH1C","ALDH2",
        "POR","NADPH","FMO1","FMO2","FMO3","FMO5",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# GO Biological Process — 30 terms, 50–200 genes each
# ─────────────────────────────────────────────────────────────────────────────
_GO_BP: Dict[str, List[str]] = {
    "Cell Proliferation": [
        "MYC","MYCN","MYCL","CCND1","CCND2","CCND3","CCNE1","CCNE2","CCNA1","CCNA2",
        "CCNB1","CCNB2","CDK1","CDK2","CDK4","CDK6","PCNA","MKI67","TOP2A","BIRC5",
        "RB1","RBL1","RBL2","E2F1","E2F2","E2F3","EGFR","ERBB2","ERBB3","MET",
        "KRAS","HRAS","NRAS","PIK3CA","AKT1","MTOR","YAP1","TAZ","WWTR1","HIPPO",
        "WNT3A","CTNNB1","TERT","MCM2","MCM3","MCM6","AURKB","PLK1","BUB1","CDC20",
        "FOXM1","MYBL2","E2F5","TFDP1","TFDP2","CDKN1A","CDKN1B","CDKN2A","PTEN",
        "STAT3","STAT5A","JAK2","IGF1R","IGF2R","IRS1","FGF2","FGFR1","PDGFRA",
    ],
    "Apoptotic Process": [
        "BAX","BAK1","BCL2","BCL2L1","BCL2L2","BCL2L11","MCL1","PMAIP1","BBC3","BID",
        "TP53","FAS","FASLG","TNF","TNFRSF1A","TNFRSF10A","TNFRSF10B",
        "TRADD","FADD","RIPK1","CASP3","CASP6","CASP7","CASP8","CASP9","CASP10",
        "CYCS","APAF1","DIABLO","SMAC","PARP1","XIAP","BIRC2","BIRC3","BIRC5",
        "CFLAR","AKT1","MAPK8","MAPK9","MAPK10","NFKB1","RELA","TRAF2",
        "DAPK1","DAPK2","DAPK3","RIP3","MLKL","PYCARD","NLRP3","CASP1",
        "LMNA","DFFA","DFFB","ENDOG","AIFM1","HTRA2","SPTAN1","ACIN1",
        "TP63","TP73","PUMA","PERP","SCOTIN","TNFRSF6B","CD27","TRAILR",
        "ATG5","BECN1","MAP1LC3A","SQSTM1","AMBRA1","BNIP3","BNIP3L",
    ],
    "Inflammatory Response": [
        "TNF","IL1A","IL1B","IL6","CXCL8","IL18","IL33","HMGB1","IL17A","IL17F",
        "NFKB1","RELA","NFKB2","RELB","IKBKA","IKBKB","NFKBIA","TRAF6","RIPK2",
        "NLRP3","PYCARD","CASP1","IL1R1","IL1RAP","TNFRSF1A","TNFRSF1B",
        "MYD88","IRAK1","IRAK4","TIRAP","TRIF","TLR2","TLR4","TLR9",
        "STAT3","JAK1","JAK2","IL6R","IL6ST","SOCS1","SOCS3","IRF1","IRF3","IRF5",
        "COX2","PTGS2","PTGS1","ALOX5","ALOX12","ALOX15","LTA4H","LTC4S","TBXAS1",
        "MMP1","MMP3","MMP9","MMP13","ADAM17","ADAM10",
        "CCL2","CCL3","CCL4","CCL5","CCL7","CXCL1","CXCL2","CXCL10","CXCL11",
        "VCAM1","ICAM1","ICAM2","SELP","SELE","SELL","CD44","ITGA4","ITGB2",
        "CR3","C3","C5","C5AR1","C3AR1","CFB","CFD","SERPING1",
        "P38","MAPK11","MAPK12","MAPK13","MAPK14","MKK3","MKK6","MSK1",
    ],
    "Immune Response": [
        "IFNG","IFNA1","IFNB1","IFNGR1","IFNGR2","IFNAR1","IFNAR2",
        "CD4","CD8A","CD8B","CD3D","CD3E","CD3G","CD247",
        "CD28","CD80","CD86","CTLA4","PDCD1","CD274","PDCD1LG2",
        "LAG3","HAVCR2","TIGIT","CD226","BTLA","VSIR","CD160",
        "FOXP3","IL2","IL2RA","IL2RB","IL2RG","IL7","IL15","IL21",
        "TNF","IL1B","IL6","IL10","IL12A","IL12B","IL18","IL23A",
        "MHC1","MHC2","B2M","TAP1","TAP2","TAPBP","CALR","HSP90B1",
        "GZMB","GZMK","PRF1","NKG2D","NKG7","EOMES","TBX21","GATA3","RORC","IRF4",
        "CXCR3","CCR5","CCR7","CXCR4","CXCR5","CCR4","CXCR6",
        "STAT1","STAT2","STAT3","STAT4","STAT5A","JAK1","JAK2","JAK3","TYK2",
        "SOCS1","SOCS3","IRF1","IRF3","IRF7","IRF9","OAS1","MX1","MX2","ISG15",
        "HLA-A","HLA-B","HLA-C","HLA-DRA","HLA-DRB1","HLA-DQA1","HLA-DQB1","HLA-DPA1","HLA-DPB1",
    ],
    "DNA Repair": [
        "BRCA1","BRCA2","PALB2","RAD51","RAD51B","RAD51C","RAD51D","XRCC2","XRCC3","RAD52","RAD54L",
        "ATM","ATR","CHEK1","CHEK2","BRIP1","BARD1","NBN","MRE11","RAD50",
        "MLH1","MLH3","MSH2","MSH3","MSH6","PMS1","PMS2","EXONUCLEASE",
        "PARP1","PARP2","PARP3","XRCC1","LIG1","LIG3","LIG4","XRCC4","NHEJ1","DCLRE1C","DNTT",
        "XPC","ERCC2","ERCC3","ERCC4","ERCC5","ERCC6","ERCC8","DDB1","DDB2","CSA","CSB",
        "FANCD2","FANCI","FANCC","FANCA","FANCE","FANCF","FANCG","FANCL","FANCM","PALB2",
        "MGMT","ALKBH2","ALKBH3","MPG","NEIL1","NEIL2","NEIL3","OGG1","MBD4","SMUG1","UNG",
        "PCNA","RFC1","RFC2","RFC3","RFC4","RFC5","RPA1","RPA2","RPA3",
        "POLH","POLI","POLK","POLL","POLM","POLN","POLQ","POLD1","POLE",
        "H2AFX","MDC1","53BP1","RIF1","PTIP","RBBP8","CtIP","EXOSC",
    ],
    "Cell Cycle Process": [
        "CDK1","CDK2","CDK4","CDK6","CCNA1","CCNA2","CCNB1","CCNB2","CCND1","CCND2","CCND3",
        "CCNE1","CCNE2","CCNH","RB1","E2F1","E2F2","E2F3","CDKN1A","CDKN1B","CDKN2A",
        "CDC20","CDC25A","CDC25B","CDC25C","BUB1","BUB1B","BUB3","MAD1L1","MAD2L1",
        "PLK1","PLK2","PLK3","AURKA","AURKB","INCENP","SURVIVIN","BOREALIN",
        "WEE1","MYT1","PKMYT1","CHEK1","CHEK2","ATM","ATR","MKI67","TOP2A",
        "MCM2","MCM3","MCM4","MCM5","MCM6","MCM7","CDC6","CDC7","CDT1","ORC1",
        "PTTG1","PTTG2","ESPL1","RAD21","STAG1","STAG2","SMC1A","SMC3","REC8",
        "NDE1","NDEL1","CENPF","CENPA","CENPB","CENPE","CENPH","KIF11","KIF23","KIF14",
        "TP53","MDM2","SKP2","FBXW7","RBX1","GADD45A","PCNA","POLA1","POLD1","POLE",
    ],
    "Transcription Regulation": [
        "TP53","TP63","TP73","MYC","MYCN","MYCL","JUN","FOS","FOSL1","FOSL2",
        "SP1","SP2","SP3","KLF4","KLF5","KLF6","ETS1","ETS2","ERG","FLI1","ETV4","ETV5",
        "NFKB1","RELA","NFKB2","RELB","REL","CREB1","ATF1","ATF2","ATF3","ATF4",
        "YAP1","TAZ","WWTR1","TEAD1","TEAD2","TEAD3","TEAD4",
        "RUNX1","RUNX2","RUNX3","GATA1","GATA2","GATA3","GATA4","GATA6",
        "PPARG","RXRA","RARA","RARB","ESR1","AR","NR3C1","NR2F2","NR4A1",
        "TWIST1","TWIST2","SNAI1","SNAI2","ZEB1","ZEB2","HMGA1","HMGA2",
        "E2F1","E2F2","E2F3","TFDP1","TFDP2","RBX1","SKP2","MED1",
        "HDAC1","HDAC2","HDAC3","HDAC4","HDAC5","HDAC6","HDAC7","HDAC8","SIRT1","SIRT2",
        "EP300","CREBBP","KAT2A","KAT2B","KAT5","KAT6A","KAT6B","KAT8",
        "EZH2","SUZ12","EED","JARID2","BMI1","RING1","RNF2","CBX2","CBX4","CBX6","CBX7","CBX8",
        "BRD2","BRD3","BRD4","BRD7","BRD9","BRDT","PBRM1","SMARCA4","SMARCB1","SMARCC1",
    ],
    "mRNA Processing / Splicing": [
        "SF3B1","SF3B2","SF3B3","SF3B4","SF3A1","SF3A2","SF3A3",
        "U2AF1","U2AF2","SRSF1","SRSF2","SRSF3","SRSF4","SRSF5","SRSF6","SRSF7","SRSF9","SRSF10",
        "HNRNPA1","HNRNPA2B1","HNRNPC","HNRNPD","HNRNPF","HNRNPH1","HNRNPH2","HNRNPK",
        "HNRNPL","HNRNPM","HNRNPR","HNRNPU","HNRNPAB","PCBP1","PCBP2",
        "PTBP1","PTBP2","PTBP3","RBFOX1","RBFOX2","RBFOX3",
        "SNRNP70","SNRNP200","SNRNPA","SNRNPB","SNRNPC","SNRNPD1","SNRNPD2","SNRNPD3",
        "PRPF3","PRPF4","PRPF6","PRPF8","PRPF18","PRPF19","PRPF31","PRPF38A","PRPF40A",
        "DDX5","DDX17","DDX21","DDX23","DHX8","DHX9","DHX15","DHX16","DHX38",
        "RBM3","RBM5","RBM6","RBM10","RBM17","RBM22","RBM25","RBM27","RBM39",
        "CELF1","CELF2","MBNL1","MBNL2","MBNL3","TRA2A","TRA2B","CLK1","CLK2","CLK3","CLK4",
        "ESRP1","ESRP2","NOVA1","NOVA2","ELAVL1","ELAVL2","ELAVL3","ELAVL4",
    ],
    "Translation / Ribosome Biogenesis": [
        "EIF4E","EIF4EBP1","EIF4EBP2","EIF4G1","EIF4G2","EIF4G3","EIF4A1","EIF4A2",
        "EIF3A","EIF3B","EIF3C","EIF3D","EIF3E","EIF3F","EIF3G","EIF3H","EIF3I","EIF3J",
        "EIF2S1","EIF2S2","EIF2S3","EIF2B1","EIF2B2","EIF2B3","EIF2B4","EIF2B5",
        "EIF1","EIF1AX","EIF5","EIF5B","EIF6","MTOR","RPS6KB1","RPS6KB2",
        "GCN2","HRI","PKR","PERK","EIF2AK1","EIF2AK2","EIF2AK3","EIF2AK4",
        "DDX56","DDX10","DDX21","WDR12","WDR43","WDR75","UTP3","UTP6","UTP14A",
        "RRP1","RRP5","RRP7A","RRP9","RRP12","RRP14","RRP15","RRP36",
        "BYSL","NOB1","NOG1","NOG2","LSG1","NMD3","RBM28","NOC4L","PNO1",
        "RPSA","RPS2","RPS3","RPS5","RPS6","RPS7","RPS8","RPS10","RPS14","RPS19",
        "RPL3","RPL4","RPL5","RPL7","RPL8","RPL10","RPL11","RPL13","RPL14","RPL22",
    ],
    "Angiogenesis": [
        "VEGFA","VEGFB","VEGFC","VEGFD","PGF","KDR","FLT1","FLT4","NRP1","NRP2",
        "ANGPT1","ANGPT2","ANGPT4","TIE1","TEK","TEK2","EphB4","EPHB2",
        "HIF1A","EPAS1","ARNT","EGLN1","EGLN3","VHL","NRP","SEMA3A","SEMA3C","SEMA3F",
        "PDGFRA","PDGFRB","PDGFA","PDGFB","PDGFC","PDGFD",
        "FGF1","FGF2","FGFR1","FGFR2","EGF","EGFR","ERBB2","IGF1","IGF1R",
        "MMP2","MMP9","MMP14","UPAR","PLAUR","PAI1","SERPINE1","TPA","UPA",
        "COL4A1","COL4A2","LAMA4","LAMA5","LAMC1","FN1","VTN","TNC","SPARC",
        "PIK3CA","AKT1","MTOR","MAPK1","MAPK3","SRC","FAK1","PLCγ","PRKCB",
        "NOTCH1","DLL4","JAG1","HEY1","HES1","ETS1","ETV2","TAL1","GATA2",
        "RASA1","RASIP1","ADGRB1","TSP1","TSP2","PEDF","EMIA1","A2M",
    ],
    "Signal Transduction": [
        "EGFR","ERBB2","ERBB3","ERBB4","INSR","IGF1R","PDGFRA","PDGFRB",
        "FGFR1","FGFR2","FGFR3","FGFR4","KIT","RET","MET","AXL","ROR2",
        "SRC","ABL1","ABL2","LCK","FYN","YES1","FGR","LYN","HCK","BLK",
        "JAK1","JAK2","JAK3","TYK2","STAT1","STAT2","STAT3","STAT5A","STAT5B",
        "AKT1","AKT2","AKT3","PIK3CA","PTEN","PDK1","MTOR","TSC1","TSC2","RHEB",
        "KRAS","HRAS","NRAS","BRAF","RAF1","CRAF","MAP2K1","MAP2K2","MAPK1","MAPK3",
        "GRB2","SOS1","SOS2","SHC1","SHC2","SHC3","GAB1","GAB2","IRS1","IRS2",
        "PLCγ1","PLCG2","PLCB1","PLCB2","IP3","DAG","PRKCB","PRKCA",
        "RASA1","NF1","RIN1","TIAM1","SOS1","DOCK1","DOCK2","DOCK3",
        "RAC1","RHOA","RHOB","CDC42","ROCK1","ROCK2","PAK1","PAK2","PAK3",
        "PTPN11","PTPN6","PTPN1","PTPRA","DUSP1","DUSP4","DUSP6","PP2A",
        "EP300","CREBBP","SKP2","FBXW7","KEAP1","NFE2L2","SIAH1","HERC2",
    ],
    "Cell Migration": [
        "ACTB","ACTG1","ACTA1","ACTA2","MYH9","MYH10","MYL6","MYL12A",
        "RHOA","RHOB","RHOC","RND1","RND2","RND3","RAC1","RAC2","CDC42",
        "ROCK1","ROCK2","LIMK1","LIMK2","CFL1","CFL2","SSH1","SSH2","SSH3",
        "DIAPH1","DIAPH2","DIAPH3","FHOD1","VASP","ENAH","ENA","VCL",
        "FAK1","PTK2","PTK2B","SRC","PXN","BCAR1","TLN1","TLN2","VIN",
        "ITGB1","ITGB3","ITGB5","ITGA2","ITGA3","ITGA5","ITGA6","ITGAV",
        "MMP1","MMP2","MMP3","MMP7","MMP9","MMP14","ADAM10","ADAM17",
        "CDH1","CDH2","CDH11","CDH13","NECTIN1","NECTIN2","NECTIN3","NRXN1",
        "CXCR4","CXCL12","CXCR3","CXCL10","CCR7","CCL19","CCL21","ACKR3",
        "HGF","MET","TWIST1","SNAI1","SNAI2","ZEB1","ZEB2","VIM","FN1",
        "SEMA3A","NRP1","PLXNA1","PLXNA2","SEMA4D","PLXNB1",
        "EZR","RDX","MSN","CD44","CD168","CLN6","CLIC4","PARVA",
    ],
    "Oxidative Stress Response": [
        "SOD1","SOD2","SOD3","CAT","GPX1","GPX2","GPX3","GPX4","GPX5","GPX6","GPX7","GPX8",
        "PRDX1","PRDX2","PRDX3","PRDX4","PRDX5","PRDX6",
        "TXNRD1","TXNRD2","TXNRD3","TXN1","TXN2","GLRX1","GLRX2",
        "HMOX1","HMOX2","FTL","FTH1","SLC40A1","HERC2","KEAP1","NFE2L2","NRF2",
        "NQO1","NQO2","GCLC","GCLM","GSS","GGT1","GGT5","GGT6","GGT7",
        "GSTP1","GSTM1","GSTM2","GSTM3","GSTA1","GSTA2","GSTA4","GSTT1",
        "MT1A","MT1B","MT1E","MT1F","MT1G","MT1H","MT1M","MT1X","MT2A","MT3",
        "PARK7","PINK1","FBXO7","HSPB1","HSPB8","DNAJB1","DNAJB2","HSPA1A",
        "FOXO1","FOXO3","FOXO4","SESN1","SESN2","SESTRIN3","AIMP2","CSTB",
        "ATM","RAD51","BRCA1","MUTYH","OGG1","NTHL1","NEIL1","NEIL2",
        "CYBB","NCF1","NCF2","NCF4","RAC1","CYBA","NOX1","NOX2","NOX3","NOX4","DUOX1","DUOX2",
        "XDH","AOX1","MPO","PTGS2","LOX","ALOX5","ALOX12","ALOX15",
    ],
    "Metabolic Process": [
        "HK1","HK2","GPI","PFKL","PFKM","PFKP","ALDOA","GAPDH","PGK1","ENO1","PKM","LDHA",
        "FASN","ACACA","ACLY","CS","IDH1","IDH2","SDHA","FH","MDH2",
        "G6PD","PGD","TKT","TALDO1","RPIA","RPE",
        "GLUD1","GLS","GLS2","ASNS","PHGDH","PSAT1","PSPH","SHMT1","SHMT2",
        "ACSL1","CPT1A","HADHA","ACADM","ACAA2","HMGCR","HMGCS1","SQLE","LSS",
        "PPARG","PPARA","RXRA","FASN","SCD","ELOVL1","ELOVL5","FADS1","FADS2",
        "UGT1A1","CYP3A4","CYP2D6","CYP2C9","GSTA1","GSTP1","NAT2","TPMT",
        "MAT1A","AHCY","CBS","CTH","ODC1","AMD1","SAT1","SMOX",
        "MTHFR","MTR","MTRR","DHFR","TYMS","GART","ATIC","PAICS","ADSL","ADSLT",
        "UMPS","CAD","DHODH","DPYD","TYMP","UCK1","UCK2","CMPK1","CMPK2",
        "PRPS1","PRPS2","PPAT","PFAS","GMPS","IMPDH1","IMPDH2","ADSS","ADSL",
    ],
    "Epigenetic Regulation": [
        "EZH1","EZH2","SUZ12","EED","JARID2","RBBP4","RBBP7","MTF2","PHF1","PHF19",
        "BMI1","RING1","RNF2","CBX2","CBX4","CBX6","CBX7","CBX8","PHC1","PHC2","PHC3",
        "KDM1A","KDM1B","KDM2A","KDM2B","KDM3A","KDM3B","KDM4A","KDM4B","KDM4C","KDM4D",
        "KDM5A","KDM5B","KDM5C","KDM5D","KDM6A","KDM6B","KDM7A","UTX","JMJD3",
        "NSD1","NSD2","NSD3","SETD2","SETD7","SETDB1","SETDB2","SMYD2","SMYD3",
        "KMT2A","KMT2B","KMT2C","KMT2D","KMT2E","KMT2F","DOT1L","EHMT1","EHMT2",
        "DNMT1","DNMT3A","DNMT3B","DNMT3L","TET1","TET2","TET3","NTMT1","ALKBH5","FTO",
        "HDAC1","HDAC2","HDAC3","HDAC4","HDAC5","HDAC6","HDAC7","HDAC8","HDAC9","HDAC10","HDAC11",
        "SIRT1","SIRT2","SIRT3","SIRT4","SIRT5","SIRT6","SIRT7",
        "EP300","CREBBP","KAT2A","KAT2B","KAT5","KAT6A","KAT6B","KAT8","KAT14",
        "BRD2","BRD3","BRD4","BRD7","BRD9","BRDT","PBRM1",
        "SMARCA4","SMARCB1","SMARCC1","SMARCC2","SMARCD1","SMARCE1","ARID1A","ARID1B","ARID2",
        "CHD1","CHD2","CHD3","CHD4","CHD5","CHD6","CHD7","CHD8","CHD9",
        "ATRX","DAXX","HIRA","ASF1A","ASF1B","CAF1A","CABIN1","UBN1",
    ],
    "p53 Pathway": [
        "TP53","TP63","TP73","MDM2","MDM4","CDKN1A","CDKN2A","CDK2","CDK4","CDK6",
        "CCNE1","CCND1","RB1","E2F1","ATM","ATR","CHEK1","CHEK2",
        "GADD45A","GADD45B","GADD45G","BAX","BBC3","PMAIP1","FAS","PERP",
        "APAF1","CASP3","CASP9","CYCS","PARP1","XIAP","BCL2","BCL2L1",
        "SESN1","SESN2","SESN3","TIGAR","SCO2","GLS2","DRAM1","DRAM2",
        "DDB2","XPC","POLK","POLH","POLI","RRM2","RRM2B","AEN","LIF",
        "ZMAT3","CYCLIN_G1","BTG2","REPRIMO","PIDD1","REDD1","REDD2",
        "PTEN","SFN","BRCA1","BRCA2","CABLES1","WWOX","HIC1","ASPP1","ASPP2",
        "MDM2","PIRH2","COP1","PJA2","HAUSP","USP7","HDMX","BCL3",
        "P53AIP1","APRIN","HIPK2","HIPK3","DAPK1","VHL","ARF",
    ],
    "Ubiquitin Proteasome": [
        "UBA1","UBA2","UBA3","UBA6","UBA7","UBB","UBC","UBD",
        "UBE2A","UBE2B","UBE2C","UBE2D1","UBE2D2","UBE2D3","UBE2E1",
        "UBE2G1","UBE2G2","UBE2I","UBE2K","UBE2L3","UBE2M","UBE2N",
        "UBE2Q1","UBE2R1","UBE2S","UBE2T","UBE2W","CDC34","BIRC6",
        "CUL1","CUL2","CUL3","CUL4A","CUL4B","CUL5","CUL7","CUL9",
        "RBX1","RBX2","SKP1","SKP2","FBXW7","FBXO6","FBXO11","FBXL2","BTRC",
        "VHL","SPOP","KEAP1","HERC1","HERC2","NEDD4","NEDD4L","ITCH","WWP1","WWP2",
        "MDM2","BRCA1","RING1","RNF2","RNF8","RNF168","TRAF2","TRAF6",
        "USP7","USP14","USP10","USP15","USP16","USP22","USP28","USP39",
        "OTUB1","OTUD5","OTUD7B","BAP1","BRCC3","CYLD","A20","OTULIN",
        "PSMA1","PSMA2","PSMA3","PSMA4","PSMA5","PSMA6","PSMA7",
        "PSMB1","PSMB2","PSMB3","PSMB4","PSMB5","PSMB6","PSMB7","PSMB8","PSMB9","PSMB10",
        "PSMC1","PSMC2","PSMC3","PSMC4","PSMC5","PSMC6",
        "PSMD1","PSMD2","PSMD3","PSMD4","PSMD6","PSMD7","PSMD11","PSMD12","PSMD14",
    ],
    "Autophagy Regulation": [
        "ULK1","ULK2","ULK3","ULK4","ATG13","ATG101","RB1CC1","AMBRA1",
        "PIK3C3","PIK3R4","BECN1","BECN2","UVRAG","RUBCN","SH3GLB1","ATG14",
        "ATG2A","ATG2B","WIPI1","WIPI2","ATG9A","ATG9B","VMP1",
        "ATG3","ATG5","ATG7","ATG10","ATG12","ATG16L1","ATG16L2",
        "MAP1LC3A","MAP1LC3B","MAP1LC3C","GABARAP","GABARAPL1","GABARAPL2",
        "SQSTM1","NBR1","OPTN","NDP52","TAX1BP1","CALCOCO2","TOLLIP",
        "RAB7A","RAB11A","RAB12","PLEKHM1","LAMP1","LAMP2","RAB9A","RAB24",
        "TFEB","TFE3","MITF","ZKSCAN3","FOXO1","FOXO3",
        "MTOR","RPTOR","AKT1","PTEN","TSC1","TSC2","RHEB","AMPK",
        "PRKAA1","PRKAA2","PRKAB1","PRKAB2","PRKAG1","PRKAG2","PRKAG3",
        "PINK1","PRKN","MFN1","MFN2","DNM1L","OPA1","PHB2","MUL1","MARCH5",
        "BNIP3","BNIP3L","FUNDC1","FUNDC2","FKBP8","NIPSNAP1","NIPSNAP2",
        "TP53","DAPK1","DAPK2","ATG4A","ATG4B","ATG4C","ATG4D",
    ],
    "Cell Senescence": [
        "TP53","RB1","RBL1","RBL2","CDKN1A","CDKN2A","CDKN2B","MDM2","E2F1",
        "IL6","CXCL8","IL1A","IL1B","CCL2","CCL3","CCL5","VEGFA","GDF15",
        "MMP1","MMP3","MMP9","MMP10","MMP13","PAI1","SERPINE1","IGFBP3",
        "HMGA1","HMGA2","LMNB1","LMNB2","LMNA","SUN1","SUN2","BANF1",
        "SIRT1","SIRT6","EP300","CBP","BMI1","EZH2","ARID1A",
        "ATM","ATR","CHEK1","CHEK2","GADD45A","H2AFX","MDC1","53BP1","RAD51",
        "TERT","DKC1","TERC","POT1","TRF1","TRF2","TIN2","TPP1","RAP1",
        "NF1","PTEN","VHL","RAS","BRAF","MEK1","ERK1","ETS1","ETS2",
        "MAPK14","P38","MKK3","MKK6","MK2","HSP27","HSPB1",
        "IRF1","IRF3","STAT1","STAT3","NF-κB","STING","CGAS","TBK1",
        "SASP","GDF15","WNT3A","DKK1","NOTCH1","BMP4","IL10","TGFB1",
    ],
    "Chromatin Organization": [
        "SMARCA4","SMARCB1","SMARCC1","SMARCC2","SMARCD1","SMARCD2","SMARCD3","SMARCE1",
        "ARID1A","ARID1B","ARID2","PBRM1","BRD7","BRD9","BCL7A","BCL7B","BCL7C",
        "CHD3","CHD4","MBD2","MBD3","HDAC1","HDAC2","RBBP4","RBBP7","MTA1","MTA2","MTA3",
        "CHD1","CHD2","CHD5","CHD6","CHD7","CHD8","CHD9",
        "ATRX","DAXX","HIRA","ASF1A","ASF1B","CAF1A","CAF1B","NAP1L1","CABIN1",
        "H3F3A","H3F3B","HIST1H3A","HIST2H3A","H1F0","HIST1H1A","HIST1H2AA",
        "EZH2","EZH1","SUZ12","EED","JARID2","RBBP4","RBBP7","MTF2","PHF1","PHF19",
        "KDM1A","KDM6A","KDM6B","SETD2","SETDB1","DOT1L","KMT2A","KMT2B","KMT2D",
        "EP300","CREBBP","KAT2A","KAT2B","KAT5","BRD4","BRD3","BRD2",
        "DNMT1","DNMT3A","DNMT3B","TET1","TET2","TET3","CTCF","CTCFL","WAPL",
        "SMC1A","SMC3","RAD21","STAG1","STAG2","SMC2","SMC4","NCAPD2","NCAPG",
        "TOPBP1","TOP2A","TOP2B","TOP1","MRE11","NBS1","RAD50",
    ],
    "RNA Processing": [
        "DICER1","DROSHA","DGCR8","XPO5","AGO1","AGO2","AGO3","AGO4",
        "TARBP2","PACT","TNRC6A","TNRC6B","TNRC6C","GW182","MOV10",
        "XRN1","XRN2","DCP1A","DCP1B","DCP2","LSM1","LSM7","EDC4",
        "EXOSC3","EXOSC4","EXOSC5","EXOSC6","EXOSC7","EXOSC8","EXOSC9",
        "PAPOLA","PAPOLB","PAPOLG","PABPN1","PABPC1","PABPC4","NUDT21",
        "RNMT","CMTR1","CMTR2","NSUN2","METTL3","METTL14","WTAP","FTO","ALKBH5",
        "CSTF1","CSTF2","CSTF3","CLP1","SYMPK","PCF11","FIP1L1",
        "TUT1","TUT4","TUT7","ZCCHC6","ZCCHC11","ANGEL1","ANGEL2",
        "HNRNPA1","HNRNPC","HNRNPD","HNRNPH1","HNRNPK","PTBP1","ELAVL1",
        "DDX6","DDX3X","DDX3Y","DDX21","DDX41","DHX9","DHX36","DHX58",
        "U1","U2","U4","U5","U6","SNRNP70","PRPF8","PRPF19",
    ],
    "Chromosome Segregation": [
        "BUB1","BUB1B","BUB3","MAD1L1","MAD2L1","MAD2L2","BUBR1","KNSTC1",
        "CENPA","CENPB","CENPC1","CENPF","CENPH","CENPI","CENPJ","CENPK",
        "CENPL","CENPM","CENPN","CENPO","CENPP","CENPQ","CENPR","CENPS","CENPT","CENPU",
        "CENPW","CENPX","NDC80","NUF2","SPC24","SPC25","ZWINT","MIS12","KNL1",
        "DSN1","NSL1","PMF1","MIS18A","MIS18B","MIS18BP1","HJURP",
        "KIF11","KIF14","KIF18A","KIF18B","KIF20A","KIF23","KIF2A","KIF2B","KIF2C",
        "NDE1","NDEL1","NUMA1","DYNLL1","DYNLT1","DYNEIN",
        "PLK1","PLK4","AURKA","AURKB","INCENP","SURVIVIN","BOREALIN","HASPIN",
        "ESPL1","RAD21","SMC1A","SMC3","STAG1","STAG2","SA1","SA2",
        "PTTG1","PTTG2","SECURIN","CDCA5","SORORIN","WAPL","PDS5A","PDS5B",
        "ATM","ATR","CHEK1","CHEK2","TP53","53BP1","H2AFX","MDC1",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Reactome — 30 pathways, 50–130 genes each
# ─────────────────────────────────────────────────────────────────────────────
_REACTOME: Dict[str, List[str]] = {
    "Cell Cycle": [
        "CDK1","CDK2","CDK4","CDK6","CDK7","CCNA1","CCNA2","CCNB1","CCNB2","CCND1",
        "CCND2","CCND3","CCNE1","CCNE2","CCNH","RB1","RBL1","RBL2","E2F1","E2F2","E2F3",
        "CDKN1A","CDKN1B","CDKN2A","CDC20","CDC25A","CDC25B","CDC25C",
        "BUB1","BUB1B","BUB3","MAD1L1","MAD2L1","PTTG1","ESPL1",
        "PLK1","AURKA","AURKB","MKI67","TOP2A","PCNA",
        "MCM2","MCM3","MCM4","MCM5","MCM6","MCM7","CDC6","CDC7","CDT1",
        "ATM","ATR","CHEK1","CHEK2","TP53","MDM2","GADD45A","SFN","FBXW7",
        "WEE1","MYT1","PKMYT1","SMC1A","SMC3","RAD21","STAG1","STAG2",
    ],
    "Apoptosis": [
        "BAX","BAK1","BCL2","BCL2L1","BCL2L2","BCL2L11","MCL1","PMAIP1","BBC3",
        "TP53","FAS","FASLG","TNF","TNFRSF1A","TNFRSF10A","TNFRSF10B",
        "TRADD","FADD","RIPK1","CASP3","CASP6","CASP7","CASP8","CASP9","CASP10",
        "CYCS","APAF1","DIABLO","SMAC","HTRA2","PARP1","XIAP","BIRC2","BIRC3","BIRC5",
        "CFLAR","AKT1","PIK3CA","MAPK8","MAPK9","NFKB1","RELA","TRAF2",
        "LMNA","DFFA","DFFB","ENDOG","AIFM1","SPTAN1","ACIN1",
        "BID","NLRP3","PYCARD","CASP1","RIPK3","MLKL","HMGB1","DAPK1",
    ],
    "DNA Repair": [
        "BRCA1","BRCA2","PALB2","RAD51","RAD51B","RAD51C","RAD51D","XRCC2","XRCC3",
        "ATM","ATR","CHEK1","CHEK2","NBN","MRE11","RAD50","MDC1","H2AFX","53BP1",
        "MLH1","MSH2","MSH3","MSH6","PMS1","PMS2","MLH3","EXONUCLEASE1",
        "PARP1","PARP2","XRCC1","LIG1","LIG3","LIG4","XRCC4","NHEJ1","DCLRE1C",
        "XPC","ERCC2","ERCC3","ERCC4","ERCC5","ERCC6","ERCC8","DDB2","CSA",
        "FANCD2","FANCI","FANCC","FANCA","FANCE","FANCF","FANCG","FANCL","FANCM",
        "MGMT","OGG1","NEIL1","NEIL2","UNG","SMUG1","MPG","MUTYH",
        "PCNA","RFC1","RPA1","RPA2","POLH","POLK","REV1","POLM","POLQ",
    ],
    "Immune System": [
        "IFNG","IFNA1","IFNB1","IFNGR1","IFNGR2","IFNAR1","IFNAR2",
        "JAK1","JAK2","JAK3","TYK2","STAT1","STAT2","STAT3","STAT4","STAT5A","STAT5B",
        "IRF1","IRF3","IRF7","IRF9","MX1","OAS1","ISG15","ISG20","IFIT1","IFIT2",
        "CD4","CD8A","CD8B","CD3D","CD3E","CD3G","CD247","ZAP70","LCK","FYN",
        "NFKB1","RELA","NFKB2","RELB","IKBKA","IKBKB","NFKBIA","TRAF2","TRAF6",
        "MYD88","IRAK1","IRAK4","TIRAP","TRIF","TLR4","TLR9","NLRP3",
        "CD274","PDCD1","LAG3","HAVCR2","TIGIT","CTLA4","CD28","CD80","CD86",
        "GZMB","PRF1","NKG2D","NKG7","EOMES","TBX21","FOXP3","IL2","IL10","IL12A",
        "TNF","IL1B","IL6","IL18","CXCL10","CCL2","CCL5","CXCL8",
        "HLA-A","HLA-B","HLA-C","HLA-DRA","HLA-DRB1","HLA-DQA1","HLA-DPB1","B2M",
    ],
    "Signal Transduction": [
        "EGFR","ERBB2","ERBB3","ERBB4","INSR","IGF1R","PDGFRA","PDGFRB","FGFR1","FGFR2",
        "KIT","RET","MET","AXL","SRC","ABL1","LCK","FYN","YES1","LYN",
        "PIK3CA","PIK3CB","PIK3CD","PIK3R1","PTEN","AKT1","AKT2","AKT3","PDK1","MTOR",
        "KRAS","HRAS","NRAS","BRAF","RAF1","MAP2K1","MAP2K2","MAPK1","MAPK3",
        "JAK1","JAK2","JAK3","TYK2","STAT1","STAT3","STAT5A","STAT5B","SOCS1","SOCS3",
        "NFKB1","RELA","IKBKA","IKBKB","NFKBIA","TRAF6","TAK1","TAB1",
        "GRB2","SOS1","SHC1","GAB1","IRS1","IRS2","SHP2","PTPN11",
        "RAC1","RHOA","CDC42","ROCK1","ROCK2","PAK1","PAK2","CFL1","LIMK1",
        "WNT3A","FZD1","LRP6","DVL1","AXIN1","APC","GSK3B","CTNNB1","LEF1",
        "NOTCH1","DLL4","JAG1","RBPJ","HES1","SHH","PTCH1","SMO","GLI1","GLI2",
        "TGFB1","TGFBR1","TGFBR2","SMAD2","SMAD3","SMAD4","BMP4","BMPR1A","SMAD1",
    ],
    "Metabolism": [
        "HK1","HK2","GPI","PFKL","PFKM","ALDOA","GAPDH","PGK1","ENO1","PKM","LDHA",
        "FASN","ACACA","ACACB","ACLY","CS","IDH1","IDH2","OGDH","SDHA","FH","MDH2",
        "G6PD","PGD","TKT","TALDO1","GLUD1","GLS","GLS2","ASNS","PHGDH","PSAT1",
        "ACSL1","CPT1A","CPT2","HADHA","HADHB","ACADM","ACADL","ACADVL","ACAA2",
        "HMGCR","HMGCS1","HMGCS2","SQLE","LSS","CYP51A1","DHCR7","DHCR24",
        "PPARA","PPARG","PPARD","RXRA","NR1H3","ABCA1","ABCG1","ABCG5","ABCG8",
        "UGT1A1","CYP3A4","CYP2C9","GSTA1","GSTP1","NAT2","TPMT","COMT","MAOA",
        "MTHFR","MTR","DHFR","TYMS","GART","IMPDH1","IMPDH2","PRPS1","PPAT",
        "PDK1","PDK2","PDK3","PDK4","PDHB","PDHA1","PC","PCK1","PCK2","G6PC","FBP1",
    ],
    "Gene Expression": [
        "TP53","MYC","MYCN","MYCL","JUN","FOS","FOSL1","SP1","KLF4","KLF5","ETS1","ETS2",
        "NFKB1","RELA","CREB1","ATF1","ATF2","ATF3","ATF4","YAP1","TAZ","TEAD1","TEAD4",
        "RUNX1","RUNX2","RUNX3","GATA3","GATA6","PPARG","RXRA","ESR1","AR","NR3C1",
        "HDAC1","HDAC2","HDAC3","SIRT1","EZH2","DNMT3A","KDM6A","SETD2","BRD4",
        "EP300","CREBBP","KAT2A","MED1","MED12","MED14","CDK8","CDK19",
        "RNA_Pol_II","TBP","TAF1","TAF4","TAF6","TAF7","GTF2B","GTF2F1","GTF2H1",
        "TWIST1","SNAI1","ZEB1","ZEB2","HMGA1","HMGA2","ID1","ID2","ID3","ID4",
        "ESRP1","ESRP2","SRSF1","PTBP1","RBFOX2","ELAVL1","HNRNPA1",
    ],
    "Extracellular Matrix Organisation": [
        "COL1A1","COL1A2","COL2A1","COL3A1","COL4A1","COL4A2","COL5A1","COL5A2","COL5A3",
        "COL6A1","COL6A2","COL6A3","COL7A1","COL8A1","COL9A1","COL10A1","COL11A1","COL12A1",
        "COL14A1","COL15A1","COL16A1","COL17A1","COL18A1","COL19A1","COL20A1",
        "FN1","VTN","TNC","THBS1","THBS2","THBS3","THBS4","POSTN","SPARC","SPARCL1","CTGF",
        "LAMA1","LAMA2","LAMA3","LAMA4","LAMA5","LAMB1","LAMB2","LAMB3","LAMC1","LAMC2","LAMC3",
        "NID1","NID2","HSPG2","AGRN","HAPLN1","HAPLN3","ACAN","VCAN","BCAN","CSPG4",
        "MMP1","MMP2","MMP3","MMP7","MMP8","MMP9","MMP10","MMP11","MMP12","MMP13","MMP14",
        "MMP15","MMP16","MMP17","MMP19","MMP20","MMP21","MMP23A","MMP24","MMP25","MMP26",
        "ADAM10","ADAM12","ADAM17","ADAM28","ADAMTS1","ADAMTS4","ADAMTS5","ADAMTS12",
        "TIMP1","TIMP2","TIMP3","TIMP4","LTBP1","LTBP2","LTBP3","LTBP4",
        "ITGA1","ITGA2","ITGA3","ITGA4","ITGA5","ITGA6","ITGA7","ITGA8","ITGA9",
        "ITGAV","ITGAX","ITGB1","ITGB2","ITGB3","ITGB4","ITGB5","ITGB6","ITGB7",
    ],
    "Developmental Biology": [
        "WNT1","WNT2","WNT3A","WNT4","WNT5A","WNT5B","WNT7A","WNT10B","WNT11","WNT16",
        "FZD1","FZD2","FZD3","FZD4","FZD5","FZD7","FZD8","FZD9","LRP5","LRP6",
        "AXIN1","AXIN2","APC","GSK3B","CTNNB1","TCF7L2","LEF1","MYC","CCND1","MMP7",
        "SHH","IHH","DHH","PTCH1","PTCH2","SMO","GLI1","GLI2","GLI3","SUFU",
        "NOTCH1","NOTCH2","NOTCH3","NOTCH4","DLL1","DLL3","DLL4","JAG1","JAG2",
        "RBPJ","HES1","HES5","HEY1","HEY2","HEYL","MAML1","MAML2","MAML3",
        "TGFB1","TGFB2","TGFB3","BMP2","BMP4","BMP7","NODAL","LEFTY1","LEFTY2",
        "SMAD1","SMAD2","SMAD3","SMAD4","SMAD5","SMAD9","ID1","ID2","ID3","ID4",
        "FGF1","FGF2","FGF4","FGF8","FGF10","FGFR1","FGFR2","FGFR3","FGFR4",
        "EGFR","ERBB2","EGF","HGF","MET","NGF","NTRK1","BDNF","NTRK2","NTF3","NTRK3",
        "RUNX1","RUNX2","GATA1","GATA2","GATA3","PAX6","PAX7","SOX2","SOX9","SOX17",
        "NKX2-1","NKX2-5","TBX1","TBX5","TBX20","HAND1","HAND2","MEF2A","MEF2C",
    ],
    "PI3K/AKT Activation": [
        "PIK3CA","PIK3CB","PIK3CD","PIK3CG","PIK3R1","PIK3R2","PIK3R3","PIK3R4","PIK3R5",
        "PTEN","PDPK1","AKT1","AKT2","AKT3","MTOR","RPTOR","RICTOR","MLST8","DEPTOR",
        "TSC1","TSC2","RHEB","RRAGB","RRAGC","RPS6KB1","RPS6KB2","EIF4EBP1","EIF4EBP2",
        "GSK3A","GSK3B","FOXO1","FOXO3","FOXO4","MDM2","TP53","BCL2","BCL2L1","MCL1","BAD",
        "CDKN1A","CDKN1B","CDKN2A","CCND1","CCND2","CDK4","CDK6","RB1","E2F1",
        "IRS1","IRS2","INSR","IGF1R","EGFR","ERBB2","PDGFRA","PDGFRB","KIT","FLT3",
        "SRC","GRB2","SHC1","GAB1","GAB2","SOS1","KRAS","HRAS","RAF1","MAP2K1","MAPK1",
        "ITGA1","ITGA5","ITGAV","ITGB1","ITGB3","FAK1","PTK2","ILK","PINCH1","PARVA",
        "SGK1","SGK3","PHLPP1","PHLPP2","PP2A","PPP2CA","THEM4",
        "PRKCA","PRKCB","PRKCD","PRKCΕ","PKCA","PKC","PKD1","PKD2","PDK2",
    ],
    "VEGF Ligand-Receptor Interactions": [
        "VEGFA","VEGFB","VEGFC","VEGFD","PGF","KDR","FLT1","FLT4","NRP1","NRP2",
        "ANGPT1","ANGPT2","TIE1","TEK","PDGFRA","PDGFRB","PDGFA","PDGFB","PDGFC","PDGFD",
        "FGF1","FGF2","FGF4","FGF7","FGF10","FGFR1","FGFR2","FGFR3",
        "HIF1A","EPAS1","ARNT","EGLN1","EGLN3","VHL","HMOX1",
        "PIK3CA","AKT1","MTOR","SRC","MAPK1","MAPK3","PLCγ1","PRKCB",
        "NOS3","NO","sGC","PRKG1","PDE5A","MMP2","MMP9","MMP14",
        "RAC1","RHOA","ROCK1","PAK1","PAK2","CFL1","LIMK1","VASP","VCL",
        "ANGPTL1","ANGPTL2","ANGPTL3","ANGPTL4","ANGPTL6","ANGPTL7",
        "DLL4","NOTCH1","NOTCH4","HEY1","HES1","ETS1","ETV2","ERG","FLI1",
    ],
    "MAPK Signaling": [
        "HRAS","KRAS","NRAS","ARAF","BRAF","RAF1","MAP3K1","MAP3K2","MAP3K3",
        "MAP3K4","MAP3K5","MAP3K7","MAP3K8","MAP2K1","MAP2K2","MAP2K3","MAP2K4",
        "MAP2K5","MAP2K6","MAP2K7","MAPK1","MAPK3","MAPK7","MAPK8","MAPK9","MAPK10",
        "MAPK11","MAPK12","MAPK13","MAPK14","RPS6KA1","RPS6KA2","RPS6KA3","RPS6KA5",
        "EGF","EGFR","ERBB2","FGF2","FGFR1","NGF","NTRK1","PDGFRA",
        "MYC","JUN","FOS","ELK1","SRF","ATF2","SP1","MEF2C","DDIT3",
        "DUSP1","DUSP4","DUSP6","DUSP8","DUSP10","DUSP16",
        "TRAF2","TRAF6","TAB1","GRB2","SHC1","GAB1","SOS1",
        "PAK1","RAC1","CDC42","FLNB","STMN1","NF1","RASA1","RASSF1",
    ],
    "TGF-β Receptor Signaling": [
        "TGFB1","TGFB2","TGFB3","TGFBR1","TGFBR2","TGFBR3",
        "ACVR1","ACVR1B","ACVR1C","ACVR2A","ACVR2B","BMPR1A","BMPR1B","BMPR2",
        "SMAD1","SMAD2","SMAD3","SMAD4","SMAD5","SMAD6","SMAD7","SMAD9",
        "SMURF1","SMURF2","SKI","SKIL","TGIF1","TGIF2","SIK","LEMD3",
        "BMP2","BMP4","BMP5","BMP6","BMP7","GDF1","GDF5","GDF8","NODAL","LEFTY1","LEFTY2",
        "INHBA","INHBB","AMH","GDF11","MSTN",
        "LTBP1","LTBP2","LTBP3","LTBP4","THBS1","THBS2","CD109","NRROS",
        "ID1","ID2","ID3","ID4","MYC","JUN","SP1","RUNX2","RUNX3","JUNB","SNAI1","SNAI2",
        "BAMBI","FKBP1A","NEDD4L","PPM1A","SARA","APC","EP300","CREBBP",
    ],
    "Cytokine Signaling in Immune System": [
        "IL6","IL2","IL10","IL12A","IL12B","IL15","IL21","IL23A","IL27","IL35",
        "IL6R","IL6ST","IL2RA","IL2RB","IL2RG","IL10RA","IL10RB","IL12RB1","IL12RB2",
        "IFNG","IFNA1","IFNB1","IFNGR1","IFNGR2","IFNAR1","IFNAR2",
        "TNF","TNFRSF1A","TNFRSF1B","LTA","LTB","LTBR",
        "JAK1","JAK2","JAK3","TYK2","STAT1","STAT2","STAT3","STAT4","STAT5A","STAT5B","STAT6",
        "SOCS1","SOCS2","SOCS3","SOCS4","SOCS5","SOCS6","PIAS1","PIAS3","PIAS4",
        "IRF1","IRF3","IRF5","IRF7","IRF8","IRF9","MX1","OAS1","IFIT1","IFIT2","ISG15",
        "NFKB1","RELA","NFKB2","RELB","REL","IKBKA","IKBKB","NFKBIA","TRAF2","TRAF6",
        "IL4","IL4R","IL13","IL13RA1","IL5","IL5RA","IL3","IL3RA","CSF2RA","CSF2RB",
        "GHR","PRLR","LEPR","THPO","MPL","EPOR","CSF3R","CSF1R","KIT","FLT3",
        "BCL2","BCL2L1","MCL1","BAD","CDKN1A","CDKN1B","CCND1","CDK4","MYC",
    ],
    "Chromatin Organization": [
        "SMARCA4","SMARCB1","SMARCC1","SMARCC2","SMARCD1","SMARCE1","ARID1A","ARID1B","ARID2","PBRM1",
        "EZH2","EZH1","SUZ12","EED","JARID2","BMI1","RING1","RNF2","CBX7","CBX8",
        "KDM1A","KDM1B","KDM2A","KDM3A","KDM4A","KDM4B","KDM4C","KDM5A","KDM5B","KDM5C","KDM6A","KDM6B",
        "SETD2","SETDB1","SETDB2","DOT1L","NSD1","NSD2","NSD3","SMYD2","SMYD3",
        "KMT2A","KMT2B","KMT2C","KMT2D","EHMT1","EHMT2","KAT6A","KAT6B",
        "DNMT1","DNMT3A","DNMT3B","DNMT3L","TET1","TET2","TET3","MBD2","MBD3",
        "HDAC1","HDAC2","HDAC3","HDAC4","HDAC5","HDAC6","SIRT1","SIRT2","SIRT3","SIRT6",
        "EP300","CREBBP","KAT2A","KAT2B","KAT5","BRD4","BRD3","BRD2","BRD7","BRD9",
        "CHD1","CHD3","CHD4","CHD7","CHD8","ATRX","DAXX","HIRA","ASF1A","ASF1B",
        "CTCF","CTCFL","SMC1A","SMC3","RAD21","STAG1","STAG2","WAPL","PDS5A","PDS5B",
        "TOP2A","TOP2B","TOP1","TOPBP1","RBBP4","RBBP7","MTA1","MTA2","NuRD",
    ],
    "RNA Metabolism": [
        "SF3B1","U2AF1","SRSF2","SRSF1","HNRNPA1","HNRNPC","PTBP1","DDX5","DHX9",
        "DICER1","DROSHA","DGCR8","XPO5","AGO1","AGO2","TNRC6A","GW182",
        "XRN1","XRN2","DCP1A","DCP2","LSM1","LSM4","EXOSC3","EXOSC9","DIS3","EXOSC10",
        "PAPOLA","PABPN1","PABPC1","RNMT","CMTR1","NSUN2","METTL3","METTL14","WTAP","FTO",
        "STAU1","STAU2","ELAVL1","ELAVL2","ELAVL3","ZFP36","ZFP36L1","ZFP36L2",
        "DHX36","DDX3X","DDX21","DDX41","DDX56","NXF1","NXT1","NXF2","NXT2","GANP","THOC1",
        "SMG1","SMG5","SMG6","SMG7","UPF1","UPF2","UPF3A","UPF3B","DHX34","NBAS",
        "CNOT1","CNOT2","CNOT3","CNOT4","CNOT6","CNOT7","CNOT8","CCR4","TOB1","TOB2",
        "RBM3","RBM5","RBM10","RBM17","RBM22","RBM25","CELF1","MBNL1","MBNL2",
    ],
    "Programmed Cell Death": [
        "BAX","BAK1","BCL2","BCL2L1","BCL2L2","BCL2L11","MCL1","PMAIP1","BBC3","BID",
        "CASP3","CASP6","CASP7","CASP8","CASP9","CASP10","CASP1","CASP4","CASP5",
        "CYCS","APAF1","DIABLO","HTRA2","SMAC","PARP1","PARP2","XIAP","BIRC2","BIRC3","BIRC5",
        "FAS","FASLG","TNFRSF1A","TNFRSF10A","TNFRSF10B","TRADD","FADD","RIPK1","CFLAR",
        "PYCARD","NLRP1","NLRP3","NLRP6","NLRC4","NLRP12","AIM2","NAIP",
        "RIPK3","MLKL","PGAM5","FADD","RIP1","MIXED_lineage",
        "CYLD","LUBAC","SHARPIN","HOIL","HOIP","OTULIN","LUBAC",
        "ATG5","BECN1","MAP1LC3A","SQSTM1","BNIP3","DAPK1","DAPK2",
        "TP53","AKT1","NFKB1","MAPK8","MAPK9","MAPK10","TRAF2","TRAF3","TRAF6",
        "AIFM1","ENDOG","DFFA","DFFB","LMNA","SPTAN1","ACIN1",
    ],
    "p53-Dependent G1/S DNA damage checkpoint": [
        "TP53","TP63","TP73","MDM2","MDM4","ATM","ATR","CHEK1","CHEK2","HIPK2","DYRK2",
        "CDKN1A","CDKN2A","CDKN2B","CDK2","CDK4","CDK6","CCNE1","CCNE2","CCND1","CCND2",
        "RB1","RBL1","RBL2","E2F1","E2F2","E2F3","E2F4","TFDP1","TFDP2",
        "GADD45A","GADD45B","GADD45G","PCNA","SFN","14-3-3sigma",
        "PMAIP1","BBC3","BAX","FAS","PERP","BTG2","DDB2","XPC","POLK",
        "PPM1D","WIP1","PIDD1","RAIDD","CASP2","CRADD",
        "BRCA1","H2AFX","MDC1","RNF8","RNF168","53BP1","TOPBP1","CLASPIN",
        "CDC25A","CDC25B","CDC25C","WEE1","MYT1","PKMYT1","PLK1","AURKA",
        "RRM2","RRM2B","GLS2","TIGAR","SCO2","DRAM1","SESN1","SESN2","REDD1",
    ],
    "ESR-mediated signalling": [
        "ESR1","ESR2","GPER1","SP1","SP3","AP1","NFKB1","RELA",
        "ERBB2","ERBB3","SRC","PIK3CA","PIK3R1","AKT1","MTOR","MAPK1","MAPK3",
        "NCOA1","NCOA2","NCOA3","NCOR1","NCOR2","HDAC3","GPS2","TBL1X","TBLR1",
        "EP300","CREBBP","MED1","CDK7","CCNH","TFIIH","POLII","BRD4",
        "PGR","AR","NR3C1","NR2F1","NR2F2","RARA","RXRA","VDR","THR1","LXR",
        "CCND1","CDK4","CDK6","RB1","E2F1","CDKN1B","CDKN1A","MYC",
        "VEGFA","MMP1","MMP2","MMP9","MMP13","BCL2","BCL2L1","MCL1","BAD",
        "CXCL12","CXCR4","IGF1R","IGF1","IGFBP1","IGFBP3","IGFBP5",
        "GATA3","FOXA1","FOXA2","FOXC1","KLF4","TFF1","PGR","XBP1","CCNE1",
    ],
    "Transcriptional Regulation by Small Molecules": [
        "RXRA","RXRB","RXRG","PPARA","PPARG","PPARD","RARA","RARB","RARG",
        "NR3C1","NR3C2","ESR1","ESR2","AR","THRΑ","THRB","VDR","LXRa","LXRb","FXR","PXR","CAR",
        "NCOA1","NCOA2","NCOA3","NCOA4","NCOA6","NCOA7","SRC1","GRIP1","ACTR",
        "NCOR1","NCOR2","HDAC3","HDAC4","HDAC5","HDAC7","GPS2","TBL1X","TBLR1",
        "EP300","CREBBP","BRD4","MED1","CDK8","MEDIATOR",
        "FASN","ACACA","SCD","LPL","APOC2","G6PD","PDK4","UCP1",
        "CYP7A1","CYP7B1","CYP27A1","ABCG5","ABCG8","ABCA1","SHP","FGF15","FGF19",
        "CYP3A4","CYP2B6","CYP2C9","ABCB1","ABCC2","GST","SULT1A1",
        "HMGCR","HMGCS1","LDLR","PCSK9","SOAT1","NPC1","NPC2","SCAP","INSIG1","INSIG2",
    ],
    "Innate Immune System": [
        "TLR1","TLR2","TLR3","TLR4","TLR5","TLR6","TLR7","TLR8","TLR9","TLR10",
        "MYD88","TIRAP","TRIF","TRAM","IRAK1","IRAK2","IRAK4","IRAK3","TRAF3","TRAF6",
        "TAK1","TAB1","TAB2","TAB3","RIPK1","FADD","CASP8",
        "IRF1","IRF3","IRF5","IRF7","IFNA1","IFNB1","IFNG",
        "NFKB1","RELA","IKBKA","IKBKB","IKBKG","MAP3K1","MAP3K7",
        "NLRP1","NLRP3","NLRP6","NLRC4","AIM2","PYCARD","CASP1","IL1B","IL18","GSDMD",
        "CGAS","STING","TBK1","IKKε","IRF3","IRF7","TRIM56","TRIM32","TRIM25",
        "DDX58","IFIH1","DHX58","MAVS","RIGI","MDA5","LGP2",
        "NLRX1","NOD1","NOD2","RIPK2","CARD9","BCL10","MALT1",
        "MBL2","FCN1","FCN2","MASP1","MASP2","C3","C5","C5AR1","C3AR1",
        "NK_CELLS","KIR2DL1","KIR3DL1","KIR2DS1","NKG2A","NKG2C","NKG2D","NCR1","NCR3",
        "GZMB","GZMK","PRF1","EOMES","TBX21",
    ],
    "Adaptive Immune System": [
        "CD4","CD8A","CD8B","CD3D","CD3E","CD3G","CD247","ZAP70","LCK","FYN","LAT","SLP76",
        "CD19","CD21","CD79A","CD79B","SYK","BTK","BLNK","PLCG2","BCAP","CD81",
        "MHC1","MHC2","HLA-A","HLA-B","HLA-C","HLA-DRA","HLA-DRB1","HLA-DQA1","HLA-DQB1","HLA-DPA1","HLA-DPB1",
        "CD28","CTLA4","ICOS","CD40L","CD40","PD1","PDCD1","PDL1","CD274","PDL2",
        "TCR","BCR","NFAT","AP-1","NF-κB","IRF4","BATF","BATF3","STAT6",
        "IL2","IL2RA","IL4","IL4R","IL5","IL21","IL21R","TGFB1","TGFBR1","TGFBR2",
        "FOXP3","GATA3","TBX21","RORC","BCL6","IRF4","TCF7","LEF1","CCR7",
        "GZMB","GZMK","PRF1","NKG7","EOMES","KLRG1","CX3CR1",
        "ICOS","ICOSL","OX40","OX40L","4-1BB","4-1BBL","GITR","GITRL","BTLA",
        "SOCE","ORAI1","STIM1","NFAT1","NFAT2","CALM1","PPP3CA","NFATC1","NFATC2",
    ],
    "Hemostasis": [
        "F2","F3","F5","F7","F8","F9","F10","F11","F12","F13A1","F13B",
        "VWF","GP1BA","GP1BB","GP9","GP6","ITGA2","ITGB3","ITGAV",
        "THBD","PROC","PROS1","PROZ","PROTEIN_C","PROTEIN_S",
        "SERPINC1","SERPINE1","PAI1","TFPI","A2M","PLAT","PLAU","PLAUR","PLG","FBN1",
        "SELP","SELE","SELL","ICAM1","VCAM1","CD62L","CD62P","CD62E",
        "COL1A1","FN1","VTN","THBS1","THBS2","COMP","GPVI","CLEC1B","CLEC2",
        "ADP","ATP","TXA2","TXAS1","PGI2","PTGIS","TBXA2R","PTGIR",
        "ADP","P2RX1","P2RY1","P2RY12","ADORA2A","ADORA3",
        "PKCα","PIK3","AKT1","RAP1","RIAM","TLN1","VCL","KINDLIN3",
        "CAL_PATHWY","MAPK_PATHWAY","NFKB","RHOA","ROCK1","MLC",
    ],
}

_ALL_DATABASES = {"kegg": _KEGG, "go_bp": _GO_BP, "reactome": _REACTOME}


def list_databases() -> List[str]:
    """Return available database names."""
    return list(_ALL_DATABASES.keys())


def get_pathway_genes(database: str, pathway: str) -> List[str]:
    """Return gene list for a specific pathway in a database."""
    return _ALL_DATABASES.get(database, {}).get(pathway, [])


# ─────────────────────────────────────────────────────────────────────────────
# Odds-ratio helper  (Wald 95 % CI, log-space)
# ─────────────────────────────────────────────────────────────────────────────

def _odds_ratio_ci(a: int, b: int, c: int, d: int, ci: float = 0.95):
    """
    Compute odds ratio and its Wald 95 % confidence interval.

    OR = (a*d) / (b*c)
    95 % CI: exp(log(OR) ± z × SE)
    SE = sqrt(1/a + 1/b + 1/c + 1/d)

    Returns (or_val, or_lo, or_hi) — all rounded to 3 dp.
    All zero-cell corrections: add 0.5 to each cell if any cell == 0.
    """
    # Haldane-Anscombe correction for zero cells
    if 0 in (a, b, c, d):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    or_val = (a * d) / (b * c)
    se     = math.sqrt(1/a + 1/b + 1/c + 1/d)
    z      = 1.959964  # z_{0.975} for 95 % CI
    lo     = math.exp(math.log(or_val) - z * se)
    hi     = math.exp(math.log(or_val) + z * se)
    return round(or_val, 3), round(lo, 3), round(hi, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Core enrichment engine  (v3.0 — with OR, CI, gene ratio, activation z-score)
# ─────────────────────────────────────────────────────────────────────────────

def _run_enrichment(
    gene_list: List[str],
    pathway_db: Dict[str, List[str]],
    db_name: str,
    background: int = 20_000,
    correction: str = "fdr_bh",
    gene_lfc_map: Optional[Dict[str, float]] = None,
    gene_nlp_map: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Core Fisher-exact pathway enrichment with extended output columns.

    New columns vs v2:
        odds_ratio          — OR from the 2×2 Fisher table
        or_ci_low           — Lower bound of Wald 95 % CI on OR
        or_ci_high          — Upper bound of Wald 95 % CI on OR
        gene_ratio          — overlap_count / query_size  (like clusterProfiler)
        activation_zscore   — (N_up − N_down) / √N_tot   (IPA-style)
        direction           — 'Activated' | 'Inhibited' | 'Inconclusive'
        effect_weighted_score — Σ |LFC| × −log10(padj) for overlap genes

    Args:
        gene_list      : Query gene symbols (upper-case).
        pathway_db     : {pathway_name: [gene_symbols]}.
        db_name        : Label for the 'database' column.
        background     : Estimated genome-wide gene count for Fisher table.
        correction     : Multiple-testing method (statsmodels multipletests).
        gene_lfc_map   : {GENE: log2FC} — enables directional scoring.
        gene_nlp_map   : {GENE: −log10(padj)} — enables EWS.
    """
    gene_set = {g.strip().upper() for g in gene_list if g.strip()}
    if not gene_set:
        return pd.DataFrame()

    lfc_map = gene_lfc_map or {}
    nlp_map = gene_nlp_map or {}

    rows = []
    for pw, pw_genes in pathway_db.items():
        pw_set  = {g.upper() for g in pw_genes}
        overlap = gene_set & pw_set
        n_over  = len(overlap)
        if n_over == 0:
            continue

        # 2×2 contingency table
        # background must be >= a+b+c so that d (genes in neither set) is ≥ 0.
        # If the caller passes a background smaller than the combined set sizes
        # (e.g. a 30-gene demo dataset), clamp it up to avoid a degenerate table.
        a = n_over
        b = len(pw_set) - n_over
        c = len(gene_set) - n_over
        safe_bg = max(background, a + b + c + 1)
        d = safe_bg - a - b - c        # guaranteed ≥ 1

        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")

        # Odds ratio + Wald 95 % CI
        or_val, or_lo, or_hi = _odds_ratio_ci(a, b, c, max(1, d))

        # Gene ratio (clusterProfiler convention)
        gene_ratio = round(n_over / max(1, len(gene_set)), 4)

        # Activation z-score (IPA-style)
        overlap_genes = sorted(overlap)
        lfc_arr = np.array([lfc_map[g] for g in overlap_genes if g in lfc_map])
        if len(lfc_arr) > 0:
            n_up   = int((lfc_arr > 0).sum())
            n_dn   = int((lfc_arr < 0).sum())
            n_tot  = n_up + n_dn
            az     = round((n_up - n_dn) / math.sqrt(n_tot), 3) if n_tot > 0 else 0.0
            if az >= 2.0:
                direction = "Activated"
            elif az <= -2.0:
                direction = "Inhibited"
            else:
                direction = "Inconclusive"
            # Effect-weighted score
            ews = round(sum(
                abs(lfc_map.get(g, 0.0)) * nlp_map.get(g, 0.0)
                for g in overlap_genes if g in lfc_map
            ), 2)
        else:
            az, direction, ews = 0.0, "Inconclusive", 0.0

        rows.append({
            "pathway":               pw,
            "database":              db_name,
            "p_value":               float(p),
            "adjusted_p_value":      float(p),   # placeholder — corrected below
            "overlap_count":         n_over,
            "pathway_size":          len(pw_set),
            "query_size":            len(gene_set),
            "gene_ratio":            gene_ratio,
            "odds_ratio":            or_val,
            "or_ci_low":             or_lo,
            "or_ci_high":            or_hi,
            "enrichment_score":      (-np.log10(p) if p > 0 else 300.0),
            "activation_zscore":     az,
            "direction":             direction,
            "effect_weighted_score": ews,
            "genes":                 ";".join(overlap_genes),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if len(df) > 1:
        _, adj = multipletests(df["p_value"], method=correction)[:2]
        df["adjusted_p_value"] = adj
    # Recompute enrichment_score from the BH-corrected value so bubble/bar
    # charts always reflect FDR-adjusted significance, never raw p-value.
    df["enrichment_score"] = -np.log10(df["adjusted_p_value"].clip(lower=1e-300))
    df = df.sort_values("adjusted_p_value").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public API  (backward-compatible; enriches app.py imports unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class EnrichmentAnalyzer:
    """
    Single-database enrichment wrapper.

    v3.0 changes:
    • padj_threshold default tightened to 0.05 (was 0.5 — too permissive).
    • Accepts gene_lfc_map for activation z-score computation.
    • Exposes background parameter.
    """

    def __init__(self, database: str = "kegg", species: str = "human"):
        if database not in _ALL_DATABASES:
            raise ValueError(
                f"Database '{database}' not available. Choose from {list(_ALL_DATABASES)}"
            )
        self.database_name = database
        self.pathways      = _ALL_DATABASES[database]

    def run_enrichment(
        self,
        gene_list: List[str],
        padj_threshold: float = 0.05,
        background: int = 20_000,
        strict: bool = True,
        gene_lfc_map: Optional[Dict[str, float]] = None,
        gene_nlp_map: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        # Normalise gene symbols here so callers don't need to
        gene_list = [g.strip().upper() for g in gene_list if g and g.strip()]
        df = _run_enrichment(
            gene_list, self.pathways, self.database_name,
            background=background,
            gene_lfc_map=gene_lfc_map,
            gene_nlp_map=gene_nlp_map,
        )
        if df.empty:
            return df
        # Strict mode: filter on BH-adjusted p-value (robust, conservative)
        # Relaxed mode: filter on nominal p-value (more results, exploratory)
        filter_col = "adjusted_p_value" if strict else "p_value"
        return df[df[filter_col] <= padj_threshold].reset_index(drop=True)

    def get_enriched_pathways(
        self,
        gene_list: List[str],
        padj_threshold: float = 0.05,
        top_n: Optional[int] = None,
        background: int = 20_000,
        strict: bool = True,
        gene_lfc_map: Optional[Dict[str, float]] = None,
        gene_nlp_map: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        df = self.run_enrichment(
            gene_list, padj_threshold, background, strict, gene_lfc_map, gene_nlp_map
        )
        return df.head(top_n) if (top_n and not df.empty) else df


class MultiDatabaseEnrichment:
    """
    Multi-database enrichment aggregator.

    v3.0 changes:
    • padj_threshold default tightened to 0.05.
    • Forwards gene_lfc_map and gene_nlp_map to each sub-analyzer.
    • Exposes background parameter.
    """

    def __init__(self, databases: List[str] = None, species: str = "human"):
        dbs = databases or ["kegg", "go_bp", "reactome"]
        self.analyzers = {
            d: EnrichmentAnalyzer(d, species)
            for d in dbs if d in _ALL_DATABASES
        }

    def run_multi_enrichment(
        self,
        gene_list: List[str],
        padj_threshold: float = 0.05,
        background: int = 20_000,
        strict: bool = True,
        gene_lfc_map: Optional[Dict[str, float]] = None,
        gene_nlp_map: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        parts = [
            a.run_enrichment(
                gene_list, padj_threshold, background, strict, gene_lfc_map, gene_nlp_map
            )
            for a in self.analyzers.values()
        ]
        parts = [p for p in parts if not p.empty]
        if not parts:
            return pd.DataFrame()
        combined = pd.concat(parts, ignore_index=True).sort_values("adjusted_p_value")
        return combined.reset_index(drop=True)
