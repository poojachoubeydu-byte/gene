"""
Gene Annotations Database
=========================
Built-in curated data for:
  - Drug-Gene Interactions (FDA-approved / clinical-stage drugs)
  - Cancer Gene Census (COSMIC-inspired oncogenes & TSGs)
  - Known biomarker classifications

All data is offline-safe — no network calls required.
"""

# ── Drug-Gene Interactions ────────────────────────────────────────────────────
# Structure: GENE_SYMBOL -> list of drug dicts
# 'type': mechanism class  |  'indication': primary cancer/disease use
# 'fda': True = FDA-approved  |  'class': broader drug class
DRUG_GENE_INTERACTIONS = {
    'EGFR': [
        {'drug': 'Erlotinib',    'type': 'TKI',     'indication': 'NSCLC / Pancreatic', 'fda': True},
        {'drug': 'Gefitinib',    'type': 'TKI',     'indication': 'NSCLC',              'fda': True},
        {'drug': 'Afatinib',     'type': 'TKI',     'indication': 'NSCLC',              'fda': True},
        {'drug': 'Osimertinib',  'type': 'TKI',     'indication': 'NSCLC (T790M)',      'fda': True},
        {'drug': 'Cetuximab',    'type': 'Antibody', 'indication': 'CRC / HNSCC',       'fda': True},
        {'drug': 'Panitumumab',  'type': 'Antibody', 'indication': 'CRC',               'fda': True},
    ],
    'ERBB2': [
        {'drug': 'Trastuzumab',  'type': 'Antibody', 'indication': 'Breast / Gastric',  'fda': True},
        {'drug': 'Pertuzumab',   'type': 'Antibody', 'indication': 'Breast',            'fda': True},
        {'drug': 'Lapatinib',    'type': 'TKI',     'indication': 'Breast',             'fda': True},
        {'drug': 'Neratinib',    'type': 'TKI',     'indication': 'Breast',             'fda': True},
        {'drug': 'Tucatinib',    'type': 'TKI',     'indication': 'Breast',             'fda': True},
        {'drug': 'T-DM1',        'type': 'ADC',     'indication': 'Breast',             'fda': True},
    ],
    'VEGFA': [
        {'drug': 'Bevacizumab',  'type': 'Antibody',          'indication': 'Multiple cancers', 'fda': True},
        {'drug': 'Ramucirumab',  'type': 'Antibody',          'indication': 'Gastric / NSCLC',  'fda': True},
        {'drug': 'Sunitinib',    'type': 'Multi-TKI',         'indication': 'RCC / GIST',       'fda': True},
        {'drug': 'Sorafenib',    'type': 'Multi-TKI',         'indication': 'HCC / RCC',        'fda': True},
    ],
    'MTOR': [
        {'drug': 'Everolimus',   'type': 'mTOR inhibitor',    'indication': 'RCC / Breast / PNET', 'fda': True},
        {'drug': 'Temsirolimus', 'type': 'mTOR inhibitor',    'indication': 'RCC',               'fda': True},
    ],
    'CDK4': [
        {'drug': 'Palbociclib',  'type': 'CDK4/6 inhibitor', 'indication': 'Breast',            'fda': True},
        {'drug': 'Ribociclib',   'type': 'CDK4/6 inhibitor', 'indication': 'Breast',            'fda': True},
        {'drug': 'Abemaciclib',  'type': 'CDK4/6 inhibitor', 'indication': 'Breast',            'fda': True},
    ],
    'CDK6': [
        {'drug': 'Palbociclib',  'type': 'CDK4/6 inhibitor', 'indication': 'Breast',            'fda': True},
        {'drug': 'Ribociclib',   'type': 'CDK4/6 inhibitor', 'indication': 'Breast',            'fda': True},
        {'drug': 'Abemaciclib',  'type': 'CDK4/6 inhibitor', 'indication': 'Breast',            'fda': True},
    ],
    'BRAF': [
        {'drug': 'Vemurafenib',  'type': 'BRAF inhibitor',   'indication': 'Melanoma',          'fda': True},
        {'drug': 'Dabrafenib',   'type': 'BRAF inhibitor',   'indication': 'Melanoma / NSCLC',  'fda': True},
        {'drug': 'Encorafenib',  'type': 'BRAF inhibitor',   'indication': 'Melanoma / CRC',    'fda': True},
    ],
    'MEK1': [
        {'drug': 'Trametinib',   'type': 'MEK inhibitor',    'indication': 'Melanoma / NSCLC',  'fda': True},
        {'drug': 'Cobimetinib',  'type': 'MEK inhibitor',    'indication': 'Melanoma',           'fda': True},
        {'drug': 'Binimetinib',  'type': 'MEK inhibitor',    'indication': 'Melanoma / CRC',    'fda': True},
    ],
    'ALK': [
        {'drug': 'Crizotinib',   'type': 'ALK inhibitor',    'indication': 'NSCLC',             'fda': True},
        {'drug': 'Alectinib',    'type': 'ALK inhibitor',    'indication': 'NSCLC',             'fda': True},
        {'drug': 'Brigatinib',   'type': 'ALK inhibitor',    'indication': 'NSCLC',             'fda': True},
        {'drug': 'Lorlatinib',   'type': 'ALK inhibitor',    'indication': 'NSCLC',             'fda': True},
    ],
    'ABL1': [
        {'drug': 'Imatinib',     'type': 'TKI',     'indication': 'CML / GIST',                'fda': True},
        {'drug': 'Dasatinib',    'type': 'TKI',     'indication': 'CML',                        'fda': True},
        {'drug': 'Nilotinib',    'type': 'TKI',     'indication': 'CML',                        'fda': True},
        {'drug': 'Ponatinib',    'type': 'TKI',     'indication': 'CML / ALL',                  'fda': True},
    ],
    'PDCD1': [
        {'drug': 'Pembrolizumab','type': 'Checkpoint inhibitor', 'indication': 'Pan-cancer',    'fda': True},
        {'drug': 'Nivolumab',   'type': 'Checkpoint inhibitor', 'indication': 'Pan-cancer',     'fda': True},
        {'drug': 'Cemiplimab',  'type': 'Checkpoint inhibitor', 'indication': 'CSCC / NSCLC',   'fda': True},
    ],
    'CD274': [
        {'drug': 'Atezolizumab', 'type': 'Checkpoint inhibitor', 'indication': 'NSCLC / BC',   'fda': True},
        {'drug': 'Durvalumab',   'type': 'Checkpoint inhibitor', 'indication': 'NSCLC / UC',   'fda': True},
        {'drug': 'Avelumab',     'type': 'Checkpoint inhibitor', 'indication': 'MCC / UC',      'fda': True},
    ],
    'CTLA4': [
        {'drug': 'Ipilimumab',   'type': 'Checkpoint inhibitor', 'indication': 'Melanoma / RCC','fda': True},
        {'drug': 'Tremelimumab', 'type': 'Checkpoint inhibitor', 'indication': 'NSCLC / HCC',   'fda': True},
    ],
    'BCL2': [
        {'drug': 'Venetoclax',   'type': 'BCL2 inhibitor',   'indication': 'CLL / AML',         'fda': True},
        {'drug': 'Navitoclax',   'type': 'BCL2/XL inhibitor','indication': 'Hematologic',       'fda': False},
    ],
    'PARP1': [
        {'drug': 'Olaparib',     'type': 'PARP inhibitor',   'indication': 'Breast / Ovarian',  'fda': True},
        {'drug': 'Niraparib',    'type': 'PARP inhibitor',   'indication': 'Ovarian',            'fda': True},
        {'drug': 'Rucaparib',    'type': 'PARP inhibitor',   'indication': 'Ovarian / Prostate', 'fda': True},
        {'drug': 'Talazoparib',  'type': 'PARP inhibitor',   'indication': 'Breast',             'fda': True},
    ],
    'IDH1': [
        {'drug': 'Enasidenib',   'type': 'IDH2 inhibitor',   'indication': 'AML',                'fda': True},
        {'drug': 'Ivosidenib',   'type': 'IDH1 inhibitor',   'indication': 'AML / CCA',          'fda': True},
    ],
    'IDH2': [
        {'drug': 'Enasidenib',   'type': 'IDH2 inhibitor',   'indication': 'AML',                'fda': True},
    ],
    'FLT3': [
        {'drug': 'Midostaurin',  'type': 'Multi-kinase',     'indication': 'AML',                'fda': True},
        {'drug': 'Gilteritinib', 'type': 'FLT3 inhibitor',   'indication': 'AML',                'fda': True},
        {'drug': 'Quizartinib',  'type': 'FLT3 inhibitor',   'indication': 'AML',                'fda': True},
    ],
    'JAK2': [
        {'drug': 'Ruxolitinib',  'type': 'JAK1/2 inhibitor', 'indication': 'MF / PV',            'fda': True},
        {'drug': 'Fedratinib',   'type': 'JAK2 inhibitor',   'indication': 'MF',                 'fda': True},
    ],
    'STAT3': [
        {'drug': 'Ruxolitinib',  'type': 'JAK1/2 inhibitor', 'indication': 'MF / PV',            'fda': True},
    ],
    'PIK3CA': [
        {'drug': 'Alpelisib',    'type': 'PI3Kα inhibitor',  'indication': 'Breast',             'fda': True},
        {'drug': 'Copanlisib',   'type': 'PI3K inhibitor',   'indication': 'Follicular lymphoma','fda': True},
        {'drug': 'Idelalisib',   'type': 'PI3Kδ inhibitor',  'indication': 'CLL / FL',           'fda': True},
    ],
    'AKT1': [
        {'drug': 'Capivasertib', 'type': 'AKT inhibitor',    'indication': 'Breast',             'fda': True},
        {'drug': 'Ipatasertib',  'type': 'AKT inhibitor',    'indication': 'Breast / Prostate',  'fda': False},
    ],
    'KRAS': [
        {'drug': 'Sotorasib',    'type': 'KRAS G12C inhibitor','indication': 'NSCLC',            'fda': True},
        {'drug': 'Adagrasib',    'type': 'KRAS G12C inhibitor','indication': 'NSCLC / CRC',      'fda': True},
    ],
    'NRAS': [
        {'drug': 'Binimetinib',  'type': 'MEK inhibitor',    'indication': 'Melanoma',           'fda': True},
    ],
    'RET': [
        {'drug': 'Selpercatinib','type': 'RET inhibitor',    'indication': 'NSCLC / Thyroid',    'fda': True},
        {'drug': 'Pralsetinib',  'type': 'RET inhibitor',    'indication': 'NSCLC / Thyroid',    'fda': True},
        {'drug': 'Vandetanib',   'type': 'Multi-TKI',        'indication': 'Thyroid',            'fda': True},
    ],
    'MET': [
        {'drug': 'Crizotinib',   'type': 'Multi-TKI',        'indication': 'NSCLC (MET ex14)',   'fda': True},
        {'drug': 'Capmatinib',   'type': 'MET inhibitor',    'indication': 'NSCLC',              'fda': True},
        {'drug': 'Tepotinib',    'type': 'MET inhibitor',    'indication': 'NSCLC',              'fda': True},
    ],
    'FGFR1': [
        {'drug': 'Pemigatinib',  'type': 'FGFR inhibitor',   'indication': 'CCA / MF',           'fda': True},
        {'drug': 'Infigratinib', 'type': 'FGFR inhibitor',   'indication': 'CCA',                'fda': True},
    ],
    'FGFR2': [
        {'drug': 'Pemigatinib',  'type': 'FGFR inhibitor',   'indication': 'CCA',                'fda': True},
        {'drug': 'Futibatinib',  'type': 'FGFR inhibitor',   'indication': 'CCA',                'fda': True},
    ],
    'FGFR3': [
        {'drug': 'Erdafitinib',  'type': 'Pan-FGFR inhibitor','indication': 'Bladder cancer',    'fda': True},
    ],
    'ESR1': [
        {'drug': 'Tamoxifen',    'type': 'SERM',             'indication': 'Breast',             'fda': True},
        {'drug': 'Fulvestrant',  'type': 'SERD',             'indication': 'Breast',             'fda': True},
        {'drug': 'Elacestrant',  'type': 'SERD',             'indication': 'Breast (ESR1-mut)',  'fda': True},
        {'drug': 'Letrozole',    'type': 'Aromatase inhibitor','indication': 'Breast',           'fda': True},
        {'drug': 'Anastrozole',  'type': 'Aromatase inhibitor','indication': 'Breast',           'fda': True},
    ],
    'AR': [
        {'drug': 'Enzalutamide', 'type': 'AR antagonist',    'indication': 'Prostate',           'fda': True},
        {'drug': 'Apalutamide',  'type': 'AR antagonist',    'indication': 'Prostate',           'fda': True},
        {'drug': 'Darolutamide', 'type': 'AR antagonist',    'indication': 'Prostate',           'fda': True},
        {'drug': 'Abiraterone',  'type': 'CYP17 inhibitor',  'indication': 'Prostate',           'fda': True},
    ],
    'CDK1': [
        {'drug': 'Dinaciclib',   'type': 'CDK inhibitor',    'indication': 'Hematologic',        'fda': False},
    ],
    'CDK2': [
        {'drug': 'Fadraciclib',  'type': 'CDK2 inhibitor',   'indication': 'Various',            'fda': False},
    ],
    'HDAC1': [
        {'drug': 'Vorinostat',   'type': 'HDAC inhibitor',   'indication': 'CTCL',              'fda': True},
        {'drug': 'Romidepsin',   'type': 'HDAC inhibitor',   'indication': 'CTCL / PTCL',       'fda': True},
    ],
    'DNMT3A': [
        {'drug': 'Azacitidine',  'type': 'Demethylating',    'indication': 'MDS / AML',          'fda': True},
        {'drug': 'Decitabine',   'type': 'Demethylating',    'indication': 'MDS / AML',          'fda': True},
    ],
    'EZH2': [
        {'drug': 'Tazemetostat', 'type': 'EZH2 inhibitor',   'indication': 'FL / ES',            'fda': True},
    ],
    'SMO': [
        {'drug': 'Vismodegib',   'type': 'SMO inhibitor',    'indication': 'BCC',                'fda': True},
        {'drug': 'Sonidegib',    'type': 'SMO inhibitor',    'indication': 'BCC',                'fda': True},
    ],
    'SRC': [
        {'drug': 'Dasatinib',    'type': 'SRC/ABL inhibitor','indication': 'CML / Ph+ ALL',      'fda': True},
        {'drug': 'Bosutinib',    'type': 'SRC/ABL inhibitor','indication': 'CML',                'fda': True},
    ],
    'TNF': [
        {'drug': 'Adalimumab',   'type': 'TNF-α antibody',   'indication': 'RA / IBD',           'fda': True},
        {'drug': 'Infliximab',   'type': 'TNF-α antibody',   'indication': 'RA / IBD',           'fda': True},
        {'drug': 'Etanercept',   'type': 'TNF-α fusion',     'indication': 'RA / PsA',           'fda': True},
    ],
    'IL6': [
        {'drug': 'Tocilizumab',  'type': 'IL-6R antibody',   'indication': 'RA / CRS',           'fda': True},
        {'drug': 'Sarilumab',    'type': 'IL-6R antibody',   'indication': 'RA',                 'fda': True},
        {'drug': 'Siltuximab',   'type': 'IL-6 antibody',    'indication': 'Castleman disease',  'fda': True},
    ],
    'MMP9': [
        {'drug': 'Andecaliximab','type': 'MMP9 inhibitor',   'indication': 'Gastric (clinical)', 'fda': False},
    ],
    'HIF1A': [
        {'drug': 'Belzutifan',   'type': 'HIF2α inhibitor',  'indication': 'VHL disease / RCC',  'fda': True},
    ],
    'TERT': [
        {'drug': 'Imetelstat',   'type': 'Telomerase inhibitor','indication': 'MF / MDS',        'fda': True},
    ],
    'MDM2': [
        {'drug': 'Milademetan',  'type': 'MDM2 inhibitor',   'indication': 'Liposarcoma',        'fda': False},
        {'drug': 'Navtemadlin',  'type': 'MDM2 inhibitor',   'indication': 'Various',            'fda': False},
    ],
    'CCND1': [
        {'drug': 'Palbociclib',  'type': 'CDK4/6 inhibitor', 'indication': 'Breast',            'fda': True},
        {'drug': 'Abemaciclib',  'type': 'CDK4/6 inhibitor', 'indication': 'Breast',            'fda': True},
    ],
    'CASP3': [
        {'drug': 'Navitoclax',   'type': 'Pro-apoptotic',    'indication': 'Hematologic',       'fda': False},
    ],
    'MYC': [
        {'drug': 'OTX015',       'type': 'BET inhibitor',    'indication': 'Various (clinical)', 'fda': False},
    ],
    'BRCA1': [
        {'drug': 'Olaparib',     'type': 'PARP inhibitor',   'indication': 'Breast / Ovarian',  'fda': True},
        {'drug': 'Niraparib',    'type': 'PARP inhibitor',   'indication': 'Ovarian',            'fda': True},
    ],
    'BRCA2': [
        {'drug': 'Olaparib',     'type': 'PARP inhibitor',   'indication': 'Breast / Ovarian / Prostate', 'fda': True},
        {'drug': 'Rucaparib',    'type': 'PARP inhibitor',   'indication': 'Ovarian / Prostate', 'fda': True},
    ],
}

# ── Cancer Gene Census ────────────────────────────────────────────────────────
# role: 'Oncogene', 'TSG' (tumor suppressor), or 'Both'
# tier: 1 = highest confidence, 2 = moderate
# hallmarks: cancer hallmarks affected
CANCER_GENE_CENSUS = {
    'TP53':   {'role': 'TSG',      'tier': 1, 'cancer_types': ['Pan-cancer'],               'hallmarks': ['apoptosis', 'cell cycle', 'DNA repair']},
    'MYC':    {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['Lymphoma', 'Breast', 'Lung'],'hallmarks': ['proliferation', 'metabolism', 'immortalization']},
    'KRAS':   {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['Pancreas', 'Lung', 'CRC'],   'hallmarks': ['proliferation', 'growth signaling']},
    'BRCA1':  {'role': 'TSG',      'tier': 1, 'cancer_types': ['Breast', 'Ovarian'],          'hallmarks': ['DNA repair', 'genome stability']},
    'BRCA2':  {'role': 'TSG',      'tier': 1, 'cancer_types': ['Breast', 'Ovarian', 'Pancreas'],'hallmarks': ['DNA repair', 'genome stability']},
    'EGFR':   {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['NSCLC', 'GBM', 'HNSCC'],     'hallmarks': ['proliferation', 'growth signaling']},
    'ERBB2':  {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['Breast', 'Gastric'],          'hallmarks': ['proliferation', 'growth signaling']},
    'PTEN':   {'role': 'TSG',      'tier': 1, 'cancer_types': ['Endometrial', 'Prostate', 'GBM'],'hallmarks': ['growth signaling', 'survival']},
    'RB1':    {'role': 'TSG',      'tier': 1, 'cancer_types': ['Retinoblastoma', 'Osteosarcoma'],'hallmarks': ['cell cycle', 'senescence']},
    'APC':    {'role': 'TSG',      'tier': 1, 'cancer_types': ['CRC', 'Stomach'],             'hallmarks': ['Wnt signaling', 'cell cycle']},
    'VEGFA':  {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['Pan-cancer'],                 'hallmarks': ['angiogenesis']},
    'CDKN2A': {'role': 'TSG',      'tier': 1, 'cancer_types': ['Melanoma', 'Lung', 'Pancreas'],'hallmarks': ['cell cycle', 'senescence']},
    'PIK3CA': {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['Breast', 'CRC', 'Endometrial'],'hallmarks': ['growth signaling', 'survival']},
    'BRAF':   {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['Melanoma', 'CRC', 'Thyroid'], 'hallmarks': ['proliferation', 'growth signaling']},
    'CDK4':   {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['Melanoma', 'Sarcoma'],        'hallmarks': ['cell cycle']},
    'MDM2':   {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['Sarcoma', 'Pan-cancer'],      'hallmarks': ['apoptosis evasion', 'TP53 suppression']},
    'MET':    {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['NSCLC', 'Gastric', 'HCC'],   'hallmarks': ['invasion', 'proliferation']},
    'ALK':    {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['NSCLC', 'ALCL', 'Neuroblastoma'],'hallmarks': ['proliferation', 'survival']},
    'ABL1':   {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['CML', 'ALL'],                 'hallmarks': ['proliferation', 'differentiation block']},
    'NRAS':   {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['Melanoma', 'AML'],            'hallmarks': ['proliferation', 'differentiation']},
    'IDH1':   {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['GBM', 'AML', 'CCA'],          'hallmarks': ['epigenetic reprogramming']},
    'IDH2':   {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['AML', 'MDS'],                 'hallmarks': ['epigenetic reprogramming']},
    'FLT3':   {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['AML'],                        'hallmarks': ['differentiation block', 'proliferation']},
    'JAK2':   {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['MPN', 'ALL'],                 'hallmarks': ['proliferation', 'differentiation']},
    'STAT3':  {'role': 'Oncogene', 'tier': 2, 'cancer_types': ['HNSCC', 'Breast', 'Lymphoma'],'hallmarks': ['survival', 'immune evasion']},
    'BCL2':   {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['Follicular lymphoma', 'CLL'], 'hallmarks': ['apoptosis evasion']},
    'CCND1':  {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['Breast', 'Mantle cell lymphoma'],'hallmarks': ['cell cycle']},
    'E2F1':   {'role': 'Both',     'tier': 2, 'cancer_types': ['Lung', 'Bladder'],            'hallmarks': ['cell cycle', 'apoptosis']},
    'HIF1A':  {'role': 'Oncogene', 'tier': 2, 'cancer_types': ['RCC', 'Pan-cancer'],          'hallmarks': ['angiogenesis', 'metabolism']},
    'VHL':    {'role': 'TSG',      'tier': 1, 'cancer_types': ['RCC', 'VHL disease'],         'hallmarks': ['HIF signaling', 'angiogenesis']},
    'ESR1':   {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['Breast'],                     'hallmarks': ['proliferation', 'survival']},
    'AR':     {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['Prostate'],                   'hallmarks': ['proliferation', 'differentiation']},
    'MTOR':   {'role': 'Oncogene', 'tier': 2, 'cancer_types': ['RCC', 'Breast', 'Endometrial'],'hallmarks': ['metabolism', 'growth signaling']},
    'MMP9':   {'role': 'Oncogene', 'tier': 2, 'cancer_types': ['Pan-cancer'],                 'hallmarks': ['invasion', 'metastasis']},
    'TERT':   {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['Pan-cancer'],                 'hallmarks': ['immortalization']},
    'SMO':    {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['BCC', 'Medulloblastoma'],     'hallmarks': ['proliferation']},
    'EZH2':   {'role': 'Both',     'tier': 1, 'cancer_types': ['Lymphoma', 'Prostate'],       'hallmarks': ['epigenetic silencing']},
    'DNMT3A': {'role': 'Both',     'tier': 1, 'cancer_types': ['AML', 'MDS'],                 'hallmarks': ['epigenetic reprogramming']},
    'BAX':    {'role': 'TSG',      'tier': 2, 'cancer_types': ['Hematologic'],                'hallmarks': ['apoptosis']},
    'CASP3':  {'role': 'TSG',      'tier': 2, 'cancer_types': ['Pan-cancer'],                 'hallmarks': ['apoptosis']},
    'CDKN1A': {'role': 'TSG',      'tier': 2, 'cancer_types': ['Pan-cancer'],                 'hallmarks': ['cell cycle', 'senescence']},
    'PARP1':  {'role': 'Both',     'tier': 2, 'cancer_types': ['BRCA-mutant cancers'],        'hallmarks': ['DNA repair']},
    'ATM':    {'role': 'TSG',      'tier': 1, 'cancer_types': ['CLL', 'Breast', 'AT'],        'hallmarks': ['DNA repair', 'apoptosis']},
    'RAD51':  {'role': 'TSG',      'tier': 2, 'cancer_types': ['Breast', 'Ovarian'],          'hallmarks': ['DNA repair', 'genome stability']},
    'TOP2A':  {'role': 'Oncogene', 'tier': 2, 'cancer_types': ['Breast', 'Lung'],             'hallmarks': ['proliferation']},
    'MKI67':  {'role': 'Oncogene', 'tier': 2, 'cancer_types': ['Pan-cancer'],                 'hallmarks': ['proliferation']},
    'SRC':    {'role': 'Oncogene', 'tier': 2, 'cancer_types': ['Breast', 'CRC', 'Lung'],      'hallmarks': ['invasion', 'proliferation']},
    'TNF':    {'role': 'Both',     'tier': 2, 'cancer_types': ['Pan-cancer'],                 'hallmarks': ['inflammation', 'apoptosis']},
    'IL6':    {'role': 'Oncogene', 'tier': 2, 'cancer_types': ['Multiple myeloma', 'Liver'],  'hallmarks': ['proliferation', 'survival', 'immune evasion']},
    'NOTCH1': {'role': 'Both',     'tier': 1, 'cancer_types': ['T-ALL', 'CLL', 'TNBC'],       'hallmarks': ['differentiation', 'proliferation']},
    'RET':    {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['Thyroid', 'NSCLC'],           'hallmarks': ['proliferation', 'survival']},
    'FGFR1':  {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['Breast', 'Lung', 'MF'],       'hallmarks': ['proliferation', 'survival']},
    'FGFR2':  {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['CCA', 'Gastric', 'Endometrial'],'hallmarks': ['proliferation', 'differentiation']},
    'FGFR3':  {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['Bladder', 'MM'],              'hallmarks': ['proliferation', 'differentiation']},
    'CTNNB1': {'role': 'Oncogene', 'tier': 1, 'cancer_types': ['CRC', 'HCC', 'Endometrial'],  'hallmarks': ['Wnt signaling', 'proliferation']},
}

# ── Biomarker classifications ─────────────────────────────────────────────────
ESTABLISHED_BIOMARKERS = {
    'EGFR':  {'type': 'Predictive', 'test': 'Mutation testing', 'context': 'NSCLC (TKI response)'},
    'ERBB2': {'type': 'Predictive', 'test': 'IHC / FISH',       'context': 'Breast cancer (HER2 therapy)'},
    'KRAS':  {'type': 'Predictive', 'test': 'Mutation testing', 'context': 'CRC (anti-EGFR non-response)'},
    'BRAF':  {'type': 'Predictive', 'test': 'Mutation testing', 'context': 'Melanoma (BRAF inhibitor)'},
    'ESR1':  {'type': 'Predictive', 'test': 'IHC',              'context': 'Breast cancer (hormone therapy)'},
    'BRCA1': {'type': 'Predictive', 'test': 'BRCA testing',     'context': 'Breast/Ovarian (PARP inhibitor)'},
    'BRCA2': {'type': 'Predictive', 'test': 'BRCA testing',     'context': 'Breast/Ovarian/Prostate (PARP inhibitor)'},
    'ALK':   {'type': 'Predictive', 'test': 'FISH / IHC / NGS', 'context': 'NSCLC (ALK inhibitor)'},
    'MET':   {'type': 'Predictive', 'test': 'NGS',              'context': 'NSCLC (MET inhibitor)'},
    'RET':   {'type': 'Predictive', 'test': 'NGS',              'context': 'NSCLC/Thyroid (RET inhibitor)'},
    'PDCD1': {'type': 'Predictive', 'test': 'IHC (PD-L1)',      'context': 'Multiple cancers (checkpoint therapy)'},
    'TP53':  {'type': 'Prognostic', 'test': 'Mutation testing', 'context': 'Multiple cancers (poor prognosis)'},
    'MYC':   {'type': 'Prognostic', 'test': 'IHC / FISH',       'context': 'Lymphoma / Breast (poor prognosis)'},
    'IL6':   {'type': 'Diagnostic', 'test': 'Serum ELISA',      'context': 'Inflammation / Multiple myeloma'},
    'TNF':   {'type': 'Diagnostic', 'test': 'Serum ELISA',      'context': 'Inflammation / Autoimmune'},
    'VEGFA': {'type': 'Predictive', 'test': 'IHC / serum',      'context': 'Anti-angiogenic therapy response'},
    'IDH1':  {'type': 'Predictive', 'test': 'Mutation testing', 'context': 'AML/GBM (IDH inhibitor response)'},
    'FLT3':  {'type': 'Predictive', 'test': 'Mutation testing', 'context': 'AML (FLT3 inhibitor response)'},
    'TERT':  {'type': 'Diagnostic', 'test': 'Mutation testing', 'context': 'Thyroid / Melanoma'},
    'MMP9':  {'type': 'Prognostic', 'test': 'IHC',              'context': 'Poor prognosis, invasion'},
}


# ── Helper functions ──────────────────────────────────────────────────────────

def get_drug_targets(gene_list):
    """
    Given a list of gene symbols, return a DataFrame of drug-gene interactions.
    Returns DataFrame with columns: gene, drug, type, indication, fda
    """
    import pandas as pd
    rows = []
    for g in gene_list:
        g_up = str(g).strip().upper()
        for entry in DRUG_GENE_INTERACTIONS.get(g_up, []):
            rows.append({'gene': g_up, **entry})
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=['gene', 'drug', 'type', 'indication', 'fda'])


def get_cancer_gene_info(gene_list):
    """
    Given a list of gene symbols, return cancer gene annotations.
    Returns DataFrame with columns: gene, role, tier, cancer_types, hallmarks
    """
    import pandas as pd
    rows = []
    for g in gene_list:
        g_up = str(g).strip().upper()
        info = CANCER_GENE_CENSUS.get(g_up)
        if info:
            rows.append({
                'gene': g_up,
                'role': info['role'],
                'tier': info['tier'],
                'cancer_types': ', '.join(info['cancer_types'][:3]),
                'hallmarks': ', '.join(info['hallmarks'][:3]),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=['gene', 'role', 'tier', 'cancer_types', 'hallmarks'])


def get_biomarker_info(gene_list):
    """Return established biomarker info for genes in the list."""
    import pandas as pd
    rows = []
    for g in gene_list:
        g_up = str(g).strip().upper()
        info = ESTABLISHED_BIOMARKERS.get(g_up)
        if info:
            rows.append({'gene': g_up, **info})
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=['gene', 'type', 'test', 'context'])
