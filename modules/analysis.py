import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import gseapy as gp
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


def run_gsea_preranked(df):
    if len(df) < 15:
        return {
            'error': 'Preranked GSEA requires ≥15 genes',
            'results': pd.DataFrame(),
            'method': 'GSEApy preranked'
        }

    try:
        ranked_df = df.copy()
        if 'pvalue' not in ranked_df.columns:
            ranked_df['pvalue'] = ranked_df['padj']

        ranked_df['rank_metric'] = (
            np.sign(ranked_df['log2FC']) *
            -np.log10(ranked_df['pvalue'].clip(lower=1e-300))
        )

        ranked = (ranked_df[['symbol', 'rank_metric']]
                  .dropna()
                  .drop_duplicates('symbol')
                  .set_index('symbol')['rank_metric']
                  .sort_values(ascending=False))

        pre_res = gp.prerank(
            rnk=ranked,
            gene_sets=['KEGG_2021_Human', 'Reactome_2022'],
            outdir=None,
            min_size=5,
            max_size=500,
            permutation_num=100,
            seed=42,
            verbose=False
        )

        results = pre_res.res2d.copy()
        results = results[results['FDR q-val'] < 0.25].copy()
        return {
            'results': results,
            'error': None,
            'method': 'GSEApy preranked'
        }
    except Exception as e:
        return {'error': str(e), 'results': pd.DataFrame(), 'method': 'GSEApy preranked'}


def run_pca_3d(df, n_clusters=3):
    if len(df) < 6:
        return {'error': 'Minimum 6 genes required for PCA',}

    df = df.copy()
    df['-log10p'] = -np.log10(df['padj'].clip(lower=1e-300))
    features = ['log2FC', '-log10p']
    if 'baseMean' in df.columns:
        df['log_baseMean'] = np.log1p(pd.to_numeric(df['baseMean'], errors='coerce').fillna(0))
        features.append('log_baseMean')

    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(3, len(features), X_scaled.shape[0])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X_scaled)
    if coords.shape[1] < 3:
        padding = np.zeros((coords.shape[0], 3 - coords.shape[1]))
        coords = np.hstack([coords, padding])

    n_clust = min(n_clusters, max(2, len(df) // 3))
    kmeans = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(coords)

    return {
        'pc1': coords[:, 0].tolist(),
        'pc2': coords[:, 1].tolist(),
        'pc3': coords[:, 2].tolist(),
        'clusters': clusters.tolist(),
        'genes': df['symbol'].tolist(),
        'explained_variance': ([round(float(v) * 100, 1) for v in pca.explained_variance_ratio_]
                               + [0.0] * (3 - n_components)),
        'n_clusters': n_clust
    }


def run_wgcna_lite(df, n_clusters=5, max_genes=500):
    if len(df) < 6:
        return {
            'error': 'Minimum 6 genes required for WGCNA-lite',
            'modules': {},
            'graph': None,
            'labels': {},
            'n_modules': 0,
            'n_edges': 0
        }

    df = df.copy()
    # Cap gene count to keep correlation matrix and nested loop tractable
    if len(df) > max_genes:
        df = df.nlargest(max_genes, 'log2FC').reset_index(drop=True)

    df['-log10p'] = -np.log10(df['padj'].clip(lower=1e-300))
    X = df[['log2FC', '-log10p']].fillna(0).values
    corr_matrix = np.corrcoef(X)

    linkage_matrix = linkage(X, method='ward')
    n_clust = min(n_clusters, max(2, len(df) // 3))
    labels = fcluster(linkage_matrix, n_clust, criterion='maxclust')

    genes = df['symbol'].tolist()
    G = nx.Graph()
    G.add_nodes_from(genes)
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            if abs(corr_matrix[i, j]) > 0.7:
                G.add_edge(genes[i], genes[j], weight=float(corr_matrix[i, j]))

    modules = {}
    for gene, label in zip(genes, labels):
        modules.setdefault(int(label), []).append(gene)

    return {
        'modules': modules,
        'graph': G,
        'labels': dict(zip(genes, labels.tolist() if hasattr(labels, 'tolist') else labels)),
        'n_modules': n_clust,
        'n_edges': G.number_of_edges(),
        'error': None
    }


def compute_meta_score(df):
    df = df.copy()
    sig = -np.log10(df['padj'].clip(lower=1e-300))
    sig_norm = (sig - sig.min()) / (sig.max() - sig.min() + 1e-9)
    lfc_abs = df['log2FC'].abs()
    lfc_norm = (lfc_abs - lfc_abs.min()) / (lfc_abs.max() - lfc_abs.min() + 1e-9)
    df['meta_score'] = ((sig_norm * 0.5) + (lfc_norm * 0.5)) * 100
    df['meta_score'] = df['meta_score'].round(1)
    return df
