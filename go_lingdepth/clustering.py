"""KMeans + cluster-validity metrics + Kruskal-Wallis depth-by-cluster effect size."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.stats import kruskal


def kmeans_labels(X, k: int, seed: int = 0) -> np.ndarray:
    return KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(X)


def kselection(X, ks, sample_size: int = 5000, seed: int = 0) -> pd.DataFrame:
    rows = []
    n = X.shape[0]
    ss = min(sample_size, n)  # silhouette is O(n^2); subsample for 25k points
    for k in ks:
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit(X)
        lab = km.labels_
        sil = silhouette_score(X, lab, sample_size=ss, random_state=seed) if k > 1 else float("nan")
        db = davies_bouldin_score(X, lab) if k > 1 else float("nan")
        rows.append(dict(k=int(k), inertia=float(km.inertia_),
                         silhouette=float(sil), davies_bouldin=float(db)))
    return pd.DataFrame(rows)


def kw_depth_by_cluster(labels, depths) -> dict:
    labels = np.asarray(labels)
    depths = np.asarray(depths, dtype=float)
    groups = [depths[labels == c] for c in np.unique(labels)]
    H, p = kruskal(*groups)
    n = len(depths)
    k = len(groups)
    eta2 = (H - k + 1) / (n - k) if n > k else float("nan")
    return {"H": float(H), "p": float(p), "dof": k - 1, "eta2": float(eta2), "k": k, "n": n}
