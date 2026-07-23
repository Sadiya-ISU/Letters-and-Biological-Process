"""Deterministic null models: label/depth permutation and the depth-wise
Monte-Carlo entropy envelope (the strongest non-parametric evidence for the
diversification-specialization curve — elevated in the Major #3 reframe)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .linguistics import tokenize, shannon_entropy, _POLICIES


def permutation_null_corr(depths, lengths, n_iter: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d = np.asarray(depths, float)
    L = np.asarray(lengths, float)
    obs = spearmanr(d, L).statistic
    name_null = np.empty(n_iter)
    depth_null = np.empty(n_iter)
    for i in range(n_iter):
        name_null[i] = spearmanr(d, rng.permutation(L)).statistic
        depth_null[i] = spearmanr(rng.permutation(d), L).statistic
    return {"observed": float(obs),
            "name_perm_p": float(np.mean(np.abs(name_null) >= abs(obs))),
            "depth_perm_p": float(np.mean(np.abs(depth_null) >= abs(obs)))}


def mc_entropy_envelope(names, depths, n_iter: int = 1000, seed: int = 42,
                        stopwords: str = "english") -> pd.DataFrame:
    stop = _POLICIES[stopwords]
    depths = np.asarray(depths)
    levels = np.arange(depths.min(), depths.max() + 1)
    pre = [tokenize(n, stop) for n in names]
    observed = np.array([shannon_entropy([t for j in np.where(depths == d)[0] for t in pre[j]])
                         for d in levels])
    rng = np.random.default_rng(seed)
    null = np.empty((n_iter, len(levels)))
    n = len(names)
    for it in range(n_iter):
        perm = rng.permutation(n)
        for i, d in enumerate(levels):
            idx = perm[depths == d]
            null[it, i] = shannon_entropy([t for j in idx for t in pre[j]])
    lo1, hi99, med = (np.percentile(null, q, axis=0) for q in (1, 99, 50))
    outside = (observed < lo1) | (observed > hi99)
    return pd.DataFrame(dict(depth=levels, observed=observed, null_lo1=lo1,
                             null_hi99=hi99, null_med=med, outside=outside))
