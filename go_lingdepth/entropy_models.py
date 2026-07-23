"""Descriptive entropy-vs-depth summaries.

Major comment #3: with only ~14 depth points, R^2 model 'selection' is fragile.
We therefore (a) keep the quadratic ONLY as a descriptive trend line, (b) report
the empirical peak with a leave-one-depth-out stability interval, and (c) expose
AICc for supplementary transparency rather than headline validation.
"""
from __future__ import annotations
import numpy as np


def _r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _aicc(y, yhat, k_params):
    n = len(y)
    rss = float(np.sum((y - yhat) ** 2))
    if rss <= 0 or n <= k_params + 1:
        return float("nan")
    aic = n * np.log(rss / n) + 2 * k_params
    return aic + (2 * k_params * (k_params + 1)) / (n - k_params - 1)


def fit_models(depths, entropy) -> dict:
    d = np.asarray(depths, dtype=float)
    h = np.asarray(entropy, dtype=float)
    out = {}
    # linear
    a, b = np.polyfit(d, h, 1)
    yhat = a * d + b
    out["linear"] = {"params": [float(a), float(b)], "r2": _r2(h, yhat), "aicc": _aicc(h, yhat, 2)}
    # logarithmic (depth>0 only; log undefined at 0)
    m = d > 0
    la, lb = np.polyfit(np.log(d[m]), h[m], 1)
    yhat = la * np.log(d[m]) + lb
    out["log"] = {"params": [float(la), float(lb)], "r2": _r2(h[m], yhat), "aicc": _aicc(h[m], yhat, 2)}
    # quadratic
    c2, c1, c0 = np.polyfit(d, h, 2)
    yhat = c2 * d * d + c1 * d + c0
    out["quadratic"] = {"params": [float(c2), float(c1), float(c0)], "r2": _r2(h, yhat),
                        "aicc": _aicc(h, yhat, 3)}
    return out


def quadratic_peak(depths, entropy) -> float:
    c2, c1, _ = np.polyfit(np.asarray(depths, float), np.asarray(entropy, float), 2)
    return float(-c1 / (2 * c2))


def peak_leave_one_out(depths, entropy):
    d = np.asarray(depths, float)
    h = np.asarray(entropy, float)
    peaks = []
    for i in range(len(d)):
        keep = np.arange(len(d)) != i
        peaks.append(quadratic_peak(d[keep], h[keep]))
    return float(np.min(peaks)), float(np.max(peaks))
