"""Major comment #1: do the length->depth and entropy->depth findings survive
alternative depth definitions? Emit a tidy grid the manuscript text reads from."""
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from go_lingdepth import ROOT_BP
from go_lingdepth.obo_parser import parse_obo, parents, terms_in_namespace, reachable_from
from go_lingdepth.depth import compute_depths
from go_lingdepth.linguistics import letter_count, depthwise_entropy

ROOT = Path(__file__).resolve().parents[1]
OBO = ROOT / "data" / "go-basic.obo"
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)
ROOT_ID = ROOT_BP


def node_universe(o, selection, pmap):
    if selection == "namespace":
        return (terms_in_namespace(o, "biological_process") | {ROOT_ID})
    return reachable_from(pmap, ROOT_ID) | {ROOT_ID}


def main():
    o = parse_obo(str(OBO))
    rows, ld_rows = [], []
    for selection, relset, metric in itertools.product(
            ["namespace", "reachable"], ["is_a", "is_a+part_of"], ["shortest", "longest", "average"]):
        pmap = parents(o, relset)
        universe = node_universe(o, selection, pmap)
        depths = compute_depths(pmap, ROOT_ID, metric)
        items = sorted((t, depths[t]) for t in universe if t in depths and t in o.names)
        if len(items) < 50:
            continue
        ids = [t for t, _ in items]
        d = np.array([dep for _, dep in items], dtype=float)
        L = np.array([letter_count(o.names[t]) for t in ids], dtype=float)
        pr, pp = pearsonr(d, L)
        sr, sp = spearmanr(d, L)
        slope, intercept = np.polyfit(d, L, 1)
        # entropy peak under this depth metric (integer-binned for non-shortest)
        ent = depthwise_entropy([o.names[t] for t in ids],
                                np.rint(d).astype(int), stopwords="english")
        peak = int(max(ent, key=ent.get)) if ent else -1
        ent_rho, _ = spearmanr(sorted(ent), [ent[k] for k in sorted(ent)]) if len(ent) > 2 else (np.nan, np.nan)
        rows.append(dict(namespace="BP", selection=selection, relation_set=relset, metric=metric,
                         n_terms=len(ids), max_depth=float(np.max(d)), mean_depth=float(np.mean(d)),
                         pearson_r=pr, pearson_p=pp, spearman_rho=sr, spearman_p=sp,
                         ols_slope=slope, ols_intercept=intercept,
                         entropy_peak_depth=peak, entropy_spearman_rho=ent_rho))
        for t, dep, ll in zip(ids, d, L):
            ld_rows.append(dict(selection=selection, relation_set=relset, metric=metric,
                                go_id=t, depth=dep, letters=ll))
    pd.DataFrame(rows).to_csv(RESULTS / "depth_robustness.csv", index=False)
    pd.DataFrame(ld_rows).to_csv(RESULTS / "length_depth_by_metric.csv", index=False)
    print(pd.DataFrame(rows)[["selection", "relation_set", "metric", "n_terms",
                              "max_depth", "pearson_r", "spearman_rho", "ols_slope",
                              "entropy_peak_depth"]].to_string(index=False))


if __name__ == "__main__":
    main()
