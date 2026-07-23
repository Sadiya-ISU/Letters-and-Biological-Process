"""Major #4: full pipeline across BP/MF/CC. Does the diversification-specialization
entropy peak generalize? Emit per-namespace summaries + entropy curves."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from go_lingdepth import NAMESPACE_ROOTS
from go_lingdepth.obo_parser import parse_obo, parents, terms_in_namespace
from go_lingdepth.depth import compute_depths
from go_lingdepth.linguistics import letter_count, depthwise_entropy
from go_lingdepth.embeddings import embed_terms, pca_reduce
from go_lingdepth.clustering import kmeans_labels, kw_depth_by_cluster

ROOT = Path(__file__).resolve().parents[1]
OBO = ROOT / "data" / "go-basic.obo"
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)


def main():
    o = parse_obo(str(OBO))
    pmap = parents(o, "is_a+part_of")
    summary, ent_long, kw_rows = [], [], []
    for ns, root in NAMESPACE_ROOTS.items():
        depths = compute_depths(pmap, root, "shortest")
        ids = sorted(t for t in (terms_in_namespace(o, ns) | {root}) if t in depths and t in o.names)
        names = [o.names[t] for t in ids]
        dep = np.array([depths[t] for t in ids])
        L = np.array([letter_count(n) for n in names], dtype=float)
        pr, _ = pearsonr(dep, L)
        sr, _ = spearmanr(dep, L)
        slope, _ = np.polyfit(dep, L, 1)
        ent = depthwise_entropy(names, dep, stopwords="english")
        peak = int(max(ent, key=ent.get))
        for d, h in sorted(ent.items()):
            ent_long.append(dict(namespace=ns, depth=d, entropy=h))
        X = pca_reduce(embed_terms(names, cache_path=str(RESULTS / f"emb_{ns}.npy")), 50, seed=0)
        kw = kw_depth_by_cluster(kmeans_labels(X, 20, seed=0), dep)
        kw_rows.append(dict(namespace=ns, k=20, kw_H=kw["H"], kw_p=kw["p"], eta2=kw["eta2"]))
        summary.append(dict(namespace=ns, n_terms=len(ids), max_depth=int(dep.max()),
                            length_pearson_r=pr, length_spearman_rho=sr, ols_slope=slope,
                            entropy_peak_depth=peak, entropy_peak_value=float(ent[peak]),
                            kw_H_k20=kw["H"], kw_eta2_k20=kw["eta2"]))
    pd.DataFrame(summary).to_csv(RESULTS / "namespace_summary.csv", index=False)
    pd.DataFrame(ent_long).to_csv(RESULTS / "namespace_entropy_by_depth.csv", index=False)
    pd.DataFrame(kw_rows).to_csv(RESULTS / "namespace_kw.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
