"""Major comment #2: justify k. Sweep k=10..50, report validity metrics AND show
the depth-by-cluster Kruskal-Wallis effect is robust across k."""
from pathlib import Path
import numpy as np
import pandas as pd
from go_lingdepth import ROOT_BP
from go_lingdepth.obo_parser import parse_obo, parents, terms_in_namespace
from go_lingdepth.depth import compute_depths
from go_lingdepth.embeddings import embed_terms, pca_reduce
from go_lingdepth.clustering import kmeans_labels, kselection, kw_depth_by_cluster

ROOT = Path(__file__).resolve().parents[1]
OBO = ROOT / "data" / "go-basic.obo"
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)


def main():
    o = parse_obo(str(OBO))
    pmap = parents(o, "is_a+part_of")
    depths = compute_depths(pmap, ROOT_BP, "shortest")
    ids = sorted(t for t in (terms_in_namespace(o, "biological_process") | {ROOT_BP})
                 if t in depths and t in o.names)
    names = [o.names[t] for t in ids]
    dep = np.array([depths[t] for t in ids])
    X = pca_reduce(embed_terms(names, cache_path=str(RESULTS / "emb_BP.npy")), 50, seed=0)
    ks = list(range(10, 51, 5))
    val = kselection(X, ks, sample_size=5000, seed=0)
    kw_rows = []
    for k in ks:
        res = kw_depth_by_cluster(kmeans_labels(X, k, seed=0), dep)
        kw_rows.append(dict(k=k, kw_H=res["H"], kw_p=res["p"], eta2=res["eta2"]))
    df = val.merge(pd.DataFrame(kw_rows), on="k")
    df.to_csv(RESULTS / "kselection.csv", index=False)
    print(df.to_string(index=False))
    print("\nk* by silhouette:", int(df.loc[df.silhouette.idxmax(), "k"]),
          "| by Davies-Bouldin(min):", int(df.loc[df.davies_bouldin.idxmin(), "k"]))
    print("KW p<1e-300 at every k:", bool((df.kw_p < 1e-300).all()))


if __name__ == "__main__":
    main()
