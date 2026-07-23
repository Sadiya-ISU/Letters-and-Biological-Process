"""Depth-wise vocabulary entropy for BP under none/english/domain stop-words.
Shows the bell-shaped peak is not an artifact of which generic tokens are kept."""
from pathlib import Path
import numpy as np
import pandas as pd
from go_lingdepth import ROOT_BP
from go_lingdepth.obo_parser import parse_obo, parents, terms_in_namespace
from go_lingdepth.depth import compute_depths
from go_lingdepth.linguistics import depthwise_entropy

ROOT = Path(__file__).resolve().parents[1]
OBO = ROOT / "data" / "go-basic.obo"
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)


def main():
    o = parse_obo(str(OBO))
    pmap = parents(o, "is_a+part_of")              # matches submitted baseline
    depths = compute_depths(pmap, ROOT_BP, "shortest")
    ids = sorted(t for t in (terms_in_namespace(o, "biological_process") | {ROOT_BP})
                 if t in depths and t in o.names)
    names = [o.names[t] for t in ids]
    dep = np.array([depths[t] for t in ids])
    out = {pol: depthwise_entropy(names, dep, stopwords=pol) for pol in ("none", "english", "domain")}
    levels = sorted(set().union(*[set(d) for d in out.values()]))
    counts = pd.Series(dep).value_counts().to_dict()
    df = pd.DataFrame([dict(namespace="BP", depth=L,
                            entropy_none=out["none"].get(L, np.nan),
                            entropy_english=out["english"].get(L, np.nan),
                            entropy_domain=out["domain"].get(L, np.nan),
                            n_terms=int(counts.get(L, 0))) for L in levels])
    df.to_csv(RESULTS / "entropy_by_depth.csv", index=False)
    print(df.to_string(index=False))
    for pol in ("none", "english", "domain"):
        peak = max(out[pol], key=out[pol].get)
        print(f"  peak depth ({pol} stop-words): {peak}")


if __name__ == "__main__":
    main()
