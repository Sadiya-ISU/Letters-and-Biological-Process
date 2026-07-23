import json
from pathlib import Path
import numpy as np
from go_lingdepth import ROOT_BP
from go_lingdepth.obo_parser import parse_obo, parents, terms_in_namespace
from go_lingdepth.depth import compute_depths
from go_lingdepth.linguistics import letter_count
from go_lingdepth.nulls import permutation_null_corr, mc_entropy_envelope

ROOT = Path(__file__).resolve().parents[1]
OBO = ROOT / "data" / "go-basic.obo"
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)


def main():
    o = parse_obo(str(OBO))
    depths = compute_depths(parents(o, "is_a+part_of"), ROOT_BP, "shortest")
    ids = sorted(t for t in (terms_in_namespace(o, "biological_process") | {ROOT_BP})
                 if t in depths and t in o.names)
    names = [o.names[t] for t in ids]
    dep = np.array([depths[t] for t in ids])
    L = np.array([letter_count(n) for n in names])
    corr = permutation_null_corr(dep, L, n_iter=1000, seed=0)
    env = mc_entropy_envelope(names, dep, n_iter=1000, seed=42, stopwords="english")
    env.to_csv(RESULTS / "entropy_null_envelope.csv", index=False)
    (RESULTS / "null_corr_summary.json").write_text(json.dumps(corr, indent=2))
    print("permutation nulls:", corr)
    print(env.to_string(index=False))
    print("depths outside 1-99% null:", env.loc[env.outside, "depth"].tolist())


if __name__ == "__main__":
    main()
