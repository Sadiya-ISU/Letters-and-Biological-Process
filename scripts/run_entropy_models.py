"""Materialize results/entropy_model_summary.json for the manuscript text.

Reports the EMPIRICAL entropy peak (argmax over depth) as the headline value and
the quadratic-fit vertex + leave-one-out stability interval as descriptive
secondary values (Major #3 reframe: empirical first, parametric guide second).
The linear/log/quadratic fit r2/aicc are retained for the supplementary figure
only, never as headline model selection.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from go_lingdepth.entropy_models import fit_models, quadratic_peak, peak_leave_one_out

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main():
    df = pd.read_csv(RESULTS / "entropy_by_depth.csv")
    d = df["depth"].to_numpy(dtype=float)
    h = df["entropy_english"].to_numpy(dtype=float)
    empirical_peak = int(d[int(np.argmax(h))])
    vertex = quadratic_peak(d, h)
    lo, hi = peak_leave_one_out(d, h)
    fits = fit_models(d, h)
    summary = {
        "namespace": "BP",
        "stopwords": "english",
        "empirical_peak_depth": empirical_peak,
        "empirical_peak_entropy_bits": float(np.max(h)),
        "quadratic_vertex": round(float(vertex), 2),
        "loo_lo": round(float(lo), 2),
        "loo_hi": round(float(hi), 2),
        "fits": {k: {"r2": round(v["r2"], 3),
                     "aicc": (None if not np.isfinite(v["aicc"]) else round(v["aicc"], 2))}
                 for k, v in fits.items()},
    }
    (RESULTS / "entropy_model_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
