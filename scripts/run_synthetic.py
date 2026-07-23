from pathlib import Path
from go_lingdepth.synthetic import signal_noise_separation

RESULTS = Path(__file__).resolve().parents[1] / "results"; RESULTS.mkdir(exist_ok=True)
df = signal_noise_separation(reps=50, seed=0)
df.to_csv(RESULTS / "synthetic_corrs.csv", index=False)
print(df.to_string(index=False))
