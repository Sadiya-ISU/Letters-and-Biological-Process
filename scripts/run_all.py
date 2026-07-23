"""Run the full revision pipeline end to end."""
import os
import runpy
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ORDER = ["reproduce_baseline.py", "run_depth_robustness.py", "run_entropy.py",
         "run_entropy_models.py", "run_clustering_kselection.py", "run_nulls.py",
         "run_synthetic.py", "run_namespaces.py", "make_figures.py"]


def main():
    seed = os.environ.get("PYTHONHASHSEED")
    if seed != "0":
        print(
            "\n*** WARNING: PYTHONHASHSEED is not set to '0' ***\n"
            "    Exact reproduction of the clustering step (KMeans) requires:\n"
            "        PYTHONHASHSEED=0 python3 scripts/run_all.py\n"
            "    Without it, set-based term ordering may shift cluster partitions.\n"
            "    (Note: setting os.environ inside this process has no effect —\n"
            "     the seed must be set BEFORE Python starts.)\n",
            flush=True,
        )

    for s in ORDER:
        print(f"\n{'=' * 70}\n# {s}\n{'=' * 70}", flush=True)
        runpy.run_path(str(SCRIPTS / s), run_name="__main__")
    print("\nALL DONE")


if __name__ == "__main__":
    main()
