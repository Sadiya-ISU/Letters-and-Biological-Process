"""Seeded synthetic ontologies: signal/noise/mixed discrimination and a
vocabulary-evolution simulator (inheritance + mutation + branch specialization)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

GENERIC = ["cell", "process", "protein", "signal", "pathway", "metabolism",
           "cycle", "regulation", "transport", "binding"]
SPECIAL = ["mitochondrial", "apoptotic", "ribosomal", "transcriptional",
           "glycolytic", "phosphorylation", "lysosomal", "cytoskeletal"]
NOISE = ["alpha", "beta", "gamma", "theta", "sigma", "random", "delta"]
MODIFIERS = ["positive", "negative", "activation", "inhibition", "regulation",
             "response", "assembly", "disassembly"]


def generate_dataset(mode: str, n_terms: int = 5000, max_depth: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    levels = rng.integers(0, max_depth, n_terms)
    names = []
    for lvl in levels:
        if mode == "signal":
            n_words = 2 + int(lvl / 2)
            vocab = GENERIC + SPECIAL[: int(lvl) % len(SPECIAL)]
        elif mode == "noise":
            n_words = int(rng.integers(2, 7))
            vocab = GENERIC + SPECIAL + NOISE
        elif mode == "mixed":
            if rng.random() < 0.5:
                n_words = 2 + int(lvl / 2)
                vocab = GENERIC + SPECIAL
            else:
                n_words = int(rng.integers(2, 7))
                vocab = GENERIC + SPECIAL + NOISE
        else:
            raise ValueError(mode)
        words = [vocab[i] for i in rng.integers(0, len(vocab), max(1, n_words))]
        names.append(" ".join(words))
    df = pd.DataFrame({"Name": names, "Level": levels})
    df["Length"] = df["Name"].str.len()
    return df


def signal_noise_separation(reps: int = 50, seed: int = 0) -> pd.DataFrame:
    rows = []
    for mode in ("signal", "noise", "mixed"):
        corrs = [spearmanr(df["Level"], df["Length"]).statistic
                 for r in range(reps)
                 for df in [generate_dataset(mode, seed=seed * 1000 + r)]]
        rows.append(dict(mode=mode, mean_corr=float(np.mean(corrs)), sd_corr=float(np.std(corrs))))
    return pd.DataFrame(rows)


def simulate_vocab_evolution(max_depth: int = 12, branching: int = 4,
                             max_nodes: int = 25000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def mutate(parent_tokens, depth):
        tokens = list(parent_tokens)
        if rng.random() < min(0.3 + depth * 0.05, 0.8):
            tokens.append(SPECIAL[int(rng.integers(0, len(SPECIAL)))])
        if rng.random() < 0.4:
            tokens.insert(0, MODIFIERS[int(rng.integers(0, len(MODIFIERS)))])
        if rng.random() < 0.2 and len(tokens) > 1:
            tokens[int(rng.integers(0, len(tokens)))] = SPECIAL[int(rng.integers(0, len(SPECIAL)))]
        return tokens

    rows = [dict(node=0, depth=0, tokens=["biological", "process"])]
    frontier = [0]
    nid = 1
    while frontier and nid < max_nodes:
        parent = frontier.pop(0)
        pd_depth = rows[parent]["depth"]
        if pd_depth >= max_depth:
            continue
        for _ in range(int(rng.integers(1, branching + 1))):
            if nid >= max_nodes:
                break
            child = dict(node=nid, depth=pd_depth + 1, tokens=mutate(rows[parent]["tokens"], pd_depth))
            rows.append(child)
            frontier.append(nid)
            nid += 1
    out = pd.DataFrame([dict(node=r["node"], depth=r["depth"], name=" ".join(r["tokens"]),
                             length=len(" ".join(r["tokens"]))) for r in rows])
    return out
