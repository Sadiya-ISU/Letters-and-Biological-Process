"""Regenerate all figures from results/. Run after Tasks 4,6,9,10,11,12."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from go_lingdepth import ROOT_BP
from go_lingdepth.obo_parser import parse_obo, parents, terms_in_namespace
from go_lingdepth.depth import compute_depths
from go_lingdepth.linguistics import letter_count
from go_lingdepth.embeddings import embed_terms, pca_reduce
from go_lingdepth.clustering import kmeans_labels
from go_lingdepth.entropy_models import quadratic_peak, peak_leave_one_out, fit_models
from go_lingdepth.plotting import save, dump_alt_text

ROOT = Path(__file__).resolve().parents[1]
OBO = ROOT / "data" / "go-basic.obo"
RES = ROOT / "results"
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)


def _bp_frame(o):
    depths = compute_depths(parents(o, "is_a+part_of"), ROOT_BP, "shortest")
    ids = sorted(t for t in (terms_in_namespace(o, "biological_process") | {ROOT_BP})
                 if t in depths and t in o.names)
    return pd.DataFrame(dict(go=ids, name=[o.names[t] for t in ids],
                             depth=[depths[t] for t in ids],
                             letters=[letter_count(o.names[t]) for t in ids]))


def fig1_length_depth(df):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    ax[0].scatter(df.depth, df.letters, s=6, alpha=0.3, color="teal")
    s, b = np.polyfit(df.depth, df.letters, 1)
    xs = np.arange(df.depth.min(), df.depth.max() + 1)
    ax[0].plot(xs, s * xs + b, "r-", lw=2, label=f"OLS: {s:.2f}·depth + {b:.2f}")
    ax[0].set(xlabel="GO depth", ylabel="Term-name letters"); ax[0].legend()
    sns.violinplot(data=df, x="depth", y="letters", cut=0, density_norm="width",
                   color="#a6cee3", ax=ax[1])
    ax[1].set(xlabel="GO depth", ylabel="Term-name letters")
    fig.tight_layout()
    save(fig, FIG, "fig1_length_depth",
         "Two panels. Left: scatter of GO Biological Process term-name letter count "
         "versus ontology depth with a positive ordinary-least-squares trend line. "
         "Right: violin plots by depth showing higher and broader letter counts at intermediate-to-deep levels.")


def fig2_umap_clusters(df):
    import umap
    X = pca_reduce(embed_terms(df.name.tolist(), cache_path=str(RES / "emb_BP.npy")), 50, seed=0)
    labels = kmeans_labels(X, 20, seed=0)
    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(X)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=labels, palette="tab20", s=6,
                    alpha=0.6, legend=False, ax=ax[0])
    ax[0].set(xlabel="UMAP-1", ylabel="UMAP-2", title="Semantic space (k=20)")
    d = pd.DataFrame(dict(Cluster=labels, Depth=df.depth.values))
    sns.boxplot(data=d, x="Cluster", y="Depth", fliersize=1, ax=ax[1])
    ax[1].set(title="Depth by cluster")
    fig.tight_layout()
    save(fig, FIG, "fig2_umap_clusters",
         "Two panels. Left: UMAP projection of PCA-reduced sentence-transformer embeddings of GO term "
         "names, colored by k-means cluster, showing separated semantic groups. Right: boxplots of "
         "ontology depth per cluster, showing clusters occupy systematically different depth ranges.")


def fig3_entropy(df):
    ent = pd.read_csv(RES / "entropy_by_depth.csv")
    env = pd.read_csv(RES / "entropy_null_envelope.csv")
    d = ent.depth.values.astype(float)
    h = ent.entropy_english.values.astype(float)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), dpi=300)
    # Panel A: empirical + DESCRIPTIVE dashed quadratic guide (no R^2 claim)
    ax[0].plot(d, h, "o-", color="tab:blue", lw=1.8, label="Observed entropy")
    c2, c1, c0 = np.polyfit(d, h, 2)
    xs = np.linspace(d.min(), d.max(), 100)
    ax[0].plot(xs, c2 * xs**2 + c1 * xs + c0, "--", color="0.4", lw=1.6,
               label="Descriptive quadratic guide")
    peak = quadratic_peak(d, h); lo, hi = peak_leave_one_out(d, h)
    emp_peak = int(d[int(np.argmax(h))])
    ax[0].axvspan(lo, hi, color="orange", alpha=0.15)
    ax[0].axvline(emp_peak, color="tab:orange", ls="-", lw=2.0,
                  label=f"Empirical peak (depth {emp_peak})")
    ax[0].axvline(peak, color="0.5", ls=":", lw=1.4,
                  label=f"Quadratic vertex ≈ {peak:.1f} (LOO {lo:.1f}–{hi:.1f})")
    ax[0].set(xlabel="GO depth", ylabel="Vocabulary entropy (bits)"); ax[0].legend(fontsize=8)
    # Panel B: Monte-Carlo null envelope
    ax[1].fill_between(env.depth, env.null_lo1, env.null_hi99, color="0.8", label="Null 1–99%")
    ax[1].plot(env.depth, env.null_med, "--", color="0.3", lw=1.2, label="Null median")
    ax[1].plot(env.depth, env.observed, "o-", color="tab:blue", lw=1.8, label="Observed")
    ax[1].set(xlabel="GO depth", ylabel="Vocabulary entropy (bits)"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, FIG, "fig3_entropy",
         "Two panels. Left: observed bell-shaped vocabulary-entropy curve over GO depth with a dashed "
         "descriptive quadratic guide and a shaded leave-one-out interval marking the intermediate peak. "
         "Right: Monte-Carlo null envelope (1–99 percent) from random term-to-node reassignment, with the "
         "observed entropy curve falling outside the null band across most depths.")


def fig4_synthetic():
    from go_lingdepth.synthetic import generate_dataset, simulate_vocab_evolution
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    for mode, col in [("signal", "tab:green"), ("mixed", "tab:orange"), ("noise", "0.5")]:
        corrs = [generate_dataset(mode, seed=r).corr(numeric_only=True).loc["Level", "Length"]
                 for r in range(50)]
        ax[0].hist(corrs, bins=20, alpha=0.6, color=col, label=mode)
    ax[0].set(xlabel="Detected Spearman correlation", ylabel="Frequency",
              title="Signal vs noise discrimination"); ax[0].legend()
    ev = simulate_vocab_evolution(max_depth=12, branching=4, max_nodes=25000, seed=0)
    g = ev.groupby("depth")["length"].mean()
    ax[1].plot(g.index, g.values, "o-", color="tab:purple")
    ax[1].set(xlabel="Simulated depth", ylabel="Mean name length",
              title="Vocabulary-evolution model")
    fig.tight_layout()
    save(fig, FIG, "fig4_validation",
         "Two panels. Left: overlaid histograms of detected length–depth correlations for synthetic "
         "ontologies under signal, mixed, and noise regimes, well separated. Right: mean simulated "
         "term-name length increasing with depth under the inheritance–mutation–specialization model.")


def figS1_depth_robustness():
    dr = pd.read_csv(RES / "depth_robustness.csv")
    sub = dr[(dr.selection == "reachable") & (dr.relation_set == "is_a+part_of")]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.bar(sub.metric, sub.spearman_rho, color=["#1b9e77", "#d95f02", "#7570b3"])
    for x, (_, r) in zip(range(len(sub)), sub.iterrows()):
        ax.text(x, r.spearman_rho + 0.005, f"peak d={int(r.entropy_peak_depth)}", ha="center", fontsize=9)
    ax.set(ylabel="Length–depth Spearman ρ", title="Robustness across depth definitions")
    fig.tight_layout()
    save(fig, FIG, "figS1_depth_robustness",
         "Bar chart of the length–depth Spearman correlation under shortest, longest, and average path "
         "depth definitions, all positive and similar in magnitude, each annotated with its intermediate entropy-peak depth.")


def figS2_kselection():
    k = pd.read_csv(RES / "kselection.csv")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), dpi=300)
    ax[0].plot(k.k, k.inertia, "o-"); ax[0].set(xlabel="k", ylabel="Inertia", title="Elbow")
    ax[1].plot(k.k, k.silhouette, "o-", color="tab:green"); ax[1].plot(k.k, -k.davies_bouldin, "s--", color="tab:red")
    ax[1].set(xlabel="k", title="Silhouette (green) / −Davies–Bouldin (red)")
    ax[2].semilogy(k.k, k.kw_p.clip(lower=1e-320), "o-", color="tab:purple")
    ax[2].set(xlabel="k", ylabel="Kruskal–Wallis p (log)", title="Depth-by-cluster p across k")
    fig.tight_layout()
    save(fig, FIG, "figS2_kselection",
         "Three panels. Left: k-means inertia versus k (elbow). Middle: silhouette score and negative "
         "Davies–Bouldin index versus k. Right: Kruskal–Wallis p-value for depth differences across clusters, "
         "remaining astronomically small for every k from 10 to 50.")


def figS3_namespaces():
    ent = pd.read_csv(RES / "namespace_entropy_by_depth.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    for ns, col in [("biological_process", "tab:blue"), ("molecular_function", "tab:green"),
                    ("cellular_component", "tab:orange")]:
        s = ent[ent.namespace == ns]
        ax.plot(s.depth, s.entropy, "o-", color=col, label=ns.replace("_", " "))
    ax.set(xlabel="GO depth", ylabel="Vocabulary entropy (bits)",
           title="Entropy–depth across GO namespaces"); ax.legend()
    fig.tight_layout()
    save(fig, FIG, "figS3_namespaces",
         "Vocabulary-entropy curves over depth for Biological Process, Molecular Function, and Cellular "
         "Component namespaces, showing whether the intermediate-depth diversification peak recurs in each.")


def figS4_stopwords():
    e = pd.read_csv(RES / "entropy_by_depth.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    for col, lab, sty in [("entropy_none", "no stop-words", "o-"),
                          ("entropy_english", "English stop-words", "s--"),
                          ("entropy_domain", "English + domain", "^:")]:
        ax.plot(e.depth, e[col], sty, label=lab)
    ax.set(xlabel="GO depth", ylabel="Vocabulary entropy (bits)",
           title="Entropy peak is robust to stop-word policy"); ax.legend()
    fig.tight_layout()
    save(fig, FIG, "figS4_stopwords",
         "Three vocabulary-entropy curves over depth computed with no stop-words, English stop-words, and "
         "English-plus-domain stop-words; all retain the same intermediate-depth peak.")


def figS5_entropy_models():
    e = pd.read_csv(RES / "entropy_by_depth.csv")
    d = e.depth.values.astype(float); h = e.entropy_english.values.astype(float)
    m = fit_models(d, h)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    ax.plot(d, h, "ko", label="Observed", ms=5)
    xs = np.linspace(max(d.min(), 0.01), d.max(), 200)
    la, lb = m["linear"]["params"]; ax.plot(xs, la * xs + lb, "-", color="tab:blue", lw=2, label="Linear")
    ga, gb = m["log"]["params"]; ax.plot(xs[xs > 0], ga * np.log(xs[xs > 0]) + gb, "--", color="tab:red", lw=2, label="Logarithmic")
    c2, c1, c0 = m["quadratic"]["params"]; ax.plot(xs, c2 * xs**2 + c1 * xs + c0, ":", color="tab:green", lw=2.5, label="Quadratic")
    ax.set(xlabel="GO depth", ylabel="Vocabulary entropy (bits)",
           title="Descriptive trend comparison (no formal model selection)"); ax.legend()
    fig.tight_layout()
    save(fig, FIG, "figS5_entropy_models",
         "Observed entropy points with three descriptive trend lines — linear (solid blue), logarithmic "
         "(dashed red), and quadratic (dotted green) — distinguished by style and color; the quadratic "
         "tracks the non-monotonic peak while the others do not. Shown for description only, not model selection.")


def main():
    o = parse_obo(str(OBO))
    df = _bp_frame(o)
    fig1_length_depth(df)
    fig2_umap_clusters(df)
    fig3_entropy(df)
    fig4_synthetic()
    figS1_depth_robustness()
    figS2_kselection()
    figS3_namespaces()
    figS4_stopwords()
    figS5_entropy_models()
    dump_alt_text(str(FIG))
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
