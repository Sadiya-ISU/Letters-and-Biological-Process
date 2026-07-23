"""Smoke and regression tests for go_lingdepth.

Scope is deliberately narrow: these pin the behaviours the manuscript makes claims
about, and they guard the two defects that made v1.0.0 of this archive unusable
(a package that could not be imported, and an ontology file that was not where the
scripts look for it). They do not attempt to re-run the full pipeline, which needs
sentence-transformers and several minutes of compute.

Run with:  pytest -q
"""
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------- packaging guards
def test_package_exports_namespace_roots():
    """Guards the v1.0.0 blocker: the init file was committed as `_init_`, so Python
    treated go_lingdepth as a namespace package and these names did not exist. Seven
    scripts began with `from go_lingdepth import ROOT_BP` and died on that line."""
    import go_lingdepth as g

    assert g.ROOT_BP == "GO:0008150"
    assert g.ROOT_MF == "GO:0003674"
    assert g.ROOT_CC == "GO:0005575"
    assert g.NAMESPACE_ROOTS["biological_process"] == g.ROOT_BP
    assert Path(g.__file__).name == "__init__.py"


def test_ontology_is_where_the_scripts_look_for_it():
    """Every script hardcodes ROOT/'data'/'go-basic.obo'."""
    assert (ROOT / "data" / "go-basic.obo").exists()


# --------------------------------------------------------------- linguistics
def test_letter_count_counts_letters_not_characters():
    """The manuscript reports a slope in *letters* per depth level; letter_count
    excludes spaces, digits and punctuation."""
    from go_lingdepth.linguistics import letter_count

    assert letter_count("cell death") == 9           # not 10: the space is excluded
    assert letter_count("T-cell activation 2") == 15  # hyphen, space and digit excluded
    assert letter_count("") == 0


def test_tokenize_drops_single_characters_and_lowercases():
    from go_lingdepth.linguistics import tokenize

    assert tokenize("Regulation of B cell") == ["regulation", "of", "cell"]


def test_shannon_entropy_matches_closed_form_in_bits():
    """Eq. 3 of the manuscript is base-2; a uniform distribution over n symbols has
    entropy log2(n)."""
    from go_lingdepth.linguistics import shannon_entropy

    assert shannon_entropy(["a", "b", "c", "d"]) == pytest.approx(2.0)
    assert shannon_entropy(["a"] * 7) == pytest.approx(0.0)
    assert shannon_entropy(list("abcdefgh")) == pytest.approx(math.log2(8))


def test_domain_stopword_list_matches_the_published_list():
    """Sec. 2.2 prints this list verbatim; it must not drift from the code."""
    from go_lingdepth.linguistics import _DOMAIN_EXTRA

    published = {
        "activity", "biological", "cell", "cellular", "dependent", "function",
        "involved", "mediated", "molecular", "negative", "positive", "process",
        "regulated", "regulating", "regulation", "response", "system", "via",
    }
    assert set(_DOMAIN_EXTRA) == published


# --------------------------------------------------------------- depth
def _toy_dag():
    """R -> a -> c, R -> b -> c, and a shortcut R -> c.

    c has three distinct root paths of length 1, 2 and 2, which is exactly the
    multi-parent shortcut case Reviewer 1 raised in major comment 1.
    """
    return {"a": ["R"], "b": ["R"], "c": ["a", "b", "R"], "R": []}


def test_compute_depths_shortest_longest_average_differ_on_a_shortcut():
    from go_lingdepth.depth import compute_depths

    parents = _toy_dag()
    shortest = compute_depths(parents, "R", metric="shortest")
    longest = compute_depths(parents, "R", metric="longest")
    average = compute_depths(parents, "R", metric="average")

    assert shortest["R"] == 0 and longest["R"] == 0
    assert shortest["a"] == longest["a"] == 1
    # the curated shortcut R->c makes c look shallow under shortest-path
    assert shortest["c"] == 1
    assert longest["c"] == 2
    assert 1 < average["c"] < 2


# --------------------------------------------------------------- obo parser
TINY_OBO = """format-version: 1.2
data-version: releases/2026-05-19

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process

[Term]
id: GO:0009987
name: cellular process
namespace: biological_process
is_a: GO:0008150 ! biological_process

[Term]
id: GO:0000001
name: obsolete thing
namespace: biological_process
is_obsolete: true

[Term]
id: GO:0044238
name: primary metabolic process
namespace: biological_process
is_a: GO:0008150 ! biological_process
relationship: part_of GO:0009987 ! cellular process
"""


def test_obo_parser_skips_obsolete_and_separates_relation_sets(tmp_path):
    from go_lingdepth.obo_parser import parse_obo, parents, terms_in_namespace

    f = tmp_path / "tiny.obo"
    f.write_text(TINY_OBO)
    o = parse_obo(str(f))

    assert o.names["GO:0009987"] == "cellular process"
    assert "GO:0000001" in o.obsolete
    # terms_in_namespace is the accessor the scripts use, and it filters obsolete terms
    assert terms_in_namespace(o, "biological_process") == {
        "GO:0008150", "GO:0009987", "GO:0044238"}

    is_a_only = parents(o, "is_a")
    both = parents(o, "is_a+part_of")
    assert set(is_a_only["GO:0044238"]) == {"GO:0008150"}
    assert set(both["GO:0044238"]) == {"GO:0008150", "GO:0009987"}, (
        "part_of must add a parent, otherwise the is_a / is_a+part_of comparison "
        "reported in Sec. 5.1 is meaningless")


# --------------------------------------------------------------- results regression
def _read_csv(name):
    import pandas as pd
    return pd.read_csv(ROOT / "results" / name)


def test_headline_numbers_in_the_manuscript_match_the_result_tables():
    """Ties the figures quoted in the paper to the shipped result tables, so a rerun
    that shifts them cannot silently disagree with the text."""
    import json

    dr = _read_csv("depth_robustness.csv")
    primary = dr[(dr.relation_set == "is_a+part_of") & (dr.metric == "shortest")
                 & (dr.selection == "namespace")].iloc[0]
    assert primary.n_terms == 24136                       # Sec. 2.1
    assert primary.max_depth == 13                        # "span depths 0--13"
    assert round(primary.pearson_r, 3) == 0.339           # Sec. 5.1
    assert round(primary.spearman_rho, 3) == 0.366        # Sec. 5.1
    assert round(primary.ols_slope, 2) == 3.22            # Eq. 7
    assert round(primary.ols_intercept, 2) == 19.68       # Eq. 7

    ks = _read_csv("kselection.csv")
    k20 = ks[ks.k == 20].iloc[0]
    assert round(k20.kw_H) == 5244                        # Abstract, Sec. 5.2
    assert round(k20.eta2, 3) == 0.217                    # Abstract, Sec. 5.2
    assert 0.16 < ks.eta2.min() and ks.eta2.max() < 0.26  # Sec. 2.3 range

    ent = json.loads((ROOT / "results" / "entropy_model_summary.json").read_text())
    assert ent["empirical_peak_depth"] == 4                        # Sec. 5.3
    assert round(ent["empirical_peak_entropy_bits"], 2) == 9.22    # Sec. 5.3
    assert ent["quadratic_vertex"] == 6.24                         # Sec. 5.3
    assert (ent["loo_lo"], ent["loo_hi"]) == (5.73, 6.37)          # "5.7--6.4"


def test_entropy_exceeds_the_null_band_only_at_depths_3_and_4():
    """The manuscript says the observed curve falls OUTSIDE the 1-99% band at 12 of
    14 depths. It is above the band at depths 3-4 and below it at 1 and 5-13 -- the
    direction is what makes the profile peaked rather than merely non-random."""
    env = _read_csv("entropy_null_envelope.csv")
    above = set(env[env.observed > env.null_hi99].depth)
    below = set(env[env.observed < env.null_lo1].depth)
    inside = set(env[~(env.observed > env.null_hi99) & ~(env.observed < env.null_lo1)].depth)

    assert above == {3, 4}
    assert below == {1, 5, 6, 7, 8, 9, 10, 11, 12, 13}
    assert inside == {0, 2}
    assert len(above | below) == 12


def test_no_entropy_permutation_p_value_is_published():
    """Guards a claim the manuscript used to make. mc_entropy_envelope returns the
    envelope only; the 0.0 in null_corr_summary.json is the LENGTH-DEPTH permutation
    p-value (its `observed` field is the length-depth rho), not an entropy test."""
    import json

    env = _read_csv("entropy_null_envelope.csv")
    assert not any("p" == c.lower() or c.lower().endswith("_p") for c in env.columns)

    summary = json.loads((ROOT / "results" / "null_corr_summary.json").read_text())
    assert round(summary["observed"], 3) == 0.366, (
        "null_corr_summary.json describes the length-depth correlation, not entropy")
