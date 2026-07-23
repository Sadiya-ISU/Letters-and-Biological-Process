"""Token/letter/entropy primitives with explicit, documented stop-word policy.

Answers Minor comment #3: the manuscript must state exactly which stop-words are
removed. We expose THREE policies and report entropy under each so the
diversification-specialization peak can be shown robust to the choice.
"""
from __future__ import annotations
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS_ENGLISH = frozenset(ENGLISH_STOP_WORDS)

# GO-ubiquitous generic tokens that dominate token frequencies (reviewer's
# examples: "process", "regulation", "cell"). Kept explicit for the manuscript.
_DOMAIN_EXTRA = frozenset({
    "process", "regulation", "regulated", "regulating", "positive", "negative",
    "cell", "cellular", "activity", "response", "involved", "mediated",
    "dependent", "via", "system", "biological", "molecular", "function",
})
STOPWORDS_DOMAIN = STOPWORDS_ENGLISH | _DOMAIN_EXTRA

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_POLICIES = {"none": frozenset(), "english": STOPWORDS_ENGLISH, "domain": STOPWORDS_DOMAIN}


def letter_count(name: str) -> int:
    return sum(1 for c in name if c.isalpha())


def tokenize(name: str, stopwords=frozenset()) -> list:
    return [t for t in _TOKEN_RE.findall(name.lower()) if len(t) > 1 and t not in stopwords]


def shannon_entropy(tokens, base: float = 2.0) -> float:
    if not tokens:
        return 0.0
    counts = np.fromiter(Counter(tokens).values(), dtype=float)
    p = counts / counts.sum()
    return float(-(p * (np.log(p) / np.log(base))).sum())


def depthwise_entropy(names, depths, stopwords: str = "english", base: float = 2.0) -> dict:
    stop = _POLICIES[stopwords]
    bucket: dict = {}
    for nm, d in zip(names, depths):
        bucket.setdefault(int(d), []).extend(tokenize(nm, stop))
    return {d: shannon_entropy(toks, base) for d, toks in bucket.items() if toks}
