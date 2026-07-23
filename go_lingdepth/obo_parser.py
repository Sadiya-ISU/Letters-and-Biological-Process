"""Parse go-basic.obo into explicit, separated relation sets.

Unlike the original notebook this (a) records namespace, (b) keeps is_a and
part_of in SEPARATE maps so depth can be computed over either edge set, and
(c) never silently folds part_of into 'is_a relations'.
"""
from __future__ import annotations
import collections
from dataclasses import dataclass, field


@dataclass
class Ontology:
    names: dict = field(default_factory=dict)
    namespaces: dict = field(default_factory=dict)
    is_a: dict = field(default_factory=lambda: collections.defaultdict(set))
    part_of: dict = field(default_factory=lambda: collections.defaultdict(set))
    obsolete: set = field(default_factory=set)
    data_version: str = "UNKNOWN"


def parse_obo(path: str) -> Ontology:
    o = Ontology()
    cid = None
    ns = None
    nm = None
    isa = set()
    pof = set()
    is_obs = False

    def flush():
        if cid is None:
            return
        if is_obs:
            o.obsolete.add(cid)
            return
        if nm is not None:
            o.names[cid] = nm
        if ns is not None:
            o.namespaces[cid] = ns
        if isa:
            o.is_a[cid] |= isa
        if pof:
            o.part_of[cid] |= pof

    with open(path, "r", encoding="utf-8") as f:
        in_term = False
        for raw in f:
            line = raw.rstrip("\n").strip()
            if line.startswith("data-version:"):
                o.data_version = line.split("data-version:", 1)[1].strip()
                continue
            if line == "[Term]":
                flush()
                in_term, cid, ns, nm, isa, pof, is_obs = True, None, None, None, set(), set(), False
                continue
            if line.startswith("[") and line != "[Term]":
                flush()
                in_term, cid = False, None
                continue
            if not in_term or not line:
                continue
            if line.startswith("id: "):
                cid = line[4:].strip()
            elif line.startswith("namespace: "):
                ns = line[11:].strip()
            elif line.startswith("name: "):
                nm = line[6:].strip()
            elif line.startswith("is_obsolete: "):
                is_obs = line.split("is_obsolete:", 1)[1].strip().lower() == "true"
            elif line.startswith("is_a: "):
                isa.add(line[6:].split(" ! ")[0].strip())
            elif line.startswith("relationship: "):
                parts = line.split()
                if len(parts) >= 3 and parts[1] == "part_of":
                    pof.add(parts[2].strip())
        flush()
    return o


def parents(o: Ontology, relation_set: str) -> dict:
    assert relation_set in ("is_a", "is_a+part_of")
    out = collections.defaultdict(set)
    for c, ps in o.is_a.items():
        out[c] |= ps
    if relation_set == "is_a+part_of":
        for c, ps in o.part_of.items():
            out[c] |= ps
    return out


def terms_in_namespace(o: Ontology, ns: str) -> set:
    return {t for t, v in o.namespaces.items() if v == ns and t not in o.obsolete}


def reachable_from(parent_map: dict, root: str) -> set:
    children = collections.defaultdict(set)
    for c, ps in parent_map.items():
        for p in ps:
            children[p].add(c)
    seen, stack = set(), [root]
    while stack:
        n = stack.pop()
        for ch in children.get(n, ()):
            if ch not in seen:
                seen.add(ch)
                stack.append(ch)
    return seen
