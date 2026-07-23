"""Confirm the parser reproduces the submitted manuscript's fingerprint.

Submitted notebook (is_a+part_of, reachable from GO:0008150) reported:
  reachable nodes = 25153 ; max shortest-path depth = 13.
If the current GO release does not match, the qualitative findings still hold,
but to reproduce the EXACT published numbers re-run download_go.py against the
release that matches this fingerprint (try a few release_archive dates).
"""
import collections
from pathlib import Path
from go_lingdepth import ROOT_BP
from go_lingdepth.obo_parser import parse_obo, parents, reachable_from

OBO = Path(__file__).resolve().parents[1] / "data" / "go-basic.obo"


def shortest_max_depth(parent_map, root, nodes):
    children = collections.defaultdict(set)
    for c, ps in parent_map.items():
        for p in ps:
            children[p].add(c)
    level = {root: 0}
    q = collections.deque([root])
    while q:
        n = q.popleft()
        for ch in children.get(n, ()):
            if ch not in level or level[n] + 1 < level[ch]:
                level[ch] = level[n] + 1
                q.append(ch)
    return max(level[n] for n in nodes if n in level)


def main():
    o = parse_obo(str(OBO))
    print("data-version:", o.data_version)
    pm = parents(o, "is_a+part_of")
    reach = reachable_from(pm, ROOT_BP)
    nodes = reach | {ROOT_BP}
    md = shortest_max_depth(pm, ROOT_BP, nodes)
    print(f"reachable-from-BP (is_a+part_of): {len(reach)}  (submitted: 25153)")
    print(f"max shortest-path depth: {md}  (submitted: 13)")
    print(f"BP namespace terms: {len([t for t,v in o.namespaces.items() if v=='biological_process' and t not in o.obsolete])}")


if __name__ == "__main__":
    main()
