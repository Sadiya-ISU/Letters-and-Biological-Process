"""Shortest, longest, and average path depth from a root over a chosen edge set.

go-basic is acyclic, so longest/average are well defined via memoized DP up the
parent links. Average path length counts EVERY distinct root->node path, which
is what makes 'shallow shortcut' edges visibly diverge from the longest path.
"""
from __future__ import annotations
import collections


def _children(parent_map):
    ch = collections.defaultdict(set)
    for c, ps in parent_map.items():
        for p in ps:
            ch[p].add(c)
    return ch


def _shortest(parent_map, root):
    children = _children(parent_map)
    level = {root: 0}
    q = collections.deque([root])
    while q:
        n = q.popleft()
        for c in children.get(n, ()):
            cand = level[n] + 1
            if c not in level or cand < level[c]:
                level[c] = cand
                q.append(c)
    return level


def _topo_from_root(parent_map, root):
    """Nodes reachable from root, in topological order (parents before children)."""
    children = _children(parent_map)
    reach, stack = {root}, [root]
    while stack:
        n = stack.pop()
        for c in children.get(n, ()):
            if c not in reach:
                reach.add(c)
                stack.append(c)
    indeg = {n: 0 for n in reach}
    for n in reach:
        for c in children.get(n, ()):
            if c in reach:
                indeg[c] += 1  # only count in-edges from reachable parents
    # Kahn's algorithm seeded at root
    order, q = [], collections.deque([root])
    seen_indeg = dict(indeg)
    while q:
        n = q.popleft()
        order.append(n)
        for c in children.get(n, ()):
            if c in reach:
                seen_indeg[c] -= 1
                if seen_indeg[c] == 0:
                    q.append(c)
    return reach, order, children


def _longest(parent_map, root):
    reach, order, children = _topo_from_root(parent_map, root)
    depth = {root: 0}
    for n in order:
        base = depth.get(n, 0)
        for c in children.get(n, ()):
            if c in reach:
                depth[c] = max(depth.get(c, 0), base + 1)
    return depth


def _average(parent_map, root):
    reach, order, children = _topo_from_root(parent_map, root)
    npaths = {root: 1}
    sumlen = {root: 0}
    for n in order:  # parents finalized before children
        for c in children.get(n, ()):
            if c in reach:
                npaths[c] = npaths.get(c, 0) + npaths[n]
                sumlen[c] = sumlen.get(c, 0) + sumlen[n] + npaths[n]
    return {n: (0.0 if n == root else sumlen[n] / npaths[n]) for n in reach}


def compute_depths(parent_map: dict, root: str, metric: str) -> dict:
    if metric == "shortest":
        return _shortest(parent_map, root)
    if metric == "longest":
        return _longest(parent_map, root)
    if metric == "average":
        return _average(parent_map, root)
    raise ValueError(f"unknown metric {metric!r}")
