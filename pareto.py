"""
Pareto frontier utilities for multi-objective ranking.

A point dominates another if it is at least as good on every objective
and strictly better on at least one. The Pareto frontier (Pareto set)
is the set of non-dominated points — no point on the frontier is
strictly worse than any other on every dimension.

Conventions used here:
    Higher is better on every objective.
    Callers normalize each objective to [0, 1] before passing it in
    (cost_score = 1 - cost, latency_score = 1 - latency, etc.).
"""

from typing import List, Sequence, Tuple


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """True iff a dominates b (>= on all, > on at least one)."""
    any_strictly_better = False
    for ai, bi in zip(a, b):
        if ai < bi:
            return False
        if ai > bi:
            any_strictly_better = True
    return any_strictly_better


def pareto_frontier_indices(points: List[Sequence[float]]) -> List[int]:
    """
    Return the indices of points that lie on the Pareto frontier.
    O(n^2) — fine for our scale (single-digit candidates).
    """
    n = len(points)
    frontier: List[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if dominates(points[j], points[i]):
                dominated = True
                break
        if not dominated:
            frontier.append(i)
    return frontier


def weighted_pick(
    points: List[Sequence[float]],
    preferences: Sequence[float],
    restrict_to_frontier: bool = True,
) -> Tuple[int, float]:
    """
    Pick the index of the point that maximizes preference @ point,
    optionally restricted to the Pareto frontier.

    For strictly positive preference weights, restricting to the
    frontier is a no-op (the dot-product maximizer is always Pareto-
    optimal). The flag exists for the constrained / mixed-sign case
    and for the eval, where we want to *report* whether the chosen
    point lies on the frontier.
    """
    if not points:
        raise ValueError("empty points list")
    if len(preferences) != len(points[0]):
        raise ValueError(
            f"preference dim {len(preferences)} != point dim {len(points[0])}"
        )

    if restrict_to_frontier:
        idxs = pareto_frontier_indices(points)
    else:
        idxs = list(range(len(points)))

    best_idx = idxs[0]
    best_score = sum(p * v for p, v in zip(preferences, points[best_idx]))
    for i in idxs[1:]:
        score = sum(p * v for p, v in zip(preferences, points[i]))
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx, best_score


def normalize_preferences(prefs: dict, keys: Sequence[str]) -> List[float]:
    """
    Project a preferences dict onto `keys` in order, missing keys -> 0.
    Normalize so the projection sums to 1 (avoids weighting bias from
    different callers using different scales).
    """
    raw = [float(prefs.get(k, 0.0)) for k in keys]
    s = sum(raw)
    if s <= 0:
        # Default to uniform if the user gave us nothing usable.
        return [1.0 / len(keys)] * len(keys)
    return [r / s for r in raw]
