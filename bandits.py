"""
Pluggable bandit policies for AgentRank.

A BanditPolicy converts (candidates, observed history) into a ranked
list of agent records. Concrete implementations:

  UCB1Bandit   -- the classic UCB1 over aggregated metrics. Honors
                  concept-drift decay via half_life_calls. Context-blind.
  LinUCBBandit -- disjoint LinUCB (Li et al., 2010). Per-agent linear
                  reward model in feature space; the same agent can be
                  best for short text and worst for long text.

Both implementations share the same result schema so AgentRankService
treats them interchangeably.
"""

import json
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import numpy as np

from log_store import LogStore


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class BanditPolicy(ABC):
    name: str = "abstract"

    @abstractmethod
    def rank(
        self,
        candidates: List[str],
        weights: Dict[str, float],
        log_store: LogStore,
        context_features: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        ...


def _per_call_reward(success: int, quality: float, latency_ms: int, weights: Dict[str, float]) -> float:
    latency_score = 1.0 - min(float(latency_ms) / 3000.0, 1.0)
    return (
        weights["success_rate"] * success
        + weights["quality_score"] * quality
        + weights["latency_score"] * latency_score
        + weights["failure_rate"] * (1 - success)
    )


def _scalar_score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    return (
        weights["success_rate"] * metrics["success_rate"]
        + weights["quality_score"] * metrics["quality_score"]
        + weights["latency_score"] * metrics["latency_score"]
        + weights["failure_rate"] * metrics["failure_rate"]
    )


# ---------------------------------------------------------------------------
# UCB1
# ---------------------------------------------------------------------------


class UCB1Bandit(BanditPolicy):
    """
    Classic UCB1 with optional concept-drift decay.

      score(a) = base(a) + alpha * sqrt( ln(1 + N) / (1 + n_a) )

    When half_life_calls is set, both base metrics and the counts (N, n_a)
    are exponentially down-weighted by age.
    """
    name = "ucb1"

    def __init__(self, alpha: float, half_life_calls: Optional[float] = None):
        self._alpha = alpha
        self._half_life = half_life_calls

    def rank(self, candidates, weights, log_store, context_features=None):
        total_w, per_agent = log_store.get_weighted_counts(self._half_life)
        out: List[Dict[str, Any]] = []
        for agent_id in candidates:
            m = log_store.compute_metrics(agent_id, half_life_calls=self._half_life)
            base = _scalar_score(m, weights)
            n_a = per_agent.get(agent_id, 0.0)
            bonus = self._alpha * math.sqrt(math.log(1 + total_w) / (1 + n_a))
            out.append({
                "agent_id": agent_id,
                "score": round(base + bonus, 4),
                "base_score": round(base, 4),
                "exploration_bonus": round(bonus, 4),
                "n_a": round(n_a, 2),
                "metrics": m,
                "bandit": self.name,
            })
        out.sort(key=lambda r: r["score"], reverse=True)
        return out


# ---------------------------------------------------------------------------
# LinUCB (disjoint)
# ---------------------------------------------------------------------------


class LinUCBBandit(BanditPolicy):
    """
    Disjoint LinUCB. Per agent a:

      A_a   = ridge * I + sum_t x_t x_t^T   (over a's observations)
      b_a   = sum_t r_t * x_t
      theta = A_a^-1 b_a
      score = theta^T x + alpha * sqrt( x^T A_a^-1 x )

    Cold-start handling: LinUCB's confidence term is the only signal at
    cold start, and it ties across all agents for any fixed context, so
    naive selection just picks the first candidate. That candidate then
    accumulates data across *all* contexts before the others get a turn,
    which destroys per-context learning. We mitigate this with an
    explicit warm-start: until every candidate has been tried at least
    `warm_start_n` times, the least-explored candidate is forced to the
    top of the ranking. Once all agents clear the threshold, LinUCB
    takes over.

    The fit is reconstructed from logs each call. For our scale (low
    thousands of rows, d=4) the cost is negligible. For production
    scale you'd cache A and b and update incrementally.
    """
    name = "linucb"

    def __init__(
        self,
        alpha: float,
        feature_dim: int,
        ridge: float = 1.0,
        warm_start_n: int = 3,
    ):
        self._alpha = alpha
        self._d = feature_dim
        self._ridge = ridge
        self._warm_start_n = warm_start_n

    def rank(self, candidates, weights, log_store, context_features=None):
        if context_features is None:
            raise ValueError("LinUCBBandit requires context_features.")
        x = np.asarray(context_features, dtype=float).flatten()
        if x.shape[0] != self._d:
            raise ValueError(
                f"Feature dim mismatch: expected {self._d}, got {x.shape[0]}"
            )

        # Warm-start: bypass LinUCB until every candidate has at least
        # warm_start_n observations. The forced order is by ascending
        # call count, so all agents accumulate samples roughly evenly.
        if self._warm_start_n > 0:
            counts = {a: log_store.calls_for_agent(a) for a in candidates}
            under = [a for a in candidates if counts[a] < self._warm_start_n]
            if under:
                ordered = sorted(candidates, key=lambda a: counts[a])
                return [
                    {
                        "agent_id": a,
                        "score": 0.0,
                        "base_score": 0.0,
                        "exploration_bonus": 0.0,
                        "n_a": float(counts[a]),
                        "metrics": {
                            "linucb_mean": 0.0,
                            "linucb_confidence": 0.0,
                            "success_rate": 0.0,
                            "quality_score": 0.0,
                            "latency_score": 0.0,
                            "failure_rate": 0.0,
                        },
                        "bandit": self.name + "_warmup",
                    }
                    for a in ordered
                ]

        out: List[Dict[str, Any]] = []
        for agent_id in candidates:
            A, b, n_used = self._fit(agent_id, log_store, weights)
            A_inv = np.linalg.inv(A)
            theta = A_inv @ b
            mean = float(theta @ x)
            conf = self._alpha * float(np.sqrt(max(0.0, x @ A_inv @ x)))
            out.append({
                "agent_id": agent_id,
                "score": round(mean + conf, 4),
                "base_score": round(mean, 4),
                "exploration_bonus": round(conf, 4),
                "n_a": float(n_used),
                "metrics": {
                    # Surface LinUCB internals where the UCB1 schema
                    # expected the aggregate metrics. We zero those out
                    # so the formatting in agent_client.py still works.
                    "linucb_mean": mean,
                    "linucb_confidence": conf,
                    "success_rate": 0.0,
                    "quality_score": 0.0,
                    "latency_score": 0.0,
                    "failure_rate": 0.0,
                },
                "bandit": self.name,
            })
        out.sort(key=lambda r: r["score"], reverse=True)
        return out

    def _fit(self, agent_id, log_store, weights):
        A = self._ridge * np.eye(self._d)
        b = np.zeros(self._d)
        n_used = 0
        for row in log_store.get_contextual_logs(agent_id):
            feats_json = row["context_features"]
            if not feats_json:
                continue
            try:
                feats = np.asarray(json.loads(feats_json), dtype=float)
            except (ValueError, TypeError):
                continue
            if feats.shape != (self._d,):
                continue
            r = _per_call_reward(
                int(row["success"]),
                float(row["quality_score"]),
                int(row["latency_ms"]),
                weights,
            )
            A += np.outer(feats, feats)
            b += r * feats
            n_used += 1
        return A, b, n_used


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_bandit(policy: Dict[str, Any], feature_dim: int = 0) -> BanditPolicy:
    """
    Construct a bandit instance from a domain policy dict.

    bandit_params.alpha overrides exploration.alpha when present, so a
    contextual bandit can run hotter than the flat UCB1 default without
    affecting the demo's default policy.
    """
    kind = policy.get("bandit", "ucb1")
    bp = policy.get("bandit_params", {}) or {}
    alpha = float(bp.get("alpha", policy.get("exploration", {}).get("alpha", 0.5)))

    if kind == "ucb1":
        half_life = policy.get("drift", {}).get("half_life_calls")
        return UCB1Bandit(alpha=alpha, half_life_calls=half_life)

    if kind == "linucb":
        if feature_dim <= 0:
            raise ValueError(
                "LinUCB requires feature_dim > 0; configure feature_extractor "
                "for this domain."
            )
        ridge = float(bp.get("ridge", 1.0))
        warm = int(bp.get("warm_start_n", 3))
        return LinUCBBandit(
            alpha=alpha,
            feature_dim=feature_dim,
            ridge=ridge,
            warm_start_n=warm,
        )

    raise ValueError(f"Unknown bandit kind: {kind!r}")
