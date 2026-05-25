"""
AgentRank service: rank candidate agents using log-derived base metrics
plus a UCB exploration bonus.

Score formula (per agent a, in domain d):

    base(a)    = sum_k weight_d[k] * metric_a[k]
    explore(a) = alpha_d * sqrt( ln(1 + N) / (1 + n_a) )
    score(a)   = base(a) + explore(a)

Where N is the total number of recorded invocations and n_a is the
number for this agent. The exploration term encourages trying agents
with few observations even when their current point estimate is lower
than the current leader. This is the standard UCB1 trick.
"""

import math
from typing import List, Dict, Any

from log_store import LogStore
from domain_registry import DomainRegistry
from config_loader import ScoringConfig


class AgentRankService:
    def __init__(
        self,
        log_store: LogStore,
        registry: DomainRegistry,
        config: ScoringConfig,
    ):
        self.log_store = log_store
        self.registry = registry
        self.config = config

    def rank(self, domain: str, task_type: str, payload: str) -> List[Dict[str, Any]]:
        """
        Returns candidate agents sorted by score (highest first).
        Each entry exposes the base score and exploration bonus separately
        so callers can show why a given agent was chosen.

        Honors the per-domain drift policy: when half_life_calls is set,
        both the metric averages and the UCB counts use exponentially-
        decayed weights, so old observations stop dominating the ranker.
        """
        candidates = self.registry.get_agents(domain, task_type)
        domain_key = f"{domain}/{task_type}"
        policy = self.config.policy_for(domain_key)
        weights = policy["weights"]
        alpha = policy["exploration"]["alpha"]
        half_life = policy.get("drift", {}).get("half_life_calls")

        # Single batched read of decayed counts so we don't re-scan logs
        # once per candidate.
        total_weight, per_agent_weight = self.log_store.get_weighted_counts(half_life)

        results: List[Dict[str, Any]] = []

        for agent_id in candidates:
            m = self.log_store.compute_metrics(agent_id, half_life_calls=half_life)

            base_score = (
                weights["success_rate"] * m["success_rate"]
                + weights["quality_score"] * m["quality_score"]
                + weights["latency_score"] * m["latency_score"]
                + weights["failure_rate"] * m["failure_rate"]
            )

            n_a = per_agent_weight.get(agent_id, 0.0)
            exploration_bonus = alpha * math.sqrt(
                math.log(1 + total_weight) / (1 + n_a)
            )

            final_score = base_score + exploration_bonus

            results.append(
                {
                    "agent_id": agent_id,
                    "score": round(final_score, 4),
                    "base_score": round(base_score, 4),
                    "exploration_bonus": round(exploration_bonus, 4),
                    "n_a": round(n_a, 2),
                    "metrics": m,
                }
            )

        results.sort(key=lambda r: r["score"], reverse=True)
        return results
