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
        """
        candidates = self.registry.get_agents(domain, task_type)
        domain_key = f"{domain}/{task_type}"
        policy = self.config.policy_for(domain_key)
        weights = policy["weights"]
        alpha = policy["exploration"]["alpha"]

        total_calls = self.log_store.total_calls()
        results: List[Dict[str, Any]] = []

        for agent_id in candidates:
            m = self.log_store.compute_metrics(agent_id)

            base_score = (
                weights["success_rate"] * m["success_rate"]
                + weights["quality_score"] * m["quality_score"]
                + weights["latency_score"] * m["latency_score"]
                + weights["failure_rate"] * m["failure_rate"]
            )

            n_a = self.log_store.calls_for_agent(agent_id)
            exploration_bonus = alpha * math.sqrt(
                math.log(1 + total_calls) / (1 + n_a)
            )

            final_score = base_score + exploration_bonus

            results.append(
                {
                    "agent_id": agent_id,
                    "score": round(final_score, 4),
                    "base_score": round(base_score, 4),
                    "exploration_bonus": round(exploration_bonus, 4),
                    "n_a": n_a,
                    "metrics": m,
                }
            )

        results.sort(key=lambda r: r["score"], reverse=True)
        return results
