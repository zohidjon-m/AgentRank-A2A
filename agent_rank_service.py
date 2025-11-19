# agent_rank_service.py
"""
AgentRank service: given a domain/task, rank candidate agents using log-based metrics.
"""

from typing import List, Dict, Any
from log_store import LogStore
from domain_registry import DomainRegistry


class AgentRankService:
    def __init__(self, log_store: LogStore, registry: DomainRegistry):
        self.log_store = log_store
        self.registry = registry

    def rank(self, domain: str, task_type: str, payload: str) -> List[Dict[str, Any]]:
        """
        Returns a sorted list of agents with scores and underlying metrics.
        Highest score first.
        """
        candidates = self.registry.get_agents(domain, task_type)
        results: List[Dict[str, Any]] = []

        for agent_id in candidates:
            m = self.log_store.compute_metrics(agent_id)

            # very simple scoring function for demo
            score = (
                0.4 * m["success_rate"]
                + 0.3 * m["quality_score"]
                + 0.2 * m["latency_score"]
                - 0.1 * m["failure_rate"]
            )

            results.append(
                {
                    "agent_id": agent_id,
                    "score": round(score, 4),
                    "metrics": m,
                }
            )

        results.sort(key=lambda r: r["score"], reverse=True)
        return results
