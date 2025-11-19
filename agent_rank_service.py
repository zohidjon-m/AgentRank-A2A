"""
AgentRankService (updated):
- Config-driven scoring weights
- Domain/task-specific scoring configs
- UCB-style exploration bonus for online learning
"""

import math
from typing import List, Dict, Any, Tuple
from log_store import LogStore
from domain_registry import DomainRegistry

# 1. GLOBAL DEFAULT WEIGHTS
DEFAULT_GLOBAL_CONFIG = {
    "weights": {
        "success_rate": 0.40,
        "quality_score": 0.30,
        "latency_score": 0.20,
        "failure_rate": -0.10,
    }
}

# 2. DOMAIN/TASK SPECIFIC WEIGHTS
# These override the above if the domain/task matches.
DOMAIN_SCORING_CONFIG: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("nlp", "summarize"): {
        "weights": {
            "success_rate": 0.30,
            "quality_score": 0.50,
            "latency_score": 0.10,
            "failure_rate": -0.10,
        }
    },
    ("nlp", "chat"): {
        "weights": {
            "success_rate": 0.40,
            "quality_score": 0.20,
            "latency_score": 0.30,
            "failure_rate": -0.10,
        }
    },
}


class AgentRankService:
    def __init__(
        self,
        log_store: LogStore,
        registry: DomainRegistry,
        default_config: Dict[str, Any] = None,
        exploration_coef: float = 0.20,
    ):
        self.log_store = log_store
        self.registry = registry
        self.default_config = default_config or DEFAULT_GLOBAL_CONFIG
        self.exploration_coef = exploration_coef

    # Get scoring config for this domain/task
    def _get_config(self, domain: str, task_type: str) -> Dict[str, Any]:
        return DOMAIN_SCORING_CONFIG.get((domain, task_type), self.default_config)


    # Compute base weighted score from metrics
    def _compute_base_score(self, metrics: Dict[str, float], config: Dict[str, Any]) -> float:
        weights = config["weights"]
        score = 0.0
        for k, w in weights.items():
            if k in metrics:
                score += w * metrics[k]
        return score

    # Add exploration bonus (UCB-like)
    def _apply_exploration_bonus(self, agent_id: str, base_score: float) -> float:
        N = self.log_store.total_calls()
        n_a = self.log_store.calls_for_agent(agent_id)

        if n_a == 0:
            # strong boost if never tested
            return base_score + self.exploration_coef

        bonus = self.exploration_coef * math.sqrt(
            math.log(1 + N) / (1 + n_a)
        )
        return base_score + bonus


    # Main ranking function
    def rank(self, domain: str, task_type: str, payload: str) -> List[Dict[str, Any]]:
        cfg = self._get_config(domain, task_type)
        candidates = self.registry.get_agents(domain, task_type)
        results: List[Dict[str, Any]] = []

        for agent_id in candidates:
            metrics = self.log_store.compute_metrics(agent_id)
            base = self._compute_base_score(metrics, cfg)
            final_score = self._apply_exploration_bonus(agent_id, base)

            results.append(
                {
                    "agent_id": agent_id,
                    "score": round(final_score, 4),
                    "base_score": round(base, 4),
                    "metrics": metrics,
                }
            )

        # Sort by final score
        results.sort(key=lambda r: r["score"], reverse=True)
        return results
