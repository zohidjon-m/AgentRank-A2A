"""
Synthetic agents with parametric reward distributions for evaluation.

Each agent is defined by a TruthSpec describing its *true* behavior:
success probability, conditional quality distribution, and latency
distribution. The simulator samples outcomes from these distributions
with a seeded RNG so eval runs are fully reproducible.

The reward function mirrors the AgentRank scoring formula, so the eval
measures exactly what the ranker is asked to optimize.
"""

import random
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TruthSpec:
    """Ground-truth behavior of a synthetic agent."""
    agent_id: str
    success_prob: float           # P(success)
    quality_mean: float           # E[quality | success]
    quality_std: float = 0.0      # 0 means deterministic quality
    latency_min_ms: int = 100
    latency_max_ms: int = 100

    def sample(self, rng: random.Random) -> Dict[str, Any]:
        success = 1 if rng.random() < self.success_prob else 0
        if success:
            if self.quality_std > 0:
                q = rng.gauss(self.quality_mean, self.quality_std)
                q = max(0.0, min(1.0, q))
            else:
                q = self.quality_mean
        else:
            q = 0.0
        latency_ms = rng.randint(self.latency_min_ms, self.latency_max_ms)
        return {
            "agent_id": self.agent_id,
            "success": success,
            "quality_score": q,
            "latency_ms": latency_ms,
            "failure_reason": None if success else "synthetic_failure",
        }

    def expected_reward(self, weights: Dict[str, float]) -> float:
        """Analytic expected per-call reward under the given weight policy."""
        p = self.success_prob
        e_quality = p * self.quality_mean  # quality is 0 on failure
        avg_latency = (self.latency_min_ms + self.latency_max_ms) / 2
        e_latency_score = 1.0 - min(avg_latency / 3000.0, 1.0)
        return (
            weights["success_rate"] * p
            + weights["quality_score"] * e_quality
            + weights["latency_score"] * e_latency_score
            + weights["failure_rate"] * (1 - p)
        )


def call_reward(outcome: Dict[str, Any], weights: Dict[str, float]) -> float:
    """Compute the scalar reward for a single observed call."""
    s = int(outcome["success"])
    q = float(outcome["quality_score"])
    latency_score = 1.0 - min(float(outcome["latency_ms"]) / 3000.0, 1.0)
    return (
        weights["success_rate"] * s
        + weights["quality_score"] * q
        + weights["latency_score"] * latency_score
        + weights["failure_rate"] * (1 - s)
    )
