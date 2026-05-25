"""
Synthetic agents with parametric reward distributions for evaluation.

Each agent is defined by a TruthSpec describing its *true* behavior:
success probability, conditional quality distribution, and latency
distribution. The simulator samples outcomes from these distributions
with a seeded RNG so eval runs are fully reproducible.

The reward function mirrors the AgentRank scoring formula, so the eval
measures exactly what the ranker is asked to optimize.

To model agents that misreport their own quality (a security threat
worth modelling), TruthSpec accepts a `claimed_quality_mean` that
defaults to the true quality_mean. When the agent lies, the simulator
returns both `quality_score` (what the agent claims) and
`true_quality_score` (what the user actually experiences). Strategies
that lack a judge see only the claim; strategies that use OracleJudge
in eval see the truth.
"""

import random
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class TruthSpec:
    """Ground-truth behavior of a synthetic agent."""
    agent_id: str
    success_prob: float                      # P(success)
    quality_mean: float                      # E[true quality | success]
    quality_std: float = 0.0                 # 0 means deterministic
    latency_min_ms: int = 100
    latency_max_ms: int = 100
    claimed_quality_mean: Optional[float] = None  # what the agent self-reports

    @property
    def claim_mean(self) -> float:
        return (
            self.claimed_quality_mean
            if self.claimed_quality_mean is not None
            else self.quality_mean
        )

    def is_honest(self) -> bool:
        return abs(self.claim_mean - self.quality_mean) < 1e-9

    def sample(self, rng: random.Random) -> Dict[str, Any]:
        success = 1 if rng.random() < self.success_prob else 0
        if success:
            if self.quality_std > 0:
                true_q = rng.gauss(self.quality_mean, self.quality_std)
                true_q = max(0.0, min(1.0, true_q))
            else:
                true_q = self.quality_mean
        else:
            true_q = 0.0

        claim_q = self.claim_mean if success else 0.0
        latency_ms = rng.randint(self.latency_min_ms, self.latency_max_ms)
        return {
            "agent_id": self.agent_id,
            "success": success,
            "quality_score": claim_q,            # what the agent reports
            "true_quality_score": true_q,        # what really happened
            "latency_ms": latency_ms,
            "failure_reason": None if success else "synthetic_failure",
        }

    def expected_reward(self, weights: Dict[str, float]) -> float:
        """Analytic expected per-call reward using TRUE quality."""
        p = self.success_prob
        e_quality = p * self.quality_mean   # 0 on failure
        avg_latency = (self.latency_min_ms + self.latency_max_ms) / 2
        e_latency_score = 1.0 - min(avg_latency / 3000.0, 1.0)
        return (
            weights["success_rate"] * p
            + weights["quality_score"] * e_quality
            + weights["latency_score"] * e_latency_score
            + weights["failure_rate"] * (1 - p)
        )


def call_reward(outcome: Dict[str, Any], weights: Dict[str, float]) -> float:
    """
    Scalar reward for a single observed call. Uses TRUE quality if the
    outcome carries it (i.e. came from the simulator), otherwise falls
    back to whatever quality_score is present. This ensures regret is
    always measured against what the user *actually* received, not what
    the agent claimed.
    """
    s = int(outcome["success"])
    q = float(outcome.get("true_quality_score", outcome["quality_score"]))
    latency_score = 1.0 - min(float(outcome["latency_ms"]) / 3000.0, 1.0)
    return (
        weights["success_rate"] * s
        + weights["quality_score"] * q
        + weights["latency_score"] * latency_score
        + weights["failure_rate"] * (1 - s)
    )
