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
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable

import numpy as np


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
    # Per-call cost in cents (e.g. tokens * $/token estimate). Used by
    # Pareto / multi-objective scoring. Defaults to 1 cent, which puts
    # cost_score near 0.9 with the default 10c reference.
    cost_cents: float = 1.0
    # Optional context-conditional quality: a function mapping a feature
    # vector to the agent's TRUE quality mean for that request. When set,
    # this overrides quality_mean. Used to model "agent A is great on
    # short text, agent B on long" so contextual bandits have something
    # to learn.
    context_quality_fn: Optional[Callable[[np.ndarray], float]] = field(
        default=None, repr=False
    )

    @property
    def claim_mean(self) -> float:
        return (
            self.claimed_quality_mean
            if self.claimed_quality_mean is not None
            else self.quality_mean
        )

    def is_honest(self) -> bool:
        return abs(self.claim_mean - self.quality_mean) < 1e-9

    def true_quality_mean_for(self, features: Optional[np.ndarray]) -> float:
        if self.context_quality_fn is not None and features is not None:
            return float(self.context_quality_fn(features))
        return self.quality_mean

    def sample(
        self,
        rng: random.Random,
        features: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        success = 1 if rng.random() < self.success_prob else 0
        q_mean = self.true_quality_mean_for(features)
        if success:
            if self.quality_std > 0:
                true_q = rng.gauss(q_mean, self.quality_std)
                true_q = max(0.0, min(1.0, true_q))
            else:
                true_q = q_mean
        else:
            true_q = 0.0

        # Claim semantics:
        #   - claimed_quality_mean unset  -> HONEST: claim = true_q each call
        #     (the agent reports what it actually delivered, which can
        #     vary by context).
        #   - claimed_quality_mean set    -> LIAR: claim = fixed inflated
        #     value regardless of context (the lying-agent scenario).
        if not success:
            claim_q = 0.0
        elif self.claimed_quality_mean is None:
            claim_q = true_q
        else:
            claim_q = self.claimed_quality_mean

        latency_ms = rng.randint(self.latency_min_ms, self.latency_max_ms)
        return {
            "agent_id": self.agent_id,
            "success": success,
            "quality_score": claim_q,            # what the agent reports
            "true_quality_score": true_q,        # what really happened
            "latency_ms": latency_ms,
            "failure_reason": None if success else "synthetic_failure",
            "cost_cents": float(self.cost_cents),
        }

    def expected_reward(
        self,
        weights: Dict[str, float],
        features: Optional[np.ndarray] = None,
    ) -> float:
        """
        Analytic expected per-call reward using TRUE quality. When
        features are supplied and context_quality_fn is set, the
        expected reward conditions on context.
        """
        p = self.success_prob
        q_mean = self.true_quality_mean_for(features)
        e_quality = p * q_mean   # 0 on failure
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


def preference_reward(outcome: Dict[str, Any], preferences: Dict[str, float]) -> float:
    """
    Multi-objective reward for a single call given a per-request
    preference vector. Mirrors ParetoBandit's objective set:

        [quality_score, latency_score, cost_score, success_rate]

    All terms are in [0, 1] (higher is better). Used by scenarios where
    the caller's preferred tradeoff varies per request — UCB1's fixed
    weights can't optimize for that, but ParetoBandit can.

    Preferences are auto-normalized to sum to 1 so callers can pass any
    positive scaling.
    """
    s = int(outcome["success"])
    q = float(outcome.get("true_quality_score", outcome["quality_score"]))
    latency_score = 1.0 - min(float(outcome["latency_ms"]) / 3000.0, 1.0)
    cost_score = 1.0 - min(
        float(outcome.get("cost_cents", 1.0)) / 10.0, 1.0
    )
    objs = {
        "quality_score": q,
        "latency_score": latency_score,
        "cost_score": cost_score,
        "success_rate": float(s),
    }
    total = sum(max(0.0, float(preferences.get(k, 0.0))) for k in objs)
    if total <= 0:
        return 0.0
    return sum(
        objs[k] * max(0.0, float(preferences.get(k, 0.0))) / total
        for k in objs
    )
