"""
Selection strategies for the evaluation harness.

Each strategy implements the same minimal interface:
    select(candidates) -> agent_id
    update(agent_id, outcome) -> None

So the runner can swap them in and out without knowing internals.
"""

import random
from typing import List, Dict, Any, Optional

from log_store import LogStore
from domain_registry import DomainRegistry
from agent_rank_service import AgentRankService
from config_loader import ScoringConfig
from judge import QualityJudge


class Strategy:
    name: str = "base"

    def select(self, candidates: List[str], payload: str = "", preferences=None) -> str:
        raise NotImplementedError

    def update(self, agent_id: str, outcome: Dict[str, Any]) -> None:
        pass


class RandomStrategy(Strategy):
    """Uniformly random over candidates. Lower-bound baseline."""
    name = "random"

    def __init__(self, rng: random.Random):
        self._rng = rng

    def select(self, candidates: List[str], payload: str = "", preferences=None) -> str:
        return self._rng.choice(candidates)


class RoundRobinStrategy(Strategy):
    """Cycles through candidates in order."""
    name = "round_robin"

    def __init__(self):
        self._i = 0

    def select(self, candidates: List[str], payload: str = "", preferences=None) -> str:
        choice = candidates[self._i % len(candidates)]
        self._i += 1
        return choice


class GreedyPriorStrategy(Strategy):
    """
    Always picks the agent with the highest cold-start prior score.
    Never updates, never explores. This is what 'no AgentRank' looks
    like when the priors happen to be right.
    """
    name = "greedy_prior"

    def __init__(self, priors: Dict[str, Dict[str, float]], weights: Dict[str, float]):
        self._priors = priors
        self._weights = weights
        self._neutral = {
            "success_rate": 0.5,
            "quality_score": 0.5,
            "latency_score": 0.5,
            "failure_rate": 0.5,
        }

    def _score(self, agent_id: str) -> float:
        p = self._priors.get(agent_id, self._neutral)
        return (
            self._weights["success_rate"] * p["success_rate"]
            + self._weights["quality_score"] * p["quality_score"]
            + self._weights["latency_score"] * p["latency_score"]
            + self._weights["failure_rate"] * p["failure_rate"]
        )

    def select(self, candidates: List[str], payload: str = "", preferences=None) -> str:
        return max(candidates, key=self._score)


class EpsilonGreedyStrategy(Strategy):
    """
    Picks the best observed agent (by empirical reward) with probability
    1-epsilon, otherwise picks uniformly at random. Classic exploration
    baseline to compare UCB against.

    Note: epsilon-greedy uses the *claimed* quality from the outcome
    (same as AgentRank without a judge). If a lying agent is in the
    mix, this baseline will be fooled too.
    """
    name = "epsilon_greedy"

    def __init__(self, weights: Dict[str, float], rng: random.Random, epsilon: float = 0.1):
        from collections import defaultdict
        self._weights = weights
        self._rng = rng
        self._epsilon = epsilon
        self._reward_sum: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)

    def _mean_reward(self, agent_id: str) -> float:
        if self._counts[agent_id] == 0:
            return float("inf")  # try untried agents first
        return self._reward_sum[agent_id] / self._counts[agent_id]

    def select(self, candidates: List[str], payload: str = "", preferences=None) -> str:
        if self._rng.random() < self._epsilon:
            return self._rng.choice(candidates)
        return max(candidates, key=self._mean_reward)

    def update(self, agent_id: str, outcome: Dict[str, Any]) -> None:
        # epsilon-greedy learns from what the agent *claims*, not the truth.
        # We reconstruct reward from the claimed quality_score, not the true one.
        weights = self._weights
        s = int(outcome["success"])
        q = float(outcome["quality_score"])  # CLAIMED — that's the realistic case
        latency_score = 1.0 - min(float(outcome["latency_ms"]) / 3000.0, 1.0)
        observed_reward = (
            weights["success_rate"] * s
            + weights["quality_score"] * q
            + weights["latency_score"] * latency_score
            + weights["failure_rate"] * (1 - s)
        )
        self._reward_sum[agent_id] += observed_reward
        self._counts[agent_id] += 1


class AgentRankStrategy(Strategy):
    """
    The real AgentRank service running on an in-memory SQLite log store.

    When `judge` is supplied, the judge's score replaces the agent's
    self-reported quality_score before logging. In eval, pair this with
    OracleJudge — the simulator emits both claimed and true quality on
    each call, and OracleJudge returns the true value via the `hint`
    argument. This isolates the judge's *contribution* to the ranker.
    """
    name = "agent_rank"

    def __init__(
        self,
        base_config: ScoringConfig,
        domain: str,
        task_type: str,
        candidates: List[str],
        priors: Dict[str, Dict[str, float]],
        judge: Optional[QualityJudge] = None,
        variant_suffix: str = "",
    ):
        # Compose a per-scenario config view so the strategy sees the
        # scenario's priors and registry without touching the global config.
        cfg = base_config.with_priors(priors).with_registry(
            f"{domain}/{task_type}", candidates
        )
        self._log_store = LogStore(db_path=":memory:", config=cfg)
        registry = DomainRegistry(cfg)
        self._service = AgentRankService(self._log_store, registry, cfg)
        self._domain = domain
        self._task_type = task_type
        self._judge = judge
        if variant_suffix:
            self.name = f"agent_rank{variant_suffix}"

    def select(self, candidates: List[str], payload: str = "", preferences=None) -> str:
        ranking = self._service.rank(
            self._domain, self._task_type, payload, preferences=preferences,
        )
        return ranking[0]["agent_id"]

    def update(self, agent_id: str, outcome: Dict[str, Any]) -> None:
        # Defensive copy so we don't mutate the simulator's outcome dict
        # (other strategies in the same trial share the same record... actually
        # they don't — runner.py samples fresh per strategy — but copying is cheap).
        outcome = dict(outcome)
        if self._judge is not None and outcome.get("success"):
            verdict = self._judge.score(
                payload="",
                output={},
                hint=outcome.get("true_quality_score"),
            )
            outcome["agent_claimed_quality"] = outcome["quality_score"]
            outcome["quality_score"] = verdict.score
            outcome["judge_name"] = verdict.judge_name
            outcome["judge_reason"] = verdict.reason
        self._log_store.record_invocation(outcome)

    def close(self) -> None:
        self._log_store.close()
