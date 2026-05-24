"""
Selection strategies for the evaluation harness.

Each strategy implements the same minimal interface:
    select(candidates) -> agent_id
    update(agent_id, outcome) -> None

So the runner can swap them in and out without knowing internals.
"""

import math
import random
from typing import List, Dict, Any, Optional

from log_store import LogStore
from domain_registry import DomainRegistry
from agent_rank_service import AgentRankService
from config_loader import ScoringConfig


class Strategy:
    name: str = "base"

    def select(self, candidates: List[str]) -> str:
        raise NotImplementedError

    def update(self, agent_id: str, outcome: Dict[str, Any]) -> None:
        pass


class RandomStrategy(Strategy):
    """Uniformly random over candidates. Lower-bound baseline."""
    name = "random"

    def __init__(self, rng: random.Random):
        self._rng = rng

    def select(self, candidates: List[str]) -> str:
        return self._rng.choice(candidates)


class RoundRobinStrategy(Strategy):
    """Cycles through candidates in order."""
    name = "round_robin"

    def __init__(self):
        self._i = 0

    def select(self, candidates: List[str]) -> str:
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

    def select(self, candidates: List[str]) -> str:
        return max(candidates, key=self._score)


class EpsilonGreedyStrategy(Strategy):
    """
    Picks the best observed agent (by empirical reward) with probability
    1-epsilon, otherwise picks uniformly at random. Classic exploration
    baseline to compare UCB against.
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

    def select(self, candidates: List[str]) -> str:
        if self._rng.random() < self._epsilon:
            return self._rng.choice(candidates)
        return max(candidates, key=self._mean_reward)

    def update(self, agent_id: str, outcome: Dict[str, Any]) -> None:
        from .simulator import call_reward
        # epsilon-greedy needs the reward, not the raw outcome; we recompute it.
        r = call_reward(outcome, self._weights)
        self._reward_sum[agent_id] += r
        self._counts[agent_id] += 1


class AgentRankStrategy(Strategy):
    """
    The real AgentRank service running on an in-memory SQLite log store.
    This is the strategy we're trying to validate.
    """
    name = "agent_rank"

    def __init__(self, config: ScoringConfig, domain: str, task_type: str):
        self._log_store = LogStore(db_path=":memory:", config=config)
        registry = DomainRegistry(config)
        self._service = AgentRankService(self._log_store, registry, config)
        self._domain = domain
        self._task_type = task_type

    def select(self, candidates: List[str]) -> str:
        # Note: AgentRank uses its own registry; we trust it matches candidates.
        ranking = self._service.rank(self._domain, self._task_type, "")
        return ranking[0]["agent_id"]

    def update(self, agent_id: str, outcome: Dict[str, Any]) -> None:
        # Outcome already has agent_id set by the simulator.
        self._log_store.record_invocation(outcome)

    def close(self) -> None:
        self._log_store.close()
