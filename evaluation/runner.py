"""
Trial runner: applies a strategy to a scenario and returns per-step data.
"""

import random
from typing import Dict, Any, List

from .scenarios import Scenario
from .simulator import call_reward
from .strategies import Strategy, AgentRankStrategy


def run_trial(
    strategy: Strategy,
    scenario: Scenario,
    n_steps: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Run one trial. Returns history: list of per-step records."""
    rng = random.Random(seed)
    spec_by_id = {a.agent_id: a for a in scenario.agents}
    candidates = scenario.candidates()
    history: List[Dict[str, Any]] = []

    for t in range(n_steps):
        choice = strategy.select(candidates)
        outcome = spec_by_id[choice].sample(rng)
        reward = call_reward(outcome, scenario.weights)
        strategy.update(choice, outcome)
        history.append({
            "t": t,
            "choice": choice,
            "reward": reward,
            "success": outcome["success"],
            "quality": outcome["quality_score"],
            "latency_ms": outcome["latency_ms"],
        })

    # Close persistent resources if the strategy needs it.
    if isinstance(strategy, AgentRankStrategy):
        strategy.close()

    return history
