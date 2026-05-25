"""
Trial runner: applies a strategy to a scenario and returns per-step data.
Honors scenario.drift_at by swapping the active TruthSpec mid-trial.
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
    candidates = scenario.candidates()
    history: List[Dict[str, Any]] = []

    for t in range(n_steps):
        # Pick the active specs for this step (drift-aware).
        specs = {a.agent_id: a for a in scenario.specs_at_step(t)}
        choice = strategy.select(candidates)
        outcome = specs[choice].sample(rng)
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

    if isinstance(strategy, AgentRankStrategy):
        strategy.close()

    return history
