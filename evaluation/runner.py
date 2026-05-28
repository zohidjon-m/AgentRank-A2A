"""
Trial runner: applies a strategy to a scenario and returns per-step data.
Handles drift via scenario.specs_at_step(t) and context via
scenario.gen_payload(rng) + feature extraction.
"""

import random
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Make the project root importable for feature_extractor.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .scenarios import Scenario
from .simulator import call_reward, preference_reward
from .strategies import Strategy, AgentRankStrategy


def _extractor_for(scenario: Scenario):
    """Returns the FeatureExtractor instance if this scenario uses one."""
    name = scenario.enable_linucb_variant_extractor
    if not name:
        return None
    from feature_extractor import get_extractor
    return get_extractor(name)


def run_trial(
    strategy: Strategy,
    scenario: Scenario,
    n_steps: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Run one trial. Returns history: list of per-step records."""
    rng = random.Random(seed)
    candidates = scenario.candidates()
    extractor = _extractor_for(scenario)
    history: List[Dict[str, Any]] = []

    for t in range(n_steps):
        payload = scenario.gen_payload(rng)
        features = extractor.extract(payload) if extractor is not None else None
        prefs = scenario.gen_preferences(rng)

        specs = {a.agent_id: a for a in scenario.specs_at_step(t)}
        choice = strategy.select(candidates, payload=payload, preferences=prefs)
        outcome = specs[choice].sample(rng, features=features)
        # Persist features alongside outcome so contextual strategies
        # can learn from them.
        if features is not None:
            outcome["context_features"] = features.tolist()
        # Reward depends on which oracle the scenario uses:
        # preference-conditional for Pareto scenarios, fixed-weight
        # otherwise.
        if prefs is not None:
            reward = preference_reward(outcome, prefs)
        else:
            reward = call_reward(outcome, scenario.weights)
        strategy.update(choice, outcome)
        history.append({
            "t": t,
            "choice": choice,
            "reward": reward,
            "success": outcome["success"],
            "quality": outcome["quality_score"],
            "latency_ms": outcome["latency_ms"],
            "features": features.tolist() if features is not None else None,
            "preferences": dict(prefs) if prefs is not None else None,
        })

    if isinstance(strategy, AgentRankStrategy):
        strategy.close()

    return history


def oracle_per_step(
    scenario: Scenario,
    n_steps: int,
    seed: int,
) -> List[float]:
    """
    Compute the per-step oracle reward. For contextual / preference
    scenarios this is regenerated with the same RNG as the trial so the
    payload + preference stream matches step-for-step.
    """
    rng = random.Random(seed)
    extractor = _extractor_for(scenario)
    out: List[float] = []
    for t in range(n_steps):
        payload = scenario.gen_payload(rng)
        features = extractor.extract(payload) if extractor is not None else None
        prefs = scenario.gen_preferences(rng)
        out.append(scenario.oracle_reward_at_step(
            t, features=features, preferences=prefs,
        ))
    return out
