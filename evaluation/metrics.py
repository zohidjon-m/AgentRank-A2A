"""
Metrics computed over a trial history.
"""

from collections import Counter
from typing import Dict, Any, List


def cumulative_reward(history: List[Dict[str, Any]]) -> List[float]:
    """Cumulative reward at each step (for plotting reward curves)."""
    out = []
    total = 0.0
    for h in history:
        total += h["reward"]
        out.append(total)
    return out


def cumulative_regret(
    history: List[Dict[str, Any]],
    oracle_reward_per_call,
) -> List[float]:
    """
    Cumulative regret vs. an omniscient oracle that always picks the best
    agent. `oracle_reward_per_call` may be either a scalar (stationary
    oracle) or a list aligned to history (for drift scenarios where the
    oracle changes over time).
    """
    if isinstance(oracle_reward_per_call, (int, float)):
        per_step = [float(oracle_reward_per_call)] * len(history)
    else:
        per_step = list(oracle_reward_per_call)
        if len(per_step) != len(history):
            raise ValueError(
                f"oracle list length {len(per_step)} != history length {len(history)}"
            )
    out = []
    total = 0.0
    for h, oracle in zip(history, per_step):
        total += oracle - h["reward"]
        out.append(total)
    return out


def selection_share(history: List[Dict[str, Any]]) -> Dict[str, float]:
    """Fraction of trials each agent was selected."""
    counts = Counter(h["choice"] for h in history)
    n = len(history)
    return {agent: count / n for agent, count in counts.items()}


def final_summary(
    history: List[Dict[str, Any]],
    oracle_reward_per_call,
) -> Dict[str, Any]:
    """
    One-shot summary of a trial. `oracle_reward_per_call` may be either
    a scalar or a list aligned to history (see cumulative_regret).
    """
    n = len(history)
    total_reward = sum(h["reward"] for h in history)
    if isinstance(oracle_reward_per_call, (int, float)):
        total_oracle = float(oracle_reward_per_call) * n
        oracle_summary = float(oracle_reward_per_call)
    else:
        total_oracle = float(sum(oracle_reward_per_call))
        oracle_summary = total_oracle / n if n else 0.0
    total_regret = total_oracle - total_reward
    return {
        "n_steps": n,
        "total_reward": total_reward,
        "avg_reward_per_call": total_reward / n if n else 0.0,
        "total_regret": total_regret,
        "avg_regret_per_call": total_regret / n if n else 0.0,
        "selection_share": selection_share(history),
        "oracle_reward_per_call": oracle_summary,
    }
