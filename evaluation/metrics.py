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


def cumulative_regret(history: List[Dict[str, Any]], oracle_reward_per_call: float) -> List[float]:
    """Cumulative regret vs. an omniscient oracle that always picks the best agent."""
    out = []
    total = 0.0
    for t, h in enumerate(history, 1):
        total += oracle_reward_per_call - h["reward"]
        out.append(total)
    return out


def selection_share(history: List[Dict[str, Any]]) -> Dict[str, float]:
    """Fraction of trials each agent was selected."""
    counts = Counter(h["choice"] for h in history)
    n = len(history)
    return {agent: count / n for agent, count in counts.items()}


def final_summary(
    history: List[Dict[str, Any]],
    oracle_reward_per_call: float,
) -> Dict[str, Any]:
    """One-shot summary of a trial."""
    n = len(history)
    total_reward = sum(h["reward"] for h in history)
    total_regret = oracle_reward_per_call * n - total_reward
    return {
        "n_steps": n,
        "total_reward": total_reward,
        "avg_reward_per_call": total_reward / n if n else 0.0,
        "total_regret": total_regret,
        "avg_regret_per_call": total_regret / n if n else 0.0,
        "selection_share": selection_share(history),
        "oracle_reward_per_call": oracle_reward_per_call,
    }
