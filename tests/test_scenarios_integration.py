"""
Integration smoke tests: every scenario in evaluation/scenarios.py runs
without crashing and the documented best strategy wins (regret-wise).

These are slower than the unit tests but they're the canary that catches
"refactored a component and accidentally broke the headline result."
"""

import pytest

from config_loader import ScoringConfig
from evaluation.scenarios import ALL_SCENARIOS
from evaluation.run_eval import build_strategies
from evaluation.runner import run_trial, oracle_per_step
from evaluation.metrics import final_summary


CONFIG = ScoringConfig.load("config/scoring.json")


# (scenario_name, expected_winner_strategy_name)
EXPECTED_WINNERS = {
    "priors_correct": "greedy_prior",
    "hidden_gem": "agent_rank",
    "lying_agent": "agent_rank_judged",
    "concept_drift": "agent_rank",
    "context_aware": "agent_rank_linucb",
    "preference_dependent": "agent_rank_pareto",
    "sybil_attack": "agent_rank_protected",
}


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.name)
def test_scenario_runs_without_error(scenario):
    """Short trial of each strategy on each scenario — no exceptions."""
    strategies = build_strategies(scenario, CONFIG, seed=42)
    for strat in strategies:
        history = run_trial(strat, scenario, n_steps=30, seed=42)
        assert len(history) == 30
        assert all("reward" in h for h in history)


@pytest.mark.parametrize(
    "scenario,expected_winner",
    [(s, EXPECTED_WINNERS[s.name]) for s in ALL_SCENARIOS],
    ids=lambda v: v.name if hasattr(v, "name") else str(v),
)
def test_documented_winner_actually_wins(scenario, expected_winner):
    """
    Run all strategies for 200 steps × 3 trials, verify the expected
    winner has the lowest mean regret. Catches headline-breaking
    regressions without paying full eval cost (~ a few seconds).
    """
    regrets_by_strategy: dict = {}
    for trial in range(3):
        seed = 1234 + trial
        oracle = oracle_per_step(scenario, n_steps=200, seed=seed)
        strategies = build_strategies(scenario, CONFIG, seed=seed)
        for strat in strategies:
            history = run_trial(strat, scenario, n_steps=200, seed=seed)
            summary = final_summary(history, oracle)
            regrets_by_strategy.setdefault(strat.name, []).append(
                summary["total_regret"]
            )

    means = {name: sum(rs) / len(rs) for name, rs in regrets_by_strategy.items()}
    winner = min(means.items(), key=lambda kv: kv[1])

    assert winner[0] == expected_winner, (
        f"{scenario.name}: expected {expected_winner} to win, "
        f"got {winner[0]}. Full means: {means}"
    )


def test_all_expected_winners_have_test_coverage():
    """Guard against new scenarios being added without an expected-winner row."""
    actual = {s.name for s in ALL_SCENARIOS}
    documented = set(EXPECTED_WINNERS.keys())
    missing = actual - documented
    extra = documented - actual
    assert not missing, f"new scenarios without expected winner: {missing}"
    assert not extra, f"expected winners for unknown scenarios: {extra}"
