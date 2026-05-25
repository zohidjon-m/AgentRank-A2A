"""
Evaluation suite entry point.

Runs every strategy on every scenario, averages over multiple seeds,
and writes a JSON summary plus (optional) matplotlib plots into
evaluation/results/.

    python -m evaluation.run_eval
    python -m evaluation.run_eval --steps 1000 --trials 20

Plots are skipped silently if matplotlib is not installed.
"""

import argparse
import json
import random
import sys
import statistics
from pathlib import Path
from typing import Dict, Any, List

# Make the project root importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config_loader import ScoringConfig
from judge import OracleJudge
from evaluation.scenarios import ALL_SCENARIOS, Scenario
from evaluation.strategies import (
    RandomStrategy,
    RoundRobinStrategy,
    GreedyPriorStrategy,
    EpsilonGreedyStrategy,
    AgentRankStrategy,
)
from evaluation.runner import run_trial, oracle_per_step
from evaluation.metrics import (
    cumulative_regret,
    final_summary,
)


def build_strategies(scenario: Scenario, config: ScoringConfig, seed: int):
    """Fresh strategy instances for one trial."""
    rng_a = random.Random(seed + 1000)
    rng_b = random.Random(seed + 2000)
    strategies = [
        RandomStrategy(rng_a),
        RoundRobinStrategy(),
        GreedyPriorStrategy(scenario.priors, scenario.weights),
        EpsilonGreedyStrategy(scenario.weights, rng_b, epsilon=0.1),
        AgentRankStrategy(
            config, domain="nlp", task_type="summarize",
            candidates=scenario.candidates(), priors=scenario.priors,
        ),
    ]
    if scenario.enable_judge_variant:
        strategies.append(
            AgentRankStrategy(
                config, domain="nlp", task_type="summarize",
                candidates=scenario.candidates(), priors=scenario.priors,
                judge=OracleJudge(),
                variant_suffix="_judged",
            )
        )
    if scenario.enable_decay_variant_half_life is not None:
        decayed_cfg = config.with_drift_half_life(
            "nlp/summarize",
            scenario.enable_decay_variant_half_life,
        )
        strategies.append(
            AgentRankStrategy(
                decayed_cfg, domain="nlp", task_type="summarize",
                candidates=scenario.candidates(), priors=scenario.priors,
                variant_suffix="_decayed",
            )
        )
    if scenario.enable_linucb_variant_extractor is not None:
        # LinUCB's exploration term scales with feature magnitudes;
        # alpha=0.5 ensures enough early exploration to discover
        # per-context optima. (UCB1 keeps its lower alpha — it doesn't
        # have the same feature-vector scaling.)
        linucb_cfg = config.with_bandit(
            "nlp/summarize",
            kind="linucb",
            feature_extractor=scenario.enable_linucb_variant_extractor,
            bandit_params={"alpha": 0.5, "ridge": 1.0},
        )
        strategies.append(
            AgentRankStrategy(
                linucb_cfg, domain="nlp", task_type="summarize",
                candidates=scenario.candidates(), priors=scenario.priors,
                variant_suffix="_linucb",
            )
        )
    return strategies


def run_scenario(
    scenario: Scenario,
    config: ScoringConfig,
    n_steps: int,
    n_trials: int,
) -> Dict[str, Any]:
    """Run all strategies on a scenario across n_trials seeds and average."""
    per_strategy: Dict[str, Dict[str, Any]] = {}

    # We accumulate regret curves across trials, then average pointwise.
    regret_accumulators: Dict[str, List[List[float]]] = {}
    summaries: Dict[str, List[Dict[str, Any]]] = {}
    oracle_means_per_trial: List[float] = []

    for trial in range(n_trials):
        seed = 1234 + trial
        # Oracle is per-step and per-trial (so contextual scenarios
        # match the payload stream the strategies see).
        oracle = oracle_per_step(scenario, n_steps, seed=seed)
        oracle_means_per_trial.append(sum(oracle) / n_steps if n_steps else 0.0)

        strategies = build_strategies(scenario, config, seed)
        for strat in strategies:
            history = run_trial(strat, scenario, n_steps, seed=seed)
            regret_curve = cumulative_regret(history, oracle)
            summary = final_summary(history, oracle)

            regret_accumulators.setdefault(strat.name, []).append(regret_curve)
            summaries.setdefault(strat.name, []).append(summary)

    oracle_mean = statistics.mean(oracle_means_per_trial) if oracle_means_per_trial else 0.0

    # Average across trials.
    for name, curves in regret_accumulators.items():
        avg_curve = [statistics.mean(col) for col in zip(*curves)]
        trial_summaries = summaries[name]
        per_strategy[name] = {
            "avg_regret_curve": avg_curve,
            "final_regret_mean": statistics.mean(s["total_regret"] for s in trial_summaries),
            "final_regret_stdev": (
                statistics.stdev(s["total_regret"] for s in trial_summaries)
                if n_trials > 1 else 0.0
            ),
            "avg_reward_per_call": statistics.mean(s["avg_reward_per_call"] for s in trial_summaries),
            "selection_share_mean": _avg_share(
                [s["selection_share"] for s in trial_summaries]
            ),
        }

    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "oracle_agent": scenario.oracle_agent(),
        "oracle_reward_per_call": oracle_mean,
        "drift_at": scenario.drift_at,
        "n_steps": n_steps,
        "n_trials": n_trials,
        "strategies": per_strategy,
    }


def _avg_share(shares: List[Dict[str, float]]) -> Dict[str, float]:
    """Average a list of selection_share dicts."""
    all_keys = set()
    for s in shares:
        all_keys.update(s.keys())
    return {
        k: statistics.mean(s.get(k, 0.0) for s in shares)
        for k in sorted(all_keys)
    }


def write_plots(results: List[Dict[str, Any]], out_dir: Path) -> bool:
    """Returns True if plots were written, False if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    for r in results:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, data in r["strategies"].items():
            ax.plot(data["avg_regret_curve"], label=name, linewidth=1.5)
        ax.set_title(f"Cumulative regret - scenario: {r['scenario']}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative regret (lower is better)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"regret_{r['scenario']}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return True


def print_summary(results: List[Dict[str, Any]]) -> None:
    for r in results:
        print()
        print("=" * 70)
        print(f"Scenario: {r['scenario']}")
        print(f"  {r['description']}")
        drift_note = f" | drift_at={r['drift_at']}" if r.get("drift_at") is not None else ""
        print(
            f"  Oracle (step 0): {r['oracle_agent']} "
            f"(avg E[reward/call] = {r['oracle_reward_per_call']:.4f}){drift_note}"
        )
        print(f"  Settings: {r['n_steps']} steps, {r['n_trials']} trials")
        print("-" * 70)
        header = f"  {'strategy':<22} {'final_regret':>14} {'+/-stdev':>10} {'avg_reward':>12}"
        print(header)
        # Sort by final regret (lower = better).
        ranked = sorted(
            r["strategies"].items(),
            key=lambda kv: kv[1]["final_regret_mean"],
        )
        for name, data in ranked:
            print(
                f"  {name:<22} "
                f"{data['final_regret_mean']:>14.3f} "
                f"{data['final_regret_stdev']:>10.3f} "
                f"{data['avg_reward_per_call']:>12.4f}"
            )
        print("  Selection share:")
        for name, data in ranked:
            shares = data["selection_share_mean"]
            share_str = "  ".join(f"{a}={p:.0%}" for a, p in shares.items())
            print(f"    {name:<22} {share_str}")


def main():
    parser = argparse.ArgumentParser(description="Run the AgentRank eval suite.")
    parser.add_argument("--steps", type=int, default=300, help="Steps per trial.")
    parser.add_argument("--trials", type=int, default=10, help="Trials per scenario.")
    parser.add_argument(
        "--config", default=str(ROOT / "config" / "scoring.json"),
        help="Scoring config path.",
    )
    parser.add_argument(
        "--out", default=str(ROOT / "evaluation" / "results"),
        help="Output directory.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = ScoringConfig.load(args.config)

    results = [
        run_scenario(s, config, n_steps=args.steps, n_trials=args.trials)
        for s in ALL_SCENARIOS
    ]

    # Write JSON.
    summary_path = out_dir / "summary.json"
    slim = [
        {
            **{k: v for k, v in r.items() if k != "strategies"},
            "strategies": {
                name: {k: v for k, v in data.items() if k != "avg_regret_curve"}
                for name, data in r["strategies"].items()
            },
        }
        for r in results
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2)

    curves_path = out_dir / "regret_curves.json"
    with open(curves_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "scenario": r["scenario"],
                    "curves": {
                        name: data["avg_regret_curve"]
                        for name, data in r["strategies"].items()
                    },
                }
                for r in results
            ],
            f,
        )

    plotted = write_plots(results, out_dir)

    print_summary(results)
    print()
    print(f"JSON summary  -> {summary_path}")
    print(f"Regret curves -> {curves_path}")
    if plotted:
        print(f"Plots         -> {out_dir}/regret_<scenario>.png")
    else:
        print("Plots         -> skipped (matplotlib not installed; pip install matplotlib)")


if __name__ == "__main__":
    main()
