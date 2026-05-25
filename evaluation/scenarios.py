"""
Evaluation scenarios.

Each scenario specifies:
  - the ground-truth agents (TruthSpec list)
  - which cold-start priors to feed strategies that need them
  - the scoring policy (weights) used for both reward and regret

By varying the relationship between truth and priors we can isolate
*why* AgentRank wins (or loses) against simpler baselines.
"""

from dataclasses import dataclass, field
from typing import List, Dict

from .simulator import TruthSpec


# The default scoring policy from config/scoring.json (nlp/summarize domain).
SUMMARIZE_WEIGHTS: Dict[str, float] = {
    "success_rate": 0.25,
    "quality_score": 0.50,
    "latency_score": 0.15,
    "failure_rate": -0.10,
}


@dataclass
class Scenario:
    name: str
    description: str
    agents: List[TruthSpec]
    priors: Dict[str, Dict[str, float]]
    weights: Dict[str, float] = field(default_factory=lambda: dict(SUMMARIZE_WEIGHTS))
    # Hint for run_eval: enable the judged AgentRank variant on this scenario.
    enable_judge_variant: bool = False

    def candidates(self) -> List[str]:
        return [a.agent_id for a in self.agents]

    def oracle_agent(self) -> str:
        """The agent with the highest *true* expected reward."""
        return max(self.agents, key=lambda a: a.expected_reward(self.weights)).agent_id

    def oracle_reward_per_call(self) -> float:
        return max(a.expected_reward(self.weights) for a in self.agents)


# ---------------------------------------------------------------------------
# Scenario A: priors match reality.
#
# This is the "happy path" for greedy: the agent with the best prior really
# is the best agent. UCB pays an exploration tax here and should *slightly*
# underperform greedy. The point of including this scenario is to show that
# AgentRank is not catastrophically worse on easy problems.
# ---------------------------------------------------------------------------
SCENARIO_PRIORS_CORRECT = Scenario(
    name="priors_correct",
    description=(
        "Cold-start priors correctly identify SummarizerQuality as the best "
        "agent. Greedy should win on cumulative reward; AgentRank should "
        "track closely while paying a small exploration cost."
    ),
    agents=[
        TruthSpec("SummarizerFast", success_prob=1.00, quality_mean=0.50,
                  latency_min_ms=80, latency_max_ms=120),
        TruthSpec("SummarizerQuality", success_prob=1.00, quality_mean=0.90,
                  latency_min_ms=700, latency_max_ms=900),
        TruthSpec("SummarizerHallucinator", success_prob=0.70, quality_mean=0.20,
                  latency_min_ms=200, latency_max_ms=1200),
    ],
    priors={
        "SummarizerFast": {"success_rate": 0.95, "quality_score": 0.5,
                           "latency_score": 0.9, "failure_rate": 0.05},
        "SummarizerQuality": {"success_rate": 0.98, "quality_score": 0.9,
                              "latency_score": 0.7, "failure_rate": 0.02},
        "SummarizerHallucinator": {"success_rate": 0.7, "quality_score": 0.2,
                                   "latency_score": 0.6, "failure_rate": 0.3},
    },
)


# ---------------------------------------------------------------------------
# Scenario B: priors are misleading.
#
# A "hidden gem" agent has a mediocre prior but is actually the best. The
# previously-favored agent has been over-rated. Greedy is *stuck* on the
# over-rated agent forever; AgentRank should explore, discover the gem,
# and converge to it.
# ---------------------------------------------------------------------------
SCENARIO_HIDDEN_GEM = Scenario(
    name="hidden_gem",
    description=(
        "Cold-start priors say SummarizerQuality is best, but in reality "
        "SummarizerFast has secretly become much better (e.g., model "
        "upgrade) and is now the optimal choice. Greedy never discovers "
        "this; AgentRank should."
    ),
    agents=[
        TruthSpec("SummarizerFast", success_prob=1.00, quality_mean=0.85,
                  latency_min_ms=80, latency_max_ms=120),
        TruthSpec("SummarizerQuality", success_prob=0.90, quality_mean=0.55,
                  latency_min_ms=700, latency_max_ms=900),
        TruthSpec("SummarizerHallucinator", success_prob=0.70, quality_mean=0.20,
                  latency_min_ms=200, latency_max_ms=1200),
    ],
    priors={
        "SummarizerFast": {"success_rate": 0.95, "quality_score": 0.5,
                           "latency_score": 0.9, "failure_rate": 0.05},
        "SummarizerQuality": {"success_rate": 0.98, "quality_score": 0.9,
                              "latency_score": 0.7, "failure_rate": 0.02},
        "SummarizerHallucinator": {"success_rate": 0.7, "quality_score": 0.2,
                                   "latency_score": 0.6, "failure_rate": 0.3},
    },
)


# ---------------------------------------------------------------------------
# Scenario C: a lying agent.
#
# SummarizerLiar self-reports quality 0.95 on every call but actually
# delivers 0.20. Without an external judge, every strategy that learns
# from observed quality (AgentRank, epsilon-greedy) is fooled and
# over-selects the liar. With OracleJudge, AgentRank sees the truth and
# steers clear.
#
# This is the threat model the LLM-as-judge stage is designed to address.
# ---------------------------------------------------------------------------
SCENARIO_LYING_AGENT = Scenario(
    name="lying_agent",
    description=(
        "SummarizerLiar self-reports quality 0.95 but delivers 0.20. "
        "Without a judge, strategies that learn from observed (claimed) "
        "quality get fooled. AgentRank + OracleJudge ignores the claim "
        "and selects the genuinely-best agent."
    ),
    agents=[
        TruthSpec("SummarizerFast", success_prob=1.00, quality_mean=0.50,
                  latency_min_ms=80, latency_max_ms=120),
        TruthSpec("SummarizerQuality", success_prob=1.00, quality_mean=0.90,
                  latency_min_ms=700, latency_max_ms=900),
        TruthSpec("SummarizerLiar", success_prob=1.00,
                  quality_mean=0.20, claimed_quality_mean=0.95,
                  latency_min_ms=100, latency_max_ms=200),
    ],
    priors={
        "SummarizerFast": {"success_rate": 0.95, "quality_score": 0.5,
                           "latency_score": 0.9, "failure_rate": 0.05},
        "SummarizerQuality": {"success_rate": 0.98, "quality_score": 0.9,
                              "latency_score": 0.7, "failure_rate": 0.02},
        # Liar advertises an attractive prior matching its (false) claim.
        "SummarizerLiar": {"success_rate": 1.00, "quality_score": 0.95,
                           "latency_score": 0.9, "failure_rate": 0.0},
    },
    enable_judge_variant=True,
)


ALL_SCENARIOS = [SCENARIO_PRIORS_CORRECT, SCENARIO_HIDDEN_GEM, SCENARIO_LYING_AGENT]
