"""
Evaluation scenarios.

Each scenario specifies:
  - the ground-truth agents (TruthSpec list)
  - which cold-start priors to feed strategies that need them
  - the scoring policy (weights) used for both reward and regret

By varying the relationship between truth and priors we can isolate
*why* AgentRank wins (or loses) against simpler baselines.
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

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
    # Concept-drift: at step `drift_at`, replace `agents` with `drift_agents`.
    # Both should describe the same agent_ids — only their behavior changes.
    drift_at: Optional[int] = None
    drift_agents: Optional[List[TruthSpec]] = None
    # When set, run_eval registers an AgentRank variant with this half-life.
    enable_decay_variant_half_life: Optional[float] = None
    # When set, run_eval registers a LinUCB variant using this feature
    # extractor name (must be present in feature_extractor.FEATURE_EXTRACTORS).
    enable_linucb_variant_extractor: Optional[str] = None
    # Optional payload generator. Returns the text payload for one
    # request given an RNG. Defaults to empty string (no context).
    payload_generator: Optional[Callable[[random.Random], str]] = field(
        default=None, repr=False
    )

    def gen_payload(self, rng: random.Random) -> str:
        if self.payload_generator is None:
            return ""
        return self.payload_generator(rng)

    def candidates(self) -> List[str]:
        return [a.agent_id for a in self.agents]

    def specs_at_step(self, t: int) -> List[TruthSpec]:
        """Returns the active TruthSpec list at logical step t."""
        if self.drift_at is not None and t >= self.drift_at and self.drift_agents:
            return self.drift_agents
        return self.agents

    def oracle_reward_at_step(self, t: int, features=None) -> float:
        """
        Best achievable expected reward at step t (post-drift if applicable).
        When features are supplied, picks the best agent *for that context*
        — this matters for contextual scenarios where different agents win
        for different inputs.
        """
        specs = self.specs_at_step(t)
        return max(s.expected_reward(self.weights, features=features) for s in specs)

    def oracle_agent(self) -> str:
        """The agent with the highest true expected reward at step 0."""
        return max(self.agents, key=lambda a: a.expected_reward(self.weights)).agent_id

    def oracle_reward_per_call(self) -> float:
        """Step-0 oracle reward. For drift scenarios use oracle_reward_at_step."""
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


# ---------------------------------------------------------------------------
# Scenario D: concept drift.
#
# SummarizerQuality is the best agent until step 150. After step 150 it
# silently degrades (e.g. an upstream model rollback) while SummarizerFast
# improves. Strategies that aggregate over all history will keep favoring
# the now-broken SummarizerQuality. The AgentRank variant with exponential
# decay (half_life_calls=40) should rapidly forget the pre-drift logs and
# switch to SummarizerFast.
# ---------------------------------------------------------------------------
SCENARIO_CONCEPT_DRIFT = Scenario(
    name="concept_drift",
    description=(
        "SummarizerQuality is best for the first 150 calls, then silently "
        "degrades while SummarizerFast improves. Strategies that average "
        "over all history stay stuck on the now-broken SummarizerQuality. "
        "AgentRank with exponential decay should detect the switch."
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
    drift_at=150,
    drift_agents=[
        # SummarizerFast: model upgraded; now high-quality.
        TruthSpec("SummarizerFast", success_prob=1.00, quality_mean=0.85,
                  latency_min_ms=80, latency_max_ms=120),
        # SummarizerQuality: rolled back to a broken model.
        TruthSpec("SummarizerQuality", success_prob=0.80, quality_mean=0.25,
                  latency_min_ms=700, latency_max_ms=900),
        TruthSpec("SummarizerHallucinator", success_prob=0.70, quality_mean=0.20,
                  latency_min_ms=200, latency_max_ms=1200),
    ],
    enable_decay_variant_half_life=40.0,
)


# ---------------------------------------------------------------------------
# Scenario E: context-aware ranking.
#
# Same three agents, but now their quality is conditional on input length:
#   SummarizerFast:    great on SHORT text  (quality ~0.90), poor on LONG (~0.30)
#   SummarizerQuality: poor on SHORT (~0.45), great on LONG (~0.92)
#   SummarizerHallucinator: bad everywhere.
#
# No single agent wins on average — the right answer depends on the
# request. UCB1 can only learn one global preference and pays heavy
# regret. LinUCB learns a per-agent reward function in feature space
# and routes each request to the right agent.
# ---------------------------------------------------------------------------


def _short_specialist_quality(features) -> float:
    # features = [intercept, norm_log_words, short_ind, long_ind]
    # Sharp specialist: great on short, terrible on long.
    short_ind = float(features[2])
    long_ind = float(features[3])
    if short_ind > 0.5:
        return 0.95
    if long_ind > 0.5:
        return 0.15
    return 0.45  # medium-length text


def _long_specialist_quality(features) -> float:
    # Inverse specialist: great on long, terrible on short.
    short_ind = float(features[2])
    long_ind = float(features[3])
    if long_ind > 0.5:
        return 0.95
    if short_ind > 0.5:
        return 0.20
    return 0.60  # medium-length text


def _flat_low_quality(features) -> float:
    return 0.20


def _varied_length_payload(rng: random.Random) -> str:
    """Generates payloads of varied length (short / medium / long)."""
    bucket = rng.choice(("short", "short", "medium", "long", "long"))
    if bucket == "short":
        n_words = rng.randint(5, 25)
    elif bucket == "medium":
        n_words = rng.randint(60, 130)
    else:  # long
        n_words = rng.randint(220, 380)
    return " ".join(["word"] * n_words)


SCENARIO_CONTEXT_AWARE = Scenario(
    name="context_aware",
    description=(
        "Per-request optimum: SummarizerFast wins on short text, "
        "SummarizerQuality wins on long text, SummarizerHallucinator "
        "is bad everywhere. UCB1 can only learn one global preference; "
        "LinUCB learns per-agent reward as a function of input length "
        "features and routes each request to the right agent."
    ),
    agents=[
        TruthSpec("SummarizerFast", success_prob=1.00, quality_mean=0.50,
                  latency_min_ms=80, latency_max_ms=120,
                  context_quality_fn=_short_specialist_quality),
        TruthSpec("SummarizerQuality", success_prob=1.00, quality_mean=0.70,
                  latency_min_ms=700, latency_max_ms=900,
                  context_quality_fn=_long_specialist_quality),
        TruthSpec("SummarizerHallucinator", success_prob=1.00, quality_mean=0.20,
                  latency_min_ms=200, latency_max_ms=1200,
                  context_quality_fn=_flat_low_quality),
    ],
    priors={
        "SummarizerFast": {"success_rate": 0.95, "quality_score": 0.5,
                           "latency_score": 0.9, "failure_rate": 0.05},
        "SummarizerQuality": {"success_rate": 0.98, "quality_score": 0.7,
                              "latency_score": 0.7, "failure_rate": 0.02},
        "SummarizerHallucinator": {"success_rate": 1.0, "quality_score": 0.2,
                                   "latency_score": 0.6, "failure_rate": 0.0},
    },
    payload_generator=_varied_length_payload,
    enable_linucb_variant_extractor="length_bucket",
)


ALL_SCENARIOS = [
    SCENARIO_PRIORS_CORRECT,
    SCENARIO_HIDDEN_GEM,
    SCENARIO_LYING_AGENT,
    SCENARIO_CONCEPT_DRIFT,
    SCENARIO_CONTEXT_AWARE,
]
