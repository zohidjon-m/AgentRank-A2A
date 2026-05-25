# AgentRank Evaluation Harness

The eval harness is what turns every claim in the README from "I think
this works" into a reproducible number. This document covers how to
run it, what the outputs mean, and how to extend it.

For the design of the underlying ranker, see
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## What the harness does

```
evaluation/
├── simulator.py      # TruthSpec: parametric synthetic agents
├── strategies.py     # selection policies: Random, RoundRobin, Greedy, ε-Greedy, AgentRank
├── scenarios.py      # 5 scenarios, each isolating one failure mode
├── runner.py         # one-trial executor
├── metrics.py        # cumulative regret, selection share
└── run_eval.py       # CLI entry — sweeps all (scenario, strategy) pairs
```

For every (scenario, strategy) pair, the harness:

1. Generates a workload (per-step payloads, possibly varied-length).
2. For each step, asks the strategy which agent to pick.
3. Samples the chosen agent's outcome from its `TruthSpec` (success
   probability, quality distribution, latency range, optionally
   context-conditional).
4. Computes the reward using **true** quality (so we measure what the
   user actually received).
5. Feeds the outcome back to the strategy for learning.
6. Tracks cumulative regret against an omniscient oracle that always
   picks the truly-best agent for the current context.

Each scenario is run for `--trials` seeded trials. Results are
averaged before being written out.

---

## Running it

```bash
# default: 300 steps, 10 trials per (scenario, strategy)
python -m evaluation.run_eval

# more careful run, longer horizon
python -m evaluation.run_eval --steps 1000 --trials 20

# write somewhere else
python -m evaluation.run_eval --out /tmp/agentrank-eval

# use a different config (e.g. to try alternate scoring weights)
python -m evaluation.run_eval --config path/to/scoring.json
```

Outputs go to `evaluation/results/` by default:

- `summary.json` — per-scenario, per-strategy regret means and stdevs,
  average reward per call, and selection share.
- `regret_curves.json` — full averaged cumulative-regret curves, one
  series per strategy per scenario. For custom plotting.
- `regret_<scenario>.png` — matplotlib regret-curve plots (skipped
  silently if `matplotlib` isn't installed).

---

## Reading the output

Each scenario section in the console summary looks like:

```
======================================================================
Scenario: lying_agent
  SummarizerLiar self-reports quality 0.95 but delivers 0.20. [...]
  Oracle (step 0): SummarizerQuality (avg E[reward/call] = 0.8100)
  Settings: 300 steps, 10 trials
----------------------------------------------------------------------
  strategy                 final_regret   +/-stdev   avg_reward
  agent_rank_judged               1.339      0.059       0.8055
  round_robin                    48.263      0.033       0.6491
  random                         48.364      1.762       0.6488
  agent_rank                     85.279      0.171       0.5257
  epsilon_greedy                 90.329      1.326       0.5089
  greedy_prior                   95.263      0.032       0.4925
  Selection share:
    agent_rank_judged      SummarizerFast=2%  SummarizerLiar=0%  SummarizerQuality=98%
    [...]
```

What each column means:

- **`final_regret`**: cumulative regret at the end of the trial,
  averaged across trials. Lower is better. A regret of `R` means the
  strategy collected `R` less total reward than an oracle would have
  over the same `n_steps` calls.
- **`+/-stdev`**: standard deviation across trials. Tight stdev means
  the strategy's behavior is reproducible; wide stdev means it's
  sensitive to the RNG seed (which is itself useful info).
- **`avg_reward`**: total reward divided by `n_steps`. Useful for
  comparing to the oracle's `E[reward/call]`.
- **Selection share**: fraction of calls each agent was selected. Tells
  you *why* a strategy did well or badly. A 100% share means the
  strategy locked onto one agent; a balanced share means it explored
  or routed.

The PNG plots show cumulative regret vs. step number for each
strategy, on the same axes. Strategies that adapt have curves that
flatten out; strategies that are stuck have curves that grow
linearly forever.

---

## The five scenarios

Each is designed to isolate one specific failure mode. The point isn't
to show AgentRank beating baselines on a single benchmark — it's to
show that **different baselines fail in different ways**, and only
AgentRank (with the right policy plugins) handles all of them.

| Scenario | Failure mode it isolates | Best strategy |
|---|---|---|
| `priors_correct` | Cold-start priors happen to match reality | `greedy_prior` |
| `hidden_gem` | Cold-start priors are wrong | `agent_rank` (UCB1 explores → finds truth) |
| `lying_agent` | Agent inflates its own quality score | `agent_rank_judged` (judge catches lie) |
| `concept_drift` | Best agent changes mid-trial | `agent_rank` / `agent_rank_decayed` |
| `context_aware` | Best agent depends on the request | `agent_rank_linucb` (per-context routing) |

The judged / decayed / linucb variants of `agent_rank` are registered
automatically when a scenario opts in via:

```python
enable_judge_variant = True              # registers agent_rank_judged
enable_decay_variant_half_life = 40.0    # registers agent_rank_decayed
enable_linucb_variant_extractor = "..."  # registers agent_rank_linucb
```

So a single scenario can compare bare AgentRank against the
plugin-enhanced variant directly, which is exactly the comparison
that proves each plugin's value.

---

## Extending the harness

### Adding a new scenario

In `evaluation/scenarios.py`:

```python
SCENARIO_MY_THING = Scenario(
    name="my_thing",
    description="What this scenario isolates and why it matters.",
    agents=[
        TruthSpec("AgentA", success_prob=0.9, quality_mean=0.8,
                  latency_min_ms=200, latency_max_ms=500),
        TruthSpec("AgentB", success_prob=0.7, quality_mean=0.6,
                  latency_min_ms=100, latency_max_ms=200),
    ],
    priors={
        "AgentA": {"success_rate": 0.9, "quality_score": 0.7,
                   "latency_score": 0.7, "failure_rate": 0.1},
        "AgentB": {"success_rate": 0.7, "quality_score": 0.5,
                   "latency_score": 0.9, "failure_rate": 0.3},
    },
    # Optional opt-ins:
    # enable_judge_variant=True,
    # enable_decay_variant_half_life=40.0,
    # enable_linucb_variant_extractor="length_bucket",
    # drift_at=150, drift_agents=[...],
    # payload_generator=my_workload_fn,
)

ALL_SCENARIOS = [..., SCENARIO_MY_THING]
```

Then `python -m evaluation.run_eval` picks it up automatically.

### Adding a new selection strategy

In `evaluation/strategies.py`:

```python
class MyStrategy(Strategy):
    name = "my_strategy"

    def __init__(self, ...):
        ...

    def select(self, candidates: List[str], payload: str = "") -> str:
        ...

    def update(self, agent_id: str, outcome: Dict[str, Any]) -> None:
        ...
```

Then add it to `build_strategies` in `evaluation/run_eval.py`:

```python
strategies = [
    ...
    MyStrategy(...),
    ...
]
```

It will compete alongside the existing baselines on every scenario.

### Lying / drifting / contextual agents in a custom scenario

The `TruthSpec` dataclass supports three orthogonal axes:

- **Lying**: set `claimed_quality_mean=X` to make the agent report a
  fixed `X` regardless of its true behavior. Without this, agents
  honestly report what they delivered on each call.
- **Drift**: set `Scenario.drift_at=T` and `Scenario.drift_agents=[...]`
  to swap the active TruthSpec list at logical step `T`. The oracle
  flips with it.
- **Contextual**: set `TruthSpec.context_quality_fn=fn` where
  `fn(features) -> quality_mean`. With a `payload_generator` that
  varies inputs and an `enable_linucb_variant_extractor`, the eval
  measures per-context routing.

These can be combined: an agent can lie *and* drift *and* be
context-dependent, all at once.

---

## Why the numbers are honest

Three things to know:

### 1. Reward is measured against TRUE quality, not claimed quality

`simulator.call_reward` reads `outcome["true_quality_score"]` when
available (which it always is for synthetic agents). Regret is
`oracle_reward - delivered_reward` where `delivered_reward` uses what
the user *actually got*, not what the agent claimed. If a lying agent
fools the strategy, the strategy's reward is low even though its
*logged* quality looks fine.

### 2. The oracle is per-step and per-trial

For drift scenarios the oracle's best-agent changes mid-trial — the
oracle reward at step `t` uses the active TruthSpec at step `t`. For
contextual scenarios the oracle is recomputed per-trial using the same
RNG seed as the strategies see, so the oracle's payload stream
matches the strategies' payload stream exactly. Otherwise the oracle
would be averaging over a different distribution than what the
strategies experienced, and regret would be junk.

### 3. Strategies are reset per trial

Each trial constructs fresh strategy instances with a fresh
`LogStore(":memory:")`. There's no contamination across trials. Random
seeds are deterministic (`1234 + trial`), so re-running the eval
produces identical numbers — if you change a result, it's because you
changed code.

---

## Limitations and honest caveats

- **Synthetic agents.** All numbers above use parametric TruthSpecs.
  A real workload of LLM-backed agents would have noisier rewards,
  longer-tail latency, and more interesting context structure.
  The harness is *designed* to be plugged into real agents — the
  `Strategy` interface is the same — but the headline numbers come
  from synthetic runs.
- **Feature extractor is rudimentary.** `LengthBucketExtractor` has
  4 dims. Real contextual ranking would use embeddings, content
  features, user metadata. The framework supports any
  `PayloadFeatureExtractor`; we just haven't built more.
- **`OracleJudge` is a strawman.** In the eval, the judge has
  perfect access to ground truth via the `hint` argument. A real
  judge (the AnthropicJudge stub) approximates this and will have
  its own error rate. The `lying_agent` headline number is therefore
  an upper bound on what a real LLM-as-judge can achieve — it shows
  the *value* of judging, not the *accuracy* of any specific judge.
- **Concept-drift decay is not yet a clear win.** In the current
  drift scenario the change is dramatic enough that UCB1's
  built-in exploration handles it without help. The mechanism is in
  place for subtler drifts and longer horizons. See the discussion
  in the `concept_drift` scenario notes.
- **No multi-objective scoring yet.** The single weighted-sum score
  pre-commits to a quality/latency tradeoff. Stage 5 (Pareto) would
  let the caller specify the tradeoff per-request.

The point of listing these is that the eval harness *can* surface
each limitation — add a noisier `TruthSpec`, a longer-horizon scenario,
or an imperfect judge, and watch the numbers move.
