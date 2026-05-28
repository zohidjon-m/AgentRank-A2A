# AgentRank Architecture

This document is the design reference: what each component does, why it's
shaped the way it is, and the math behind the bandit policies. For
results and motivation, see [the main README](../README.md). For how to
run the eval harness, see [EVAL.md](EVAL.md).

---

## Design principles

1. **Data-driven everything.** Every scoring knob lives in
   `config/scoring.json`. Code changes are reserved for adding new
   mechanisms (a new bandit policy, a new feature extractor); tuning
   live behavior is a config edit.
2. **Pluggable policies, single ranker.** There's one
   `AgentRankService.rank()` entry point. It dispatches to a
   `BanditPolicy` chosen per-domain. Adding Thompson sampling later
   means adding one class in `bandits.py` and one config value.
3. **Quality is verified, not trusted.** Agents are not trusted to score
   themselves. A `QualityJudge` re-scores the output. Without this, any
   learning strategy is gameable by an agent that claims `quality=1.0`.
4. **Stateless logic, persistent state.** Bandits are stateless — they
   reconstruct their fit from logs each call. State lives in SQLite, so
   the demo survives restarts and multiple processes can share a DB.
5. **The eval harness is a first-class deliverable.** Every claim in the
   README is a number from `run_eval`. Each new feature should ship with
   a scenario that demonstrates it's necessary.

---

## Request flow

A single call through
`agent_client.AgentClient.handle_task(domain, task_type, payload, preferences=None)`:

```
1. rank
   AgentRankService.rank_with_features(domain, task_type, payload, preferences)
   ├─ load policy from ScoringConfig.policy_for("nlp/summarize")
   │  → {weights, exploration, drift, bandit, feature_extractor,
   │     bandit_params, trust}
   ├─ optionally extract features = LengthBucketExtractor.extract(payload)
   ├─ build_bandit(policy, feature_dim) → UCB1 / LinUCB / Pareto
   ├─ bandit.rank(candidates, weights, log_store,
   │              context_features=..., preferences=...)
   │  → sorted list of {agent_id, score, base_score, exploration_bonus,
   │                    n_a, metrics, bandit, ...}
   └─ ProbationPolicy(trust).adjust(ranking, log_store)
      → same shape with `trust_status` added per entry; reorders so
        trusted agents come first when the probation share cap fires.

2. dispatch
   send_message({"performative": "request", "receiver": best, ...})
   → response with {content, metrics: {success, quality_score,
                                       latency_ms, cost_cents, ...}}

3. judge (if configured)
   judge.score(payload=payload, output=response.content, hint=None)
   → JudgeResult(score, reason, judge_name)
   metrics.quality_score := judge_result.score
   metrics.agent_claimed_quality := <original>
   metrics.judge_name, metrics.judge_reason := <metadata>

4. log
   metrics.context_features := features.tolist() (if extractor was used)
   metrics.cost_cents := <from agent or default>
   log_store.record_invocation(metrics)
```

That's the entire pipeline. Every later concern — drift handling,
contextual ranking, lying-agent defense, per-request preferences,
sybil resistance — fits into this flow without changing the shape.
The `preferences` arg is `None` for UCB1 / LinUCB (they ignore it)
and a `{quality: w, latency: w, cost: w, success: w}` dict for
ParetoBandit. The trust pass is a no-op under the permissive default
config; non-default `trust` configs cap exposure of probation /
flagged agents.

---

## Components

### `config_loader.py` — `ScoringConfig`

Read-only view over `config/scoring.json`. The interesting methods:

- `policy_for(domain_key)`: returns a fully-filled policy dict (weights,
  exploration, drift, bandit, feature_extractor, bandit_params, trust)
  for the given domain, falling back to defaults for missing keys.
- `agent_default(agent_id)`: cold-start prior for an agent before any
  logs exist. Returns the per-agent prior if configured, else the
  generic `_fallback`, else `0.5` for everything.
- `with_priors(priors)`, `with_registry(domain_key, agents)`,
  `with_drift_half_life(...)`, `with_bandit(...)`, `with_trust(...)`:
  return cloned configs with overrides applied. Used by the eval
  harness to compose per-scenario views without touching the persisted
  config file.

The "defensive fill" pattern in `policy_for` (`{**default, **explicit}`)
means a policy can omit any subset of keys and still produce a valid
config. New keys can be added with sensible defaults without breaking
older configs.

### `log_store.py` — `LogStore`

SQLite-backed. Single table `invocations`:

```sql
id                     INTEGER PRIMARY KEY AUTOINCREMENT
agent_id               TEXT
success                INTEGER (0/1)
quality_score          REAL  -- post-judge if judge ran, else agent self-report
latency_ms             INTEGER
failure_reason         TEXT
timestamp              REAL  -- time.time() at write
judge_name             TEXT  -- which judge produced quality_score (NULL if none)
judge_reason           TEXT  -- one-line rationale
agent_claimed_quality  REAL  -- agent's self-report before judge override
context_features       TEXT  -- JSON-encoded feature vector (NULL if no extractor)
cost_cents             REAL  -- per-call cost (NULL if not reported)
```

Auto-migration: each later-added column has an idempotent
`ALTER TABLE ... ADD COLUMN` guarded by a try/except. Existing databases
upgrade silently.

Key reads:

- `compute_metrics(agent_id, half_life_calls=None)`: aggregate
  success / quality / latency / failure. When half-life is set, log
  entries are weighted by
  `0.5 ** ((current_tick - entry.id) / half_life_calls)`. Cold start
  (no logs) returns the agent's prior from the config.
- `get_objective_vector(agent_id, half_life_calls, cost_reference_cents)`:
  the same metrics plus a `cost_score = 1 - min(avg_cost/cost_ref, 1)`
  dimension. Used by `ParetoBandit`. Cost samples falling back to the
  config's per-agent prior when no costs have been logged.
- `get_weighted_counts(half_life_calls=None)`: `(total_weight,
  {agent: agent_weight})` — used by UCB1 / Pareto for the exploration
  bonus. Batched single read, not per-agent.
- `get_contextual_logs(agent_id)`: rows that have a stored
  `context_features` value. Used by LinUCB to refit.
- `recent_selections(n)`: last `n` `agent_id`s, newest last. Used by
  `ProbationPolicy` to compute the rolling probation share.
- `recent_claimed_quality(agent_id, n)`: last `n` claimed-quality
  scores for the agent (`agent_claimed_quality` if a judge ran, else
  `quality_score`). Used by the inflated-claim anomaly detector.

The id column is also the logical clock: `MAX(id)` is "now" in
invocation counts, and the age of an entry is `now - entry.id`. We use
the id rather than `timestamp` because the eval runs hundreds of
invocations per millisecond — wall-clock has no meaningful resolution at
that rate.

### `domain_registry.py` — `DomainRegistry`

`(domain, task_type) → [agent_id]`. Loaded from config; preserves
insertion order so candidate iteration is deterministic. A small but
non-trivial concern: `bandits.LinUCBBandit` ties at cold start, and
stable sort means the first candidate wins. That's why we have an
explicit warm-start phase (see below).

### `bandits.py` — `BanditPolicy`, `UCB1Bandit`, `LinUCBBandit`, `ParetoBandit`, `build_bandit`

The factory `build_bandit(policy, feature_dim)` reads
`policy["bandit"]` and constructs the right instance. All bandits
share the same `rank(candidates, weights, log_store,
context_features=None, preferences=None)` signature so
`AgentRankService` treats them interchangeably; each implementation
ignores the kwargs it doesn't need. Three implementations today:

#### UCB1Bandit (context-blind)

```
score(a) = base(a) + α · √( ln(1 + N_eff) / (1 + n_a_eff) )

base(a) = w_SR·SR(a) + w_QS·QS(a) + w_LS·LS(a) + w_FR·FR(a)
```

Where `N_eff` and `n_a_eff` are total and per-agent counts, either raw
or exponentially-weighted by drift half-life. With no decay, this is
literal UCB1. With decay, it's "discounted UCB" — the standard
adaptation for non-stationary bandits, where stale agents become
exploration-worthy again as their effective count shrinks.

#### LinUCBBandit (contextual, disjoint LinUCB)

```
A_a   = ridge · I + Σ_t  x_t x_tᵀ
b_a   = Σ_t  r_t · x_t                  (r_t recomputed from logs)
θ_a   = A_a⁻¹ b_a
score(a, x) = θ_aᵀ x + α · √( xᵀ A_a⁻¹ x )
```

Reward `r_t` for a logged call is recomputed at fit time using the
current `weights` — so changing scoring weights doesn't invalidate the
log. Only entries with a stored `context_features` are used (so logs
predating LinUCB rollout simply don't contribute).

**Cold-start warm-up.** Naive LinUCB ties at cold start (θ=0 for all
agents, identical confidence for any fixed x), so the first candidate
in the list wins and accumulates *all* the early data. That destroys
per-context learning: by the time later candidates get a turn, the
leader has a strong (but context-mixed) fit and is hard to dislodge.

The fix: until every agent has `warm_start_n` (default 3)
observations, the least-explored agent is forced to the top of the
ranking. After warm-up, LinUCB takes over.

#### ParetoBandit (multi-objective, context-blind)

```
objective(a) = [quality_score, latency_score, cost_score, success_rate]
optimism(a)  = α · √( ln(1 + N_eff) / (1 + n_a_eff) )      # one scalar
ucb(a)       = min(1, objective(a) + optimism(a))           # per-dim
score(a)     = preferences · ucb(a)
```

Per request the caller passes a `preferences` dict (`{"quality_score":
0.7, "latency_score": 0.2, "cost_score": 0.1, "success_rate": 0.0}`).
ParetoBandit projects each agent's optimistic objective vector onto
the preference direction and ranks by the resulting scalar. It also
computes the Pareto frontier of the optimistic vectors and tags each
entry with `on_frontier` for observability — under strictly positive
preferences the dot-product maximizer is always on the frontier, so
the frontier tag is informational. It becomes load-bearing for the
planned constrained-query extension ("min cost subject to quality
≥ 0.7").

Cost handling: `LogStore.get_objective_vector` divides observed cost
by `bandit_params.cost_reference_cents` (default `10.0`) and inverts
so higher is better, matching the convention of the other objectives.

#### Reward function

UCB1 and LinUCB use the same per-call reward, computed in `_per_call_reward`:

```
r = w_SR · success + w_QS · quality + w_LS · latency_score + w_FR · (1 - success)
```

Where `latency_score = 1 - min(latency_ms / 3000, 1)` and `quality` is
the judge's verdict (if a judge ran) else the agent's self-report. The
formula is the same one used by UCB1's "base" — so LinUCB's reward
matches what UCB1 sees aggregated.

ParetoBandit doesn't reduce its objectives to a scalar reward; it
keeps them as a vector and picks based on the per-request preferences.

### `pareto.py` — `dominates`, `pareto_frontier_indices`, `weighted_pick`, `normalize_preferences`

Pure utility module — no I/O, no state. Three helpers used by
`ParetoBandit`:

- `dominates(a, b)`: True iff `a` is `>=` `b` on every objective and
  `>` on at least one.
- `pareto_frontier_indices(points)`: O(n²) Pareto frontier. Fine for
  our scale (single-digit candidates per request); swap for a faster
  algorithm if you ever have hundreds of candidates.
- `weighted_pick(points, preferences, restrict_to_frontier=True)`:
  argmax of `preferences · point`, optionally restricted to the
  frontier. The restriction is a no-op for strictly positive
  preferences but matters for the planned constrained queries.
- `normalize_preferences(prefs_dict, keys)`: projects a preferences
  dict onto `keys` in order, fills missing keys with 0, normalizes so
  the projection sums to 1. Defaults to uniform if the input is empty
  / non-positive.

### `feature_extractor.py` — `PayloadFeatureExtractor`, `LengthBucketExtractor`

Maps a payload to a fixed-length numeric vector. Today we have a
4-dim extractor for summarization (`[intercept, normalized log word
count, short indicator, long indicator]`). Adding new extractors is
a single class plus a registry entry; new domains can use different
extractors via the config's `feature_extractor` field.

The features matter: too few dims and LinUCB can't distinguish
contexts; too many and you'll overfit with limited data. Length
bucketing is the bare minimum to make "this is short, that is long"
representable.

### `trust.py` — `TrustConfig`, `ProbationPolicy`

Stage 6's defense against sybil floods and inflated self-reports. The
policy applies *after* the bandit ranks candidates — the bandit still
sees and scores everyone; the trust pass rearranges the final order
when the probation-share quota is exhausted, and tags each entry
with `trust_status` for observability:

- `"trusted"` — agent is on the allowlist OR has earned trust through
  accumulation.
- `"probation"` — not on allowlist, hasn't earned trust yet.
- `"flagged_inflated"` — anomaly detector tripped (recent claimed
  quality is suspiciously uniform-and-high).
- `"demoted"` — was in probation AND the share cap was hit, so the
  policy moved the entry below the trusted ones.

`TrustConfig` knobs:

| Knob | Default | Effect |
|---|---|---|
| `trusted_agents` | `[]` | Allowlist of pre-vetted agents (in production: signed-identity registry). Always trusted, regardless of call count. |
| `min_trusted_invocations` | `0` | Calls needed to earn trust by accumulation (when not on the allowlist). `0` disables the accumulation path. |
| `max_probation_share` | `1.0` | Probation+flagged agents combined can occupy at most this fraction of the rolling window. `1.0` = no cap (default behavior). |
| `window_size` | `50` | How many recent selections the share is measured over. |
| `detect_inflated_claims` | `false` | Run the anomaly detector. |
| `inflated_quality_floor` | `0.95` | Flag if recent claimed quality mean ≥ this. Calibrated so 0.99-saturating sybils get caught but a genuinely-consistent 0.9 agent does not. |
| `inflated_stdev_ceiling` | `0.02` | Flag if recent claimed-quality stdev ≤ this (i.e. unnaturally uniform). |
| `inflated_min_calls` | `5` | Minimum sample size before the detector fires. |

The two unimplemented mechanisms from the Stage 6 design
(signed-identity attestation, per-agent rate limits) require
multi-process infrastructure that single-process eval can't exercise.
The allowlist field is the integration point for attestation: in
production, agents whose ed25519 signatures verify against an
external identity registry get added to `trusted_agents`.

### `judge.py` — `QualityJudge`, three implementations

The judge is the answer to "agents can lie about themselves." Interface:

```python
score(*, payload: str, output: Any, hint: Optional[float]) -> JudgeResult
```

- `MockHeuristicJudge` — offline. `0.6 · grounding + 0.4 · length_score`
  where `grounding` is the fraction of output tokens present in the
  input (catches hallucinations) and `length_score` rewards summaries
  in the 10%-50% length-ratio band. Good enough for the demo.
- `OracleJudge` — eval-only. Returns the ground-truth quality from the
  `hint` argument. Raises if `hint` is missing so production code that
  accidentally instantiates it fails loudly.
- `AnthropicJudge` — Claude as judge with a rubric system prompt
  (faithfulness / coverage / conciseness / coherence). Includes prompt
  caching on the rubric so per-call latency and cost stay low. Gated
  by `ANTHROPIC_API_KEY` — fail-loud constructor; `run_demo.py` falls
  back to `MockHeuristicJudge` if the key isn't present.

The judge is optional. Without one, the bandit learns from agent
self-reports — fine if you trust them, catastrophic if you don't.
See the `lying_agent` scenario for the proof.

### `agent_rank_service.py` — `AgentRankService`

Intentionally thin. Builds the bandit, extracts features, delegates,
then runs the trust pass. Two public methods:

- `rank(domain, task_type, payload, preferences=None)` — returns the
  ranked list only. Backward-compatible signature.
- `rank_with_features(domain, task_type, payload, preferences=None)` —
  returns `(ranking, features)` so `AgentClient` can persist features
  alongside the invocation.

`preferences` is forwarded to the bandit (ParetoBandit uses it; UCB1
and LinUCB ignore it). The `ProbationPolicy` runs unconditionally
but is a no-op under the permissive default config.

### `agent_client.py` — `AgentClient`

Owns the full request lifecycle:
rank → dispatch → judge → log. Signature is
`handle_task(domain, task_type, payload, preferences=None)`; the
preferences flow through to the bandit when ParetoBandit is
configured.

Important: the judge is called *after* dispatch (we need real output
to score) and the result *replaces* the agent's self-reported
`quality_score` before logging. The original claim is preserved in
`agent_claimed_quality` so you can later answer "did the agent ever
inflate?" The trust policy's inflated-claim anomaly detector reads
exactly that column.

If the judge raises, the error is captured in `metrics.judge_error` but
doesn't break the request flow — a degraded log entry is preferable to
a 500.

### `a2a_protocol.py` — minimal A2A layer

A toy implementation of A2A request/response: a dict in, a dict out,
with `performative` set to `inform` or `failure`. In a real deployment
this is replaced by the official A2A library. The ranker doesn't care
which.

### `agents/` — three demo summarizers

- `summarizer_fast.py` — truncates to 80 chars. Always succeeds.
- `summarizer_quality.py` — first two sentences. Always succeeds, slow.
- `summarizer_hallucinator.py` — 30% failure rate, 70% absurd answers.

These exist only to demonstrate the protocol end-to-end. The eval
harness uses parametric synthetic agents instead — `evaluation/simulator.py`.

---

## Configuration reference

`config/scoring.json` has four top-level sections:

```json
{
  "scoring": {
    "default": { ...policy... },
    "domains": {
      "nlp/summarize": { ...policy override... }
    }
  },
  "agent_defaults": {
    "SummarizerFast": { ... },
    "SummarizerQuality": { ... },
    "SummarizerHallucinator": { ... },
    "_fallback": { ... }
  },
  "registry": {
    "nlp/summarize": ["SummarizerFast", "SummarizerQuality", "SummarizerHallucinator"]
  },
  "persistence": {
    "db_path": "agentrank.db"
  }
}
```

A complete policy:

```json
{
  "weights": {
    "success_rate": 0.25,
    "quality_score": 0.50,
    "latency_score": 0.15,
    "failure_rate": -0.10
  },
  "exploration": { "alpha": 0.2 },
  "drift": { "half_life_calls": null },
  "bandit": "ucb1",
  "feature_extractor": null,
  "bandit_params": {},
  "trust": {
    "trusted_agents": [],
    "min_trusted_invocations": 0,
    "max_probation_share": 1.0,
    "window_size": 50,
    "detect_inflated_claims": false,
    "inflated_quality_floor": 0.95,
    "inflated_stdev_ceiling": 0.02,
    "inflated_min_calls": 5
  }
}
```

For LinUCB:

```json
{
  "bandit": "linucb",
  "feature_extractor": "length_bucket",
  "bandit_params": { "alpha": 0.5, "ridge": 1.0, "warm_start_n": 3 }
}
```

For ParetoBandit:

```json
{
  "bandit": "pareto",
  "bandit_params": {
    "alpha": 0.2,
    "cost_reference_cents": 10.0,
    "default_preferences": {
      "quality_score": 0.5,
      "latency_score": 0.2,
      "cost_score": 0.2,
      "success_rate": 0.1
    }
  }
}
```

For trust-protected ranking (any bandit can opt into a non-default
trust policy):

```json
{
  "bandit": "ucb1",
  "trust": {
    "trusted_agents": ["SummarizerFast", "SummarizerQuality"],
    "min_trusted_invocations": 25,
    "max_probation_share": 0.10,
    "detect_inflated_claims": true
  }
}
```

Any subset of these can be specified — the rest fall back to the
defaults in `_DEFAULT_POLICY`.

---

## Key design decisions and tradeoffs

### Why is reward recomputed at LinUCB fit time, not stored at write time?

If we stored `r_t` directly, changing scoring weights would invalidate
the log. Recomputing from `success / quality_score / latency_ms` and
the current weights means policies can be tuned without retraining or
re-logging. Cost: O(N) recomputation per rank, which at our scale
(low thousands of rows) is negligible.

### Why decay by `id`, not wall-clock `timestamp`?

The eval runs hundreds of invocations per millisecond. Wall-clock has
no meaningful resolution there. The id column is monotonic
(`AUTOINCREMENT`) and the natural unit for "number of invocations
ago." A real production deployment that wants time-based decay can
swap this trivially — `_decay_weight` is a single function.

### Why warm-start LinUCB instead of just tuning α?

Higher α makes LinUCB explore more, but it also makes confidence
bounds dominate over mean estimates indefinitely — the bandit never
"commits." Warm-start gives a clean separation: forced exploration
gets each agent enough samples to differentiate contexts, then α
controls steady-state exploration. Cleaner than trying to find one α
that does both jobs.

### Why is the judge optional?

Adding a real LLM-as-judge to every call is expensive (latency, cost).
The interface is in place so users can opt in. The `MockHeuristicJudge`
gives most of the benefit (catches hallucinations and pass-throughs)
for zero runtime cost. The eval's `OracleJudge` lets us isolate the
*value* of judging from the *implementation* of any specific judge.

### Why is the bandit stateless?

State in SQLite means:
- the demo survives restarts (which the original in-memory version did not)
- multiple processes can share a DB
- the bandit's "fit" is always consistent with what's in the log

The cost is recomputing on every rank. For thousands of logs and tiny
feature dims, this is fine. For millions of logs you'd want an
incremental update path — easy to add to `LinUCBBandit` without
touching its public interface.

### Why per-trial RNG seeding in the eval?

So contextual scenarios produce the same payload stream for the
strategy *and* the oracle. The oracle is recomputed per trial with
the same seed; without that, the oracle would average over a different
distribution than the strategies actually saw, and regret numbers
would be wrong. The same applies to per-request preferences in
ParetoBandit scenarios — same seed → same preference stream → honest
oracle.

### Why is trust a separate pass instead of a bandit feature?

The bandit's job is "given the metrics I see, what's the best agent?"
The trust policy's job is "is this metric stream trustworthy in the
first place?" Mixing those collapses two orthogonal concerns into
one component. Keeping trust as a post-pass means any bandit (UCB1,
LinUCB, ParetoBandit, anything we add later) gets sybil resistance
for free, and the trust policy itself can be swapped — allowlist
today, signed-identity attestation tomorrow — without touching the
bandits.

### Why does ParetoBandit inflate per-dimension instead of using a single UCB term?

The whole point of multi-objective scoring is that different agents
win on different dimensions. Adding one global exploration scalar
to the projected score would behave like UCB1 with a fancier base —
agents that are confident on quality but uncertain on latency would
get the same exploration boost across all objectives. Per-dimension
inflation lets exploration cost differ by dimension, which is the
right behavior when one agent has lots of "this is fast" evidence but
no "this is high quality" evidence.

---

## Where to add things

| You want to... | Touch this |
|---|---|
| Add a new agent | `agents/<name>.py`, `a2a_protocol.AGENT_HANDLERS`, `config/scoring.json` (`registry`, optionally `agent_defaults`) |
| Add a new domain | `config/scoring.json` (`scoring.domains`, `registry`), nothing in code |
| Add a new bandit policy | New class in `bandits.py`, dispatch in `build_bandit`, optionally new keys in `_DEFAULT_POLICY["bandit_params"]` |
| Add a new feature extractor | New class in `feature_extractor.py`, register in `FEATURE_EXTRACTORS` |
| Add a new judge | New class in `judge.py` implementing `QualityJudge`, optionally wire into `run_demo.build_judge()` |
| Add a new objective dimension | Add column in `log_store.py` schema (auto-migrate), aggregator in `get_objective_vector`, key in `ParetoBandit.OBJECTIVE_KEYS`, scoring weight in config defaults |
| Add a new trust mechanism | Either extend `TrustConfig` + `ProbationPolicy` in `trust.py`, or implement a separate policy class and chain after `ProbationPolicy.adjust` in `AgentRankService` |
| Add a new eval scenario | New `Scenario` in `evaluation/scenarios.py`, append to `ALL_SCENARIOS`. Optionally set `enable_*_variant` flags to register comparison variants. |
| Add a new selection strategy to the eval | New class in `evaluation/strategies.py`, instantiate in `run_eval.build_strategies()` |

See [EVAL.md](EVAL.md) for the eval-specific extension points.
