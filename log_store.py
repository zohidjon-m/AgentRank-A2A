"""
SQLite-backed log store + metric computation for AgentRank.

Stores one row per agent invocation and computes aggregate metrics on
demand. Cold-start priors come from the config (so adding a new agent
no longer requires editing this file).

The schema includes optional judge metadata (judge_name, judge_reason,
agent_claimed_quality) for observability — so you can later answer
questions like "did the judge ever disagree with the agent's
self-report, and by how much?"

The interface mirrors the previous in-memory store so callers do not
need to change. Pass db_path=":memory:" for tests.
"""

import sqlite3
import time
from typing import Dict, Any, Optional

from config_loader import ScoringConfig


class LogStore:
    def __init__(
        self,
        db_path: str = ":memory:",
        config: Optional[ScoringConfig] = None,
    ):
        self._db_path = db_path
        self._config = config
        # check_same_thread=False so the same connection can be reused
        # by ad-hoc scripts; we are single-writer in the demo.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ---- schema ------------------------------------------------------------

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS invocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                success INTEGER NOT NULL,
                quality_score REAL NOT NULL,
                latency_ms INTEGER NOT NULL,
                failure_reason TEXT,
                timestamp REAL NOT NULL,
                judge_name TEXT,
                judge_reason TEXT,
                agent_claimed_quality REAL,
                context_features TEXT,
                cost_cents REAL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_invocations_agent ON invocations(agent_id)"
        )

        for col, decl in [
            ("judge_name", "TEXT"),
            ("judge_reason", "TEXT"),
            ("agent_claimed_quality", "REAL"),
            ("context_features", "TEXT"),
            ("cost_cents", "REAL"),
        ]:
            try:
                self._conn.execute(f"ALTER TABLE invocations ADD COLUMN {col} {decl}")
            except sqlite3.OperationalError:
                pass  # column already exists

        self._conn.commit()

    # ---- writes ------------------------------------------------------------

    def record_invocation(self, metrics: Dict[str, Any]) -> None:
        # context_features may arrive as a list/array (from a feature
        # extractor) or as a pre-encoded JSON string. Normalize to text.
        ctx = metrics.get("context_features")
        if ctx is not None and not isinstance(ctx, str):
            import json
            try:
                ctx = json.dumps(list(ctx))
            except (TypeError, ValueError):
                ctx = None

        self._conn.execute(
            """
            INSERT INTO invocations
                (agent_id, success, quality_score, latency_ms,
                 failure_reason, timestamp,
                 judge_name, judge_reason, agent_claimed_quality,
                 context_features, cost_cents)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metrics["agent_id"],
                int(metrics["success"]),
                float(metrics["quality_score"]),
                int(metrics["latency_ms"]),
                metrics.get("failure_reason"),
                time.time(),
                metrics.get("judge_name"),
                metrics.get("judge_reason"),
                metrics.get("agent_claimed_quality"),
                ctx,
                (float(metrics["cost_cents"])
                 if metrics.get("cost_cents") is not None else None),
            ),
        )
        self._conn.commit()

    def get_contextual_logs(self, agent_id: str):
        """
        Yield rows containing the fields LinUCB needs: success,
        quality_score, latency_ms, context_features (JSON text).
        Skips entries that never had features attached.
        """
        return self._conn.execute(
            """
            SELECT success, quality_score, latency_ms, context_features
            FROM invocations
            WHERE agent_id = ? AND context_features IS NOT NULL
            """,
            (agent_id,),
        ).fetchall()

    # ---- reads -------------------------------------------------------------

    def total_calls(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM invocations").fetchone()
        return int(row[0])

    def calls_for_agent(self, agent_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM invocations WHERE agent_id = ?",
            (agent_id,),
        ).fetchone()
        return int(row[0])

    def _current_tick(self) -> int:
        """Highest id ever inserted. AUTOINCREMENT guarantees monotonicity."""
        row = self._conn.execute("SELECT MAX(id) FROM invocations").fetchone()
        return int(row[0]) if row[0] is not None else 0

    @staticmethod
    def _decay_weight(entry_id: int, current_tick: int, half_life: Optional[float]) -> float:
        if not half_life or half_life <= 0:
            return 1.0
        age = max(0, current_tick - entry_id)
        return 0.5 ** (age / half_life)

    def get_weighted_counts(
        self,
        half_life_calls: Optional[float] = None,
    ) -> tuple:
        """
        Returns (total_weight, {agent_id: agent_weight}) for use in the
        UCB exploration bonus. Without half-life, this is equivalent to
        (total_calls, {agent: calls_for_agent}). With half-life, recent
        calls weigh more so stale agents become "exploration-worthy"
        again over time.
        """
        if not half_life_calls or half_life_calls <= 0:
            total = self.total_calls()
            cur = self._conn.execute(
                "SELECT agent_id, COUNT(*) FROM invocations GROUP BY agent_id"
            )
            return float(total), {row[0]: float(row[1]) for row in cur}

        from collections import defaultdict
        current_tick = self._current_tick()
        per_agent = defaultdict(float)
        total = 0.0
        for agent_id, entry_id in self._conn.execute(
            "SELECT agent_id, id FROM invocations"
        ):
            w = self._decay_weight(int(entry_id), current_tick, half_life_calls)
            per_agent[agent_id] += w
            total += w
        return total, dict(per_agent)

    def get_objective_vector(
        self,
        agent_id: str,
        half_life_calls: Optional[float] = None,
        cost_reference_cents: float = 10.0,
    ) -> Dict[str, float]:
        """
        Returns the agent's per-objective scores for Pareto / multi-objective
        ranking. Adds a cost_score dimension on top of the existing
        compute_metrics fields. All values are in [0, 1], higher is better.

            cost_score = 1 - min(avg_cost_cents / cost_reference_cents, 1)

        Falls back to a cost_score of 0.5 if the agent has no logged costs
        and no prior is configured.
        """
        base = self.compute_metrics(agent_id, half_life_calls=half_life_calls)
        # Pull cost samples (these may be NULL for legacy rows).
        rows = self._conn.execute(
            """
            SELECT cost_cents FROM invocations
            WHERE agent_id = ? AND cost_cents IS NOT NULL
            """,
            (agent_id,),
        ).fetchall()
        if rows:
            avg_cost = sum(float(r["cost_cents"]) for r in rows) / len(rows)
            cost_score = 1.0 - min(avg_cost / cost_reference_cents, 1.0)
        else:
            # No cost observations -> fall back to prior or neutral 0.5
            cost_score = float(base.get("cost_score", 0.5))
        return {
            "success_rate": base["success_rate"],
            "quality_score": base["quality_score"],
            "latency_score": base["latency_score"],
            "failure_rate": base["failure_rate"],
            "cost_score": cost_score,
        }

    def compute_metrics(
        self,
        agent_id: str,
        half_life_calls: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute metrics from logs. If no logs exist, return the config's
        cold-start prior for the agent.

        With half_life_calls, log entries are exponentially down-weighted
        by age (measured in invocation counts, not wall-clock time). This
        is the standard adaptation for non-stationary bandits: an agent
        that was great last year and broken this year stops looking great.
        """
        rows = self._conn.execute(
            """
            SELECT id, success, quality_score, latency_ms
            FROM invocations
            WHERE agent_id = ?
            """,
            (agent_id,),
        ).fetchall()

        if not rows:
            return self._default_for(agent_id)

        if half_life_calls and half_life_calls > 0:
            current_tick = self._current_tick()
            weights = [
                self._decay_weight(int(r["id"]), current_tick, half_life_calls)
                for r in rows
            ]
        else:
            weights = [1.0] * len(rows)

        wsum = sum(weights)
        if wsum <= 0:
            # All entries decayed to zero (shouldn't happen with sane
            # half-life, but be defensive). Fall back to the prior.
            return self._default_for(agent_id)

        successes = sum(w * int(r["success"]) for w, r in zip(weights, rows))
        avg_quality = sum(w * float(r["quality_score"]) for w, r in zip(weights, rows)) / wsum
        avg_latency = sum(w * int(r["latency_ms"]) for w, r in zip(weights, rows)) / wsum

        success_rate = successes / wsum
        # Normalize latency to 1 = fast, 0 = slow (3s cutoff).
        latency_score = 1.0 - min(avg_latency / 3000.0, 1.0)

        return {
            "success_rate": success_rate,
            "quality_score": avg_quality,
            "latency_score": latency_score,
            "failure_rate": 1.0 - success_rate,
        }

    # ---- helpers -----------------------------------------------------------

    def _default_for(self, agent_id: str) -> Dict[str, float]:
        if self._config is not None:
            return self._config.agent_default(agent_id)
        return {
            "success_rate": 0.5,
            "quality_score": 0.5,
            "latency_score": 0.5,
            "failure_rate": 0.5,
        }

    def close(self) -> None:
        self._conn.close()
