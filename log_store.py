"""
SQLite-backed log store + metric computation for AgentRank.

Stores one row per agent invocation and computes aggregate metrics on
demand. Cold-start priors come from the config (so adding a new agent
no longer requires editing this file).

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
                timestamp REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_invocations_agent ON invocations(agent_id)"
        )
        self._conn.commit()

    # ---- writes ------------------------------------------------------------

    def record_invocation(self, metrics: Dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO invocations
                (agent_id, success, quality_score, latency_ms,
                 failure_reason, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                metrics["agent_id"],
                int(metrics["success"]),
                float(metrics["quality_score"]),
                int(metrics["latency_ms"]),
                metrics.get("failure_reason"),
                time.time(),
            ),
        )
        self._conn.commit()

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

    def compute_metrics(self, agent_id: str) -> Dict[str, float]:
        """
        Compute metrics from logs. If no logs exist, return the config's
        cold-start prior for the agent.
        """
        rows = self._conn.execute(
            """
            SELECT success, quality_score, latency_ms
            FROM invocations
            WHERE agent_id = ?
            """,
            (agent_id,),
        ).fetchall()

        if not rows:
            return self._default_for(agent_id)

        n = len(rows)
        successes = sum(int(r["success"]) for r in rows)
        failures = n - successes
        avg_quality = sum(float(r["quality_score"]) for r in rows) / n
        avg_latency = sum(int(r["latency_ms"]) for r in rows) / n

        # Normalize latency to 1 = fast, 0 = slow (3s cutoff).
        latency_score = 1.0 - min(avg_latency / 3000.0, 1.0)

        return {
            "success_rate": successes / n,
            "quality_score": avg_quality,
            "latency_score": latency_score,
            "failure_rate": failures / n,
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
