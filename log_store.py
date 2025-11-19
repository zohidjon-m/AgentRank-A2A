# log_store.py
"""
In-memory log store + metric computation for AgentRank.
Good enough for a hackathon demo.
"""

from typing import Dict, Any, List
import time
import math


class LogStore:
    def __init__(self):
        # each entry: {agent_id, success, quality_score, latency_ms, failure_reason, timestamp}
        self._logs: List[Dict[str, Any]] = []

    def record_invocation(self, metrics: Dict[str, Any]) -> None:
        entry = {
            "agent_id": metrics["agent_id"],
            "success": int(metrics["success"]),
            "quality_score": float(metrics["quality_score"]),
            "latency_ms": int(metrics["latency_ms"]),
            "failure_reason": metrics.get("failure_reason"),
            "timestamp": time.time(),
        }
        self._logs.append(entry)

    def _filter_by_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        return [e for e in self._logs if e["agent_id"] == agent_id]

    def compute_metrics(self, agent_id: str) -> Dict[str, float]:
        """
        Compute simple aggregate metrics for an agent.
        If no history exists, return neutral values (0.5).
        """
        entries = self._filter_by_agent(agent_id)
        if not entries:
            return {
                "success_rate": 0.5,
                "quality_score": 0.5,
                "latency_score": 0.5,
                "failure_rate": 0.5,
            }

        n = len(entries)
        successes = sum(e["success"] for e in entries)
        failures = n - successes
        avg_quality = sum(e["quality_score"] for e in entries) / n
        avg_latency = sum(e["latency_ms"] for e in entries) / n

        # map latency (0..3000+ ms) to 1..0
        latency_score = 1.0 - min(avg_latency / 3000.0, 1.0)

        success_rate = successes / n
        failure_rate = failures / n

        return {
            "success_rate": success_rate,
            "quality_score": avg_quality,
            "latency_score": latency_score,
            "failure_rate": failure_rate,
        }
