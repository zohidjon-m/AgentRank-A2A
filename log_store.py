"""
In-memory log store + metric computation for AgentRank.
Now includes:
- call count per agent (n_a)
- total call count (N)
"""

from typing import Dict, Any, List
import time


class LogStore:
    def __init__(self):
        # Each entry tracks one invocation
        self._logs: List[Dict[str, Any]] = []

    def _default_metrics_for(self, agent_id: str) -> Dict[str, float]:
        default_profiles = {
            "SummarizerFast": {
                "success_rate": 0.95,
                "quality_score": 0.5,
                "latency_score": 0.9,
                "failure_rate": 0.05,
            },
            "SummarizerQuality": {
                "success_rate": 0.98,
                "quality_score": 0.9,
                "latency_score": 0.7,
                "failure_rate": 0.02,
            },
            "SummarizerHallucinator": {
                "success_rate": 0.7,
                "quality_score": 0.2,
                "latency_score": 0.6,
                "failure_rate": 0.3,
            },
        }
        return default_profiles.get(
            agent_id,
            {
                "success_rate": 0.5,
                "quality_score": 0.5,
                "latency_score": 0.5,
                "failure_rate": 0.5,
            },
        )

    def record_invocation(self, metrics: Dict[str, Any]) -> None:
        print("DEBUG LOG ENTRY AGENT ID:", repr(metrics["agent_id"]))
        # print("DEBUG → LogStore ID used for logging:", id(self))
        entry = {
            "agent_id": metrics["agent_id"],
            "success": int(metrics["success"]),
            "quality_score": float(metrics["quality_score"]),
            "latency_ms": int(metrics["latency_ms"]),
            "failure_reason": metrics.get("failure_reason"),
            "timestamp": time.time(),
        }
        self._logs.append(entry)

    def _entries_for(self, agent_id: str) -> List[Dict[str, Any]]:
        return [e for e in self._logs if e["agent_id"] == agent_id]

    def total_calls(self) -> int:
        return len(self._logs)

    def calls_for_agent(self, agent_id: str) -> int:
        return len(self._entries_for(agent_id))

    def compute_metrics(self, agent_id: str) -> Dict[str, float]:
        """
        Compute metrics from logs. If no logs → default profile.
        """
        
        entries = self._entries_for(agent_id)
        if not entries:
            return self._default_metrics_for(agent_id)

        n = len(entries)
        successes = sum(e["success"] for e in entries)
        failures = n - successes
        avg_quality = sum(e["quality_score"] for e in entries) / n
        avg_latency = sum(e["latency_ms"] for e in entries) / n

        # Normalize latency to 1 = fast, 0 = slow (3s cutoff)
        latency_score = 1.0 - min(avg_latency / 3000.0, 1.0)

        return {
            "success_rate": successes / n,
            "quality_score": avg_quality,
            "latency_score": latency_score,
            "failure_rate": failures / n,
        }
