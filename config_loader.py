"""
Configuration loader for AgentRank.

Centralizes scoring weights, exploration parameters, agent priors,
and the agent registry. All knobs live in config/scoring.json so the
scoring policy is data-driven rather than hardcoded.
"""

import json
from typing import Dict, Any, List, Tuple


_NEUTRAL_DEFAULT = {
    "success_rate": 0.5,
    "quality_score": 0.5,
    "latency_score": 0.5,
    "failure_rate": 0.5,
}

_DEFAULT_POLICY = {
    "weights": {
        "success_rate": 0.4,
        "quality_score": 0.3,
        "latency_score": 0.2,
        "failure_rate": -0.1,
    },
    "exploration": {"alpha": 0.5},
}


class ScoringConfig:
    """Read-only view of the scoring/registry config."""

    def __init__(self, raw: Dict[str, Any]):
        self._raw = raw

    @classmethod
    def load(cls, path: str = "config/scoring.json") -> "ScoringConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    # ---- scoring policy ----------------------------------------------------

    def policy_for(self, domain_key: str) -> Dict[str, Any]:
        """
        Return the scoring policy (weights + exploration) for a domain key
        like "nlp/summarize". Falls back to the default policy if the key
        is not configured.
        """
        scoring = self._raw.get("scoring", {})
        domains = scoring.get("domains", {})
        if domain_key in domains:
            policy = domains[domain_key]
        else:
            policy = scoring.get("default", _DEFAULT_POLICY)

        # Defensive fill so callers always see all keys.
        weights = {**_DEFAULT_POLICY["weights"], **policy.get("weights", {})}
        exploration = {**_DEFAULT_POLICY["exploration"], **policy.get("exploration", {})}
        return {"weights": weights, "exploration": exploration}

    # ---- agent priors ------------------------------------------------------

    def agent_default(self, agent_id: str) -> Dict[str, float]:
        """Cold-start metric prior for an agent."""
        defaults = self._raw.get("agent_defaults", {})
        if agent_id in defaults:
            return dict(defaults[agent_id])
        if "_fallback" in defaults:
            return dict(defaults["_fallback"])
        return dict(_NEUTRAL_DEFAULT)

    # ---- registry ----------------------------------------------------------

    def registry_pairs(self) -> List[Tuple[str, str, List[str]]]:
        """
        Returns a list of (domain, task_type, [agent_ids]) tuples derived
        from registry keys of the form "domain/task_type".
        """
        out: List[Tuple[str, str, List[str]]] = []
        for key, agents in self._raw.get("registry", {}).items():
            if "/" not in key:
                continue
            domain, task_type = key.split("/", 1)
            out.append((domain, task_type, list(agents)))
        return out

    # ---- persistence -------------------------------------------------------

    def db_path(self) -> str:
        return self._raw.get("persistence", {}).get("db_path", "agentrank.db")
