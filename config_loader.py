"""
Configuration loader for AgentRank.

Centralizes scoring weights, exploration parameters, agent priors,
and the agent registry. All knobs live in config/scoring.json so the
scoring policy is data-driven rather than hardcoded.

`with_priors` / `with_registry` return cloned configs with overrides
applied — used by the eval harness to compose per-scenario views.
"""

import copy
import json
from typing import Dict, Any, List, Tuple, Optional


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
    # Concept-drift handling. half_life_calls = N means a log entry's
    # weight halves every N invocations. None / 0 disables decay
    # (flat averaging over all history — the original behavior).
    "drift": {"half_life_calls": None},
    # Bandit policy: "ucb1" (context-blind, the default) or "linucb"
    # (contextual; requires feature_extractor to be set).
    "bandit": "ucb1",
    # Optional name of a registered PayloadFeatureExtractor. Required for
    # LinUCB; ignored by UCB1.
    "feature_extractor": None,
    # Optional per-bandit parameters (e.g., {"ridge": 1.0} for LinUCB).
    "bandit_params": {},
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
        drift = {**_DEFAULT_POLICY["drift"], **policy.get("drift", {})}
        bandit_params = {**_DEFAULT_POLICY["bandit_params"], **policy.get("bandit_params", {})}
        return {
            "weights": weights,
            "exploration": exploration,
            "drift": drift,
            "bandit": policy.get("bandit", _DEFAULT_POLICY["bandit"]),
            "feature_extractor": policy.get(
                "feature_extractor", _DEFAULT_POLICY["feature_extractor"]
            ),
            "bandit_params": bandit_params,
        }

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

    # ---- overrides (for eval / tests) --------------------------------------

    def with_priors(self, priors: Dict[str, Dict[str, float]]) -> "ScoringConfig":
        """Return a clone with additional / overridden agent priors."""
        new_raw = copy.deepcopy(self._raw)
        new_raw.setdefault("agent_defaults", {})
        new_raw["agent_defaults"].update(priors)
        return ScoringConfig(new_raw)

    def with_registry(self, domain_key: str, agents: List[str]) -> "ScoringConfig":
        """Return a clone with the registry entry for domain_key replaced."""
        new_raw = copy.deepcopy(self._raw)
        new_raw.setdefault("registry", {})
        new_raw["registry"][domain_key] = list(agents)
        return ScoringConfig(new_raw)

    def with_drift_half_life(
        self,
        domain_key: str,
        half_life_calls: Optional[float],
    ) -> "ScoringConfig":
        """Return a clone with concept-drift half-life set for one domain."""
        new_raw = copy.deepcopy(self._raw)
        scoring = new_raw.setdefault("scoring", {})
        domains = scoring.setdefault("domains", {})
        if domain_key not in domains:
            domains[domain_key] = copy.deepcopy(
                scoring.get("default", _DEFAULT_POLICY)
            )
        domains[domain_key].setdefault("drift", {})
        domains[domain_key]["drift"]["half_life_calls"] = half_life_calls
        return ScoringConfig(new_raw)

    def with_bandit(
        self,
        domain_key: str,
        kind: str,
        feature_extractor: Optional[str] = None,
        bandit_params: Optional[Dict[str, Any]] = None,
    ) -> "ScoringConfig":
        """
        Return a clone with the bandit kind / feature extractor configured
        for one domain. Used by the eval harness to register a LinUCB
        variant without touching the persisted config.
        """
        new_raw = copy.deepcopy(self._raw)
        scoring = new_raw.setdefault("scoring", {})
        domains = scoring.setdefault("domains", {})
        if domain_key not in domains:
            domains[domain_key] = copy.deepcopy(
                scoring.get("default", _DEFAULT_POLICY)
            )
        domains[domain_key]["bandit"] = kind
        if feature_extractor is not None:
            domains[domain_key]["feature_extractor"] = feature_extractor
        if bandit_params is not None:
            domains[domain_key]["bandit_params"] = dict(bandit_params)
        return ScoringConfig(new_raw)
