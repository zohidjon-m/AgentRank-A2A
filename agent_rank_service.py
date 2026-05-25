"""
AgentRank service: ranks candidate agents using a pluggable bandit
policy. UCB1 is the default (context-blind); LinUCB plugs in when a
domain configures a feature extractor.

This module is intentionally thin: it loads the right bandit + extractor
for the requested domain and delegates. Everything algorithmic lives in
bandits.py and feature_extractor.py.
"""

import json
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from log_store import LogStore
from domain_registry import DomainRegistry
from config_loader import ScoringConfig
from bandits import build_bandit
from feature_extractor import get_extractor, PayloadFeatureExtractor


class AgentRankService:
    def __init__(
        self,
        log_store: LogStore,
        registry: DomainRegistry,
        config: ScoringConfig,
    ):
        self.log_store = log_store
        self.registry = registry
        self.config = config

    def rank(
        self,
        domain: str,
        task_type: str,
        payload: str,
    ) -> List[Dict[str, Any]]:
        """
        Returns candidate agents sorted by score (highest first).

        For backwards compatibility the returned dicts have the same
        shape as before: agent_id, score, base_score, exploration_bonus,
        n_a, metrics, bandit.
        """
        ranking, _ = self.rank_with_features(domain, task_type, payload)
        return ranking

    def rank_with_features(
        self,
        domain: str,
        task_type: str,
        payload: str,
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        """
        Same as rank(), but also returns the extracted feature vector
        (or None when no extractor is configured). AgentClient persists
        this alongside the invocation so the bandit can learn from it.
        """
        candidates = self.registry.get_agents(domain, task_type)
        domain_key = f"{domain}/{task_type}"
        policy = self.config.policy_for(domain_key)

        extractor = self._extractor_for(policy)
        features: Optional[np.ndarray] = None
        feature_dim = 0
        if extractor is not None:
            features = extractor.extract(payload)
            feature_dim = extractor.dim

        bandit = build_bandit(policy, feature_dim=feature_dim)
        ranking = bandit.rank(
            candidates=candidates,
            weights=policy["weights"],
            log_store=self.log_store,
            context_features=features,
        )
        return ranking, features

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _extractor_for(policy: Dict[str, Any]) -> Optional[PayloadFeatureExtractor]:
        name = policy.get("feature_extractor")
        if not name:
            return None
        return get_extractor(name)
