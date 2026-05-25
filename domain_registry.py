"""
Domain registry mapping (domain, task_type) -> list of agent IDs.

Loaded from config so adding a new agent or task does not require
editing this file.
"""

from typing import List, Tuple, Dict, Optional

from config_loader import ScoringConfig


class DomainRegistry:
    def __init__(self, config: Optional[ScoringConfig] = None):
        self._registry: Dict[Tuple[str, str], List[str]] = {}

        if config is not None:
            for domain, task_type, agents in config.registry_pairs():
                self._registry[(domain, task_type)] = agents
        else:
            # Backwards-compatible default for tests that construct
            # the registry without a config.
            self._registry[("nlp", "summarize")] = [
                "SummarizerFast",
                "SummarizerQuality",
                "SummarizerHallucinator",
            ]

    def get_agents(self, domain: str, task_type: str) -> List[str]:
        return list(self._registry.get((domain, task_type), []))
