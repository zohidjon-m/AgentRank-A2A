# domain_registry.py
"""
Static registry mapping (domain, task_type) -> list of agent IDs.
In a real system this might be dynamic or config-driven.
"""

from typing import List, Tuple, Dict


class DomainRegistry:
    def __init__(self):
        # (domain, task_type) -> [agents]
        self._registry: Dict[Tuple[str, str], List[str]] = {
            ("nlp", "summarize"): [
                "SummarizerFast",
                "SummarizerQuality",
                "SummarizerHallucinator",
            ]
        }

    def get_agents(self, domain: str, task_type: str) -> List[str]:
        return self._registry.get((domain, task_type), [])
