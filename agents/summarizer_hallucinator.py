# agents/summarizer_hallucinator.py
"""
Sometimes fails or "hallucinates".
Used to show how AgentRank penalizes bad behavior.
"""

import time
import random
from typing import Dict, Any


def handle(text: str) -> Dict[str, Any]:
    # Random latency
    time.sleep(random.uniform(0.2, 1.2))

    # 30% chance of failing
    if random.random() < 0.3:
        return {
            "summary": None,
            "agent_flavor": "hallucinator",
            "success": 0,
            "quality_score": 0.0,
            "failure_reason": "random_failure",
        }

    # 70% of the time, produce a weird / low-quality "summary"
    fake_fact = random.choice(
        [
            "This text is secretly about quantum turtles.",
            "The main point is that coffee controls the universe.",
            "The document proves that Mondays do not exist.",
        ]
    )

    return {
        "summary": fake_fact,
        "agent_flavor": "hallucinator",
        "success": 1,
        "quality_score": 0.2,
        "failure_reason": None,
    }
