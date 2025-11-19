# agents/summarizer_fast.py
"""
Fast, low-quality summarizer.
Used to show tradeoff between latency and quality.
"""

import time
from typing import Dict, Any


def handle(text: str) -> Dict[str, Any]:
    # Simulate very fast processing
    time.sleep(0.1)

    summary = text[:80] + ("..." if len(text) > 80 else "")

    return {
        "summary": summary,
        "agent_flavor": "fast",
        "success": 1,
        "quality_score": 0.5,
        "failure_reason": None,
    }
