# agents/summarizer_quality.py
"""
Slower but higher-quality summarizer.
"""

import time
from typing import Dict, Any


def handle(text: str) -> Dict[str, Any]:
    # Simulate heavier processing / model call
    time.sleep(0.8)

    # Very naive "summary", but pretend it's good
    sentences = text.split(".")
    first_two = ". ".join(s.strip() for s in sentences[:2] if s.strip())
    if not first_two:
        first_two = text[:120]

    summary = first_two + ("..." if len(text) > len(first_two) else "")

    return {
        "summary": summary,
        "agent_flavor": "high_quality",
        "success": 1,
        "quality_score": 0.9,
        "failure_reason": None,
    }
