# a2a_protocol.py
"""
Minimal A2A-style message passing layer for the demo.
In a real A2A implementation this would be replaced by the official library.
"""

import time
from typing import Dict, Any
from agents import summarizer_fast, summarizer_quality, summarizer_hallucinator


AGENT_HANDLERS = {
    "SummarizerFast": summarizer_fast.handle,
    "SummarizerQuality": summarizer_quality.handle,
    "SummarizerHallucinator": summarizer_hallucinator.handle,
}


def send_message(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate an A2A request/response cycle.

    Expected request format:
    {
        "performative": "request",
        "sender": "AgentClient",
        "receiver": "SummarizerFast",
        "domain": "nlp",
        "task_type": "summarize",
        "content": "<text>"
    }
    """
    receiver = request["receiver"]
    handler = AGENT_HANDLERS.get(receiver)

    start = time.perf_counter()
    if handler is None:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "performative": "failure",
            "sender": receiver,
            "receiver": request["sender"],
            "content": {"error": f"Unknown agent '{receiver}'"},
            "metrics": {
                "agent_id": receiver,
                "latency_ms": latency_ms,
                "success": 0,
                "quality_score": 0.0,
                "failure_reason": "unknown_agent",
            },
        }

    # Execute the agent "business logic"
    result = handler(request["content"])
    latency_ms = int((time.perf_counter() - start) * 1000)

    success = int(result.get("success", 1))
    quality = float(result.get("quality_score", 0.5))
    failure_reason = None if success else result.get("failure_reason", "unknown")

    response = {
        "performative": "inform" if success else "failure",
        "sender": receiver,
        "receiver": request["sender"],
        "content": {
            k: v for k, v in result.items()
            if k not in ("success", "quality_score", "failure_reason")
        },
        "metrics": {
            "agent_id": receiver,
            "latency_ms": latency_ms,
            "success": success,
            "quality_score": quality,
            "failure_reason": failure_reason,
        },
    }
    return response
