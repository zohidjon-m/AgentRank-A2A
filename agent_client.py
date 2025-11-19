# agent_client.py
"""
AgentClient that uses AgentRankService to pick the best agent, then talks via A2A.
"""

from typing import Any, Dict
from agent_rank_service import AgentRankService
from a2a_protocol import send_message


class AgentClient:
    def __init__(self, rank_service: AgentRankService):
        self.rank_service = rank_service

    def handle_task(self, domain: str, task_type: str, payload: str) -> Dict[str, Any]:
        # 1) Ask AgentRank for best agent
        ranking = self.rank_service.rank(domain, task_type, payload)

        if not ranking:
            print("[AgentClient] No agents available for this task.")
            return {"error": "no_agents"}

        print("\n[AgentRank] Ranking for domain=", domain, "task_type=", task_type)
        for r in ranking:
            m = r["metrics"]
            print(
                f"  - {r['agent_id']:24s} | score={r['score']:.3f} "
                f"| SR={m['success_rate']:.2f} QS={m['quality_score']:.2f} "
                f"LS={m['latency_score']:.2f} FR={m['failure_rate']:.2f}"
            )

        best = ranking[0]["agent_id"]
        print(f"\n[AgentRank] Selected best agent: {best}")

        # 2) Build A2A request message
        request = {
            "performative": "request",
            "sender": "AgentClient",
            "receiver": best,
            "domain": domain,
            "task_type": task_type,
            "content": payload,
        }

        # 3) Send via A2A protocol
        response = send_message(request)

        # 4) Record metrics into log store for future rankings
        metrics = response.get("metrics")
        if metrics:
            self.rank_service.log_store.record_invocation(metrics)

        return response
