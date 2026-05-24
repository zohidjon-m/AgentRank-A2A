"""
AgentClient: uses AgentRankService to pick the best agent for a task,
then dispatches via the A2A protocol layer and records the outcome.
"""

from typing import Any, Dict

from agent_rank_service import AgentRankService
from a2a_protocol import send_message


class AgentClient:
    def __init__(self, rank_service: AgentRankService):
        self.rank_service = rank_service

    def handle_task(self, domain: str, task_type: str, payload: str) -> Dict[str, Any]:
        # 1) Ask AgentRank for the best agent.
        ranking = self.rank_service.rank(domain, task_type, payload)

        if not ranking:
            print("[AgentClient] No agents available for this task.")
            return {"error": "no_agents"}

        print(f"\n[AgentRank] Ranking for domain={domain} task_type={task_type}")
        for r in ranking:
            m = r["metrics"]
            print(
                f"  - {r['agent_id']:24s} | score={r['score']:.3f} "
                f"(base={r['base_score']:.3f} +explore={r['exploration_bonus']:.3f}) "
                f"| n={r['n_a']:<3d} "
                f"| SR={m['success_rate']:.2f} QS={m['quality_score']:.2f} "
                f"LS={m['latency_score']:.2f} FR={m['failure_rate']:.2f}"
            )

        best = ranking[0]["agent_id"]
        print(f"\n[AgentRank] Selected best agent: {best}")

        # 2) Build the A2A request.
        request = {
            "performative": "request",
            "sender": "AgentClient",
            "receiver": best,
            "domain": domain,
            "task_type": task_type,
            "content": payload,
        }

        # 3) Dispatch.
        response = send_message(request)

        # 4) Record metrics for future rankings.
        metrics = response.get("metrics")
        if metrics:
            self.rank_service.log_store.record_invocation(metrics)

        return response
