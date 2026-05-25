"""
AgentClient: uses AgentRankService to pick the best agent for a task,
then dispatches via the A2A protocol layer and records the outcome.

If a QualityJudge is provided, it scores the agent's output and
overrides the agent's self-reported quality before the metrics are
logged. This prevents agents from gaming the ranker by inflating their
own quality_score.
"""

from typing import Any, Dict, Optional

from agent_rank_service import AgentRankService
from a2a_protocol import send_message
from judge import QualityJudge


class AgentClient:
    def __init__(
        self,
        rank_service: AgentRankService,
        judge: Optional[QualityJudge] = None,
    ):
        self.rank_service = rank_service
        self.judge = judge

    def handle_task(self, domain: str, task_type: str, payload: str) -> Dict[str, Any]:
        # 1) Ask AgentRank for the best agent (and the extracted features,
        #    which we'll persist alongside the invocation so the bandit
        #    can learn from them).
        ranking, features = self.rank_service.rank_with_features(
            domain, task_type, payload
        )

        if not ranking:
            print("[AgentClient] No agents available for this task.")
            return {"error": "no_agents"}

        bandit_kind = ranking[0].get("bandit", "ucb1")
        print(
            f"\n[AgentRank] Ranking for domain={domain} task_type={task_type} "
            f"bandit={bandit_kind}"
        )
        for r in ranking:
            m = r["metrics"]
            # LinUCB doesn't aggregate the same way UCB1 does, so we
            # render its internals when present and fall back to the
            # UCB1 fields otherwise.
            if "linucb_mean" in m:
                tail = (
                    f"| mean={m['linucb_mean']:+.3f} "
                    f"conf={m['linucb_confidence']:.3f}"
                )
            else:
                tail = (
                    f"| SR={m['success_rate']:.2f} QS={m['quality_score']:.2f} "
                    f"LS={m['latency_score']:.2f} FR={m['failure_rate']:.2f}"
                )
            print(
                f"  - {r['agent_id']:24s} | score={r['score']:.3f} "
                f"(base={r['base_score']:.3f} +explore={r['exploration_bonus']:.3f}) "
                f"| n={r['n_a']:>5.1f} {tail}"
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

        # 4) Run the judge (if configured) on the response and override
        #    the agent's self-reported quality. We only judge successful
        #    calls — a failed call has nothing meaningful to score.
        metrics = response.get("metrics") or {}
        if self.judge is not None and metrics.get("success"):
            try:
                verdict = self.judge.score(
                    payload=payload,
                    output=response.get("content", {}),
                )
                claimed = metrics.get("quality_score")
                metrics["quality_score"] = verdict.score
                metrics["judge_name"] = verdict.judge_name
                metrics["judge_reason"] = verdict.reason
                metrics["agent_claimed_quality"] = claimed
                print(
                    f"[Judge] {verdict.judge_name}: "
                    f"agent claimed {claimed:.2f}, judge says {verdict.score:.2f} "
                    f"({verdict.reason})"
                )
            except Exception as e:  # noqa: BLE001
                # A failing judge must not break the request flow.
                metrics["judge_error"] = str(e)
                print(f"[Judge] error: {e}")

        # 5) Attach the context features (if any) so contextual bandits
        #    can learn from this call, then record.
        if features is not None:
            metrics["context_features"] = features.tolist()
        if metrics:
            self.rank_service.log_store.record_invocation(metrics)

        return response
