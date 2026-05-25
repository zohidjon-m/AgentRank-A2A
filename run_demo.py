"""
Entry point for the AgentRank demo.

Wires together: config -> persistent log store -> registry ->
rank service -> client. Each invocation prints the full ranking
breakdown so you can watch the UCB bonus shift selections over time.

The MockHeuristicJudge is wired in by default. It rescores the agent's
output using length + word-overlap heuristics, which catches the
hallucinator's invented facts even when it claims success. Swap in
AnthropicJudge once you have ANTHROPIC_API_KEY set.
"""

import os

from config_loader import ScoringConfig
from log_store import LogStore
from domain_registry import DomainRegistry
from agent_rank_service import AgentRankService
from agent_client import AgentClient
from judge import MockHeuristicJudge, AnthropicJudge


def build_judge():
    """Use the Anthropic judge if a key is set, otherwise fall back to the mock."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicJudge()
        except (ImportError, RuntimeError) as e:
            print(f"[demo] AnthropicJudge unavailable ({e}); using MockHeuristicJudge")
    return MockHeuristicJudge()


def main():
    config = ScoringConfig.load("config/scoring.json")
    logs = LogStore(db_path=config.db_path(), config=config)
    registry = DomainRegistry(config)
    rank_service = AgentRankService(logs, registry, config)
    judge = build_judge()
    client = AgentClient(rank_service, judge=judge)

    print("=== AgentRank + A2A Demo (Summarization Domain) ===")
    print("Agents: SummarizerFast, SummarizerQuality, SummarizerHallucinator")
    print(f"Persistence: {config.db_path()} | prior invocations: {logs.total_calls()}")
    print(f"Judge: {judge.name}")

    try:
        while True:
            try:
                text = input("\nEnter text to summarize (or 'q' to quit): ").strip()
            except EOFError:
                break
            if text.lower() in ("q", "quit", "exit"):
                break
            if not text:
                continue

            response = client.handle_task("nlp", "summarize", text)
            print("\n[A2A] Response performative:", response.get("performative"))
            print("[A2A] Sender:", response.get("sender"))
            print("[A2A] Content:", response.get("content"))
    except KeyboardInterrupt:
        pass
    finally:
        logs.close()
        print("\nDemo finished.")


if __name__ == "__main__":
    main()
