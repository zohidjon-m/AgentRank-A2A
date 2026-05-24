"""
Entry point for the AgentRank demo.

Wires together: config -> persistent log store -> registry ->
rank service -> client. Each invocation prints the full ranking
breakdown so you can watch the UCB bonus shift selections over time.
"""

from config_loader import ScoringConfig
from log_store import LogStore
from domain_registry import DomainRegistry
from agent_rank_service import AgentRankService
from agent_client import AgentClient


def main():
    config = ScoringConfig.load("config/scoring.json")
    logs = LogStore(db_path=config.db_path(), config=config)
    registry = DomainRegistry(config)
    rank_service = AgentRankService(logs, registry, config)
    client = AgentClient(rank_service)

    print("=== AgentRank + A2A Demo (Summarization Domain) ===")
    print("Agents: SummarizerFast, SummarizerQuality, SummarizerHallucinator")
    print(f"Persistence: {config.db_path()} | prior invocations: {logs.total_calls()}")

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
