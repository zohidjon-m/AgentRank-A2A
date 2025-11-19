# run_demo.py
"""
Entry point for the hackathon demo.
Shows:
- multiple agents in same domain
- AgentRank ranking them
- A2A-style request/response
- logs updating and changing future rankings
"""

from log_store import LogStore
from domain_registry import DomainRegistry
from agent_rank_service import AgentRankService
from agent_client import AgentClient


def main():
    logs = LogStore()
    registry = DomainRegistry()
    rank_service = AgentRankService(logs, registry)
    client = AgentClient(rank_service)

    print("=== AgentRank + A2A Demo (Summarization Domain) ===")
    print("Agents in domain 'nlp/summarize': SummarizerFast, SummarizerQuality, SummarizerHallucinator")

    while True:
        try:
            text = input("\nEnter text to summarize (or 'q' to quit): ").strip()
            if text.lower() in ("q", "quit", "exit"):
                break

            response = client.handle_task("nlp", "summarize", text)

            print("\n[A2A] Response performative:", response.get("performative"))
            print("[A2A] Sender:", response.get("sender"))
            print("[A2A] Content:", response.get("content"))

        except KeyboardInterrupt:
            break

    print("\nDemo finished.")


if __name__ == "__main__":
    main()
