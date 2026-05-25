"""
Non-interactive smoke test for the Stage 1 foundation.

Runs a small batch of requests through the full pipeline and prints the
ranking trace after each call so you can see UCB exploration in action
without having to type input.

    python scripts/smoke_test.py
"""

import sys
from pathlib import Path

# Make the project root importable when running this from anywhere.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config_loader import ScoringConfig
from log_store import LogStore
from domain_registry import DomainRegistry
from agent_rank_service import AgentRankService
from agent_client import AgentClient


SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Climate scientists warn that global temperatures continue to rise, "
    "with significant impacts expected across coastal communities.",
    "Python is a high-level programming language known for its readability "
    "and broad standard library.",
    "Researchers published a new paper on multi-agent coordination.",
    "Breaking: a small earthquake was reported near the eastern coast today.",
    "Coffee consumption studies show mixed results on long-term health.",
    "The latest model release improved benchmark scores across the board.",
    "Markets opened lower today amid concerns about inflation forecasts.",
    "A new species of deep-sea fish was discovered near the Mariana Trench.",
    "Voters head to the polls today in a closely watched runoff election.",
    "Engineers unveiled a prototype electric aircraft with extended range.",
    "Historians revisited primary sources from the colonial archive.",
    "Astronomers detected unusual radio signals from a nearby star system.",
    "The annual literature festival opens tomorrow with international guests.",
    "Cybersecurity firms reported a sharp rise in phishing attempts this quarter.",
]


def main():
    # Use an in-memory db so the smoke test never leaves a file behind.
    config = ScoringConfig.load(str(ROOT / "config" / "scoring.json"))
    logs = LogStore(db_path=":memory:", config=config)
    registry = DomainRegistry(config)
    rank_service = AgentRankService(logs, registry, config)
    client = AgentClient(rank_service)

    print("=" * 70)
    print("Stage 1 smoke test — UCB + config + persistence")
    print("=" * 70)

    selections = []
    for i, text in enumerate(SAMPLE_TEXTS, 1):
        print(f"\n--- Request {i} ---")
        response = client.handle_task("nlp", "summarize", text)
        selections.append(response["sender"])

    print("\n" + "=" * 70)
    print("Selection counts across the batch")
    print("=" * 70)
    from collections import Counter
    for agent, count in Counter(selections).most_common():
        print(f"  {agent:24s} chosen {count} time(s)")
    print(f"\nTotal recorded invocations: {logs.total_calls()}")
    print(
        "If UCB is working you should see the best agent dominate but the "
        "others picked occasionally."
    )

    logs.close()


if __name__ == "__main__":
    main()
