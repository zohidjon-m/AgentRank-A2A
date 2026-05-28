"""
Trust / sybil resistance for AgentRank.

The threat: in an open A2A ecosystem, anyone can register an agent. A
malicious operator can spawn dozens of fresh agents with inflated
cold-start priors, then ride the exploration budget — UCB1 will try
each new agent at least once, and even fail-fast detection takes calls
worth of wasted exposure.

This module addresses two of the four mechanisms from the Stage 6
design (see ARCHITECTURE.md):

1. **Probation pool** — agents with fewer than `min_trusted_invocations`
   logged calls are "in probation" and collectively cannot exceed
   `max_probation_share` of recent selections. The bandit still gets
   to explore them, just bounded.

2. **Simple anomaly detection** — agents whose recent claimed quality
   is suspiciously uniform-and-high (the classic "every call is 0.99")
   get an extra penalty even after probation expires. Catches inflated
   self-reports that the judge would also catch — but cheaper.

The remaining two mechanisms (signed identity attestation, per-agent
rate limits) require multi-process infrastructure and are documented
in ARCHITECTURE.md but not implemented here.

The policy applies AFTER the bandit ranks candidates. The bandit still
sees and scores everyone; the probation policy just rearranges the
final ordering when the recent-window quota is exhausted.
"""

import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from log_store import LogStore


@dataclass
class TrustConfig:
    """Per-domain trust policy knobs."""
    # Allowlist of pre-vetted agents (e.g. from a signed registry). These
    # are always trusted, no matter their call count. In production this
    # would be the set of agents whose ed25519 identities have been
    # validated by the marketplace operator. Empty list = no allowlist
    # restriction (all agents earn trust by accumulation).
    trusted_agents: List[str] = field(default_factory=list)
    # Agents NOT on the allowlist must accumulate at least this many
    # recorded calls to escape probation. Set to 0 to disable the
    # accumulation path entirely (allowlist becomes the only way in).
    min_trusted_invocations: int = 0
    # Probation agents combined can occupy at most this fraction of
    # recent selections. 1.0 disables the cap.
    max_probation_share: float = 1.0
    # Rolling window size for measuring the share. Larger = more lenient
    # to short-term bursts of probation selections.
    window_size: int = 50
    # If True, also flag agents whose recent claimed-quality distribution
    # is suspiciously uniform-and-high.
    detect_inflated_claims: bool = False
    # Inflated-claim thresholds. Default floor is 0.95 — high enough that
    # genuinely-consistent agents (e.g. quality=0.9 on every call) are not
    # flagged, but sybils that report a saturating 0.99 every call are.
    inflated_min_calls: int = 5
    inflated_quality_floor: float = 0.95
    inflated_stdev_ceiling: float = 0.02


class ProbationPolicy:
    """
    Stateless probation + anomaly check. Reads recent selections from
    the LogStore; doesn't maintain its own state.
    """

    def __init__(self, config: Optional[TrustConfig] = None):
        self.cfg = config or TrustConfig()

    def is_trusted(self, agent_id: str, log_store: LogStore) -> bool:
        # Allowlist check first — pre-vetted agents skip probation.
        if agent_id in self.cfg.trusted_agents:
            return True
        # Without allowlist OR accumulation requirement, default to
        # trusted (permissive default keeps existing scenarios working).
        if not self.cfg.trusted_agents and self.cfg.min_trusted_invocations <= 0:
            return True
        # Otherwise, earn trust by accumulating verified calls.
        if self.cfg.min_trusted_invocations <= 0:
            return False
        return log_store.calls_for_agent(agent_id) >= self.cfg.min_trusted_invocations

    def is_flagged_for_inflated_claims(
        self,
        agent_id: str,
        log_store: LogStore,
    ) -> bool:
        """
        True iff the agent has reported a suspicious cluster of perfect
        claimed-quality scores. Cheap pre-filter; doesn't replace a
        real judge.
        """
        if not self.cfg.detect_inflated_claims:
            return False
        claims = log_store.recent_claimed_quality(
            agent_id,
            n=max(self.cfg.window_size, self.cfg.inflated_min_calls),
        )
        if len(claims) < self.cfg.inflated_min_calls:
            return False
        mean = statistics.mean(claims)
        std = statistics.pstdev(claims)
        return (
            mean >= self.cfg.inflated_quality_floor
            and std <= self.cfg.inflated_stdev_ceiling
        )

    def adjust(
        self,
        ranking: List[Dict[str, Any]],
        log_store: LogStore,
    ) -> List[Dict[str, Any]]:
        """
        Return a (possibly reordered) ranking that respects the
        probation share cap. Tags each entry with `trust_status`:
            'trusted'             — on allowlist or earned trust
            'probation'           — not on allowlist, hasn't earned trust
            'flagged_inflated'    — anomaly detector tripped
            'demoted'             — was in probation AND the cap was hit
        """
        is_noop = (
            not self.cfg.trusted_agents
            and self.cfg.min_trusted_invocations <= 0
            and not self.cfg.detect_inflated_claims
        )
        if is_noop:
            for r in ranking:
                r["trust_status"] = "trusted"
            return ranking

        # First pass: classify each candidate.
        for r in ranking:
            aid = r["agent_id"]
            if self.is_flagged_for_inflated_claims(aid, log_store):
                r["trust_status"] = "flagged_inflated"
            elif not self.is_trusted(aid, log_store):
                r["trust_status"] = "probation"
            else:
                r["trust_status"] = "trusted"

        # Second pass: enforce share cap on recent selections.
        recent = log_store.recent_selections(self.cfg.window_size)
        if recent:
            untrusted_count = sum(
                1 for aid in recent
                if not self.is_trusted(aid, log_store)
                or self.is_flagged_for_inflated_claims(aid, log_store)
            )
            share = untrusted_count / len(recent)
        else:
            share = 0.0

        if share >= self.cfg.max_probation_share:
            # Cap exceeded: rerank so all trusted agents come first.
            trusted = [r for r in ranking if r["trust_status"] == "trusted"]
            untrusted = [r for r in ranking if r["trust_status"] != "trusted"]
            for r in untrusted:
                if r["trust_status"] != "flagged_inflated":
                    r["trust_status"] = "demoted"
            return trusted + untrusted

        return ranking
