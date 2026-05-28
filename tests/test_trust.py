"""ProbationPolicy: allowlist, share cap, inflated-claim detector."""

import pytest

from log_store import LogStore
from trust import TrustConfig, ProbationPolicy


@pytest.fixture
def store():
    s = LogStore(db_path=":memory:")
    yield s
    s.close()


def _ranking(*agent_ids, base_score=0.5):
    """Build a fake bandit ranking output for these agent ids."""
    return [
        {
            "agent_id": aid,
            "score": base_score,
            "base_score": base_score,
            "exploration_bonus": 0.0,
            "n_a": 0.0,
            "metrics": {},
            "bandit": "fake",
        }
        for aid in agent_ids
    ]


def _seed(store, agent_id, n, q=0.5):
    for _ in range(n):
        store.record_invocation({
            "agent_id": agent_id, "success": 1,
            "quality_score": q, "latency_ms": 100,
            "failure_reason": None,
        })


class TestPermissiveDefaultIsNoOp:
    def test_default_config_tags_all_trusted_without_reordering(self, store):
        policy = ProbationPolicy(TrustConfig())
        ranking = _ranking("A", "B", "C")
        out = policy.adjust(ranking, store)
        assert [r["agent_id"] for r in out] == ["A", "B", "C"]
        assert all(r["trust_status"] == "trusted" for r in out)


class TestAllowlist:
    def test_allowlisted_agents_are_trusted_no_matter_what(self, store):
        cfg = TrustConfig(
            trusted_agents=["Good"],
            min_trusted_invocations=100,
        )
        policy = ProbationPolicy(cfg)
        assert policy.is_trusted("Good", store) is True
        assert policy.is_trusted("Stranger", store) is False

    def test_non_allowlisted_earns_trust_by_accumulation(self, store):
        cfg = TrustConfig(min_trusted_invocations=5)
        policy = ProbationPolicy(cfg)
        assert policy.is_trusted("A", store) is False
        _seed(store, "A", n=5)
        assert policy.is_trusted("A", store) is True


class TestShareCap:
    def test_demotion_fires_when_share_exceeds_cap(self, store):
        cfg = TrustConfig(
            trusted_agents=["Honest"],
            max_probation_share=0.10,
            window_size=10,
        )
        policy = ProbationPolicy(cfg)
        # Seed 10 recent selections, all to the sybil
        _seed(store, "Sybil", n=10)
        # Bandit's preferred order: Sybil first, Honest second
        ranking = _ranking("Sybil", "Honest")
        out = policy.adjust(ranking, store)
        # After demotion: Honest moves to the top
        assert out[0]["agent_id"] == "Honest"
        assert out[0]["trust_status"] == "trusted"
        assert out[1]["agent_id"] == "Sybil"
        assert out[1]["trust_status"] == "demoted"

    def test_no_demotion_when_share_under_cap(self, store):
        cfg = TrustConfig(
            trusted_agents=["Honest"],
            max_probation_share=0.50,
            window_size=10,
        )
        policy = ProbationPolicy(cfg)
        # 2 sybil selections out of 10 (share=0.2) < cap=0.5
        _seed(store, "Sybil", n=2)
        _seed(store, "Honest", n=8)
        ranking = _ranking("Sybil", "Honest")
        out = policy.adjust(ranking, store)
        # Sybil keeps its bandit-preferred slot
        assert out[0]["agent_id"] == "Sybil"
        assert out[0]["trust_status"] == "probation"


class TestInflatedClaimDetector:
    def test_consistent_high_claims_flagged(self, store):
        cfg = TrustConfig(
            detect_inflated_claims=True,
            inflated_quality_floor=0.95,
            inflated_stdev_ceiling=0.02,
            inflated_min_calls=5,
        )
        policy = ProbationPolicy(cfg)
        # Sybil claims 0.99 every call -> mean 0.99, std 0
        _seed(store, "Sybil", n=10, q=0.99)
        assert policy.is_flagged_for_inflated_claims("Sybil", store) is True

    def test_consistent_below_floor_not_flagged(self, store):
        """A genuinely-consistent 0.9 agent must NOT be flagged."""
        cfg = TrustConfig(
            detect_inflated_claims=True,
            inflated_quality_floor=0.95,
            inflated_stdev_ceiling=0.02,
            inflated_min_calls=5,
        )
        policy = ProbationPolicy(cfg)
        _seed(store, "Honest_Quality_Agent", n=10, q=0.9)
        assert policy.is_flagged_for_inflated_claims("Honest_Quality_Agent", store) is False

    def test_detector_disabled_by_default(self, store):
        cfg = TrustConfig()  # detect_inflated_claims=False
        policy = ProbationPolicy(cfg)
        _seed(store, "Whatever", n=10, q=0.99)
        assert policy.is_flagged_for_inflated_claims("Whatever", store) is False

    def test_below_min_calls_not_flagged(self, store):
        cfg = TrustConfig(
            detect_inflated_claims=True,
            inflated_min_calls=5,
        )
        policy = ProbationPolicy(cfg)
        _seed(store, "TooFew", n=3, q=0.99)  # < min_calls
        assert policy.is_flagged_for_inflated_claims("TooFew", store) is False
