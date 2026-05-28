"""Bandit policies: UCB1, LinUCB, Pareto. Sanity + correctness."""

import json
import pytest

from log_store import LogStore
from config_loader import ScoringConfig
from bandits import UCB1Bandit, LinUCBBandit, ParetoBandit, build_bandit


@pytest.fixture
def store():
    s = LogStore(db_path=":memory:")
    yield s
    s.close()


WEIGHTS = {
    "success_rate": 0.25,
    "quality_score": 0.50,
    "latency_score": 0.15,
    "failure_rate": -0.10,
}


def _seed(store, agent_id, n, q=0.5, lat=100, feats=None):
    for _ in range(n):
        store.record_invocation({
            "agent_id": agent_id, "success": 1,
            "quality_score": q, "latency_ms": lat,
            "failure_reason": None,
            "context_features": feats,
        })


class TestUCB1Bandit:
    def test_higher_observed_quality_wins(self, store):
        _seed(store, "Good", n=20, q=0.9)
        _seed(store, "Bad", n=20, q=0.1)
        bandit = UCB1Bandit(alpha=0.2)
        result = bandit.rank(["Good", "Bad"], WEIGHTS, store)
        assert result[0]["agent_id"] == "Good"

    def test_exploration_diversifies_after_imbalance(self, store):
        # Heavy bias toward A — UCB1 should give B a high exploration bonus
        _seed(store, "A", n=100, q=0.5)
        _seed(store, "B", n=1, q=0.5)
        bandit = UCB1Bandit(alpha=0.5)
        result = bandit.rank(["A", "B"], WEIGHTS, store)
        # B's bonus should pull it ahead
        agent_to_entry = {r["agent_id"]: r for r in result}
        assert agent_to_entry["B"]["exploration_bonus"] > agent_to_entry["A"]["exploration_bonus"]


class TestLinUCBBandit:
    def test_requires_features(self, store):
        bandit = LinUCBBandit(alpha=0.5, feature_dim=4, warm_start_n=0)
        with pytest.raises(ValueError):
            bandit.rank(["A"], WEIGHTS, store, context_features=None)

    def test_warm_start_forces_least_explored_first(self, store):
        bandit = LinUCBBandit(alpha=0.5, feature_dim=4, warm_start_n=3)
        x = [1.0, 0.5, 0.0, 0.0]
        # Seed A 5 times with features; B has 0 calls
        _seed(store, "A", n=5, feats=json.dumps(x))
        result = bandit.rank(["A", "B"], WEIGHTS, store, context_features=x)
        # B should be first (least-explored), warm-up active
        assert result[0]["agent_id"] == "B"
        assert "warmup" in result[0]["bandit"]

    def test_routes_per_context_after_warm_up(self, store):
        """
        Train Fast on short, Quality on long. Verify each wins its context.
        """
        short = [1.0, 0.3, 1.0, 0.0]
        long_ = [1.0, 0.8, 0.0, 1.0]
        # 20 observations per (agent, context) -> well past warm-up
        for _ in range(20):
            store.record_invocation({
                "agent_id": "Fast", "success": 1, "quality_score": 0.9,
                "latency_ms": 100, "failure_reason": None,
                "context_features": json.dumps(short),
            })
            store.record_invocation({
                "agent_id": "Fast", "success": 1, "quality_score": 0.1,
                "latency_ms": 100, "failure_reason": None,
                "context_features": json.dumps(long_),
            })
            store.record_invocation({
                "agent_id": "Quality", "success": 1, "quality_score": 0.4,
                "latency_ms": 800, "failure_reason": None,
                "context_features": json.dumps(short),
            })
            store.record_invocation({
                "agent_id": "Quality", "success": 1, "quality_score": 0.95,
                "latency_ms": 800, "failure_reason": None,
                "context_features": json.dumps(long_),
            })
        bandit = LinUCBBandit(alpha=0.2, feature_dim=4, warm_start_n=0)
        # Short query -> Fast
        result = bandit.rank(["Fast", "Quality"], WEIGHTS, store, context_features=short)
        assert result[0]["agent_id"] == "Fast"
        # Long query -> Quality
        result = bandit.rank(["Fast", "Quality"], WEIGHTS, store, context_features=long_)
        assert result[0]["agent_id"] == "Quality"


class TestParetoBandit:
    def test_preferences_change_winner(self, store):
        # Premium: high quality, slow, expensive
        # Budget: low quality, fast, cheap
        for _ in range(20):
            store.record_invocation({
                "agent_id": "Premium", "success": 1, "quality_score": 0.95,
                "latency_ms": 1800, "failure_reason": None, "cost_cents": 8.0,
            })
            store.record_invocation({
                "agent_id": "Budget", "success": 1, "quality_score": 0.30,
                "latency_ms": 80, "failure_reason": None, "cost_cents": 0.5,
            })

        bandit = ParetoBandit(alpha=0.0)  # alpha=0 to isolate the preference effect

        # Quality-first -> Premium
        r = bandit.rank(["Premium", "Budget"], WEIGHTS, store,
                        preferences={"quality_score": 1.0})
        assert r[0]["agent_id"] == "Premium"

        # Cost-first -> Budget
        r = bandit.rank(["Premium", "Budget"], WEIGHTS, store,
                        preferences={"cost_score": 1.0})
        assert r[0]["agent_id"] == "Budget"

        # Latency-first -> Budget
        r = bandit.rank(["Premium", "Budget"], WEIGHTS, store,
                        preferences={"latency_score": 1.0})
        assert r[0]["agent_id"] == "Budget"

    def test_pareto_frontier_tag(self, store):
        # Two non-dominated agents
        _seed(store, "Premium", n=20, q=0.9, lat=1800)
        _seed(store, "Budget", n=20, q=0.3, lat=80)
        bandit = ParetoBandit(alpha=0.0)
        r = bandit.rank(["Premium", "Budget"], WEIGHTS, store,
                        preferences={"quality_score": 0.5, "latency_score": 0.5})
        # Both should be on the frontier — neither dominates
        on_frontier = {entry["agent_id"]: entry["on_frontier"] for entry in r}
        assert on_frontier["Premium"] is True
        assert on_frontier["Budget"] is True


class TestBuildBandit:
    def test_default_kind_is_ucb1(self):
        cfg = ScoringConfig({}).policy_for("any")
        assert isinstance(build_bandit(cfg), UCB1Bandit)

    def test_linucb_requires_feature_dim(self):
        cfg = ScoringConfig({}).policy_for("any")
        cfg["bandit"] = "linucb"
        with pytest.raises(ValueError):
            build_bandit(cfg, feature_dim=0)

    def test_unknown_bandit_raises(self):
        cfg = ScoringConfig({}).policy_for("any")
        cfg["bandit"] = "magic"
        with pytest.raises(ValueError):
            build_bandit(cfg)
