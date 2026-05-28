"""LogStore math: decay arithmetic, weighted counts, cold-start fallback."""

import pytest

from log_store import LogStore
from config_loader import ScoringConfig


@pytest.fixture
def empty_store():
    """Fresh in-memory log store with no config (so cold start = neutral 0.5)."""
    s = LogStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def config():
    return ScoringConfig.load("config/scoring.json")


@pytest.fixture
def configured_store(config):
    s = LogStore(db_path=":memory:", config=config)
    yield s
    s.close()


def _record(store, agent_id, success=1, q=0.5, lat=100, cost=None):
    store.record_invocation({
        "agent_id": agent_id,
        "success": success,
        "quality_score": q,
        "latency_ms": lat,
        "failure_reason": None,
        "cost_cents": cost,
    })


class TestColdStart:
    def test_no_logs_neutral_default_without_config(self, empty_store):
        m = empty_store.compute_metrics("Anyone")
        assert m["success_rate"] == 0.5
        assert m["quality_score"] == 0.5
        assert m["latency_score"] == 0.5
        assert m["failure_rate"] == 0.5

    def test_no_logs_uses_config_prior(self, configured_store):
        m = configured_store.compute_metrics("SummarizerQuality")
        # SummarizerQuality has a per-agent prior in config/scoring.json
        assert m["quality_score"] == 0.9
        assert m["success_rate"] == 0.98


class TestFlatAveraging:
    def test_quality_is_simple_average_without_decay(self, empty_store):
        _record(empty_store, "A", q=0.4)
        _record(empty_store, "A", q=0.6)
        _record(empty_store, "A", q=0.8)
        m = empty_store.compute_metrics("A")
        assert m["quality_score"] == pytest.approx(0.6)

    def test_success_rate_correct(self, empty_store):
        for _ in range(8):
            _record(empty_store, "A", success=1)
        for _ in range(2):
            _record(empty_store, "A", success=0)
        m = empty_store.compute_metrics("A")
        assert m["success_rate"] == pytest.approx(0.8)
        assert m["failure_rate"] == pytest.approx(0.2)

    def test_latency_score_inverts_and_normalizes(self, empty_store):
        # 1500ms average -> latency_score = 1 - 0.5 = 0.5
        _record(empty_store, "A", lat=1500)
        m = empty_store.compute_metrics("A")
        assert m["latency_score"] == pytest.approx(0.5)


class TestExponentialDecay:
    def test_decay_weights_recent_observations_higher(self, empty_store):
        """
        100 old entries at q=0.9 then 10 recent entries at q=0.1.
        Flat average ~ 0.83; with half_life=10 calls, recent entries
        dominate.
        """
        for _ in range(100):
            _record(empty_store, "A", q=0.9)
        for _ in range(10):
            _record(empty_store, "A", q=0.1)

        flat = empty_store.compute_metrics("A")
        decayed = empty_store.compute_metrics("A", half_life_calls=10)

        assert flat["quality_score"] == pytest.approx((100 * 0.9 + 10 * 0.1) / 110)
        # Decayed should be substantially lower than flat
        assert decayed["quality_score"] < flat["quality_score"] - 0.2

    def test_no_decay_when_half_life_none(self, empty_store):
        for _ in range(10):
            _record(empty_store, "A", q=0.9)
        flat = empty_store.compute_metrics("A")
        none = empty_store.compute_metrics("A", half_life_calls=None)
        assert flat == none

    def test_weighted_counts_match_raw_without_decay(self, empty_store):
        for _ in range(5):
            _record(empty_store, "A")
        for _ in range(3):
            _record(empty_store, "B")
        total, per_agent = empty_store.get_weighted_counts(None)
        assert total == 8.0
        assert per_agent["A"] == 5.0
        assert per_agent["B"] == 3.0

    def test_weighted_counts_shrink_with_decay(self, empty_store):
        for _ in range(100):
            _record(empty_store, "A")
        total_raw, _ = empty_store.get_weighted_counts(None)
        total_dec, _ = empty_store.get_weighted_counts(10)
        assert total_raw == 100.0
        # 100 entries each weight 0.5^((100-i)/10), sum ~ 14-15
        assert 12 < total_dec < 18


class TestCostScore:
    def test_objective_vector_includes_cost(self, empty_store):
        _record(empty_store, "A", cost=2.0)
        _record(empty_store, "A", cost=4.0)
        objs = empty_store.get_objective_vector("A", cost_reference_cents=10.0)
        # avg cost = 3, score = 1 - 0.3 = 0.7
        assert objs["cost_score"] == pytest.approx(0.7)

    def test_no_cost_logged_falls_back(self, empty_store):
        _record(empty_store, "A")  # no cost
        objs = empty_store.get_objective_vector("A")
        # Without cost samples and without config prior -> 0.5 neutral
        assert objs["cost_score"] == 0.5


class TestRecentMethods:
    def test_recent_selections_returns_newest_last(self, empty_store):
        _record(empty_store, "A")
        _record(empty_store, "B")
        _record(empty_store, "C")
        _record(empty_store, "A")
        assert empty_store.recent_selections(3) == ["B", "C", "A"]

    def test_recent_claimed_quality(self, empty_store):
        _record(empty_store, "A", q=0.1)
        _record(empty_store, "A", q=0.5)
        _record(empty_store, "A", q=0.9)
        assert empty_store.recent_claimed_quality("A", 5) == [0.1, 0.5, 0.9]
