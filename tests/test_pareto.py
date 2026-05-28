"""Pure-function tests for the Pareto utilities."""

import pytest

from pareto import (
    dominates,
    pareto_frontier_indices,
    weighted_pick,
    normalize_preferences,
)


class TestDominates:
    def test_strictly_better_on_all_dims_dominates(self):
        assert dominates([0.9, 0.9, 0.9], [0.5, 0.5, 0.5]) is True

    def test_equal_does_not_dominate(self):
        assert dominates([0.5, 0.5], [0.5, 0.5]) is False

    def test_better_on_one_equal_on_rest_dominates(self):
        assert dominates([0.6, 0.5], [0.5, 0.5]) is True

    def test_better_on_one_worse_on_another_does_not_dominate(self):
        assert dominates([0.9, 0.1], [0.5, 0.5]) is False

    def test_worse_on_all_does_not_dominate(self):
        assert dominates([0.1, 0.1], [0.5, 0.5]) is False


class TestParetoFrontier:
    def test_singleton_is_its_own_frontier(self):
        assert pareto_frontier_indices([[0.5, 0.5]]) == [0]

    def test_dominated_point_excluded(self):
        # b dominates a → frontier is [b]
        idxs = pareto_frontier_indices([[0.1, 0.1], [0.9, 0.9]])
        assert idxs == [1]

    def test_pareto_optimal_set_when_no_one_dominates(self):
        # Three corners — each best on one axis
        pts = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        assert sorted(pareto_frontier_indices(pts)) == [0, 1, 2]

    def test_mixed(self):
        # 0: dominated by 2.  1: best on dim1 alone.  2: best mix.
        pts = [[0.3, 0.3], [0.0, 0.9], [0.6, 0.7]]
        assert sorted(pareto_frontier_indices(pts)) == [1, 2]


class TestWeightedPick:
    def test_weighted_pick_returns_argmax_under_positive_prefs(self):
        pts = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        # All-quality preference -> first point wins
        idx, score = weighted_pick(pts, [1.0, 0.0])
        assert idx == 0
        assert score == pytest.approx(1.0)
        # All-latency preference -> second point wins
        idx, _ = weighted_pick(pts, [0.0, 1.0])
        assert idx == 1
        # Balanced -> middle point wins (1.0 vs 0.5+0.5=1.0 tie; but
        # tied points: argmax returns first match, which is index 0)
        idx, _ = weighted_pick(pts, [0.5, 0.5])
        assert idx in (0, 1, 2)  # ties allowed

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            weighted_pick([[0.5, 0.5]], [1.0])

    def test_empty_points_raises(self):
        with pytest.raises(ValueError):
            weighted_pick([], [1.0])


class TestNormalizePreferences:
    def test_normalizes_to_sum_one(self):
        prefs = normalize_preferences(
            {"a": 2.0, "b": 1.0, "c": 1.0},
            ("a", "b", "c"),
        )
        assert sum(prefs) == pytest.approx(1.0)
        assert prefs[0] == pytest.approx(0.5)
        assert prefs[1] == pytest.approx(0.25)

    def test_missing_keys_become_zero(self):
        prefs = normalize_preferences(
            {"a": 1.0},
            ("a", "b"),
        )
        assert prefs == pytest.approx([1.0, 0.0])

    def test_empty_or_non_positive_returns_uniform(self):
        prefs = normalize_preferences({}, ("a", "b", "c", "d"))
        assert prefs == pytest.approx([0.25, 0.25, 0.25, 0.25])
        # All-zero is also "empty" for our purposes
        prefs = normalize_preferences({"a": 0.0, "b": 0.0}, ("a", "b"))
        assert prefs == pytest.approx([0.5, 0.5])
