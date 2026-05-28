"""ScoringConfig: default-filling, composer methods, agent priors."""

import pytest

from config_loader import ScoringConfig


def test_load_real_config():
    cfg = ScoringConfig.load("config/scoring.json")
    policy = cfg.policy_for("nlp/summarize")
    # Every documented key must be present
    for k in ("weights", "exploration", "drift", "bandit",
              "feature_extractor", "bandit_params", "trust"):
        assert k in policy
    # nlp/summarize weights override the default with quality=0.5
    assert policy["weights"]["quality_score"] == 0.5


def test_minimal_config_fills_defaults():
    raw = {}  # Completely empty
    cfg = ScoringConfig(raw)
    policy = cfg.policy_for("any/thing")
    # Default weights still present
    assert policy["weights"]["success_rate"] == 0.4
    assert policy["exploration"]["alpha"] == 0.5
    assert policy["drift"]["half_life_calls"] is None
    assert policy["bandit"] == "ucb1"
    assert policy["trust"]["max_probation_share"] == 1.0


def test_partial_policy_merges_with_defaults():
    raw = {
        "scoring": {
            "domains": {
                "my/domain": {
                    "bandit": "linucb",
                    # weights / exploration / drift / trust omitted on purpose
                }
            }
        }
    }
    policy = ScoringConfig(raw).policy_for("my/domain")
    assert policy["bandit"] == "linucb"  # explicit
    assert policy["weights"]["success_rate"] == 0.4  # defaulted
    assert policy["trust"]["min_trusted_invocations"] == 0  # defaulted


def test_agent_default_falls_back_to_fallback_then_neutral():
    cfg = ScoringConfig.load("config/scoring.json")
    # Known agent has explicit prior
    known = cfg.agent_default("SummarizerQuality")
    assert known["quality_score"] == 0.9
    # Unknown agent gets the _fallback (which the real config defines)
    unknown = cfg.agent_default("CompletelyMadeUp")
    assert unknown["quality_score"] == 0.5  # _fallback default in our config


def test_with_priors_does_not_mutate_original():
    cfg = ScoringConfig.load("config/scoring.json")
    cfg2 = cfg.with_priors({"NewAgent": {
        "success_rate": 1.0, "quality_score": 1.0,
        "latency_score": 1.0, "failure_rate": 0.0,
    }})
    assert cfg2.agent_default("NewAgent")["success_rate"] == 1.0
    # Original is unchanged
    original_unknown = cfg.agent_default("NewAgent")
    assert original_unknown != cfg2.agent_default("NewAgent")


def test_with_bandit_composer():
    cfg = ScoringConfig.load("config/scoring.json")
    cfg2 = cfg.with_bandit(
        "nlp/summarize",
        kind="linucb",
        feature_extractor="length_bucket",
        bandit_params={"alpha": 0.5},
    )
    p = cfg2.policy_for("nlp/summarize")
    assert p["bandit"] == "linucb"
    assert p["feature_extractor"] == "length_bucket"
    assert p["bandit_params"]["alpha"] == 0.5


def test_with_trust_composer():
    cfg = ScoringConfig.load("config/scoring.json")
    cfg2 = cfg.with_trust(
        "nlp/summarize",
        trusted_agents=["AgentA"],
        min_trusted_invocations=5,
    )
    p = cfg2.policy_for("nlp/summarize")
    assert p["trust"]["trusted_agents"] == ["AgentA"]
    assert p["trust"]["min_trusted_invocations"] == 5


def test_with_drift_half_life_composer():
    cfg = ScoringConfig.load("config/scoring.json")
    cfg2 = cfg.with_drift_half_life("nlp/summarize", 40)
    p = cfg2.policy_for("nlp/summarize")
    assert p["drift"]["half_life_calls"] == 40
