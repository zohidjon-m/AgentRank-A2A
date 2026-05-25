"""
Quality judges for AgentRank.

The problem this solves: in the original design, `quality_score` was
self-reported by each agent. An agent that always claims `quality=1.0`
would dominate the ranking even if its outputs were garbage. That makes
the whole ranker trivially gameable.

A QualityJudge is an external evaluator that scores the *agent's actual
output* against the *request*, ignoring what the agent claims about
itself. The AgentClient calls the judge after a successful response and
replaces the agent's self-reported quality before logging.

Three concrete judges live here:

  MockHeuristicJudge  -- length-ratio + word-overlap heuristics. Works
                         offline; good enough for the demo.
  OracleJudge         -- eval-only. Reads ground-truth quality from a
                         hint argument provided by the simulator.
  AnthropicJudge      -- interface stub. Fill in a real Claude API call
                         when an ANTHROPIC_API_KEY is available.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class JudgeResult:
    """A single judge verdict."""
    score: float            # in [0, 1]
    reason: str             # short human-readable rationale
    judge_name: str         # which judge produced this


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class QualityJudge(ABC):
    """
    Scores the *output* of an agent against the *payload* of the request.

    The hint argument exists only for eval: an OracleJudge can use it to
    return ground-truth quality. Production judges ignore it.
    """

    name: str = "abstract"

    @abstractmethod
    def score(
        self,
        *,
        payload: str,
        output: Any,
        hint: Optional[float] = None,
    ) -> JudgeResult:
        ...


# ---------------------------------------------------------------------------
# Heuristic judge (no network, works offline)
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']+")


def _tokenize(text: str) -> list:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class MockHeuristicJudge(QualityJudge):
    """
    Cheap heuristic for summarization-style tasks.

    Score = 0.6 * grounding + 0.4 * length_appropriateness

      grounding   = fraction of output tokens that appear in the input
                    (catches hallucinations: invented facts have words
                    that don't appear in the source).
      length      = output should be shorter than input but not empty.
                    Sweet spot: 10%-50% of input length.

    Returns 0.0 for empty / None outputs immediately.
    """

    name = "mock_heuristic"

    def score(
        self,
        *,
        payload: str,
        output: Any,
        hint: Optional[float] = None,
    ) -> JudgeResult:
        summary = self._extract_summary(output)
        if not summary or not summary.strip():
            return JudgeResult(0.0, "empty output", self.name)

        in_tokens = _tokenize(payload)
        out_tokens = _tokenize(summary)

        if not out_tokens:
            return JudgeResult(0.0, "no scorable tokens", self.name)
        if not in_tokens:
            # Degenerate: empty input. Give a neutral score.
            return JudgeResult(0.5, "empty payload, neutral score", self.name)

        in_set = set(in_tokens)
        grounded = sum(1 for t in out_tokens if t in in_set)
        grounding = grounded / len(out_tokens)

        ratio = len(out_tokens) / len(in_tokens)
        if ratio >= 1.0:
            length_score, length_note = 0.30, "not a summary (>= input length)"
        elif ratio < 0.05:
            length_score, length_note = 0.40, "too short"
        elif 0.10 <= ratio <= 0.50:
            length_score, length_note = 1.00, "good length"
        else:
            length_score, length_note = 0.70, "off ideal length"

        score = 0.6 * grounding + 0.4 * length_score
        reason = (
            f"grounding={grounding:.2f} ({grounded}/{len(out_tokens)} tokens in input), "
            f"length_ratio={ratio:.2f} ({length_note})"
        )
        return JudgeResult(round(score, 4), reason, self.name)

    @staticmethod
    def _extract_summary(output: Any) -> str:
        if output is None:
            return ""
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            for key in ("summary", "text", "content"):
                if key in output:
                    val = output[key]
                    if val is None:
                        return ""
                    if isinstance(val, str):
                        return val
            return ""
        return str(output)


# ---------------------------------------------------------------------------
# Oracle judge (eval only)
# ---------------------------------------------------------------------------


class OracleJudge(QualityJudge):
    """
    Returns ground-truth quality from the `hint` argument.

    Only meaningful inside the eval harness where the simulator can
    supply the true (vs. claimed) quality of each call. Raises if used
    without a hint, so production code that accidentally instantiates
    this fails loudly.
    """

    name = "oracle"

    def score(
        self,
        *,
        payload: str,
        output: Any,
        hint: Optional[float] = None,
    ) -> JudgeResult:
        if hint is None:
            raise ValueError(
                "OracleJudge requires a hint (ground-truth quality). "
                "It is an eval-only judge."
            )
        return JudgeResult(
            score=float(hint),
            reason="ground-truth from simulator",
            judge_name=self.name,
        )


# ---------------------------------------------------------------------------
# Anthropic judge (skeleton — fill in when API key is available)
# ---------------------------------------------------------------------------


_RUBRIC_SYSTEM_PROMPT = """\
You are evaluating the quality of an AI agent's output against a user request.

Score on a 0-1 scale based on:
- Faithfulness: does the output stay grounded in the input, or invent facts?
- Coverage: does it preserve the important content?
- Conciseness: is it appropriately compact for a summary?
- Coherence: is it well-formed and readable?

Respond with a single JSON object on one line:
{"score": <float 0..1>, "reason": "<one short sentence>"}

Do not include any other text.
"""


class AnthropicJudge(QualityJudge):
    """
    Real LLM-as-judge using Claude.

    Currently a skeleton. The interface and prompt structure are in
    place; the actual API call is gated until ANTHROPIC_API_KEY is
    present in the environment. Drop a key in and the constructor will
    pick it up.

    Includes prompt caching on the rubric system prompt so per-call
    latency and cost stay low even at scale.
    """

    name = "anthropic_claude"

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        max_tokens: int = 256,
    ):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "AnthropicJudge requires ANTHROPIC_API_KEY in environment. "
                "Either set the key or use MockHeuristicJudge for offline runs."
            )
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "AnthropicJudge requires the anthropic SDK. "
                "Install with: pip install anthropic"
            ) from e

        import anthropic
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def score(
        self,
        *,
        payload: str,
        output: Any,
        hint: Optional[float] = None,
    ) -> JudgeResult:
        import json

        summary = MockHeuristicJudge._extract_summary(output)
        user_msg = (
            f"REQUEST PAYLOAD:\n{payload}\n\n"
            f"AGENT OUTPUT:\n{summary}\n"
        )

        # Cache the rubric so we only pay full cost on the first call.
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=[
                {
                    "type": "text",
                    "text": _RUBRIC_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_msg}],
        )

        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        ).strip()

        try:
            parsed = json.loads(text)
            score = float(parsed.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            reason = str(parsed.get("reason", ""))[:300]
        except (ValueError, TypeError, json.JSONDecodeError):
            # Fall back to neutral score on parse failure rather than crashing.
            score = 0.5
            reason = f"parse failure (raw: {text[:80]!r})"

        return JudgeResult(score=score, reason=reason, judge_name=self.name)
