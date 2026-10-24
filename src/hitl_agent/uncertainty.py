"""Uncertainty detection: turns an LLM assessment (or its failure) into pause signals.

Signals are deliberately conservative: anything the agent cannot vouch for
pauses the run. Failing closed is the safe default for a compliance system.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from hitl_agent.llm import LLMClient
from hitl_agent.models import (
    Assessment,
    RefundRequest,
    SignalKind,
    UncertaintySignal,
)

SYSTEM_PROMPT = (
    "You assess customer refund requests for a finance team. Reply with ONLY a JSON object: "
    '{"verdict": "approve"|"deny", "confidence": <0..1>, "rationale": "<short>", '
    '"missing_info": ["<question you need answered before deciding>", ...]}. '
    "Be honest about uncertainty: lower your confidence and list missing_info "
    "when facts are absent or ambiguous."
)


@dataclass(frozen=True)
class Policy:
    confidence_threshold: float = 0.8
    high_value_amount: float = 1000.0  # at/above this a human must always approve
    samples: int = 1  # >1 enables self-consistency checking
    ticket_ttl_hours: int = 24


def build_prompt(req: RefundRequest) -> str:
    return "Refund request:\n" + req.model_dump_json(indent=2)


def parse_assessment(text: str) -> Assessment:
    """Parse model output into an Assessment; raises ValueError if it cannot."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in model output")
    try:
        return Assessment.model_validate(json.loads(match.group(0)))
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"invalid assessment: {e}") from e


def run_sample(llm: LLMClient, req: RefundRequest) -> Assessment | UncertaintySignal:
    """One model call. Failures come back as signals so the caller fails closed."""
    try:
        raw = llm.complete(SYSTEM_PROMPT, build_prompt(req))
    except Exception as e:  # provider outage, timeout, auth...
        return UncertaintySignal(kind=SignalKind.LLM_ERROR, detail=f"{type(e).__name__}: {e}")
    try:
        return parse_assessment(raw)
    except ValueError as e:
        return UncertaintySignal(kind=SignalKind.UNPARSEABLE, detail=str(e))


def evaluate(
    req: RefundRequest, results: list[Assessment | UncertaintySignal], policy: Policy
) -> tuple[Assessment | None, list[UncertaintySignal]]:
    """Combine sample results into a primary assessment plus any uncertainty signals."""
    samples = [r for r in results if isinstance(r, Assessment)]
    failures = [r for r in results if isinstance(r, UncertaintySignal)]
    primary = samples[0] if samples else None
    if primary is not None and not failures:
        return primary, detect(req, primary, policy, samples)
    # Model failed: still apply the policy rules that don't depend on it.
    return primary, failures + _policy_signals(req, policy)


def assess(
    llm: LLMClient, req: RefundRequest, policy: Policy
) -> tuple[Assessment | None, list[UncertaintySignal]]:
    """Sequential convenience wrapper (the graph runs the samples as parallel branches)."""
    results: list[Assessment | UncertaintySignal] = []
    for _ in range(max(1, policy.samples)):
        results.append(run_sample(llm, req))
        if isinstance(results[-1], UncertaintySignal):
            break
    return evaluate(req, results, policy)


def _policy_signals(req: RefundRequest, policy: Policy) -> list[UncertaintySignal]:
    if req.amount >= policy.high_value_amount:
        return [
            UncertaintySignal(
                kind=SignalKind.HIGH_VALUE,
                detail=f"amount {req.amount:.2f} >= policy limit {policy.high_value_amount:.2f}",
            )
        ]
    return []


def detect(
    req: RefundRequest, a: Assessment, policy: Policy, samples: list[Assessment] | None = None
) -> list[UncertaintySignal]:
    signals: list[UncertaintySignal] = []
    if a.confidence < policy.confidence_threshold:
        signals.append(
            UncertaintySignal(
                kind=SignalKind.LOW_CONFIDENCE,
                detail=f"confidence {a.confidence:.2f} < threshold {policy.confidence_threshold:.2f}",
            )
        )
    if a.missing_info:
        signals.append(UncertaintySignal(kind=SignalKind.MISSING_INFO, detail="; ".join(a.missing_info)))
    if samples and len({s.verdict for s in samples}) > 1:
        votes = ", ".join(s.verdict.value for s in samples)
        signals.append(UncertaintySignal(kind=SignalKind.INCONSISTENT, detail=f"samples disagree: {votes}"))
    signals.extend(_policy_signals(req, policy))
    return signals

