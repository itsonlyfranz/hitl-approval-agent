import pytest

from conftest import reply
from hitl_agent.llm import FakeLLMClient
from hitl_agent.models import SignalKind
from hitl_agent.uncertainty import Policy, evaluate, parse_assessment, run_sample


def kinds(signals):
    return {s.kind for s in signals}


def run(replies, req, policy):
    llm = FakeLLMClient(replies)
    return evaluate(req, [run_sample(llm, req) for _ in replies], policy)


def test_confident_complete_request_has_no_signals(req):
    a, signals = run([reply()], req(), Policy())
    assert a is not None and signals == []


def test_low_confidence_detected(req):
    _, signals = run([reply(confidence=0.5)], req(), Policy())
    assert kinds(signals) == {SignalKind.LOW_CONFIDENCE}


def test_confidence_exactly_at_threshold_is_not_uncertain(req):
    _, signals = run([reply(confidence=0.8)], req(), Policy(confidence_threshold=0.8))
    assert signals == []


def test_missing_info_detected_even_when_confident(req):
    _, signals = run([reply(missing=["Was it returned?"])], req(), Policy())
    assert kinds(signals) == {SignalKind.MISSING_INFO}


def test_high_value_policy_forces_pause_even_when_confident(req):
    _, signals = run([reply(confidence=0.99)], req(amount=1000), Policy(high_value_amount=1000))
    assert kinds(signals) == {SignalKind.HIGH_VALUE}


def test_inconsistent_samples_detected(req):
    _, signals = run([reply("approve"), reply("deny")], req(), Policy(samples=2))
    assert kinds(signals) == {SignalKind.INCONSISTENT}


def test_later_samples_doubt_is_not_masked_by_a_confident_first_sample(req):
    replies = [reply(confidence=0.95), reply(confidence=0.4, missing=["Was it returned?"]),
               reply(confidence=0.6, missing=["Was it returned?", "Receipt?"])]
    a, signals = run(replies, req(), Policy(samples=3))
    assert kinds(signals) == {SignalKind.LOW_CONFIDENCE, SignalKind.MISSING_INFO}
    assert a.confidence == 0.4 and a.missing_info == ["Was it returned?", "Receipt?"]


@pytest.mark.parametrize("bad", ["I think approve", "{not json}", '{"verdict": "maybe", "confidence": 0.9}',
                                 '{"verdict": "approve", "confidence": 7}'])
def test_unparseable_output_fails_closed(req, bad):
    a, signals = run([bad], req(), Policy())
    assert a is None and SignalKind.UNPARSEABLE in kinds(signals)


def test_llm_error_fails_closed_and_still_applies_policy(req):
    a, signals = run([RuntimeError("down")], req(amount=5000), Policy())
    assert a is None and kinds(signals) == {SignalKind.LLM_ERROR, SignalKind.HIGH_VALUE}


def test_json_wrapped_in_prose_is_parsed():
    assert parse_assessment("Sure!\n" + reply() + "\nthanks").confidence == 0.95
