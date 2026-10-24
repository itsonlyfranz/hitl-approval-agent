import pytest

from conftest import reply
from hitl_agent.agent import TicketNotPending
from hitl_agent.models import Decision, HumanResponse, Outcome, TicketStatus
from hitl_agent.validation import ResponseRejected

Q = "Was it returned?"


def approve(reviewer="bob", **kw):
    base = dict(reviewer=reviewer, decision=Decision.APPROVE, justification="Verified with customer.",
                answers={Q: "yes"})
    return HumanResponse(**{**base, **kw})


def paused(make_agent, req, **kw):
    agent = make_agent([reply(confidence=0.4, missing=[Q])], **kw)
    res = agent.submit(req())
    return agent, res


def test_confident_request_executes_without_human(make_agent, req):
    agent = make_agent([reply()])
    res = agent.submit(req())
    assert not res.paused and res.outcome is Outcome.EXECUTED
    assert agent.executor.entries()["r1"]["amount"] == 50.0
    assert [r["event"] for r in agent.audit.read()] == ["request_received", "assessment", "executed"]


def test_confident_deny_does_not_execute(make_agent, req):
    agent = make_agent([reply("deny")])
    assert agent.submit(req()).outcome is Outcome.DENIED
    assert agent.executor.entries() == {}


def test_uncertain_request_pauses_and_does_not_execute(make_agent, req):
    agent, res = paused(make_agent, req)
    assert res.paused and res.questions == [Q]
    assert agent.executor.entries() == {}
    assert agent.store.load(res.ticket_id).status is TicketStatus.PENDING


def test_pause_survives_process_restart(make_agent, req, tmp_path):
    agent, res = paused(make_agent, req)
    fresh = make_agent([reply()])  # new agent object, same files on disk
    out = fresh.resume(res.ticket_id, approve())
    assert out.outcome is Outcome.EXECUTED


def test_resume_executes_with_validated_context(make_agent, req):
    agent, res = paused(make_agent, req)
    out = agent.resume(res.ticket_id, approve())
    assert out.outcome is Outcome.EXECUTED and out.amount == 50.0
    t = agent.store.load(res.ticket_id)
    assert t.status is TicketStatus.RESOLVED and t.response.answers == {Q: "yes"}


def test_resume_does_not_call_llm_again(make_agent, req):
    agent, res = paused(make_agent, req)
    n = len(agent.llm_fake.calls)
    agent.resume(res.ticket_id, approve())
    assert len(agent.llm_fake.calls) == n


@pytest.mark.parametrize("kw,msg", [
    (dict(reviewer="mallory"), "not an authorised"),
    (dict(reviewer="carol"), "separation of duties"),
    (dict(justification="ok"), "justification"),
    (dict(answers={}), "unanswered question"),
    (dict(answers={Q: "  "}), "unanswered question"),
    (dict(approved_amount=10), "only valid with decision 'modify'"),
])
def test_invalid_response_refused_and_ticket_stays_pending(make_agent, req, kw, msg):
    agent, res = paused(make_agent, req)
    with pytest.raises(ResponseRejected) as e:
        agent.resume(res.ticket_id, approve(**kw))
    assert any(msg in err for err in e.value.errors)
    assert agent.store.load(res.ticket_id).status is TicketStatus.PENDING
    assert agent.executor.entries() == {}
    assert agent.audit.read()[-1]["event"] == "human_response_rejected"
    # a corrected response is then accepted
    assert agent.resume(res.ticket_id, approve()).outcome is Outcome.EXECUTED


def test_reviewer_approval_limit_enforced(make_agent, req):
    agent = make_agent([reply(confidence=0.4, missing=[Q])])
    res = agent.submit(req(amount=900))
    with pytest.raises(ResponseRejected, match="approval limit"):
        agent.resume(res.ticket_id, approve("bob"))
    assert agent.resume(res.ticket_id, approve("alice")).outcome is Outcome.EXECUTED


def test_modify_approves_reduced_amount(make_agent, req):
    agent, res = paused(make_agent, req)
    out = agent.resume(res.ticket_id, approve(decision=Decision.MODIFY, approved_amount=20))
    assert out.amount == 20 and agent.executor.entries()["r1"]["amount"] == 20


@pytest.mark.parametrize("amt", [None, 0, -5, 51])
def test_modify_amount_bounds(make_agent, req, amt):
    agent, res = paused(make_agent, req)
    with pytest.raises(ResponseRejected, match="approved_amount"):
        agent.resume(res.ticket_id, approve(decision=Decision.MODIFY, approved_amount=amt))


def test_reject_needs_no_answers_and_never_executes(make_agent, req):
    agent, res = paused(make_agent, req)
    out = agent.resume(res.ticket_id, HumanResponse(reviewer="bob", decision=Decision.REJECT,
                                                   justification="Not eligible per policy."))
    assert out.outcome is Outcome.REJECTED and agent.executor.entries() == {}


def test_ticket_cannot_be_resumed_twice(make_agent, req):
    agent, res = paused(make_agent, req)
    agent.resume(res.ticket_id, approve())
    with pytest.raises(TicketNotPending):
        agent.resume(res.ticket_id, approve())
    assert agent.audit.read()[-1]["event"] == "resume_refused"
    assert len(agent.executor.entries()) == 1


def test_expired_ticket_is_not_executed(make_agent, req, clock):
    agent, res = paused(make_agent, req)
    clock.advance(hours=25)
    out = agent.resume(res.ticket_id, approve())
    assert out.outcome is Outcome.EXPIRED and agent.executor.entries() == {}
    assert agent.store.load(res.ticket_id).status is TicketStatus.EXPIRED


def test_llm_outage_pauses_instead_of_acting(make_agent, req):
    agent = make_agent([RuntimeError("503")])
    res = agent.submit(req())
    assert res.paused and agent.executor.entries() == {}


def test_high_value_request_needs_human_despite_confidence(make_agent, req):
    agent = make_agent([reply(confidence=0.99)])
    res = agent.submit(req(amount=2000))
    assert res.paused
    assert agent.resume(res.ticket_id, approve("alice", answers={})).outcome is Outcome.EXECUTED


def test_duplicate_request_id_reports_the_amount_actually_paid(make_agent, req):
    agent = make_agent([reply()])
    agent.submit(req(amount=40))
    res = agent.submit(req(amount=90))
    assert res.outcome is Outcome.EXECUTED and res.amount == 40
    assert agent.executor.entries()["r1"]["amount"] == 40
    events = [r for r in agent.audit.read() if r["event"] in ("executed", "already_executed")]
    assert [(r["event"], r["data"]["amount"]) for r in events] == [("executed", 40), ("already_executed", 40)]


def test_human_approved_duplicate_records_the_ledger_amount(make_agent, req):
    agent = make_agent([reply()])
    agent.submit(req(amount=40))
    second, res = paused(make_agent, lambda: req(amount=45))
    out = second.resume(res.ticket_id, approve())
    assert out.amount == 40 and second.store.load(res.ticket_id).final_amount == 40
    assert second.audit.read()[-1]["event"] == "already_executed"
