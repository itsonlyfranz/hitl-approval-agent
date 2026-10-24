"""Tests for the LangGraph wiring: interrupts, checkpoints, parallel branches, limits."""

import pytest

from conftest import reply
from hitl_agent.agent import RECURSION_LIMIT
from hitl_agent.models import Decision, HumanResponse
from hitl_agent.uncertainty import Policy
from hitl_agent.validation import ResponseRejected


def thread_of(agent, res):
    return agent.store.load(res.ticket_id).thread_id


def test_pause_is_a_langgraph_interrupt_with_checkpointed_state(make_agent, req):
    agent = make_agent([reply(confidence=0.4, missing=["q?"])])
    res = agent.submit(req())
    snap = agent.graph.get_state({"configurable": {"thread_id": thread_of(agent, res)}})
    assert snap.next == ("human_review",)
    assert snap.tasks[0].interrupts[0].value["questions"] == ["q?"]
    assert snap.values["request"]["request_id"] == "r1"


def test_graph_nodes_are_explicit(make_agent):
    agent = make_agent([reply()])
    names = set(agent.graph.get_graph().nodes)
    assert {"intake", "sample", "evaluate", "open_ticket", "human_review", "validate", "execute", "reject"} <= names


def test_samples_run_as_parallel_branches(make_agent, req):
    agent = make_agent([reply(), reply(), reply()], policy=Policy(samples=3))
    res = agent.submit(req())
    assert not res.paused and len(agent.llm_fake.calls) == 3
    state = agent.graph.get_state({"configurable": {"thread_id": next(iter(_threads(agent)))}})
    assert len(state.values["samples"]) == 3


def _threads(agent):
    return {c.config["configurable"]["thread_id"] for c in agent.graph.checkpointer.list(None)}


def test_disagreeing_parallel_samples_pause(make_agent, req):
    agent = make_agent([reply("approve"), reply("deny")], policy=Policy(samples=2))
    res = agent.submit(req())
    assert res.paused and res.signals[0].kind.value == "inconsistent_samples"


def test_every_parallel_samples_doubt_reaches_the_human(make_agent, req):
    agent = make_agent([reply(), reply(confidence=0.4, missing=["Returned?"]), reply(missing=["Receipt?"])],
                       policy=Policy(samples=3))
    res = agent.submit(req())
    assert res.paused and {s.kind.value for s in res.signals} == {"low_confidence", "missing_info"}
    assert sorted(res.questions) == ["Receipt?", "Returned?"]


def test_refused_response_loops_back_to_the_interrupt(make_agent, req):
    agent = make_agent([reply(confidence=0.4, missing=["q?"])])
    res = agent.submit(req())
    cfg = {"configurable": {"thread_id": thread_of(agent, res)}}
    with pytest.raises(ResponseRejected):
        agent.resume(res.ticket_id, HumanResponse(reviewer="bob", decision=Decision.APPROVE,
                                                  justification="short"))
    assert agent.graph.get_state(cfg).next == ("human_review",)


def test_two_runs_with_same_request_id_do_not_share_state(make_agent, req):
    agent = make_agent([reply(confidence=0.4, missing=["q?"])])
    a, b = agent.submit(req()), agent.submit(req())
    assert thread_of(agent, a) != thread_of(agent, b)


def test_recursion_limit_is_configured(make_agent):
    assert RECURSION_LIMIT <= 50
    assert make_agent([reply()])._config("t")["recursion_limit"] == RECURSION_LIMIT
