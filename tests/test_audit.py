import json

import pytest

from conftest import reply
from hitl_agent.audit import AuditLog
from hitl_agent.models import Decision, HumanResponse
from hitl_agent.validation import ResponseRejected


def test_full_lifecycle_is_audited_in_order(make_agent, req):
    agent = make_agent([reply(confidence=0.3, missing=["q?"])])
    res = agent.submit(req())
    with pytest.raises(ResponseRejected):
        agent.resume(res.ticket_id, HumanResponse(reviewer="carol", decision=Decision.APPROVE, justification="x"))
    agent.resume(res.ticket_id, HumanResponse(reviewer="bob", decision=Decision.APPROVE,
                                              justification="Checked with customer.", answers={"q?": "a"}))
    events = [r["event"] for r in agent.audit.read("r1")]
    assert events == ["request_received", "assessment", "paused", "human_response_rejected",
                      "human_response_accepted", "executed"]
    assert agent.audit.verify()[0]


def test_audit_records_who_and_why(make_agent, req):
    agent = make_agent([reply(confidence=0.3, missing=["q?"])])
    res = agent.submit(req())
    agent.resume(res.ticket_id, HumanResponse(reviewer="bob", decision=Decision.APPROVE,
                                              justification="Checked with customer.", answers={"q?": "a"}))
    accepted = next(r for r in agent.audit.read() if r["event"] == "human_response_accepted")
    assert accepted["actor"] == "bob"
    assert accepted["data"]["response"]["justification"] == "Checked with customer."
    paused = next(r for r in agent.audit.read() if r["event"] == "paused")
    assert paused["data"]["signals"][0]["kind"] == "low_confidence"


@pytest.fixture
def log(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    for i in range(3):
        log.append("e", f"r{i}", "agent", {"i": i})
    return log


def rewrite(log, fn):
    lines = [json.loads(line) for line in log.path.read_text().splitlines()]
    fn(lines)
    log.path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in lines))


def test_intact_log_verifies(log):
    assert log.verify() == (True, "3 records verified")


def test_empty_log_verifies(tmp_path):
    assert AuditLog(tmp_path / "x.jsonl").verify()[0]


def test_edited_record_detected(log):
    rewrite(log, lambda ls: ls[1]["data"].update(i=99))
    ok, msg = log.verify()
    assert not ok and "record 1" in msg


def test_deleted_record_detected(log):
    rewrite(log, lambda ls: ls.pop(1))
    assert not log.verify()[0]


def test_reordered_records_detected(log):
    rewrite(log, lambda ls: ls.reverse())
    assert not log.verify()[0]


def test_head_deletion_detected(log):
    rewrite(log, lambda ls: ls.pop(0))
    assert not log.verify()[0]


def test_garbage_line_detected(log):
    with log.path.open("a") as f:
        f.write("not json\n")
    assert not log.verify()[0]
