import pytest

from hitl_agent.audit import AuditLog
from hitl_agent.cli import main
from hitl_agent.store import TicketStore, TicketNotFound


def test_offline_demo_runs_end_to_end(tmp_path, capsys):
    assert main(["--data-dir", str(tmp_path), "demo", "--offline"]) == 0
    out = capsys.readouterr().out
    assert "PAUSED" in out and "EXECUTED" in out and "refused:" in out and "VALID" in out
    assert AuditLog(tmp_path / "audit.jsonl").verify()[0]


def test_pending_and_respond_commands(tmp_path, capsys):
    main(["--data-dir", str(tmp_path), "demo", "--offline"])
    capsys.readouterr()
    assert main(["--data-dir", str(tmp_path), "pending"]) == 0
    assert capsys.readouterr().out == ""  # demo resolved everything


def test_audit_command_reports_tampering(tmp_path, capsys):
    main(["--data-dir", str(tmp_path), "demo", "--offline"])
    p = tmp_path / "audit.jsonl"
    p.write_text(p.read_text().replace("carol", "mallory", 1))
    assert main(["--data-dir", str(tmp_path), "audit"]) == 2


def test_store_rejects_path_traversal(tmp_path):
    with pytest.raises(TicketNotFound):
        TicketStore(tmp_path).load("../../etc/passwd")


def _pause_one(tmp_path):
    from hitl_agent.cli import build_agent
    from hitl_agent.llm import FakeLLMClient
    from hitl_agent.models import RefundRequest
    agent = build_agent(tmp_path, FakeLLMClient(['{"verdict": "approve", "confidence": 0.4, "missing_info": ["q?"]}']))
    return agent.submit(RefundRequest(request_id="r1", requester="carol", customer_id="c1", amount=10, reason="x"))


@pytest.mark.parametrize("ticket", ["t-missing", "../../etc/passwd"])
def test_respond_to_unknown_ticket_is_refused(tmp_path, capsys, ticket):
    rc = main(["--data-dir", str(tmp_path), "respond", ticket, "--reviewer", "bob",
               "--decision", "reject", "--justification", "Not eligible per policy."])
    assert rc == 1 and "response refused" in capsys.readouterr().out


def test_respond_to_resolved_ticket_is_refused(tmp_path, capsys):
    res = _pause_one(tmp_path)
    args = ["--data-dir", str(tmp_path), "respond", res.ticket_id, "--reviewer", "bob",
            "--decision", "reject", "--justification", "Not eligible per policy."]
    assert main(args) == 0
    capsys.readouterr()
    assert main(args) == 1 and "is resolved" in capsys.readouterr().out


def test_pending_hides_expired_tickets(tmp_path, capsys):
    res = _pause_one(tmp_path)
    store = TicketStore(tmp_path / "tickets")
    assert main(["--data-dir", str(tmp_path), "pending"]) == 0
    assert res.ticket_id in capsys.readouterr().out
    t = store.load(res.ticket_id)
    store.save(t.model_copy(update={"expires_at": t.created_at}))
    assert main(["--data-dir", str(tmp_path), "pending"]) == 0
    assert res.ticket_id not in capsys.readouterr().out


def test_respond_reports_already_executed_for_duplicate_request(tmp_path, capsys):
    from hitl_agent.cli import build_agent
    from hitl_agent.llm import FakeLLMClient
    from hitl_agent.models import RefundRequest
    build_agent(tmp_path, FakeLLMClient(['{"verdict": "approve", "confidence": 0.95}'])).submit(
        RefundRequest(request_id="r1", requester="carol", customer_id="c1", amount=10, reason="x"))
    res = _pause_one(tmp_path)
    capsys.readouterr()
    rc = main(["--data-dir", str(tmp_path), "respond", res.ticket_id, "--reviewer", "bob",
               "--decision", "approve", "--justification", "Verified with customer.", "--answer", "q?=yes"])
    assert rc == 0 and capsys.readouterr().out.strip() == f"{res.ticket_id}: ALREADY_EXECUTED"
