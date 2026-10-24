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
