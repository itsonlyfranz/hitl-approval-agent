"""CLI: `hitl demo`, `hitl submit`, `hitl pending`, `hitl respond`, `hitl audit`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from hitl_agent.agent import ApprovalAgent, TicketNotPending
from hitl_agent.audit import AuditLog
from hitl_agent.executor import LedgerExecutor
from hitl_agent.llm import AnthropicClient, FakeLLMClient, LLMClient, load_env
from hitl_agent.models import Decision, HumanResponse, RefundRequest, TicketStatus
from hitl_agent.store import TicketNotFound, TicketStore
from hitl_agent.uncertainty import Policy
from hitl_agent.validation import Reviewer, ResponseRejected

REVIEWERS = [Reviewer("alice", 5000.0), Reviewer("bob", 500.0)]


def build_agent(data_dir: Path, llm: LLMClient) -> ApprovalAgent:
    return ApprovalAgent(
        llm=llm,
        store=TicketStore(data_dir / "tickets"),
        audit=AuditLog(data_dir / "audit.jsonl"),
        executor=LedgerExecutor(data_dir / "ledger.json"),
        reviewers=REVIEWERS,
        policy=Policy(),
        checkpoint_path=data_dir / "checkpoints.sqlite",
    )


def _scripted_llm() -> FakeLLMClient:
    """Offline stand-in that behaves like a cautious model, keyed off the request text."""

    def reply(_system: str, prompt: str) -> str:
        if "late delivery" in prompt:
            v = {"verdict": "approve", "confidence": 0.95, "rationale": "Documented late delivery.", "missing_info": []}
        elif "no order" in prompt:
            v = {"verdict": "approve", "confidence": 0.55, "rationale": "Unclear claim.",
                 "missing_info": ["Was the item returned?"]}
        else:
            v = {"verdict": "approve", "confidence": 0.92, "rationale": "Looks legitimate.", "missing_info": []}
        return json.dumps(v)

    return FakeLLMClient(reply)


def _print_result(label: str, res) -> None:
    state = f"PAUSED ticket={res.ticket_id}" if res.paused else f"{res.outcome.value.upper()}"
    print(f"{label}: {state}")
    for s in res.signals if res.paused else []:
        print(f"    signal {s.kind.value}: {s.detail}")
    for q in res.questions:
        print(f"    question: {q}")


def cmd_demo(args) -> int:
    data_dir = Path(args.data_dir)
    llm = _scripted_llm() if args.offline else AnthropicClient()
    agent = build_agent(data_dir, llm)
    print("== 1. Confident, low-value request: handled autonomously")
    _print_result("req-1", agent.submit(RefundRequest(
        request_id="req-1", requester="carol", customer_id="c1", amount=40, reason="late delivery")))

    print("\n== 2. Ambiguous request: agent detects uncertainty and pauses")
    paused = agent.submit(RefundRequest(
        request_id="req-2", requester="carol", customer_id="c2", amount=120, reason="no order on file"))
    _print_result("req-2", paused)

    print("\n== 3. Invalid human input is refused (requester approving own request, no answers)")
    try:
        agent.resume(paused.ticket_id, HumanResponse(
            reviewer="carol", decision=Decision.APPROVE, justification="ok"))
    except ResponseRejected as e:
        for err in e.errors:
            print(f"    refused: {err}")

    print("\n== 4. Valid human response: agent resumes with validated context and executes")
    _print_result("req-2", agent.resume(paused.ticket_id, HumanResponse(
        reviewer="bob", decision=Decision.APPROVE, justification="Customer confirmed return by phone.",
        answers={"Was the item returned?": "Yes, tracking shows delivery to warehouse."})))

    print("\n== 5. High-value request: policy always requires a human, even if the model is confident")
    big = agent.submit(RefundRequest(
        request_id="req-3", requester="carol", customer_id="c3", amount=2500, reason="late delivery"))
    _print_result("req-3", big)
    try:
        agent.resume(big.ticket_id, HumanResponse(
            reviewer="bob", decision=Decision.APPROVE, justification="Within my authority, surely."))
    except ResponseRejected as e:
        print(f"    refused: {e.errors[0]}")
    _print_result("req-3", agent.resume(big.ticket_id, HumanResponse(
        reviewer="alice", decision=Decision.MODIFY, approved_amount=1500,
        justification="Partial refund per customer-service policy.")))

    ok, msg = agent.audit.verify()
    print(f"\n== 6. Audit trail ({data_dir / 'audit.jsonl'}): {'VALID' if ok else 'INVALID'} - {msg}")
    for r in agent.audit.read():
        print(f"    #{r['seq']:02d} {r['event']:<24} {r['request_id']:<6} by {r['actor']}")
    return 0


def _agent_from_env(args) -> ApprovalAgent:
    load_env()
    return build_agent(Path(args.data_dir), AnthropicClient())


def cmd_submit(args) -> int:
    req = RefundRequest(request_id=args.request_id, requester=args.requester, customer_id=args.customer,
                        amount=args.amount, reason=args.reason)
    _print_result(req.request_id, _agent_from_env(args).submit(req))
    return 0


def cmd_pending(args) -> int:
    now = datetime.now(timezone.utc)
    for t in TicketStore(Path(args.data_dir) / "tickets").list(TicketStatus.PENDING):
        if t.expires_at <= now:
            continue
        print(f"{t.ticket_id}  {t.request.request_id}  {t.request.amount:.2f} {t.request.currency}  "
              f"expires {t.expires_at:%Y-%m-%d %H:%M}Z")
        for q in t.questions:
            print(f"    question: {q}")
    return 0


def cmd_respond(args) -> int:
    answers = dict(a.split("=", 1) for a in args.answer)
    resp = HumanResponse(reviewer=args.reviewer, decision=Decision(args.decision),
                         justification=args.justification, answers=answers, approved_amount=args.amount)
    agent = build_agent(Path(args.data_dir), FakeLLMClient(["{}"]))  # resume never calls the LLM
    try:
        _print_result(args.ticket_id, agent.resume(args.ticket_id, resp))
    except ResponseRejected as e:
        print("response refused:", *e.errors, sep="\n  ")
        return 1
    except TicketNotFound as e:
        print(f"response refused: unknown ticket {e.args[0]}")
        return 1
    except TicketNotPending as e:
        print(f"response refused: {e}")
        return 1
    return 0


def cmd_audit(args) -> int:
    log = AuditLog(Path(args.data_dir) / "audit.jsonl")
    ok, msg = log.verify()
    for r in log.read(args.request_id):
        print(json.dumps(r))
    print(("VALID: " if ok else "INVALID: ") + msg, file=sys.stderr)
    return 0 if ok else 2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hitl", description=__doc__)
    p.add_argument("--data-dir", default=os.environ.get("HITL_DATA_DIR", ".hitl"))
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the end-to-end scenario")
    d.add_argument("--offline", action="store_true", help="scripted model, no API key needed")
    d.set_defaults(fn=cmd_demo)

    s = sub.add_parser("submit", help="submit a refund request (uses Claude)")
    s.add_argument("request_id"); s.add_argument("--requester", required=True)
    s.add_argument("--customer", required=True); s.add_argument("--amount", type=float, required=True)
    s.add_argument("--reason", required=True)
    s.set_defaults(fn=cmd_submit)

    sub.add_parser("pending", help="list tickets awaiting a human").set_defaults(fn=cmd_pending)

    r = sub.add_parser("respond", help="answer a pending ticket")
    r.add_argument("ticket_id"); r.add_argument("--reviewer", required=True)
    r.add_argument("--decision", choices=[d.value for d in Decision], required=True)
    r.add_argument("--justification", required=True)
    r.add_argument("--answer", action="append", default=[], help="'question=answer' (repeatable)")
    r.add_argument("--amount", type=float)
    r.set_defaults(fn=cmd_respond)

    a = sub.add_parser("audit", help="print and verify the audit log")
    a.add_argument("--request-id")
    a.set_defaults(fn=cmd_audit)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
