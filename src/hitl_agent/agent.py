"""The approval agent: assess -> detect uncertainty -> pause -> validate human input -> resume."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from hitl_agent.audit import AuditLog
from hitl_agent.executor import Executor
from hitl_agent.llm import LLMClient
from hitl_agent.models import (
    Decision,
    HumanResponse,
    Outcome,
    RefundRequest,
    RunResult,
    Ticket,
    TicketStatus,
    Verdict,
)
from hitl_agent.store import TicketStore
from hitl_agent.uncertainty import Policy, assess
from hitl_agent.validation import Reviewer, ResponseRejected, effective_amount, validate_response

AGENT = "agent"


class TicketNotPending(Exception):
    pass


class ApprovalAgent:
    def __init__(
        self,
        llm: LLMClient,
        store: TicketStore,
        audit: AuditLog,
        executor: Executor,
        reviewers: list[Reviewer],
        policy: Policy | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        self.llm, self.store, self.audit, self.executor = llm, store, audit, executor
        self.reviewers = {r.reviewer_id: r for r in reviewers}
        self.policy = policy or Policy()
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    # -- phase 1: decide or pause -------------------------------------------------
    def submit(self, req: RefundRequest) -> RunResult:
        self.audit.append("request_received", req.request_id, req.requester, req.model_dump(mode="json"))
        assessment, signals = assess(self.llm, req, self.policy)
        self.audit.append(
            "assessment",
            req.request_id,
            AGENT,
            {"assessment": assessment.model_dump(mode="json") if assessment else None},
        )

        if signals:
            now = self._clock()
            ticket = Ticket(
                ticket_id=f"t-{uuid.uuid4().hex[:8]}",
                request=req,
                assessment=assessment,
                signals=signals,
                questions=list(assessment.missing_info) if assessment else [],
                created_at=now,
                expires_at=now + timedelta(hours=self.policy.ticket_ttl_hours),
            )
            self.store.save(ticket)
            self.audit.append(
                "paused",
                req.request_id,
                AGENT,
                {"ticket_id": ticket.ticket_id, "signals": [s.model_dump(mode="json") for s in signals]},
            )
            return RunResult(
                request_id=req.request_id,
                paused=True,
                ticket_id=ticket.ticket_id,
                signals=signals,
                questions=ticket.questions,
            )

        # Confident, complete and within policy: act autonomously (still audited).
        assert assessment is not None
        if assessment.verdict is Verdict.DENY:
            self.audit.append("auto_denied", req.request_id, AGENT, {"rationale": assessment.rationale})
            return RunResult(request_id=req.request_id, paused=False, outcome=Outcome.DENIED)
        ref = self.executor.execute(req, req.amount)
        self.audit.append("executed", req.request_id, AGENT, {"ref": ref, "amount": req.amount, "mode": "autonomous"})
        return RunResult(request_id=req.request_id, paused=False, outcome=Outcome.EXECUTED, amount=req.amount)

    # -- phase 2: resume with validated human input ------------------------------
    def resume(self, ticket_id: str, resp: HumanResponse) -> RunResult:
        ticket = self.store.load(ticket_id)
        rid = ticket.request.request_id
        if ticket.status is not TicketStatus.PENDING:
            self.audit.append("resume_refused", rid, resp.reviewer, {"ticket_id": ticket_id, "status": ticket.status.value})
            raise TicketNotPending(f"ticket {ticket_id} is {ticket.status.value}")

        now = self._clock()
        if now >= ticket.expires_at:
            ticket.status, ticket.outcome = TicketStatus.EXPIRED, Outcome.EXPIRED
            self.store.save(ticket)
            self.audit.append("expired", rid, AGENT, {"ticket_id": ticket_id})
            return RunResult(request_id=rid, paused=False, outcome=Outcome.EXPIRED, ticket_id=ticket_id)

        try:
            validate_response(ticket, resp, self.reviewers)
        except ResponseRejected as e:
            # Ticket stays pending so the reviewer can correct and retry.
            self.audit.append(
                "human_response_rejected",
                rid,
                resp.reviewer,
                {"ticket_id": ticket_id, "errors": e.errors, "response": resp.model_dump(mode="json")},
            )
            raise

        self.audit.append(
            "human_response_accepted", rid, resp.reviewer, {"ticket_id": ticket_id, "response": resp.model_dump(mode="json")}
        )
        ticket.response, ticket.status = resp, TicketStatus.RESOLVED
        if resp.decision is Decision.REJECT:
            ticket.outcome = Outcome.REJECTED
            self.store.save(ticket)
            self.audit.append("rejected", rid, resp.reviewer, {"ticket_id": ticket_id})
            return RunResult(request_id=rid, paused=False, outcome=Outcome.REJECTED, ticket_id=ticket_id)

        # Resume with validated context merged into the request the executor sees.
        amount = effective_amount(ticket, resp)
        request = ticket.request.model_copy(update={"context": {**ticket.request.context, **resp.answers}})
        ref = self.executor.execute(request, amount)
        ticket.outcome, ticket.final_amount = Outcome.EXECUTED, amount
        self.store.save(ticket)
        self.audit.append(
            "executed", rid, AGENT, {"ref": ref, "amount": amount, "mode": "human_approved", "approver": resp.reviewer}
        )
        return RunResult(request_id=rid, paused=False, outcome=Outcome.EXECUTED, ticket_id=ticket_id, amount=amount)
