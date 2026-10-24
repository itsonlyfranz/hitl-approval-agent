"""Validation of human input before the agent is allowed to resume.

Checks authority (known reviewer, approval limit), separation of duties,
completeness (every question answered, justification given) and sanity of any
amount the reviewer changes. Nothing reaches the executor unless this passes.
"""

from __future__ import annotations

from dataclasses import dataclass

from hitl_agent.models import Decision, HumanResponse, Ticket

MIN_JUSTIFICATION = 10


@dataclass(frozen=True)
class Reviewer:
    reviewer_id: str
    approval_limit: float


class ResponseRejected(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def effective_amount(ticket: Ticket, resp: HumanResponse) -> float:
    if resp.decision is Decision.MODIFY and resp.approved_amount is not None:
        return resp.approved_amount
    return ticket.request.amount


def validate_response(ticket: Ticket, resp: HumanResponse, reviewers: dict[str, Reviewer]) -> None:
    """Raise ResponseRejected listing every problem found; return None if valid."""
    errors: list[str] = []
    reviewer = reviewers.get(resp.reviewer)
    if reviewer is None:
        errors.append(f"reviewer '{resp.reviewer}' is not an authorised reviewer")
    if resp.reviewer == ticket.request.requester:
        errors.append("reviewer cannot be the requester (separation of duties)")
    if len(resp.justification) < MIN_JUSTIFICATION:
        errors.append(f"justification must be at least {MIN_JUSTIFICATION} characters")

    if resp.decision is not Decision.REJECT:
        for q in ticket.questions:
            if not resp.answers.get(q, "").strip():
                errors.append(f"unanswered question: {q}")
        if resp.decision is Decision.MODIFY:
            amt = resp.approved_amount
            if amt is None or amt <= 0 or amt > ticket.request.amount:
                errors.append("approved_amount must be > 0 and no more than the requested amount")
        elif resp.approved_amount is not None:
            errors.append("approved_amount is only valid with decision 'modify'")
        if reviewer is not None and effective_amount(ticket, resp) > reviewer.approval_limit:
            errors.append(
                f"amount {effective_amount(ticket, resp):.2f} exceeds reviewer approval limit "
                f"{reviewer.approval_limit:.2f}"
            )
    if errors:
        raise ResponseRejected(errors)
