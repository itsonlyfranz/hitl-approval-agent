"""Pydantic data models shared across the agent."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class RefundRequest(BaseModel):
    """A request the agent must decide on (the demo domain is customer refunds)."""

    request_id: str
    requester: str = Field(description="Identity of whoever submitted the request")
    customer_id: str
    order_id: str | None = None
    amount: float = Field(gt=0)
    currency: str = "USD"
    reason: str
    context: dict[str, Any] = Field(default_factory=dict)


class Verdict(str, Enum):
    APPROVE = "approve"
    DENY = "deny"


class Assessment(BaseModel):
    """The LLM's structured opinion about a request."""

    verdict: Verdict
    confidence: float = Field(ge=0, le=1)
    rationale: str = ""
    missing_info: list[str] = Field(default_factory=list)


class SignalKind(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    MISSING_INFO = "missing_info"
    HIGH_VALUE = "high_value"
    INCONSISTENT = "inconsistent_samples"
    UNPARSEABLE = "unparseable_output"
    LLM_ERROR = "llm_error"


class UncertaintySignal(BaseModel):
    kind: SignalKind
    detail: str


class TicketStatus(str, Enum):
    PENDING = "pending"
    RESOLVED = "resolved"
    EXPIRED = "expired"


class Decision(str, Enum):
    """What a human reviewer can decide."""

    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"  # approve with a corrected amount


class HumanResponse(BaseModel):
    reviewer: str
    decision: Decision
    justification: str
    # Answers to the agent's `missing_info` questions, keyed by question.
    answers: dict[str, str] = Field(default_factory=dict)
    # Only for MODIFY: the amount the reviewer authorises.
    approved_amount: float | None = None

    @field_validator("justification")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class Outcome(str, Enum):
    EXECUTED = "executed"
    REJECTED = "rejected"
    DENIED = "denied"
    EXPIRED = "expired"


class Ticket(BaseModel):
    """Durable checkpoint: everything needed to resume after a pause."""

    ticket_id: str
    thread_id: str = ""  # LangGraph thread holding the paused run
    request: RefundRequest
    assessment: Assessment | None
    signals: list[UncertaintySignal]
    questions: list[str]
    status: TicketStatus = TicketStatus.PENDING
    created_at: datetime
    expires_at: datetime
    response: HumanResponse | None = None
    outcome: Outcome | None = None
    final_amount: float | None = None


class RunResult(BaseModel):
    """What `submit` / `resume` hand back to the caller."""

    request_id: str
    paused: bool
    outcome: Outcome | None = None
    ticket_id: str | None = None
    signals: list[UncertaintySignal] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    amount: float | None = None
