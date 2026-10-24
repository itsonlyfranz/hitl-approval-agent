"""The approval agent as a LangGraph StateGraph.

    intake ─► sample x N (parallel) ─► evaluate ─┬─► auto_execute / auto_deny ─► END
                                                 └─► open_ticket ─► human_review (interrupt)
                                                          ▲                │
                                                          └── invalid ◄─ validate ─┬─► execute ─► END
                                                                                   ├─► reject  ─► END
                                                                                   └─► expired ─► END

Pausing is a LangGraph `interrupt`: the run is checkpointed (SQLite) and resumed
later with `Command(resume=...)`. Invalid human input loops back to the interrupt.
"""

from __future__ import annotations

import operator
import sqlite3
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from hitl_agent.audit import AuditLog
from hitl_agent.executor import Executor
from hitl_agent.llm import LLMClient
from hitl_agent.models import (
    Assessment,
    Decision,
    HumanResponse,
    Outcome,
    RefundRequest,
    RunResult,
    Ticket,
    TicketStatus,
    UncertaintySignal,
    Verdict,
)
from hitl_agent.store import TicketStore
from hitl_agent.uncertainty import Policy, evaluate, run_sample
from hitl_agent.validation import ResponseRejected, Reviewer, effective_amount, validate_response

AGENT = "agent"
RECURSION_LIMIT = 40  # hard cap on graph supersteps per invocation


class TicketNotPending(Exception):
    pass


class AgentState(TypedDict, total=False):
    """Graph state. Values are JSON-friendly dicts so checkpoints stay portable."""

    request: dict
    samples: Annotated[list[dict], operator.add]  # parallel branches append here
    assessment: dict | None
    signals: list[dict]
    ticket_id: str
    questions: list[str]
    response: dict
    errors: list[str]
    outcome: str
    amount: float


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
        checkpoint_path: Path | None = None,
    ):
        self.llm, self.store, self.audit, self.executor = llm, store, audit, executor
        self.reviewers = {r.reviewer_id: r for r in reviewers}
        self.policy = policy or Policy()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        if checkpoint_path is None:
            saver = MemorySaver()
        else:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            saver = SqliteSaver(sqlite3.connect(str(checkpoint_path), check_same_thread=False))
        self.graph = self._build().compile(checkpointer=saver)

    # -- graph nodes ---------------------------------------------------------------
    def _req(self, state: AgentState) -> RefundRequest:
        return RefundRequest.model_validate(state["request"])

    def _intake(self, state: AgentState) -> dict:
        req = self._req(state)
        self.audit.append("request_received", req.request_id, req.requester, req.model_dump(mode="json"))
        return {}

    def _fan_out(self, state: AgentState) -> list[Send]:
        return [Send("sample", {"request": state["request"]}) for _ in range(max(1, self.policy.samples))]

    def _sample(self, state: dict) -> dict:
        result = run_sample(self.llm, RefundRequest.model_validate(state["request"]))
        return {"samples": [result.model_dump(mode="json") | {"_kind": type(result).__name__}]}

    def _evaluate(self, state: AgentState) -> dict:
        req = self._req(state)
        results = [
            (Assessment if s["_kind"] == "Assessment" else UncertaintySignal).model_validate(
                {k: v for k, v in s.items() if k != "_kind"}
            )
            for s in state["samples"]
        ]
        assessment, signals = evaluate(req, results, self.policy)
        dumped = assessment.model_dump(mode="json") if assessment else None
        self.audit.append("assessment", req.request_id, AGENT, {"assessment": dumped})
        return {
            "assessment": dumped,
            "signals": [s.model_dump(mode="json") for s in signals],
            "questions": list(assessment.missing_info) if assessment else [],
        }

    def _route_evaluation(self, state: AgentState) -> str:
        if state["signals"]:
            return "open_ticket"
        return "auto_deny" if state["assessment"]["verdict"] == Verdict.DENY.value else "auto_execute"

    def _auto_execute(self, state: AgentState) -> dict:
        req = self._req(state)
        ref = self.executor.execute(req, req.amount)
        self.audit.append("executed", req.request_id, AGENT, {"ref": ref, "amount": req.amount, "mode": "autonomous"})
        return {"outcome": Outcome.EXECUTED.value, "amount": req.amount}

    def _auto_deny(self, state: AgentState) -> dict:
        req = self._req(state)
        self.audit.append("auto_denied", req.request_id, AGENT, {"rationale": state["assessment"]["rationale"]})
        return {"outcome": Outcome.DENIED.value}

    def _open_ticket(self, state: AgentState, config: RunnableConfig) -> dict:
        req, now = self._req(state), self._clock()
        ticket = Ticket(
            ticket_id=f"t-{uuid.uuid4().hex[:8]}",
            thread_id=config["configurable"]["thread_id"],
            request=req,
            assessment=Assessment.model_validate(state["assessment"]) if state["assessment"] else None,
            signals=[UncertaintySignal.model_validate(s) for s in state["signals"]],
            questions=state["questions"],
            created_at=now,
            expires_at=now + timedelta(hours=self.policy.ticket_ttl_hours),
        )
        self.store.save(ticket)
        self.audit.append(
            "paused", req.request_id, AGENT, {"ticket_id": ticket.ticket_id, "signals": state["signals"]}
        )
        return {"ticket_id": ticket.ticket_id}

    def _human_review(self, state: AgentState) -> dict:
        # Kept side-effect free: LangGraph re-runs this node from the top on resume.
        response = interrupt({"ticket_id": state["ticket_id"], "questions": state["questions"]})
        return {"response": response, "errors": []}

    def _validate(self, state: AgentState) -> dict:
        ticket = self.store.load(state["ticket_id"])
        rid = ticket.request.request_id
        if self._clock() >= ticket.expires_at:
            ticket.status, ticket.outcome = TicketStatus.EXPIRED, Outcome.EXPIRED
            self.store.save(ticket)
            self.audit.append("expired", rid, AGENT, {"ticket_id": ticket.ticket_id})
            return {"outcome": Outcome.EXPIRED.value}
        resp = HumanResponse.model_validate(state["response"])
        try:
            validate_response(ticket, resp, self.reviewers)
        except ResponseRejected as e:
            # Ticket stays pending so the reviewer can correct and retry.
            self.audit.append(
                "human_response_rejected",
                rid,
                resp.reviewer,
                {"ticket_id": ticket.ticket_id, "errors": e.errors, "response": resp.model_dump(mode="json")},
            )
            return {"errors": e.errors}
        self.audit.append(
            "human_response_accepted",
            rid,
            resp.reviewer,
            {"ticket_id": ticket.ticket_id, "response": resp.model_dump(mode="json")},
        )
        return {"errors": []}

    def _route_validation(self, state: AgentState) -> str:
        if state.get("outcome") == Outcome.EXPIRED.value:
            return END
        if state["errors"]:
            return "human_review"
        return "reject" if state["response"]["decision"] == Decision.REJECT.value else "execute"

    def _resolve(self, state: AgentState, outcome: Outcome, amount: float | None = None) -> Ticket:
        ticket = self.store.load(state["ticket_id"])
        ticket.response = HumanResponse.model_validate(state["response"])
        ticket.status, ticket.outcome, ticket.final_amount = TicketStatus.RESOLVED, outcome, amount
        self.store.save(ticket)
        return ticket

    def _reject(self, state: AgentState) -> dict:
        ticket = self._resolve(state, Outcome.REJECTED)
        self.audit.append("rejected", ticket.request.request_id, ticket.response.reviewer,
                          {"ticket_id": ticket.ticket_id})
        return {"outcome": Outcome.REJECTED.value}

    def _execute(self, state: AgentState) -> dict:
        ticket = self.store.load(state["ticket_id"])
        resp = HumanResponse.model_validate(state["response"])
        amount = effective_amount(ticket, resp)
        # Resume with validated context merged into the request the executor sees.
        request = ticket.request.model_copy(update={"context": {**ticket.request.context, **resp.answers}})
        ref = self.executor.execute(request, amount)
        self._resolve(state, Outcome.EXECUTED, amount)
        self.audit.append(
            "executed", request.request_id, AGENT,
            {"ref": ref, "amount": amount, "mode": "human_approved", "approver": resp.reviewer},
        )
        return {"outcome": Outcome.EXECUTED.value, "amount": amount}

    def _build(self) -> StateGraph:
        g = StateGraph(AgentState)
        for name, fn in [
            ("intake", self._intake), ("sample", self._sample), ("evaluate", self._evaluate),
            ("auto_execute", self._auto_execute), ("auto_deny", self._auto_deny),
            ("open_ticket", self._open_ticket), ("human_review", self._human_review),
            ("validate", self._validate), ("reject", self._reject), ("execute", self._execute),
        ]:
            g.add_node(name, fn)
        g.add_edge(START, "intake")
        g.add_conditional_edges("intake", self._fan_out, ["sample"])  # parallel self-consistency samples
        g.add_edge("sample", "evaluate")
        g.add_conditional_edges("evaluate", self._route_evaluation, ["open_ticket", "auto_deny", "auto_execute"])
        g.add_edge("open_ticket", "human_review")
        g.add_edge("human_review", "validate")
        g.add_conditional_edges("validate", self._route_validation, ["human_review", "reject", "execute", END])
        for terminal in ("auto_execute", "auto_deny", "reject", "execute"):
            g.add_edge(terminal, END)
        return g

    # -- public API ----------------------------------------------------------------
    def _config(self, thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

    def _result(self, config: RunnableConfig) -> RunResult:
        snap = self.graph.get_state(config)
        v = snap.values
        req = RefundRequest.model_validate(v["request"])
        paused = bool(snap.next)
        return RunResult(
            request_id=req.request_id,
            paused=paused,
            outcome=None if paused else Outcome(v["outcome"]),
            ticket_id=v.get("ticket_id"),
            signals=[UncertaintySignal.model_validate(s) for s in v.get("signals", [])],
            questions=v.get("questions", []) if paused else [],
            amount=v.get("amount"),
        )

    def submit(self, req: RefundRequest) -> RunResult:
        config = self._config(f"{req.request_id}-{uuid.uuid4().hex[:6]}")
        self.graph.invoke({"request": req.model_dump(mode="json"), "samples": []}, config)
        return self._result(config)

    def resume(self, ticket_id: str, resp: HumanResponse) -> RunResult:
        ticket = self.store.load(ticket_id)
        if ticket.status is not TicketStatus.PENDING:
            self.audit.append("resume_refused", ticket.request.request_id, resp.reviewer,
                              {"ticket_id": ticket_id, "status": ticket.status.value})
            raise TicketNotPending(f"ticket {ticket_id} is {ticket.status.value}")
        config = self._config(ticket.thread_id)
        self.graph.invoke(Command(resume=resp.model_dump(mode="json")), config)
        result = self._result(config)
        if result.paused:  # validation refused the response; the run is waiting again
            raise ResponseRejected(self.graph.get_state(config).values["errors"])
        return result
