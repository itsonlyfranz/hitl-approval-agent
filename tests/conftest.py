import json
from datetime import datetime, timedelta, timezone

import pytest

from hitl_agent.agent import ApprovalAgent
from hitl_agent.audit import AuditLog
from hitl_agent.executor import LedgerExecutor
from hitl_agent.llm import FakeLLMClient
from hitl_agent.models import RefundRequest
from hitl_agent.store import TicketStore
from hitl_agent.uncertainty import Policy
from hitl_agent.validation import Reviewer


def reply(verdict="approve", confidence=0.95, missing=(), rationale="ok"):
    return json.dumps({"verdict": verdict, "confidence": confidence, "rationale": rationale,
                       "missing_info": list(missing)})


class Clock:
    def __init__(self):
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def make_agent(tmp_path, clock):
    def _make(replies, policy=None):
        llm = FakeLLMClient(replies)
        agent = ApprovalAgent(
            llm=llm, store=TicketStore(tmp_path / "tickets"), audit=AuditLog(tmp_path / "audit.jsonl", clock),
            executor=LedgerExecutor(tmp_path / "ledger.json"),
            reviewers=[Reviewer("alice", 5000), Reviewer("bob", 500)],
            policy=policy or Policy(), clock=clock,
            checkpoint_path=tmp_path / "checkpoints.sqlite",
        )
        agent.llm_fake = llm
        return agent
    return _make


@pytest.fixture
def req():
    def _req(rid="r1", amount=50.0, requester="carol"):
        return RefundRequest(request_id=rid, requester=requester, customer_id="c1", amount=amount, reason="damaged")
    return _req
