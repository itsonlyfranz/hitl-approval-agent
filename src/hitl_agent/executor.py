"""The side-effecting action the agent gates. Local ledger file, idempotent per request."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from hitl_agent.models import RefundRequest


class Executor(Protocol):
    def execute(self, request: RefundRequest, amount: float) -> str: ...


class LedgerExecutor:
    """Records issued refunds in a JSON file (stand-in for a payments API)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def entries(self) -> dict[str, dict]:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def execute(self, request: RefundRequest, amount: float) -> str:
        entries = self.entries()
        ref = f"refund-{request.request_id}"
        if request.request_id not in entries:  # idempotent: never pay twice
            entries[request.request_id] = {
                "ref": ref,
                "customer_id": request.customer_id,
                "amount": amount,
                "currency": request.currency,
            }
            self.path.write_text(json.dumps(entries, indent=2))
        return ref
