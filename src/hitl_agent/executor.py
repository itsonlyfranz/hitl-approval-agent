"""The side-effecting action the agent gates. Local ledger file, idempotent per request."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from hitl_agent.models import RefundRequest


class Executor(Protocol):
    def execute(self, request: RefundRequest, amount: float) -> tuple[dict, bool]:
        """Return the ledger entry for the request and whether this call created it."""
        ...


class LedgerExecutor:
    """Records issued refunds in a JSON file (stand-in for a payments API)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def entries(self) -> dict[str, dict]:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def execute(self, request: RefundRequest, amount: float) -> tuple[dict, bool]:
        entries = self.entries()
        if request.request_id in entries:  # idempotent: never pay twice
            return entries[request.request_id], False
        entry = entries[request.request_id] = {
            "ref": f"refund-{request.request_id}",
            "customer_id": request.customer_id,
            "amount": amount,
            "currency": request.currency,
        }
        self.path.write_text(json.dumps(entries, indent=2))
        return entry, True
