"""Local-file ticket store: one JSON checkpoint per ticket, written atomically."""

from __future__ import annotations

import os
from pathlib import Path

from hitl_agent.models import Ticket, TicketStatus


class TicketNotFound(KeyError):
    pass


class TicketStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, ticket_id: str) -> Path:
        if not ticket_id.replace("-", "").replace("_", "").isalnum():
            raise TicketNotFound(ticket_id)  # also blocks path traversal
        return self.root / f"{ticket_id}.json"

    def save(self, ticket: Ticket) -> None:
        path = self._path(ticket.ticket_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(ticket.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def load(self, ticket_id: str) -> Ticket:
        path = self._path(ticket_id)
        if not path.exists():
            raise TicketNotFound(ticket_id)
        return Ticket.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self, status: TicketStatus | None = None) -> list[Ticket]:
        tickets = [Ticket.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(self.root.glob("*.json"))]
        return [t for t in tickets if status is None or t.status is status]
