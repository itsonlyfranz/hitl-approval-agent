"""Append-only, hash-chained audit log (JSON Lines).

Each record embeds the SHA-256 of the previous record, so editing, deleting or
reordering any line breaks verification from that point on.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def _digest(body: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: Path, clock: Callable[[], datetime] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def append(self, event: str, request_id: str, actor: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        records = self.read()
        body = {
            "seq": len(records),
            "ts": self._clock().isoformat(),
            "event": event,
            "request_id": request_id,
            "actor": actor,
            "data": data or {},
            "prev_hash": records[-1]["hash"] if records else GENESIS,
        }
        record = {**body, "hash": _digest(body)}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def read(self, request_id: str | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [r for r in out if request_id is None or r["request_id"] == request_id]

    def verify(self) -> tuple[bool, str]:
        """Return (ok, message). On failure the message names the first bad record."""
        prev = GENESIS
        try:
            records = self.read()
        except json.JSONDecodeError as e:
            return False, f"unparseable record: {e}"
        for i, rec in enumerate(records):
            body = {k: v for k, v in rec.items() if k != "hash"}
            if rec.get("seq") != i:
                return False, f"record {i}: sequence broken (records deleted or reordered)"
            if rec.get("prev_hash") != prev:
                return False, f"record {i}: chain link broken"
            if _digest(body) != rec.get("hash"):
                return False, f"record {i}: content does not match its hash (tampered)"
            prev = rec["hash"]
        return True, f"{len(records)} records verified"
