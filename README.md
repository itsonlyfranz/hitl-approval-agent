# Human-in-the-Loop Approval Agent

An LLM agent that decides on refund requests **only when it is sure**. When it is not, it pauses, asks a human a specific question, validates the answer, and resumes — with every step written to a tamper-evident audit log.

> Project 6 of my AI-engineering portfolio. **Shows: you can build safe, compliant systems.**

**Technologies:** Python 3.11+, Pydantic v2, Anthropic Claude SDK, pytest

## Highlights

- Six distinct uncertainty signals (low confidence, missing info, inconsistent samples, unparseable output, LLM error, high-value policy) gate every action; the agent fails closed.
- Pause/resume through durable, atomically written JSON tickets: no side effect happens while a request waits, and a pending ticket survives a process restart.
- Human input is validated before resume: authorised reviewer, separation of duties, every question answered, bounded amount edits, per-reviewer approval limits.
- SHA-256 hash-chained audit log: the offline demo writes 15 verified records across 3 requests, and tests confirm edits, deletions and reordering are detected.
- The LLM sits behind a one-method interface; the 50-test suite runs offline with a fake client and an injected clock.

## What it demonstrates

| Capability | Where | Proved by |
|---|---|---|
| **Uncertainty detection** | `uncertainty.py` | `tests/test_uncertainty.py` |
| **Pause** - durable checkpoint, nothing executes while waiting; survives restarts and expires | `agent.py`, `store.py` | `test_uncertain_request_pauses...`, `test_pause_survives_process_restart`, `test_expired_ticket_is_not_executed` |
| **Request human input** - ticket carries the signals and the exact questions | `models.Ticket` | `test_uncertain_request_pauses...` |
| **Resume with validated context** | `validation.py` | `test_invalid_response_refused...`, `test_reviewer_approval_limit_enforced`, `test_modify_*` |
| **Full audit trail** | `audit.py` | `tests/test_audit.py` |

## Architecture

```
RefundRequest ─► LLM assess (JSON: verdict, confidence, missing_info)
                     │
              uncertainty.detect ── no signals ──► execute / deny autonomously ─┐
                     │ signals                                                  │
                     ▼                                                          │
              Ticket (JSON file) ── status: PENDING ── human sees questions      │
                     │                                                          │
              HumanResponse ─► validation ── refused ─► ticket stays PENDING ───┤
                     │ valid                                                    │
                     ▼                                                          ▼
         approve / modify / reject ──► Executor (idempotent ledger) ──► audit.jsonl (hash chain)
```

Components:

- **`llm.py`** - `LLMClient` interface with `AnthropicClient` (real) and `FakeLLMClient` (offline tests).
- **`uncertainty.py`** - prompts the model, parses its JSON assessment, and turns results and failures into pause signals.
- **`agent.py`** - `ApprovalAgent` orchestrating submit (decide or pause) and resume (validate, execute).
- **`store.py`** - local-file ticket checkpoints written atomically.
- **`validation.py`** - authority, separation-of-duties, completeness and amount checks on human responses.
- **`executor.py`** - idempotent ledger standing in for the side-effecting payments call.
- **`audit.py`** - append-only hash-chained JSONL log with verification.
- **`cli.py`** - `hitl demo | submit | pending | respond | audit`.

## Quickstart

```bash
uv venv && uv pip install -e ".[dev]"    # or: python -m venv .venv && pip install -e ".[dev]"
pytest                                   # fully offline, no API key
hitl demo --offline                      # scripted model, no API key
cp .env.example .env                     # add ANTHROPIC_API_KEY for the real model
hitl demo                                # same scenario against Claude
```

The demo walks through: an autonomous approval; an ambiguous request that pauses; a refused human response (requester approving their own request, unanswered question, thin justification); a valid response that resumes and executes; a high-value request that policy forces to a human even though the model is confident, including a reviewer exceeding their limit and a partial (`modify`) approval; and finally verification of the audit chain.

Interactive use with real Claude:

```bash
hitl submit req-9 --requester carol --customer c9 --amount 120 --reason "arrived broken"
hitl pending
hitl respond <ticket> --reviewer bob --decision approve --justification "Photo verified" --answer "Question text=answer"
hitl audit            # prints records, exits 2 if the chain is broken
```

Configuration: `ANTHROPIC_API_KEY`, `HITL_MODEL` (default `claude-sonnet-5-5`), `HITL_DATA_DIR` (default `.hitl`).

## Design decisions and trade-offs

- **Fail closed.** Unparseable model output or an API error is itself an uncertainty signal that pauses the run; the agent never guesses. Cost: outages create human work.
- **Policy beats confidence.** Amounts at or above the limit always go to a human; the model's confidence cannot waive that.
- **Self-reported confidence is a weak signal**, so it is combined with missing-info questions, optional multi-sample consistency (`Policy.samples`), and deterministic rules. LLM confidence is not calibrated; the deterministic rules are the real guarantee.
- **Rejected responses don't consume the ticket.** A bad answer is audited and the reviewer can retry; only an accepted answer resolves it. A ticket resolves exactly once, so double-submission cannot pay twice (the executor is also idempotent per request).
- **Resume does not re-ask the model.** The human's validated decision is authoritative and deterministic; answers are merged into the request context handed to the executor and recorded in the audit log.
- **Local files, no services.** Tickets are atomic JSON files; the audit log is JSONL. Single-writer by design — a real deployment would swap in a database with row locks and append-only storage.
- **Tamper-*evident*, not tamper-*proof*.** The hash chain detects modification, deletion and reordering, but someone with write access can rewrite the whole chain or truncate the tail; anchoring the head hash externally would close that gap.
- **Reviewer identity is a registry in code** (`REVIEWERS` in `cli.py`); authentication is out of scope.

## Layout

```
src/hitl_agent/   package
tests/            offline pytest suite (fake LLM, injected clock)
.env.example      configuration template
```
