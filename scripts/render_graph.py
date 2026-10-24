"""Regenerate the agent flow diagram in README.md from the real compiled LangGraph graph.

    python scripts/render_graph.py           # rewrite the diagram in README.md
    python scripts/render_graph.py --check   # exit 1 if README.md is out of date
    python scripts/render_graph.py --print   # print the Mermaid source only

Runs offline: the graph is built with a fake LLM and temporary storage, and never invoked.
"""
import argparse
import re
import sys
import tempfile
from pathlib import Path

from hitl_agent.agent import ApprovalAgent
from hitl_agent.audit import AuditLog
from hitl_agent.executor import LedgerExecutor
from hitl_agent.llm import FakeLLMClient
from hitl_agent.store import TicketStore

README = Path(__file__).resolve().parent.parent / "README.md"
START, END = "<!-- flow-diagram:start -->", "<!-- flow-diagram:end -->"
BLOCK = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)


def compiled_mermaid() -> str:
    """Mermaid source of the compiled graph, as `ApprovalAgent` builds it."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        agent = ApprovalAgent(
            llm=FakeLLMClient([]), store=TicketStore(root / "tickets"), audit=AuditLog(root / "audit.jsonl"),
            executor=LedgerExecutor(root / "ledger.json"), reviewers=[],
        )
        return agent.graph.get_graph().draw_mermaid().strip()


def flow_block(mermaid: str | None = None) -> str:
    return f"{START}\n```mermaid\n{mermaid or compiled_mermaid()}\n```\n{END}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if README.md differs from the compiled graph")
    ap.add_argument("--print", action="store_true", help="print the Mermaid source and exit")
    args = ap.parse_args()
    if args.print:
        print(compiled_mermaid())
        return 0
    text = README.read_text()
    if not BLOCK.search(text):
        print(f"{README.name}: missing {START} ... {END} markers", file=sys.stderr)
        return 1
    new = BLOCK.sub(lambda _: flow_block(), text, count=1)
    if args.check:
        if new != text:
            print(f"{README.name}: flow diagram is stale; run python scripts/render_graph.py", file=sys.stderr)
            return 1
        return 0
    README.write_text(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
