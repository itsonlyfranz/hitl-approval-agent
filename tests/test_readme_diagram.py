import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("render_graph", ROOT / "scripts" / "render_graph.py")
render_graph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_graph)

NODE = re.compile(r"^\s*(\w+)\(", re.MULTILINE)
EDGE = re.compile(r"^\s*(\w+) (-->|-\.->) (\w+);$", re.MULTILINE)


def _readme_flow_block() -> str:
    match = render_graph.BLOCK.search((ROOT / "README.md").read_text())
    assert match, "README.md has no flow-diagram markers"
    return match.group(0)


def test_readme_flow_diagram_matches_compiled_graph():
    block = _readme_flow_block()
    graph = render_graph.compiled_graph()
    hint = "README flow diagram drifted from the compiled graph; run `python scripts/render_graph.py`"
    assert set(NODE.findall(block)) == set(graph.nodes), hint
    readme_edges = {(src, dst, arrow == "-.->") for src, arrow, dst in EDGE.findall(block)}
    graph_edges = {(e.source, e.target, e.conditional) for e in graph.edges}
    assert readme_edges == graph_edges, hint
