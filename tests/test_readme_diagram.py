import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("render_graph", ROOT / "scripts" / "render_graph.py")
render_graph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_graph)


def _readme_flow_block() -> str:
    match = render_graph.BLOCK.search((ROOT / "README.md").read_text())
    assert match, "README.md has no flow-diagram markers"
    return match.group(0)


def test_readme_flow_diagram_matches_compiled_graph():
    assert _readme_flow_block() == render_graph.flow_block(), (
        "README flow diagram drifted from the compiled graph; run `python scripts/render_graph.py`"
    )
