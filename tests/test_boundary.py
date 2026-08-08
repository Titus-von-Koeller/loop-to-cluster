"""A study script may not import from this repo.

The one structural rule, enforced rather than requested. A script in `scripts/` exists
to be read top to bottom and modified by hand; an import the reader has to follow costs
them exactly the thing the script is for. The rule is easy to state and easy to forget
during an edit, so it is checked instead of documented.

Their generated `*_profiled.py` twins are exempt. Nobody reads those for understanding,
and sharing the measurement code is the only way results stay comparable between them.
"""

import ast
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
LOCAL_PACKAGES = {"l2c"}


def study_scripts() -> list[Path]:
    """Hand-written scripts only — the generated profiled twins may import the harness."""
    if not SCRIPTS.is_dir():
        return []
    return sorted(p for p in SCRIPTS.glob("*.py") if not p.stem.endswith("_profiled"))


def imported_roots(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level > 0:
            roots.add("<relative>")
    return roots


@pytest.mark.parametrize("script", study_scripts(), ids=lambda p: p.name)
def test_study_script_is_self_contained(script: Path):
    offenders = imported_roots(script.read_text()) & (LOCAL_PACKAGES | {"<relative>"})
    assert not offenders, (
        f"{script.name} imports {sorted(offenders)}. A study script must stand alone — "
        "inline what it needs, or move the work into its _profiled twin."
    )
