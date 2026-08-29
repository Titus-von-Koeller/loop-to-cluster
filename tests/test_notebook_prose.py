"""Hold added notebook prose to the width the rest of the repository uses.

`E501` and `W505` are switched off for `notebooks/pytorch-basics/*.py` because the
converted tutorials carry upstream's own long lines and URLs, and rewrapping those would
be editing someone else's material to satisfy a linter. The ignore is per file, though,
so it also stops checking every line added since -- which is how a 139-character
paragraph survived a formatter, a linter and a full run.

This checks the half the ignore was never meant to cover: a markdown line longer than the
limit has to be one upstream wrote.
"""

import re
import subprocess
from pathlib import Path

import pytest

# The commit that converted the PyTorch tutorials into marimo notebooks.
CONVERSION = "891febb"
NOTEBOOKS = sorted((Path(__file__).resolve().parent.parent / "notebooks" / "pytorch-basics").glob("*.py"))
MAX_WIDTH = 95
BLOCK = re.compile(r'mo\.md\(r?"""\n(.*?)\n    """\)', re.DOTALL)


def _upstream_lines(path: Path) -> set[str] | None:
    """Every markdown line as the conversion left it, or None if it is unreachable."""
    relative = path.relative_to(path.parent.parent.parent)
    result = subprocess.run(
        ["git", "show", f"{CONVERSION}:{relative}"],
        capture_output=True,
        text=True,
        cwd=path.parent.parent.parent,
        check=False,
    )
    if result.returncode != 0:
        return None
    return {line for block in BLOCK.findall(result.stdout) for line in block.splitlines()}


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_added_prose_is_wrapped(notebook: Path) -> None:
    upstream = _upstream_lines(notebook)
    if upstream is None:
        pytest.skip(f"the conversion commit {CONVERSION} is not reachable from here")

    too_long = [
        line
        for block in BLOCK.findall(notebook.read_text())
        for line in block.splitlines()
        if len(line) > MAX_WIDTH and line not in upstream
    ]
    assert not too_long, "added markdown lines wider than {}:\n{}".format(
        MAX_WIDTH, "\n".join(f"  ({len(line)}) {line.strip()[:80]}..." for line in too_long)
    )
