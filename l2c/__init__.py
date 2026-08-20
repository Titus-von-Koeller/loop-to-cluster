"""loop-to-cluster: profiling infrastructure for hand-written study scripts.

**Nothing in `scripts/` may import this package.** A study script is written to be read
top to bottom and modified by hand; an import it has to follow costs the reader the
thing the script exists to give them. Only the figure tooling under
`figures/` imports from here. `tests/test_boundary.py` enforces it.

Submodules are not imported here on purpose, so that `import l2c` stays cheap and does
not drag in torch or transformers:

    from l2c.harness import ledger, measure, predict, report, runs

Throughout, docstrings note how the concept at hand is handled in `accelerate`. The
scripts are bare torch — that is the point — but every mechanism they build by hand has
a counterpart somewhere in accelerate, and knowing which file owns it is most of what
onboarding to that codebase consists of.
"""

__version__ = "0.1.0"
