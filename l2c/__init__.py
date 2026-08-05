"""loop-to-cluster: a lab notebook for learning distributed training.

Shared code only. The lessons live in `steps/`, deliberately duplicated — see the
README's one rule.

Submodules are not imported here on purpose, so that `import l2c` stays cheap and
does not drag in torch or transformers. Import what you need:

    from l2c.harness import ledger, measure, predict, report
    from l2c.common import data, model

Throughout, docstrings note how the concept at hand is handled in `accelerate`.
The steps themselves are bare torch — that is the point — but every mechanism they
build by hand has a counterpart somewhere in accelerate, and knowing which file
owns it is most of what onboarding to that codebase consists of.
"""

__version__ = "0.1.0"
