"""Measurement primitives, shared so that numbers stay comparable across scripts.

Consistency is the whole reason this exists. A benchmark of mixed precision against a
baseline means nothing unless both were measured the same way, so the measurement code
is deliberately shared even though the scripts it measures are deliberately not.

`PROFILING.md` is the contract: what to measure, in what order, and the schema to emit.
"""
