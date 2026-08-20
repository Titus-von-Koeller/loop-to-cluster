"""Measurement primitives, shared so that numbers stay comparable across scripts.

Consistency is the whole reason this exists. A benchmark of mixed precision against a
baseline means nothing unless both were measured the same way, so the measurement code
is deliberately shared even though the scripts it measures are deliberately not.

Order matters when measuring: warm up first, because AdamW allocates its two moments on
the first `step()` and a window opened at iteration zero undercounts by 8 bytes per
parameter. The `observe` skill carries the rest.
"""
