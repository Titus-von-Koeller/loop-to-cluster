# Study scripts

You write these. One per topic, derived from the baseline by copy-and-modify.

**The rule: a script here imports nothing from this repo.** Not `l2c`, not another
script. If two scripts share fifteen identical lines, that is correct — it means you can
change one without touching the other. `tests/test_boundary.py` enforces this.

Keep each one under about 50 lines and readable top to bottom. The test to apply, line by
line, is *do I really need this?*

| | |
| --- | --- |
| `NN_topic.py` | yours, by hand. Trains, prints the loss, nothing else. |
| `NN_topic_profiled.py` | generated. Same training, plus measurement. Don't hand-edit. |

One modification per script. Don't stack features — mixed precision *and* accumulation in
one file makes the result impossible to attribute.

See `../PROFILING.md` for what the profiled twin measures and why it's a separate file.
