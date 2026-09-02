"""Series invariants for the tutorial notebooks, run by the pre-commit hook.

Two rules the polish passes established and prose alone kept losing:
- no dead pytorch.org nav links (the converted tutorials' relative *.html breadcrumbs);
- exactly one h1 per notebook — the title; sections are h2 and below.

h1 counting only looks inside mo.md string literals, because an indented Python comment
("    # Display image and label.") is not a heading.
"""

import re
import sys

failures = []
for path in sys.argv[1:]:
    with open(path) as f:
        source = f.read()
    if "_tutorial.html" in source or "(intro.html)" in source:
        failures.append(f"{path}: dead pytorch.org nav link")
    h1 = sum(
        1
        for block in re.findall(r'mo\.md\(\s*r?"""(.*?)"""', source, re.S)
        for line in block.splitlines()
        if re.match(r"\s*# [^#]", line)
    )
    if h1 != 1:
        failures.append(f"{path}: {h1} h1 headings in markdown (want exactly one, the title)")

if failures:
    print("\n".join(failures))
    sys.exit(1)
