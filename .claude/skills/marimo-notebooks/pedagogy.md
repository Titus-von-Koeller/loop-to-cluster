# pedagogy.md — framing practices, and techniques that make concepts graspable

Two shelves, kept separate on purpose. The first is about **intent**: how to frame and
pace content so it teaches. The second is **technique**: visualization and interactivity
moves with a track record of making concepts easier to explore and grasp, plus the
design standards the rendered surface is held to. Entries are hypotheses with sources
(this repo's notebooks; marimo-team/learn; the explorable-explanations canon — Victor,
Case; the information-design canon); a reader outcome that contradicts one is grounds
to propose its amendment or deletion to Titus.

The organizing observation, common to every source: **the gap between the reader's
committed model and the executed truth is what teaches.** Victor's reader critiques a
disclosed model rather than believing prose; Case's reader guesses, then causes the
surprise personally; a reactive notebook checks a causal model the instant a value is
edited; N2 and N5 apply the same law to the author. Techniques earn their place by
surfacing that gap, cheapening its testing, or holding attention on it.

## Shelf one — framing the content

- **Open with a question the reader already cares about, on concrete ground; end open**,
  so the reader leaves with somewhere to go (Case). Found tutorials tend to open with
  category prose; a rewritten opening may lead with the concrete hook.
- **Intuition → named intuition → formal definition** (marimo/learn). Variant proven
  here: plant an artifact early and explain it later — a caption carried under every
  picture before its concept gets its section.
- **Withhold the explanation while the reader can still discover it** (Case) — bounded
  by the guard rails below; withholding without structure is a sandbox, not a lesson.
- **Define at first contact** or declare the forward reference; a term met before its
  definition, undeclared, is a gap.
- **The fact rides the exhibit** (Victor: curiosity loses to laziness at any friction):
  captions at the point of performance, one keystroke to verify, never "see section 4."
- **Contrast isolates a concept better than definition**: minimal pairs — same numbers,
  different reading — with exactly one variable changed.
- **Failure is content**: the instructive error belongs in the exhibit, met safely and
  explained, because an error met in safety is recognized in the wild.
- **Prediction primes the reveal**: a one-line prose nudge to guess before the exhibit —
  a reveal without any committed guess is a display, and displays get nodded at.

### Guard rails — this genre's documented failure modes

- **Guidance over sandbox.** Victor: "The interactivity itself is not really the point…
  the author must guide the reader." A widget without an authored question is decoration.
- **Engagement is not learning.** Case: interactivity only where it is the best medium
  for the claim; anything else entertains without teaching.
- **A reveal is not a check.** Disclosing an answer validates nothing; reactivity makes
  machine-graded checks nearly free, so a check the machine cannot grade should not
  pretend to be one.

## Shelf two — visualization, interactivity, and the rendered surface

- **Reactivity is causality made visible**: "change the value above and watch the
  results change" is the one move static notebooks cannot make; use it as the lesson,
  not as decoration.
- **Draw, don't print**: a tensor rendered as its own colored numbers, an image shown as
  the image it is; thousands of values read at a glance where dozens read as text.
- **Self-identifying data**: values that name their own position (arange grids,
  digit-address tensors) make every transformation traceable by looking.
- **Safe to poke**: idempotent cells, expensive cells behind run-buttons, autorun for
  everything else — exploration dies where touching something might break it.
- **Climb the ladder both directions** (Victor): concrete instance → abstract over the
  operation → abstract over parameters, and a way back down to a single instance when
  the abstract rung loses the reader.
- **The reader causes the phenomenon** where the build cost is warranted (Case's
  Polygons): the strongest ownership of a result is having produced it.
- **The medium demonstrates its own claim** when the recursion is free (a notebook about
  memory layout that prints its own strides).

### Design standards for the surface

- **Modern information design, applied without exception**: maximize data-ink, no chart
  junk; direct labeling over legends where cheap; consistent type scale and spacing;
  generous whitespace; every picture captioned with what the object *is*. The bar is a
  polished, aesthetic document, not a lab bench.
- **Color policy** (constants and rationale in `notebooks/pytorch-basics/_viz.py`):
  categorical hues from Okabe-Ito, sequential from cividis, diverging blue-orange —
  palettes designed to stay distinguishable under red-green color-vision deficiency.
  The primary reader's deficiency is mild: hue pairs must stay *reliably* separable,
  which established palettes guarantee; lightness carries magnitude so pictures also
  survive grayscale. Hand-tuned hex values in a notebook are a bug.
- **Theme honesty**: exhibits are checked in the theme actually used for reading — which is
  plural where the OS flips it (autoDetectColorScheme: Horizon Bright by day, Horizon Bold in
  dark), so exhibits are checked in both; a
  graphic that assumes a white page carries its own background.
