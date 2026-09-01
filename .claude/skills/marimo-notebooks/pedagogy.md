# The learning experience — hypotheses under test

Same grammar as SKILL.md, one level down: SKILL.md holds the binding intents; this file is
the technique shelf. Every entry is a hypothesis about what makes these notebooks teach,
with its provenance, and is amended or deleted when a reader outcome contradicts it.
Sources: this repo's practice; marimo-team/learn; the explorable-explanations canon
(Victor, Case). Quoted fragments are evidence, not templates.

## The one sentence

**The gap between the reader's committed model and the executed truth is the only thing
that teaches.** Every technique below surfaces that gap, cheapens testing it, or keeps
attention on it. Provenance: four sources converge — Victor's reader is a critic of a
disclosed model, not an audience; Case's reader predicts, then causes the surprise
personally; marimo's edit-and-watch checks a causal model without a rerun ritual; this
repo's N2 (truth is executed) and N5 (observations outrank models) are the same law
applied to the author.

## Commit the model before the truth runs

In this repo the committed model surfaces in conversation — a question asked, a guess
ventured, an "I still don't understand." That channel is the collaboration itself and
needs no instrument; the techniques here are for the surface where corrections land.

- **Predict, then reveal.** Ask for the guess before showing — "what's the best way to
  spread out your recalls?" precedes the answer (Case, remember/); "predict the shape and
  the frozen digit before running; the mismatch is the lesson" (this repo). In prose, as
  a nudge before the exhibit — a reveal without any committed guess is a display, and
  displays get nodded at.
- **The reader causes the phenomenon.** In Polygons the reader drags shapes under a mild
  rule and personally produces segregation; the surprise is the gap between what they
  predicted and what they caused. Strongest form of ownership; costs the most to build.
- **The medium demonstrates its own claim.** "You'll use Spaced Repetition to learn about
  Spaced Repetition" (Case). This repo's analogue: the notebook about memory layout is
  itself inspected with stride captions; keep the recursion when it's free.

## Ground first, name later

- **Open with a question the reader already cares about**, on concrete ground; end open,
  so the reader "goes beyond the teacher" (Case). The upstream tutorials open with
  category prose instead — an addition may cold-open ahead of them; the upstream text
  keeps its order (N1).
- **Intuition → named intuition → formal definition** (marimo/learn, functors course).
  This repo's variant: plant the artifact early, explain it later — stride under every
  picture, "a later section is about nothing else."
- **Withhold the explanation while the reader can still discover it** (Case: learners
  "create their own data points, and form their own model"). Bounded by the guard rails
  below: withholding without structure is a sandbox, not a lesson.

## Make the model inspectable and cheap to poke

- **Reactivity is causality made visible**: "try changing the value above and watch the
  results change" (marimo/learn) — the medium's one move Jupyter cannot make. Use it as
  the lesson, not as decoration.
- **Curiosity loses to laziness at any friction** (Victor): the fact rides the exhibit —
  captions at the point of performance, one keystroke to verify, never "see section 4."
- **Safe to poke**: idempotent cells, expensive cells behind run-buttons, autorun
  everywhere else. Exploration dies where touching something might break it.
- **Self-identifying data**: values that name their own position (arange grids, digit
  tensors), so any transformation can be traced by looking.

## Move on the ladder, both directions

Concrete instance → abstract over the operation → abstract over parameters — 02 already
climbs it (one tensor drawn → the operation dropdown → the shape-pair widget). "The
deepest insights are born not at any one level of abstraction, but in the transitions
between them" (Victor) — and the descending transition is the one this repo still lacks:
when a reader is lost at the abstract rung, hand them back a single concrete instance.

## Contrast and failure carry the load

- **Minimal pairs**: same numbers, different reading (`col.expand` vs a real 5×5;
  `t[..., -1]` vs `t[:, -1]` at rank 4). One variable changes; the concept is the
  difference.
- **Failure as content**: `x.T.view(2, 6)` is in the explorer on purpose; the styled
  error panel explains what `view` refuses and why. An error met in safety is recognized
  in the wild.

## Guard rails — the documented failure modes of this whole genre

- **Guidance over sandbox.** Victor: "The interactivity itself is not really the point…
  the author must guide the reader." A widget without an authored question is decoration.
- **Engagement is not learning.** Case: "Don't try to explain everything with something
  interactive" — use interactivity only where it is the best medium for the claim.
- **A reveal is not a check.** marimo/learn's exercises disclose answers instead of
  validating them — reactivity makes machine-graded checks nearly free, and a check the
  machine cannot grade should usually not pretend to be a check.
- **The argument is the filter** (this repo): every toy serves the file's one claim, or
  it goes. "Interactive pictures" scattered on prose is the diluted form Victor disowned.

## Withdrawn: the calibration loop

A proposed instrument — prediction widgets logging guess-vs-truth into a calibration
corpus — was declined as off-center: it optimized for an imagined anonymous reader, where
the actual system is a two-agent loop in which conversation finds the gaps and the
notebook is where explanations persist. Kept as the genre temptation it is: when a round
of learning-experience thinking starts producing machinery, ask whose reader it serves.
