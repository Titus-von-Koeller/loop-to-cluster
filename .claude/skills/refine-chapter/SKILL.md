---
name: refine-chapter
description: Revise an existing textbook chapter in a separate later pass — cut, move material to its correct chapter, verify claims, and enforce the book's invariants. Use when asked to review, refine, tighten or fix a chapter or wiki page. Not for authoring a new one (use write-chapter) and not for answering a question mid-exercise (use explain).
---

# Refine a chapter

A second pass, deliberately separated from authoring. Writing a chapter and judging it are
different jobs and the same session does both badly — an author defends structure, an editor
tests it.

You are an editor whose main lever is **removal and relocation, not rewording** — most defects
in this book are material that is correct, well written, and in the wrong chapter.

**Cut, but surface suspected edits first.** Removal is wanted, not merely tolerated. The one
constraint is that Titus edits this wiki too, so before removing a passage, ask whether it
reads as *his* edit rather than as earlier Claude prose — a deliberate trim, a sharpened
sentence, an added aside, anything that does not match the design spec or the house voice.

When it does: name that specific passage, say why you think he wrote it, and get confirmation
before removing it. Everything else you cut on your own judgment. A page that differs from
what you expected has been *edited*, not damaged — "this broke" is the wrong reading and
reverting is the wrong reflex. Phrasing that looks clumsy may be a deliberate cut.

**Retrieval questions are Titus's.** Producing an answer cold is the mechanism that
consolidates, so never write or complete one — leave the heading with its gaps flagged. The
*Interrogate this section* blocks are yours, along with the prose.

## Read before you judge. All of it.

Non-negotiable, in this order:

1. **`docs/BOOK.md`** — the controlling idea, ledger, equivalence spine, structure, chapter
   contract, disclosure rule, numbers policy, out-of-scope list. These are the standard you
   are enforcing.
2. **The table of contents** in Notion.
3. **The chapter under review, in full.**
4. **The chapters immediately before and after it, in full.**
5. **The exercise it maps to**, if it exists.

A chapter's worst defects live in its relationship to its neighbors and are invisible from
inside it. The first edition of chapter 1 read as merely dense; against the table of contents
it was a compressed edition of chapters 2 through 6. That diagnosis was unavailable without
step 2.

## The passes, in order

Run them separately. Combining them means the cheap fixes hide the structural ones.

### 1. Placement

For every paragraph: *which exercise makes the reader feel this?* An answer several chapters
away means the paragraph is early — name its destination chapter. This pass usually removes
more than all the others combined.

Then: does the chapter still satisfy the contract after the cuts, or has removing borrowed
material revealed that it never had its own prediction, equivalence claim or hole?

### 2. Correctness

Every mechanism claim gets checked, not skimmed. Two classes have actually occurred here:

- **Imprecise mechanism claims.** "The scales start at 1.0, so normalization begins as an
  identity operation" — the learned affine is the identity; the normalization is fully active.
  Wrong in a way that breaks the arithmetic two sections later.
- **Summary rows contradicting their own prose.** A fold asserting "the initial loss equals
  ln(V)" beside a callout explaining that it never does and that asserting equality fails.
  **Check every fold, table and callout against the text next to it** — these drift because
  they are written last and read first.

Framework behavior claims — `accelerate`, FSDP, DeepSpeed, NCCL — are verified against
installed source with file and line, or marked unverified in the text.

**Run the existing verifiers rather than re-deriving by hand.**
`docs/_wiki_build/verify_params.py` checks the analytic parameter count against
`sum(p.numel())`; `verify_facts.py` ground-truths the optimizer, initialization and norm
claims and diffs the preset field by field against the released config. That second one
exists because `initializer_range` reached the wiki as 0.02 while the released config says
0.041666… — a field that changes no shape is invisible to a parameter count. Never source a
wiki number from the lab's own code.

### 3. Numbers

Apply the deletion test to each figure: remove the digits. If the sentence still teaches, the
number supported a general claim — keep it. If it collapses to something already known, the
number was performing rigor. Specimen measurements go to `bench/results/` or become an
instruction to the reader.

### 4. Altitude and continuity

Does the chapter hold one level throughout, or does a section drop below the book's register
and fail to return? Does it open with the ledger, row marked? Does the closing hole point at a
real later chapter?

### 5. Prose

Last, and least — but against a specific standard, not taste. `docs/BOOK.md`'s `## Prose`
section is binding: sentence construction, banned hedges, unquantified comparatives, emphasis
rationing, the failure register, and specimens to match. Also roadmap openings, terminal
deflation, arguing with its own centerpiece, unexpanded acronyms, United States spelling,
conversational residue.

## Rules

**Cut without ceremony.** "This is chapter 4's material" is a complete argument. You do not
need a replacement to justify a deletion. The single exception is a passage you suspect Titus
wrote — flag that one and confirm before removing it.

**Say where it goes.** A cut with no destination is a loss. Every relocation names its target
chapter, so the material survives the edit.

**Do not rewrite what is merely different from your preference.** The standard is `docs/BOOK.md`,
not taste. If a passage satisfies the contract and the disclosure rule, leave it.

**Report as a diff, not a rewrite.** Default output is an ordered list of changes — cut, move,
fix, verify — each with its reason and the invariant it serves, most structural first. Edit
the page in place only when asked, and never overwrite without reading the current version.

**Rank by structure, not by count.** One misplaced section outweighs twenty prose nits. Lead
with placement findings.

**Decide.** Where a defect admits two fixes, recommend one.

## Before reporting

- Is the top finding structural?
- Does every cut name its destination?
- Did I check every fold and table against its adjacent prose?
- Did I verify the framework claims, or flag them?
- Am I enforcing `docs/BOOK.md`, or my preferences?
