# Design decisions log

Newest first. One entry per adoption, change or exception: date, IDs, what, why.

## Open

- **Units (N4).** Either drop the `_gev`/`_km`/`_cm2` suffixes from public
  names and state one unit system once, or keep the suffixes.
- **Name of the organizing object (O2).** `Detector` is taken by the paper's
  fit container in `response/site_models.py`, so that class must be renamed or
  made private first.

## 2026-09-18: Protected .claude/agents/ (P4)

The `repo-auditor` subagent's file sets what the audit does and which tools
it gets, so it needs approval like the skills and hooks. Added to P4, to
CLAUDE.md, and to `permissions.ask`. Approved by the user.

## 2026-09-18: Added the Repo Audit (RA1-RA4)

Every 14 days, a `SessionStart` hook (`repo_audit_due.py`, only on new
sessions) tells Claude the audit is due. Claude explains it in 1-3 sentences
and asks; a "no" snoozes it for 3 days. When approved, the audit runs
`sp-repo-review` from its own venv in `.audit/`, then the read-only
`repo-auditor` subagent checks fixed sources per area plus one open search.
The report goes to the gitignored `.audit/`, and proposals follow
design-governance. The trigger lives in `.claude/settings.local.json`, so it
fires for the maintainer only. Approved by the user.

## 2026-09-18: Added workflow-issues (I1-I4)

The two Markdown issue templates are replaced by four issue forms (bug,
wrong or surprising result, feature, documentation) with required fields,
and blank issues are off. The 15 labels live in `.github/labels.yml`,
grouped as type, contributors, status, resolution and PR, and
`crazy-max/ghaction-github-labeler@v6` syncs them on changes to main. There
are no area labels, because the API refactor will move module boundaries.
Approved by the user.

## 2026-09-18: Required PR template (PR2, PR7)

The template has three required sections (What and why, How to check it,
Effect on results) and a checklist of judgment calls only. Anything CI
already checks is left off. "Effect on results" exists because a PR can
shift a number the paper relies on. The checklist also asks that the branch
be up to date with `main`; CONTRIBUTING.md explains how, for newcomers to git.
AI use is a checkbox (the author understands and tested every line), not a
NumPy-style named disclosure. `pr-template.yml` fails a PR with an empty
required section; GitHub itself never enforces templates. Approved by the user.

## 2026-09-18: Exception to C1 in 89e3feb

The docstring commit also carries the deletion of
`numpy-docstring-format.md`, which had been staged earlier. The user chose to
leave it. Lesson: check `git diff --cached` before committing.

## 2026-09-18: AI2 covers CONTRIBUTING.md, no Assisted-by trailer, C2 confirmed

- CONTRIBUTING.md gets an "AI-assisted contributions" paragraph: the human is
  the author and is accountable, and no AI is listed as co-author.
- The per-commit `Assisted-by:` trailer (Fedora, Linux kernel, LLVM,
  OpenTelemetry) is not adopted. For a single-author project the README note is enough.
- The 32 existing commit lines that name Claude stay. Earlier commits are never changed (C5).
- C2 is confirmed: lowercase subjects from now on. Older commits in another style stay as they are.

Approved by the user.

## 2026-09-18: Added workflow-commits, workflow-pull-requests, workflow-ai-disclosure

C1-C6 and PR1-PR6 follow cbea.ms/git-commit and Google's eng-practices
(small CLs, CL descriptions), and keep the repo's lowercase imperative
subjects. AI1-AI3: Claude is never an author, and the README carries the
disclosure. `attribution` in `.claude/settings.json` is set empty. Requested
by the user.

## 2026-09-18: Moved P and L into design-governance (L3, L5)

`design-principles` had grown to 56 lines, over the L5 limit. It is now the
index only, and the change protocol and learning rules live in their own
skill. The rule text is unchanged. Approved by the user.

## 2026-09-18: Changes need explicit approval (P2-P5, L1, L4, L6)

Protected files change only after the user explicitly approves that change,
as a change of their own, and never through Bash. The tree still learns, but
it proposes and never applies unasked. `permissions.ask` in
`.claude/settings.json` makes the harness prompt before any edit to them.
Why: the user asked on 2026-09-18 that nothing here change silently or as a
side effect.

## 2026-09-18: Added L1-L7, the tree learns

The skill tree is a living document: it picks up design choices as they are
made, prunes itself, and stays curt. CLAUDE.md records this, and a PostToolUse
hook (`.claude/hooks/design_reminder.py`) prompts it after edits to `src/softpaws/`,
`docs/`, `examples/` or `README.md`. Why: the user wants the tree to learn,
with every change visible.

## 2026-09-18: Enforced A1, D1, D5 and N5 in code

- Ruff `D` with the numpy convention, `src/` only, enforces D5. `D401`
  (imperative mood) is ignored because D1 allows a noun-phrase summary. The
  52 docstrings with backslashes got an `r` prefix, which also fixes LaTeX
  such as `\nu` that rendered as a newline.
- `tests/test_design_rules.py` checks A1 (top level has at most 20 names), D1
  (a summary exists and is at most 75 characters) and N5 (no jargon in public
  names). D1 and N5 are ratchets with 9 and 3 known offenders.

## 2026-09-18: Adopted the initial rules

A1-A5, O1-O6, N1-N5, D1-D5, R1-R5, G1-G5, P1-P4, from the ease-of-use review.
Why: 195 public names, 141 of them in `softpaws.response`, with paper
machinery mixed into the user API, and options encoded in function names.
A student could not pass the module count into the point-source limit.

Supersedes `.claude/skills/numpy-docstring-format.md`. Its format rules are
kept as D5. New: the length limits D2-D4.
