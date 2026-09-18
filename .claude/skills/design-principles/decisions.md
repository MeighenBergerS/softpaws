# Design decisions log

Newest first. One entry per adoption, change or exception: date, IDs, what, why.

## Open

Nothing open.

## 2026-09-18: Standards ship as JSON and can be rebuilt (A7)

The standards are the two instrument numbers per site from script 77 at
σ = 5% (the run scripts 83-89 read): median and 68% range, with b = 1,
λ = BGR18 and ε₀ fixed. TRIDENT comes from the 2022 bands in the same fit.
They ship as `data/standards.json`, which `softpaws.standards` reads, so a
plain pip install works. `make_standards(fit_result, out)` rebuilds the file
from a fit result, and `standards.use(path)` or `SOFTPAWS_STANDARDS` switches
to a user's own. The fit itself stays a separate step, because running it
would make the paper's fit code public. Approved by the user.

## 2026-09-18: C5 checks what's staged as its own step; exception to C1 in 2f164e0

`2f164e0` also carries the staged move of `utils/constants.py`, so that
commit alone is a tree whose imports fail; `08937e6` repairs it straight
after. The user chose to leave it. It was the second slip of this kind:
printing what's staged in the same command as the commit did not catch it.
C5 now requires checking what's staged in a separate step before
committing. Approved by the user.

## 2026-09-18: Tightening a check rides with the fix (P6)

P4 (protected files get their own commit) and C1 (the tests pass at every
commit) clashed as soon as code fixed something a check tracks: the ratchet
then fails until its baseline is lowered in the protected test file.
Tightening a check (lowering a baseline, removing a fixed offender, dropping
a dead exemption) now goes in the fix's commit without separate approval.
Loosening one is an exception under P2. Approved by the user.

## 2026-09-18: A6 refined, standards.py (A7), utils/ replaced (A4)

- A6 now covers literal numbers and arithmetic on literals, including
  default grids such as `np.linspace(3.0, 8.0, 26)`. Three kinds stay put:
  values computed by package code (`B_SCALE_FLOOR`, record-derived depths),
  numbers that define a published record or table (`sites.py`, `optics.py`),
  and paper tuning (the private `_paper/`). The check was refined to match,
  and its baseline rose from 74 to 87 because it now sees the grids.
  `_KERNEL_SCALING_TOKEN` is runtime state, not a constant, so it is renamed
  in lower case.
- A7: `standards.py` holds the current best-fit values from the paper's fits,
  each naming its script, so users never run a fit first. `constants.py`
  holds fixed numbers, `standards.py` fitted ones, and `_paper/` the tuning.
- A4: `softpaws/utils/`, which held only `constants.py`, is replaced by a
  top-level `softpaws/constants.py`, and the Architecture section of
  CLAUDE.md is updated to match.

Approved by the user.

## 2026-09-18: Units, one constants file, Detector as the central object (N4, A4, A6, O1-O3, D5, G6)

- N4: public names and arguments lose their unit suffixes. The code computes
  in GeV, cm, s and rad, the field's flux and cross-section conventions, so
  the physics needs no conversion factors. Users pass values with Geant4-style
  multipliers from `softpaws.constants` (`1.0 * km`). This was chosen over
  fixed per-quantity units, where a wrong-unit number fails silently, and
  over astropy Quantity, which is heavy and trips up newcomers.
- A6: every named number (units, physical constants, calibrated parameters,
  defaults) lives in `src/softpaws/constants.py`. Records and tabulated curves
  stay where they are.
- D5: docstrings give the dimension (`[length]`), not a unit. G6:
  `docs/units.md` explains the scheme, and the README, the quickstart and
  `llms.txt` link to it.
- O2/O3: `Detector` is the central object, and analyses are its methods; the
  separate organizing object is dropped. The paper's fit container
  `response.site_models.Detector` becomes `SiteFit` and leaves the public API (A2).
- Checks (L6): ratchets in `test_design_rules.py`, with baselines of 259
  suffixed names and arguments (N4) and 74 named numbers outside the
  constants file (A6).

Approved by the user.

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
