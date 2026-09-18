---
name: workflow-commits
description: How to split and write git commits in softpaws. Load before staging or committing anything.
---

# Commits

C1. One logical change per commit, and the tests pass at each one. Protected
    files (`design-governance` P4) always get a commit of their own.
C2. Subject: imperative, lowercase, no period, ≤50 characters (72 at most),
    saying what changes.
    Do: `add Detector.effective_area`, `fix NaN below the b_scale floor`.
    Not: `update`, `fixed stuff`, `More examples cleaning`.
C3. Add a body when the why isn't obvious: a blank line after the subject,
    wrapped at 72, covering why and what, not how. End with `Closes #N` where one applies.
C4. No AI attribution in the message (`workflow-ai-disclosure`).
C5. Commit only when the user asks. Stage files by name, never `git add -A`.
    Never amend or rewrite published history.
C6. Before committing, `ruff check .` and `pytest -m "not network"` pass.
