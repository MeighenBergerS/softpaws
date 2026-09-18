---
name: workflow-ai-disclosure
description: How softpaws discloses that it was built with AI assistance. Load before writing a commit, a PR, the README, or anything carrying authorship.
---

# AI disclosure

AI1. Claude is never an author, co-author or signer. No `Co-Authored-By`,
     `Signed-off-by` or "Generated with Claude Code" lines in commits or PRs.
     The human committer is the author and is accountable.
AI2. Disclosure happens once, for the whole project, in the README section
     "Development with AI assistance". CONTRIBUTING.md sets the same rule for
     contributors. No per-commit `Assisted-by:` line.
AI3. `attribution.commit` and `attribution.pr` in `.claude/settings.json` are
     empty, so the harness adds nothing.
