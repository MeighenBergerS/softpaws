---
name: repo-audit
description: The Repo Audit, a two-weekly comparison of softpaws with current best practices, run only after the user approves. Load when a session start says it is due, or when the user asks for /repo-audit.
---

# Repo Audit

Untracked, in `.audit/`: reports `YYYY-MM-DD.md`, `state.json`, and a tool venv `.venv`.

RA1. Ask first, in 1-3 short sentences, e.g.: "The Repo Audit compares softpaws
     with current best practices for code, docs, skills and GitHub. It reads the
     web and the repo, changes nothing, and writes an untracked report with
     proposals. Run it now?"
RA2. Declined: set `snooze_until` in `state.json` to today + 3 days. Nothing else.
RA3. Approved:
     1. Create the venv if missing (`python3 -m venv .audit/.venv`, then
        `.audit/.venv/bin/pip install -q 'sp-repo-review[cli]'`), and run
        `.audit/.venv/bin/sp-repo-review . --format json`.
     2. Hand that output and the latest earlier report to the `repo-auditor` subagent.
     3. Write its report to `.audit/<today>.md`, set `last_run` to today, and drop `snooze_until`.
     4. In chat, at most ~10 lines: the counts, the top findings, and each
        proposal as **Design change proposed**.
RA4. The audit changes nothing outside `.audit/`. Adopting a proposal follows `design-governance`.
