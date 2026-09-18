---
name: workflow-issues
description: Issue forms, labels and triage in softpaws. Load before opening, labelling or triaging an issue, or before editing labels or issue forms.
---

# Issues

I1. Every issue comes from a form in `.github/ISSUE_TEMPLATE/` and starts as
    `needs triage`. Blank issues are off; questions go to Discussions.
I2. Triage: set the type, then remove `needs triage`. Use `needs info` or
    `blocked` while waiting, and close with `duplicate`, `invalid` or `wontfix`.
I3. An issue gets `good first issue` only when it names the files to touch,
    what "done" means, and how to test it.
I4. Labels are defined in `.github/labels.yml`. Never create, edit or delete
    labels on GitHub by hand; the `labels` workflow syncs them.
