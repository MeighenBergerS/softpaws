---
name: design-principles
description: Index of the softpaws design rules. Load before adding, renaming, moving or exporting code, or writing docs or examples, to find the skill that applies.
---

# softpaws design principles

Goal: a new user, or their LLM, sets up their own analysis from a few obvious
names, without reading long docs. Robust for ~90% of users, not for every edge.

| Skill | Covers | IDs |
| --- | --- | --- |
| `design-governance` | how rules change, approval, learning | P, L |
| `design-architecture` | layers, modules, one source of truth | A |
| `design-objects` | classes, the organizing object, `Detector` | O |
| `design-naming` | names, arguments over variants, units | N |
| `design-docstrings` | docstring length and format | D |
| `design-robustness` | validation, errors, defaults | R |
| `design-docs` | docs, examples, `llms.txt` | G |
| `workflow-commits` | splitting and writing commits | C |
| `workflow-pull-requests` | scoping and describing PRs | PR |
| `workflow-ai-disclosure` | no AI authorship, one project-level note | AI |
| `workflow-issues` | issue forms, labels, triage | I |

Rules describe the target state. Existing code that breaks one is migration
debt: fix it when you touch it, don't warn about it otherwise.

A conflict with a rule, or any change to one, follows `design-governance`.
Nothing here changes without the user's explicit approval. The log is
`decisions.md` in this folder.
