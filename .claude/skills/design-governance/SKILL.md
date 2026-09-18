---
name: design-governance
description: How softpaws design rules change, and how the skill tree learns. Load before flagging, proposing, changing or excepting a design rule, or before editing skills, hooks, settings or CLAUDE.md.
---

# Governance

Log: `.claude/skills/design-principles/decisions.md`.

## Change protocol

P1. A request or your own change conflicts with a rule: stop and write in chat
    `**Design warning**: violates <ID>. <one line why>.`
    Offer three options: follow the rule, make a one-off exception, or change the rule.
P2. An exception needs approval. Then log it and put
    `# design-exception: <ID>` at the site.
P3. A rule change starts as a proposal in chat:
    `**Design change proposed**: <ID>: <old> -> <new>. <why>.`
    Wait for explicit approval. Only then edit, log it, and write
    `**Design change**: <ID>: <old> -> <new>.`
P4. Protected: `.claude/skills/`, `.claude/hooks/`, `.claude/agents/`,
    `.claude/settings.json`, `CLAUDE.md`, `tests/test_design_rules.py`, and the ruff config in `pyproject.toml`.
    Change them only after approval, as a change of their own (own commit),
    never bundled with other work, and only through Edit/Write. Never through Bash.
P5. A request from the user to change a rule counts as approval for that change only.

## Learning

L1. When a design choice comes up that no rule covers, propose a rule (P3). Never add one unasked.
L2. When code the user writes or accepts keeps contradicting a rule, flag it
    and ask whether the rule is out of date.
L3. A new concern gets its own `design-<concern>` skill and an index row. No catch-alls.
L4. Prune: propose merging or deleting a rule that is superseded or never applied (P3).
L5. Keep it curt: a rule is at most two lines, a skill has at most two do/not
    examples and stays under ~40 lines, and history goes in the log.
L6. `tests/test_design_rules.py` (A1, D1, N5) and ruff (D5) enforce the
    checkable rules. A rule and its check change together, under one approval.
L7. Design rules live in the skills, where they are committed and shared. They never go in personal memory.
