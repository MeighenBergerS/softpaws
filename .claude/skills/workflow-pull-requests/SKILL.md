---
name: workflow-pull-requests
description: How to scope and describe a softpaws pull request. Load before opening, updating or describing a PR.
---

# Pull requests

PR1. One concern per PR, and small: one self-contained change, not a whole
     feature. Keep refactors apart from behaviour changes.
PR2. The title follows C2. The description says what and why, how it was
     tested, and where to start reviewing, and links the issue.
PR3. Review your own diff first. CI is green, and no unrelated changes ride along.
PR4. List any design rule changed or excepted, as in the template.
PR5. No AI attribution in the title, the body or the commits (`workflow-ai-disclosure`).
PR6. Open, push or merge only when the user asks.
