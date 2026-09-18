"""SessionStart hook: tell Claude when the Repo Audit is due.

Due when 14 days have passed since the last run (or it never ran) and no
snooze is active. State lives in the untracked ``.audit/state.json``.
"""

import json
import os
from datetime import date, timedelta
from pathlib import Path

INTERVAL = timedelta(days=14)


def main() -> None:
    root = Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))
    state_file = root / ".audit" / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    today = date.today()

    snooze = state.get("snooze_until")
    if snooze and today < date.fromisoformat(snooze):
        return
    last = state.get("last_run")
    if last and today - date.fromisoformat(last) < INTERVAL:
        return

    since = f"last run {last}" if last else "never run"
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "SessionStart",
        "additionalContext": (
            f"The Repo Audit is due ({since}). Before anything else, load the "
            "repo-audit skill and follow RA1: explain it in 1-3 short sentences "
            "and ask the user whether to run it."
        ),
    }}))


if __name__ == "__main__":
    main()
