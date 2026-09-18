"""Fail when a pull request leaves a required template section empty.

Reads the PR description from the ``PR_BODY`` environment variable. HTML
comments (the template's guidance) do not count as content.
"""

import os
import re
import sys

REQUIRED = ("What and why", "How to check it", "Effect on results")


def sections(body):
    """Map each ``## `` heading to the text below it, comments removed."""
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    found = {}
    for block in re.split(r"^## ", body, flags=re.M)[1:]:
        heading, _, text = block.partition("\n")
        found[heading.strip()] = text.strip()
    return found


def main():
    """Print what is missing and exit nonzero, or confirm the template is filled."""
    found = sections(os.environ.get("PR_BODY") or "")
    missing = [name for name in REQUIRED if not found.get(name)]
    if missing:
        print("Please fill in these sections of the PR description:")
        for name in missing:
            print(f"  - {name}")
        print('Write "None" under "Effect on results" if no number changes.')
        sys.exit(1)
    print("All required sections are filled in.")


if __name__ == "__main__":
    main()
