---
name: design-docs
description: Structure of the softpaws docs site, README, examples and llms.txt. Load when writing or editing any page under docs/, the README, or an example.
---

# Docs

G1. Diataxis: tutorials (learn), how-to (one task), reference (generated
    from the docstrings), explanation (physics). One kind per page.
G2. The top-level API comes first. Subpackages appear only in how-to and reference pages.
G3. An example does one task in a few calls through the organizing object,
    with no paper numbers or benchmark narrative.
G4. `llms.txt` at the repo root is one page: the top-level API, the unit
    convention, and links. Update it whenever the A1 list changes.
G5. The README shows the one-screen workflow and nothing outside the top level.
G6. `docs/units.md` gives the base units, every multiplier, and how to convert
    in and out, in two lines of code. The README, the quickstart and `llms.txt` link to it.
