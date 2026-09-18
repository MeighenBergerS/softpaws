---
name: design-architecture
description: softpaws layers, module boundaries and single source of truth. Load when adding a module, moving code, exporting a name, or deciding where new code lives.
---

# Architecture

A1. Three layers.
    - `softpaws` top level holds the ~15 names a typical analysis needs, and nothing else.
    - Subpackages (`transport`, `response`, ...) hold the building blocks for users who change the physics.
    - Private (`_`-prefixed) modules and `scripts/` hold paper reproduction and internals. None of it goes in an `__all__`.
A2. A name is public only if a user workflow needs it. A name that one script
    uses lives with that script, or is private.
A3. One source of truth. A quantity or helper is defined once and imported,
    never copied into an example or a script.
A4. Dependencies flow one way: `constants`, `standards` -> `data`, `detectors`,
    `fluxes` -> `transport` -> `response` -> `comparison`. `Detector` sits on
    top of all of them, and nothing imports it.
A5. Group modules by concept, never by paper section or example number.
A6. Every named number that is a literal, or arithmetic on literals, lives in
    `src/softpaws/constants.py`. Three kinds stay put: values computed by
    package code, numbers that define a published record or table
    (`detectors/sites.py`, `detectors/optics.py`), and paper tuning (private `_paper/`).
A7. Current best-fit values ship in `src/softpaws/data/standards.json` with
    their source, and `softpaws.standards` exposes them. `make_standards()`
    rebuilds the file from a fit result, and `standards.use(path)` switches to
    a user's own. Every default that comes from a fit is read from here.
