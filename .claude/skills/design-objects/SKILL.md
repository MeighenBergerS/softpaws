---
name: design-objects
description: When softpaws uses a class, and how the organizing object and Detector are shaped. Load when adding a class, a method, or a new analysis type.
---

# Objects

O1. Use classes for the big concepts only: the organizing object, `Detector`,
    fluxes, and result objects. Everything else is a function on arrays.
O2. One organizing object is the entry point. It holds the detector(s), the
    flux, the exposure and the background, and exposes each analysis as a
    method, so typing `obj.` shows the workflow. Its name is open; see
    `decisions.md`.
O3. `Detector` holds everything that describes the instrument: geometry,
    site, medium, optics and module count. Downstream code reads these from
    it; they are never passed separately.
    Do: `Detector(..., modules=10_000)`, then `analysis.point_source(...)`.
    Not: `point_source_limit(site, ..., reach_km=...)` with the optics left behind.
O4. Every constructor argument, and every assumption a result depends on, is
    a readable attribute.
O5. Objects are immutable (frozen dataclasses). A variant is made with `.replace(...)`.
O6. Methods return arrays or small result objects. They never print.
