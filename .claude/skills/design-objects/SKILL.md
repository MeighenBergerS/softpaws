---
name: design-objects
description: When softpaws uses a class, and how Detector, the central object, is shaped. Load when adding a class, a method, or a new analysis type.
---

# Objects

O1. Use classes for the big concepts only: `Detector`, fluxes, and result
    objects. Everything else is a function on arrays.
O2. `Detector` is the central object and the entry point. Each analysis is a
    method that takes the flux, the exposure and the background, so typing
    `det.` shows the workflow.
O3. `Detector` holds everything that describes the instrument: geometry,
    site, medium, optics and module count. Downstream code reads these from
    it; they are never passed separately.
    Do: `det = Detector(..., modules=10_000)`, then `det.point_source(...)`.
    Not: `point_source_limit(site, ..., reach_km=...)` with the optics left behind.
O4. Every constructor argument, and every assumption a result depends on, is
    a readable attribute.
O5. Objects are immutable (frozen dataclasses). A variant is made with `.replace(...)`.
O6. Methods return arrays or small result objects. They never print.
