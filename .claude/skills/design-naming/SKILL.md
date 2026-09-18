---
name: design-naming
description: Naming rules for public softpaws functions, methods and arguments, and the unit convention. Load when naming or renaming anything public.
---

# Naming

N1. A name says what the user gets, in the user's words, not the method or the paper's term.
    Do: `det.effective_area(energy, zenith)`.
    Not: `derived_directional_effective_area_cm2(site, optics, ...)`.
N2. An option is an argument, never a new name.
    Do: `effective_area(..., average="sky")`.
    Not: `sky_averaged_effective_area_cm2`, `band_averaged_effective_area_cm2`.
N3. The same concept has the same argument name everywhere: `energy`,
    `zenith`, `declination`, `years`, `threshold`.
N4. Units are open (see `decisions.md`). Until they are decided, keep the
    existing convention: a unit suffix on the name, and `[unit]` in the docstring.
N5. Field-standard abbreviations are fine (`aeff`, `psf`). Project jargon is
    not (`ladder`, `halo`, `ceiling`, `rung`).
