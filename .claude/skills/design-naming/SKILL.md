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
N4. No unit suffixes on public names or arguments. The code computes in GeV,
    cm, s and rad. Users multiply by units from `softpaws.constants`
    (`radius=1.0 * km`) and divide to read a result out (`aeff / m**2`).
N5. Field-standard abbreviations are fine (`aeff`, `psf`). Project jargon is
    not (`ladder`, `halo`, `ceiling`, `rung`).
N6. Outside `data`, no public name carries a detector or dataset name: the
    detector is an argument, or a `Detector`.
    Do: `det.published_effective_area(energy)`.
    Not: `icecube_upgoing()`, `pone_allsky_cm2()`, `ic_effective_area_regenerated()`.
