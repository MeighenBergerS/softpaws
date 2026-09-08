# Your own detector

Nothing in the construction is specific to a site, so a detector that has
no published effective area is the same three calls as one that has. This
page builds a hypothetical water telescope, gives it optics, and folds a flux
of your own through it. Everything here runs on the tables that ship with the
package.

## 1. Describe the layout

A `Site` is a frozen dataclass. The published layouts are module constants,
and a new one is the same constructor with your numbers. The shape is a
sphere, an upright cylinder, or a prism with a regular polygon cross-section,
and every length is in kilometres.

```python
from softpaws.detectors import Site

DEMO = Site(
    name="Demo",
    shape="cylinder",
    latitude_deg=30.0,      # sets how declination maps to zenith
    depth_km=2.5,           # centre of the instrumented volume below the surface
    density_g_cm3=1.04,     # sea water
    radius_km=0.6,
    height_km=0.9,
    n_blocks=1,
    below_km=0.1,           # water between the bottom of the array and the seabed
)
print(DEMO.detector_volume_km3(), DEMO.mean_projected_area_km2())
```

```
1.018 1.414
```

The `below_km` field matters for upgoing tracks: a muon that arrives from
below spends only this much of its range in water and the rest in rock, and
the range is shortened accordingly. A published layout is adjusted the same
way, with `replace`:

```python
from softpaws.detectors import ARCA230

TALLER = ARCA230.replace(name="ARCA, taller blocks", height_km=0.8)
```

## 2. Compute the response

The effective area per arrival direction needs the layout, the muon
threshold of the selection, and nothing else. Leaving out the light reach
keeps the instrumented footprint and makes the result parameter-free.

```python
import numpy as np
from softpaws.response.declination import directional_effective_area_cm2

log10_e = np.array([5.0, 6.0, 7.0])
aeff = directional_effective_area_cm2(
    DEMO, np.array([-0.5]), threshold_gev=1.0e3, log10_e=log10_e
)
print(aeff[:, 0])
```

```
[1.35e+06 3.54e+06 3.46e+06]
```

That is the area in cm² at 100 TeV, 1 PeV and 10 PeV, half way to the nadir.
The turn-over at 10 PeV is the Earth absorbing the parent neutrino along
that column. Against ARCA230 in the same direction the demo layout sits at
0.78 at every energy, which is the ratio of the two projected areas: the
transport is the same, only the geometry differs.

The point-source ceiling follows as in the [quickstart](quickstart.md), band
by band in `sin(dec)`:

```python
from softpaws.response.declination import (
    band_averaged_effective_area_cm2,
    point_source_sensitivity,
)

edges = np.linspace(0.0, 1.0, 5)
banded = band_averaged_effective_area_cm2(DEMO, edges, 1.0e3)
print(point_source_sensitivity(banded, livetime_s=10 * 365.25 * 86400.0, gamma=2.0))
```

```
[2.03e-10 2.76e-10 3.83e-10 7.09e-10]
```

## 3. Give it optics

A published table carries two instrument numbers, a selection threshold and
a light reach. With an `Optics` record both are predicted from the medium
and the module instead of fitted. The two shipped records are IceCube's deep
ice and ARCA's sea water. A new one is a copy with the fields that differ,
and `dataclasses.replace` is the way to make it, because `Optics` is a plain
frozen dataclass:

```python
import dataclasses
from softpaws.detectors import ARCA_OPTICS
from softpaws.response.light_reach import (
    DEFAULT_MIN_MODULES,
    instrumented_chord_km,
    muon_threshold_gev,
)

DEMO_OPTICS = dataclasses.replace(
    ARCA_OPTICS,
    name="Demo",
    absorption_m=50.0,             # a murkier site than Capo Passero
    module_density_per_km3=3000.0,
    headroom_below_m=100.0,        # the same 0.1 km as below_km above
)
chord = instrumented_chord_km(DEMO.radius_km, DEMO.height_km, None)
print(muon_threshold_gev(DEMO_OPTICS, DEFAULT_MIN_MODULES, chord))
```

```
1688.2
```

That is the muon energy at which a track through the array first lights the
default eight modules: the threshold the light sets, before any selection.
The same optics dilate the body by the light the muon makes outside it, and
the derived response replaces the parameter-free one:

```python
from softpaws.response.declination import derived_directional_effective_area_cm2
from softpaws.response.effective_area import default_cross_section

derived = derived_directional_effective_area_cm2(
    DEMO.replace(optics=DEMO_OPTICS),
    DEMO_OPTICS,
    np.array([-0.5]),
    DEFAULT_MIN_MODULES,
    ("mu", "tau"),
    default_cross_section(),
    log10_e=log10_e,
)
print(derived[:, 0] / aeff[:, 0])
```

```
[1.00 1.22 1.36]
```

The light reach grows with energy, so the derived area pulls away from the
footprint above a PeV. The wavelength shapes inside the shipped records are
smooth parameterizations of the published curves, and a collaboration's own
tables drop in through the same fields.

## 4. Fold a flux of your own

The published fits are `PowerLawFit` and `BrokenPowerLawFit` records, so a
new power law is one line. The normalization is per flavour at 100 TeV, in
units of 10⁻¹⁸ GeV⁻¹ cm⁻² s⁻¹ sr⁻¹:

```python
from softpaws.fluxes import PowerLawFit

MINE = PowerLawFit("mine", phi0=1.5, gamma=2.4)
print(MINE.flux(1.0e5))
```

Any callable of the energy works where a shape is not a power law. The
expected count is the effective area against the flux, integrated over
energy and over the sky, and that is one quadrature with no special entry
point:

```python
from softpaws.response.declination import COMMON_LOG10_E

def my_flux(energy_gev):
    """Per flavour, GeV^-1 cm^-2 s^-1 sr^-1, with a cutoff at 3 PeV."""
    return 1.5e-18 * (energy_gev / 1.0e5) ** -2.4 * np.exp(-energy_gev / 3.0e6)

cos_theta = np.linspace(-1.0, 0.0, 21)                      # the upgoing sky
aeff_up = directional_effective_area_cm2(DEMO, cos_theta, threshold_gev=1.0e3)
energy = 10.0 ** COMMON_LOG10_E
per_direction = np.trapezoid(aeff_up * my_flux(energy)[:, None], energy, axis=0)
rate = 2.0 * np.pi * np.trapezoid(per_direction, cos_theta)  # s^-1
print(rate * 10 * 365.25 * 86400.0)
```

```
5464.0
```

Upgoing events in ten years, above a 1 TeV muon threshold and before any
selection beyond it. The `AtmosphericFlux` of `softpaws.fluxes` is a callable
of energy and declination in the same units, so the background enters the
same integral.

## Where to go next

Tutorials 11 and 12 in `examples/` inject a single event, a line and a
power law through the same response at four sites, and fit the light reach
where a published table exists to fit it against. For a site with no table,
the optics path above is the prediction the paper makes for P-ONE and
TRIDENT.
