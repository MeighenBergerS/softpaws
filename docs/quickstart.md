# Quickstart

Five calls take you from a detector to a point-source ceiling. None of them
needs a downloaded release; everything here runs on the tables that ship with
the package.

## 1. Pick a detector

A `Site` carries the published geometry, the medium and the latitude, and
knows the area it presents to each arrival direction.

```python
from softpaws.detectors import ICECUBE, ARCA230, get_site

print(ICECUBE.detector_volume_km3(), ICECUBE.mean_projected_area_km2())
```

```
1.0 1.4309...
```

`get_site("ARCA230")` looks a site up by name. The registry holds IceCube,
IceCube-Gen2, ARCA230, ARCA21, TRIDENT in both layouts, P-ONE and Baikal-GVD.

## 2. Look at the transport

The loss kernel gives the drift coefficient, the transport exponent and the
range a muon covers before it drops below a threshold.

```python
import numpy as np
from softpaws.transport import (
    drift_coefficient,
    phi_eigenvalue_at_energy,
    stochastic_muon_range_km,
)

energy = 1.0e6
print(drift_coefficient(energy))              # b_mu [km^-1] in water
print(phi_eigenvalue_at_energy(1.0, energy))  # Phi(1), which equals b_mu
print(stochastic_muon_range_km(energy, 1.0e3))
```

A 1 PeV muon loses energy at 0.38 per kilometre of water and travels 15.6 km
before it falls to a TeV.

## 3. Build an effective area

The effective area per arrival direction is the target volume the transport
implies, weighted by the cross section and the Earth transmission.

```python
from softpaws.response.declination import directional_effective_area_cm2

aeff = directional_effective_area_cm2(
    ICECUBE, np.array([-0.5]), threshold_gev=1.0e3, log10_e=np.array([5.0, 6.0])
)
print(aeff[:, 0])
```

```
[1.38e+06 3.60e+06]
```

That is 100 TeV and 1 PeV, half way to the nadir, in cm². Passing
`reach_km` grows the body with the light the muon makes; leaving it out keeps
the instrumented footprint and nothing else, which makes the result
parameter-free.

## 4. Add a flux

The published fits are named constants, so a spectrum is one line.

```python
from softpaws.fluxes import ICECUBE_TRACKS_2022, ICECUBE_BPL_2025

print(ICECUBE_TRACKS_2022.flux(1.0e5))   # per flavour, at 100 TeV
print(ICECUBE_BPL_2025.shape(1.0e7))     # relative to the pivot
```

## 5. Ask what a point source would be seen at

Averaging inside the published declination bands and folding a spectrum
gives the flux a background-free search would exclude.

```python
from softpaws.response.declination import (
    band_averaged_effective_area_cm2,
    point_source_sensitivity,
)

edges = np.linspace(0.0, 1.0, 5)          # the upgoing sky, in sin(dec)
banded = band_averaged_effective_area_cm2(ICECUBE, edges, 1.0e3, reach_km=0.0178)
ceiling = point_source_sensitivity(banded, livetime_s=10 * 365.25 * 86400.0, gamma=2.0)
print(ceiling)
```

```
[2.75e-10 4.24e-10 6.57e-10 1.37e-09]
```

`E² φ` in GeV cm⁻² s⁻¹ at 100 TeV, band by band. The ceiling worsens toward
the nadir because the Earth absorbs the high-energy end, which is where the
declination dependence lives.

## Next

The [examples](examples.md) work through each of these in more depth, and
[Reproducing the paper](reproduce.md) rebuilds every figure. The
[API reference](api-reference/) documents every function.
