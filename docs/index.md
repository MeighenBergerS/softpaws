# softpaws

softpaws builds the response of a neutrino telescope from muon transport
rather than from simulation. It solves the transport of a high-energy muon
through matter, turns that solution into the volume a detector effectively
watches, and from there into an effective area, an event rate, or the energy
of a single track. The same code serves IceCube, KM3NeT/ARCA, P-ONE, TRIDENT
and Baikal-GVD, because nothing in the construction is specific to one site.

## What it is for

A published effective area is a Monte-Carlo product. It answers what a
detector sees, but not why, and it cannot be moved to a detector that has not
been simulated. softpaws computes the same quantity from the loss kernel of
the medium, the neutrino cross section, the geometry of the instrumented
volume, and two numbers per site that the instrument itself sets: a selection
threshold and a light reach.

That makes three things possible. You can reproduce a published effective
area and see which ingredient carries each feature. You can predict the
response of a detector that has no published table yet. And you can ask what
a measurement would look like under a different loss model, which is how the
error on the transport itself is measured.

## What is inside

| Subpackage | What it holds |
| --- | --- |
| `softpaws.transport` | The loss kernel, the transport exponent, the range to threshold, the loss law, Earth geometry and attenuation, and the tau channel |
| `softpaws.detectors` | Published geometry, medium and optics for each site |
| `softpaws.fluxes` | Power laws, the published IceCube fits, and the atmospheric background |
| `softpaws.response` | Effective areas: from the light reach, from the published optics, and resolved by declination |
| `softpaws.comparison` | Likelihoods, posterior statistics, the event benchmark, and single-track energy reconstruction |
| `softpaws.data` | Loaders for the IceCube release and for the published curves of the other detectors |

## Where to start

Read [Installation](installation.md), then [Quickstart](quickstart.md) for
five calls that take you from a detector to an effective area to an event
rate. [Data](data.md) explains what ships with the package and what you have
to download. The [API reference](api-reference/) is generated from the
docstrings.

The method is described in *Analytical High-Energy Muon Transport for
Neutrino Telescopes*; see [Citation](citation.md).
