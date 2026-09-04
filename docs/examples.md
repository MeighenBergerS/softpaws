# Examples

Twelve numbered tutorials in `examples/` work through the package from the
transport up. Each calls only the public API, prints the numbers it computes,
and writes one figure to `examples/output/`. The printed output is the
result; the figure illustrates it.

Run one like this:

```sh
python examples/02_transport_exponent.py
```

Every script takes `--out-dir`, and the ones that read a release take
`--data-dir`. Pass `--help` to see the rest.

## The tutorials

| # | Script | What it shows | Needs |
| --- | --- | --- | --- |
| 01 | `load_the_release` | The events, effective areas, smearing and uptime of the DR2 release, and how much exposure it carries | DR2 |
| 02 | `transport_exponent` | `Phi(A)` from the loss kernel, against the drift-only and second-order truncations | — |
| 03 | `range_and_loss_law` | The first-passage range against the mean-loss range, and why the loss tail is not Gaussian | — |
| 04 | `earth_attenuation` | The PREM column against arrival direction, the survival it implies, and what regeneration adds back | — |
| 05 | `effective_area` | Effective areas of four detectors from the instrumented footprint alone, against IceCube's published table | DR2 |
| 06 | `declination_and_point_sources` | The response band by band, and the flux a background-free search would exclude | DR2 |
| 07 | `single_event_energy` | The neutrino energy behind KM3-230213A, under three flux priors and three loss families | — |
| 08 | `tau_induced_tracks` | What the tau channel contributes to the track rate, and where it takes over | — |
| 09 | `fitting_the_light_reach` | Scanning the one instrument number that closes the gap in tutorial 05 | DR2 |
| 10 | `atmospheric_background` | The MCEq background against the astrophysical flux, and where they cross | cached table or the `atm` extra |
| 11 | `diffuse_signal` | The flux a new diffuse signal needs before each telescope sees it, as a single event, a line and a power law | DR2 for IceCube |
| 12 | `point_source_signal` | The same three injections for a point source, against declination and site latitude | DR2 for IceCube |

Tutorials 02, 03, 07 and 08 need no downloaded data at all: everything they
use ships with the package.

## What they reproduce

Several tutorials land on numbers the method paper quotes, which makes them a
quick check that an installation is sound.

- Tutorial 02 reproduces the transport exponent and its truncations exactly.
- Tutorial 03 reproduces the loss-law survival table and the range rows.
- Tutorial 07 gives 250 PeV for the median neutrino energy of KM3-230213A
  under an `E^-2` prior with the exact kernel.
- Tutorials 05 and 09 give the level of 0.75 that the instrumented footprint
  alone leaves against the published DR2 table, and show the one number that
  closes it.
- Tutorials 11 and 12 fit that same number at all four sites, to between 0.02
  and 0.08 dex, and carry it into a sensitivity.

## The older scripts

`examples/` also holds the analysis scripts the method paper was built from,
numbered in the order they were written. They are being moved to a directory
of their own, one script per figure; see
[Reproducing the paper](reproduce.md).
