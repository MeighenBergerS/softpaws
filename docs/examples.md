# Examples

Twelve numbered examples in `examples/` work through the package from the
transport up. Each is a short script that reads top to bottom: a docstring
saying what it shows, a few named inputs, and one `main()` built from public
library calls. Each prints the numbers it computes and writes one figure to
`examples/output/`.

Run one like this:

```sh
python examples/02_transport_exponent.py
```

To change an input, such as the threshold, the exposure or the detectors
compared, edit the constants at the top of the script.

## The examples

| # | Script | What it shows | Needs |
| --- | --- | --- | --- |
| 01 | `load_the_release` | The events, effective areas and uptime of the DR2 release, and how much exposure it carries | DR2 |
| 02 | `transport_exponent` | `Phi(A)` from the loss kernel, against the drift-only and second-order truncations | — |
| 03 | `range_and_loss_law` | The first-passage range against the mean-loss range, and why the loss tail is not Gaussian | — |
| 04 | `earth_attenuation` | The PREM column against arrival direction, the survival it implies, and what regeneration adds back | — |
| 05 | `effective_area` | Sky-averaged effective areas of four detectors from the instrumented footprint, against IceCube's published table | DR2 for the comparison |
| 06 | `declination_and_point_sources` | The response band by band, with and without the light reach, and the flux a background-free search excludes | DR2 |
| 07 | `single_event_energy` | The neutrino energy behind KM3-230213A, under three flux priors and two loss models | — |
| 08 | `tau_induced_tracks` | What the tau channel contributes to the track rate, and where it takes over | — |
| 09 | `fitting_the_light_reach` | Fitting the one instrument number that closes the gap left in example 05 | DR2 |
| 10 | `atmospheric_background` | The MCEq background against the astrophysical flux, and where they cross | — |
| 11 | `diffuse_signal` | The flux a new diffuse signal needs before each telescope sees it: a single event, a line and a power law | DR2 for IceCube |
| 12 | `point_source_signal` | The flux a point-source search reaches against declination at four sites, with the atmospheric background counted | DR2 for IceCube |

Examples 02, 03, 04, 07, 08 and 10 need no downloaded data, and 05, 11 and 12
run without it and skip IceCube. Only 01, 06 and 09 need the release.

## What they reproduce

Several examples land on numbers the method paper quotes, which makes them a
quick check that an installation is sound.

- Example 02 reproduces the identities `Phi(1) = b_mu` and
  `Phi(2) = 2 b_mu - d_mu`.
- Example 03 reproduces the loss-law table at a factor of 4.5: 5.9% for the
  three-moment law against 2.0e-4 for the Gaussian.
- Example 07 gives 250 PeV for the median neutrino energy of KM3-230213A
  under an `E^-2` prior.
- Example 09 fits a light reach of 13 m per e-fold at IceCube, which moves the
  published-over-model level from 0.74 to 0.98.
- Example 12 lands on IceCube's published point-source sensitivity across the
  northern sky, with nothing fitted beyond that one number.

## Checking that they run

`tests/test_examples.py` runs every example end to end. It is marked slow:

```sh
pytest --run-slow tests/test_examples.py
```
