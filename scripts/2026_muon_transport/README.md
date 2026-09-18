# Scripts for *Analytical High-Energy Muon Transport for Neutrino Telescopes*

Every figure, table and quoted number of the paper comes from a script here.
They are numbered in the order they were written, not in the order the paper
presents them; the map below is the translation. Rename this directory to the
arXiv identifier once it exists.

For a first look at the package, read `examples/` instead. These scripts are
the analysis, not a tutorial: several take hours and cache their result in
`output/`, and later ones read those caches.

## Running them

```sh
python scripts/2026_muon_transport/run_all.py --dry-run
```

That prints the order without running anything. Drop the flag to run it. A
step whose outputs are already in `output/` is skipped unless you pass
`--force`.

Individual scripts take `--out-dir` and, where they read a release,
`--data-dir`. Pass `--help` to any of them.

## What you need

The tabulated inputs ship with the package. Beyond them:

- the IceTracks-DR2 release, for everything touching IceCube data
  (DOI 10.7910/DVN/MMIIZA; see the data guide for the layout);
- MCEq, once, for the atmospheric background: `pip install -e ".[atm]"`;
- PROPOSAL, for the kernel benchmark and the loss-model ensemble:
  `pip install -e ".[transport]"`.

## The figures

| Paper | Script | Output |
| --- | --- | --- |
| Figure 2 | `37_moment_convergence.py` | `37_exponent_vs_proposal` |
| Figure 3 | `89_sky_averaged_plug_in.py`, from the cache of 88 | `89a_sky_averaged_plug_in` |
| Figure 4 | `83_four_detector_aeff_reduced.py` | `83a_four_detector_aeff_reduced` |
| Figure 5, 12, 13 | `76_dr2_event_benchmark.py` | `76a_reco_spectrum`, `76b_declination`, `76c_declination_high` |
| Figure 6, 11 | `84_point_source_background_limited.py` | `84d_site_ceiling`, `84c_published_sensitivity` |
| Figure 7, 8, 14 | `75_km3_figures_with_bands.py` | `75a_event_energy_banded`, `75b_tension_bpl`, `75c_tension_spl` |
| Figure 9, Appendix B | `82_reduced_response_plane_trident2025.py` | `82_reduced_response_plane_trident2025` |
| Figure 10, Appendix C | `74_point_source_with_bands.py` | `74a_effective_area_bands` |

Figure 1 is a hand-drawn sketch and has no script.

## The tables

| Paper | Built from |
| --- | --- |
| Table I, the instrument numbers | `make_recipe_table.py`, from the chains of 77, 82 and 88 |
| Table II, the range to threshold | `make_transport_table.py`, checked by `39_range_moment_estimator.py` and `48_fluctuation_cost.py` |
| Table III, the loss law | `27_proposal_cross_section_and_loss.py`, `36_transport_exponent_truncations.py`, `37_moment_convergence.py` |
| Table V, the inputs of the estimate | `make_estimate_table.py`, from example 89's first-principles inputs and the library |
| Table VI, the tension ladder | `31_flux_contours_effective_area.py` |

Table E.2 is hand-collected geometry, and matches `softpaws.detectors`; its
kernel rows are printed by `make_transport_table.py`. Both table scripts write
LaTeX to `output/`.

## The quoted numbers

Several scripts make no figure but supply numbers the text quotes: 45 and 46
for the derived optics and the band-by-band residual, 47 for the point-source
residual, 48 for the range validation, 49 for the tau share, 50 and 51 for the
flavour fit, 57 for the KM3-230213A energies, 72 and 73 for the informed prior
and the four-site residuals, 76 for the event benchmark, 85 for the
closed-form response of Section III against the full one at IceCube, 86
for the same comparison drawn at all four sites (`86a_closed_form_four_detectors`),
and 87 for what each of the three corrections of Section IV is worth at
KM3NeT/ARCA230 (`87a_arca_closed_form_ladder`, no longer a paper figure).
Script 90 checks how much of Figure 2's agreement the three matched moments
guarantee. It draws two other kernels with the same moments and the envelope of
all of them (`90a_three_moment_families`, `90b_three_moment_ratio`), and the
nested envelopes for one to five matched moments
(`90c_moment_envelopes`, and as `Phi(A)` itself in
`90d_moment_envelopes_absolute`).

Every number the paper depends on is pinned in
`tests/regression/baseline.json`; check them with

```sh
pytest -q --run-slow tests/regression
```

Two blocks there are known to be stale and are regenerated rather than
trusted: the range rows of Table C.1 and the water log-loss rate of Tables C.1
and E.2 predate the current kernel table, and the two-flavour run log of
script 45 predates a column it now prints.

## The order

The dependencies run bottom-up through the caches in `output/`. `run_all.py`
encodes this list:

```
22  27  32  35  45  46  47  55  33  56  69  51  31  57
70  71  72  73  77  81  82  83  85  86  87  88  89  74  84  75  76  37
```

then `paper/make_recipe_table.py`. The expensive steps are 22 (MCEq), 69 (the
PROPOSAL ensemble), and the posterior chains of 33, 56, 72, 73, 77 and 82.
