# Changelog

All notable changes to softpaws are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- `softpaws.detectors`: published geometry, medium and optics for IceCube,
  IceCube-Gen2, KM3NeT/ARCA in two configurations, TRIDENT in two layouts,
  P-ONE and Baikal-GVD.
- `softpaws.fluxes`: single and broken power laws, the published IceCube
  fits as named constants, and the MCEq atmospheric background.
- `softpaws.transport.earth`: the PREM profile, chords, columns, the
  overburden and the zenith grid.
- `softpaws.transport.loss_ensemble` and a kernel-scaling hook, which push an
  alternative loss parametrization through every calculation.
- `softpaws.response`: the effective-area engine, the light reach derived
  from published optics, the first-principles assembly, and the
  declination-resolved response with its point-source ceiling.
- `softpaws.comparison`: posterior sampling and statistics, Feldman-Cousins
  calibration, single-track energy reconstruction, the reconstructed-energy
  likelihood, and the DR2 event benchmark.
- `softpaws.data.paths`: `SOFTPAWS_DATA_DIR` overrides where the releases are
  read from, and every loader names the DOI when a file is missing.
- A documentation site, continuous integration, and a regression fixture that
  pins the numbers the method paper quotes.

### Changed

- The tabulated inputs ship in the wheel as package data.
- The slow tests are opt-in behind `--run-slow`.

### Removed

- `softpaws.comparison.datasets`, which nothing reached.
- `softpaws.response.irfs.PointSpreadFunction` is deprecated and will be
  removed before 1.0.
