# softpaws

A first-principles forward model for ultra-high-energy neutrino telescopes,
implementing the **soft-volume** drift-diffusion approach and comparing it,
head to head, against the published IceCube instrument response functions
(effective area + smearing matrix) on real IceCube data.

## What this is

Standard neutrino point-source analyses map a flux to an event rate through
Monte-Carlo-derived instrument response functions: a binned effective area and
an energy/angular smearing matrix. `softpaws` instead builds the detector
response from the underlying muon-transport physics.

Following Palmisano, *The soft volume of ultra-high energy neutrinos
experiments* ([arXiv:2607.13143](https://arxiv.org/abs/2607.13143)), it treats
the neutrino-to-muon problem with a second-order (drift-diffusion) expansion of
the Boltzmann collision operator: soft energy losses dominate the muon
propagation and rare hard scatters are handled perturbatively. The resulting
**soft volume** — which can exceed the instrumented volume — gives a fast,
microphysically transparent alternative to full MC.

The project has three parts:

1. **Implement** the soft-volume forward model from the paper.
2. **Compare** it to the published IceCube IRF forward model
   (effective area + smearing matrix).
3. **Validate** both against the actual IceCube event data.

## Data

**Neutrino model paper**
[arXiv:2607.13143](https://arxiv.org/abs/2607.13143) — Palmisano, *The soft
volume of ultra-high energy neutrinos experiments*.

**IceCube data release** (IceTracks-DR2, April 2008 – May 2022)
DOI [10.7910/DVN/MMIIZA](https://doi.org/10.7910/DVN/MMIIZA)
| Paper: [arXiv:2605.19040](https://arxiv.org/abs/2605.19040)

The release provides reconstructed muon-track events, binned effective areas,
and smearing matrices. The raw files are not part of the package (see
[.gitignore](.gitignore)). Download the release from the DOI above and either
place it under `src/softpaws/data/dataverse_files/` (with its `events/`,
`irfs/` and `uptime/` subdirectories) or point the environment variable
`SOFTPAWS_DATA_DIR` at a directory that holds `dataverse_files/`. The HESE
7.5-year release goes under `hese/` next to it. See `softpaws.data.paths`.

## Installation

```sh
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Requires Python ≥ 3.11.

The atmospheric-neutrino background of `examples/22_atmospheric_background_mceq.py`
is computed with [MCEq](https://github.com/mceq-project/MCEq), an optional extra:

```sh
.venv/bin/pip install -e ".[atm]"
```

## Project layout

```
src/softpaws/
├── data/          # Load IceCube DR2 events, effective areas, and smearing matrices
├── transport/     # Soft-volume drift-diffusion muon transport (arXiv:2607.13143)
├── response/      # Forward models: flux → event rate (soft-volume and IRF paths)
├── comparison/    # Compare soft-volume vs IRF predictions against DR2 data
└── utils/         # Physical constants, unit conversions, shared helpers
examples/          # Numbered, runnable scripts (output in examples/output/)
styles/            # Shared matplotlib style
tests/
docs/
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).
