# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`softpaws` is a first-principles forward model for ultra-high-energy neutrino
telescopes. It implements the **soft-volume** drift-diffusion muon-transport
approach of Palmisano, *The soft volume of ultra-high energy neutrinos
experiments* ([arXiv:2607.13143](https://arxiv.org/abs/2607.13143)), and
benchmarks it against the published IceCube instrument response functions
(effective area + smearing matrix) on real IceCube data
(IceTracks-DR2, DOI [10.7910/DVN/MMIIZA](https://doi.org/10.7910/DVN/MMIIZA)).

The project has three parts, in order: (1) implement the soft-volume forward
model, (2) reimplement the published IceCube IRF forward model for comparison,
(3) validate both against actual IceCube event data.

The repo is currently a scaffold: subpackages exist with module-level
docstrings describing their intended contents, but `tests/`, `examples/`, and
`docs/` are still empty placeholders (`.gitkeep` only).

## Commands

```sh
# Setup
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Lint
ruff check .

# Tests
pytest
pytest -m "not network"          # skip tests requiring network access
pytest tests/test_foo.py::test_bar   # single test
```

## Architecture

Data flows one direction through the subpackages of `src/softpaws/`:

```
data/  →  transport/  →  response/  →  comparison/
```

- **`data/`** — loads the IceCube IceTracks-DR2 release: reconstructed
  muon-track events, binned effective areas, and smearing matrices. The raw
  release files are never committed to this repo (`.gitignore` blocks
  `*.fits`, `*.h5`, `*.csv`, `dataverse_files/`, etc.). They live instead in
  the companion `neutrino_subhalos` project, under
  `src/neutrino_subhalos/data/dataverse_files/{events,irfs,uptime}/`.
- **`transport/`** — the soft-volume drift-diffusion muon transport itself:
  a second-order expansion of the Boltzmann collision operator where soft
  energy losses dominate propagation and rare hard scatters are handled
  perturbatively. This is the physics core from arXiv:2607.13143.
- **`response/`** — two interchangeable forward-model paths, both mapping a
  neutrino flux to a predicted event rate: one built on `transport/` (soft
  volume), the other reproducing the published IceCube IRF path (effective
  area convolved with energy/angular smearing matrices). Keeping these
  interchangeable is what makes the head-to-head comparison possible.
- **`comparison/`** — places the two `response/` predictions side by side and
  validates both against the observed DR2 event distributions.
- **`utils/`** — physical constants, unit conversions, shared helpers used
  across the other subpackages.

`examples/` holds numbered, runnable scripts; their output is written to
`examples/output/` (gitignored except for `.gitkeep`). `styles/` holds a
shared matplotlib style (`beacom_conformal.mplstyle`) for figures produced by
examples and comparison plots.

## Conventions

- Docstrings use NumPy style with project-specific conventions (units in
  square brackets, `EVENTS_DTYPE` field names in double backticks) — see
  `.claude/skills/numpy-docstring-format.md` for the full rules.
- Prose in docstrings and documentation follows the
  [Microsoft Writing Style Guide](https://learn.microsoft.com/en-us/style-guide/welcome/).
- Ruff line length is 100; enabled rule sets are E, F, W, I.
- Commit messages are short and imperative (`add drift-diffusion collision
  operator`, not `update` or `fixed stuff`).
