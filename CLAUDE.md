# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`softpaws` is a first-principles forward model for ultra-high-energy neutrino
telescopes. It implements the analytic muon transport of Meighen-Berger,
*Estimating High-Energy Neutrino Effective Areas from Muon Propagation* (2026), the
method paper drafted in `paper/main.tex`. That calculation diagonalizes the
locally scale-invariant collision operator with power laws, so the whole loss
history collapses to a single transport exponent `Phi(s)`, and it uses that
exponent as a first-passage generator for the range of a muon and for the
detector response built on it.

The repo builds three things on that exponent: effective areas for IceCube,
KM3NeT/ARCA, P-ONE, TRIDENT and Baikal-GVD from two instrument numbers per
site, event rates against the IceCube IceTracks-DR2 release (DOI
[10.7910/DVN/MMIIZA](https://doi.org/10.7910/DVN/MMIIZA)), and the energy of a
single track.

An earlier analytic calculation, Palmisano, Redigolo, Tammaro and Tesi,
*The soft volume of ultra-high energy neutrinos experiments*
([arXiv:2607.13143](https://arxiv.org/abs/2607.13143)) and its companion
[arXiv:2507.10665](https://arxiv.org/abs/2507.10665), expands the same
collision operator to second order in the energy each collision removes. That
approach was implemented here first, as a check, and it is kept as the drift
limit of the transport exponent and as a cross-check of it. It is prior work,
never "the paper". In docs and docstrings refer to it by author and arXiv
number so the two are never confused. In `paper/main.tex` prose cite it by
number only (`two recent papers~\cite{...}`, `Refs.~\cite{...}`), never by
author name.

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

# Docs
mkdocs serve
```

## Architecture

Data flows one direction through the subpackages of `src/softpaws/`:

```
data/  +  detectors/  +  fluxes/  →  transport/  →  response/  →  comparison/
```

- **`data/`** — loaders for the IceCube releases and for the published curves
  of the other detectors. The tabulated inputs that ship with the package (the
  loss coefficients, the BGR18 cross section, the ARCA, P-ONE and TRIDENT
  effective areas) live here. The IceCube releases are large and are never
  committed (`.gitignore` blocks `*.fits`, `*.h5`, `dataverse_files/`); the
  user downloads them and places them under the package's `data` directory or
  sets `SOFTPAWS_DATA_DIR`; see `softpaws.data.paths`.
- **`detectors/`** — published geometry, medium and optics for each site.
- **`fluxes/`** — power laws, the published IceCube fits, and the atmospheric
  background through MCEq.
- **`transport/`** — the physics core: the loss kernel, the transport exponent
  `Phi(s)` and its eigenvalue treatment (`eigenvalue`), the range to threshold
  and its first-passage moments (`muon_range`), the log-loss law, Earth geometry and
  attenuation, and the tau channel. The drift-diffusion coefficients
  (`coefficients`) and the drift-limit soft volume (`soft_volume`) are the
  prior-work limit, kept as a cross-check.
- **`response/`** — effective areas and track rates: from the light reach,
  from the published optics, and resolved by declination. It also holds the
  published-IRF path, so a prediction can be put next to IceCube's own
  effective area and smearing matrix on the same footing.
- **`comparison/`** — likelihoods, posterior statistics, the no-fit event
  benchmark against DR2, and single-track energy reconstruction.
- **`utils/`** — physical constants, unit conversions, shared helpers used
  across the other subpackages.

`examples/` holds twelve numbered tutorials, `01` to `12`, writing to
`examples/output/` (gitignored except for `.gitkeep`).
`scripts/2026_muon_transport/` holds one script per paper figure, table and
number, and `scripts/future_bsm/` holds searches that belong to a later paper.
`styles/` holds the shared matplotlib style (`beacom_conformal.mplstyle`) used
for every figure.

## Conventions

- Design rules live in `.claude/skills/design-*`. The index is
  `design-principles`; the change protocol and the learning rules are in
  `design-governance`. Load the
  relevant skill before touching public code, docstrings, docs or examples.
  A conflict with a rule is flagged in chat as **Design warning**, a rule
  change as **Design change**, and both are logged in
  `design-principles/decisions.md`. Never change a rule silently.
- The design skill tree is a living document. Whenever development makes,
  changes or reverses a design choice, propose the skill update in chat as
  **Design change proposed**, following the learning rules in
  `design-governance`. A PostToolUse hook (`.claude/hooks/design_reminder.py`)
  prompts this after edits to public files.
- Skills, hooks, `.claude/settings.json`, this file and
  `tests/test_design_rules.py` change only after the user explicitly approves
  that change. Never silently, never bundled with other work, never through
  Bash. `permissions.ask` in `.claude/settings.json` enforces the prompt.
- Prose in docstrings and documentation follows the
  [Microsoft Writing Style Guide](https://learn.microsoft.com/en-us/style-guide/welcome/).
- Ruff line length is 100; enabled rule sets are E, F, W, I, and D (numpy docstrings, `src/` only).
- Commits and PRs follow `workflow-commits` and `workflow-pull-requests`.
  Never add Claude as an author, co-author or signer (`workflow-ai-disclosure`).
