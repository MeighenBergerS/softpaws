# Repository cleanup plan

Goal: turn `softpaws` from a research scratchpad into a pip-installable,
documented package that an outside reader can install, run, and use to
reproduce every figure and number in the paper.

Written 2026-09-03 from a full survey of the repository. Numbers below come
from that survey. Phases are ordered so that each one leaves the repository
working and the paper reproducible.

## 1. Where the repository stands

**Library.** `src/softpaws/` is 9,200 lines in 26 modules. The physics core is
sound: 285 tests pass, ruff is clean, there are no absolute paths and no
`TODO` markers. Two modules are too large (`transport/soft_volume.py` at 2,028
lines and `response/soft_volume.py` at 1,288 lines). Nineteen public defs have
no docstring and 35 lack a `Returns` section. One module is dead
(`comparison/datasets.py`), one script lives inside the package by mistake
(`data/icecube/extract_flavor2510_fig1.py`), and one class calls itself legacy
(`response.irfs.PointSpreadFunction`).

**Packaging.** The package is not installable from a wheel. `pyproject.toml`
has no `package-data` entry, so none of the shipped CSV and JSON tables would
ship. Two tables the code reads are untracked (`proposal_muon_rock.csv`,
`trident_2025_fig3a_log10aeff_m2.csv`), and `data/bounds/*.csv` is ignored
while `tests/test_bounds.py` reads it. There is no way to point the package at
the 2.5 GB DR2 release except placing it inside `src/softpaws/data/`. The
README sends readers to a private sibling project for the data.

**Examples.** `examples/` holds 80 scripts and 33,500 lines, 3.6 times the
library. They are not examples. Eleven of them are libraries in disguise
(`35`, `45`, `46` alone have 13, 13 and 11 dependents), loaded by 40 copies of
an `importlib` helper. The site geometry for six detectors is declared in three
incompatible forms across twelve files, with two real conflicts (IceCube radius
0.62 vs 0.564 km, Gen2 8.0 km³ x 1.0 km vs 7.9 km³ x 1.25 km). Around 60
physics helpers are duplicated two to five times. Ten scripts treat
`examples/output/` as a cache input.

**Paper.** `docs/main.tex` uses 12 generated figures from 6 scripts (37, 74,
75, 76, 82, 83), one machine-written table (`make_recipe_table.py` from 77 and
82), and quoted numbers from about 15 more scripts' run logs. Building the
paper from scratch needs 27 scripts in a fixed order, three cached chains, MCEq
and PROPOSAL. Nothing records that order.

**Docs.** There is no docs site. `docs/` mixes the manuscript, its backups, four
derivation notes, three style guides, referee reports and six revision plans.

## 2. Target layout

```
softpaws/
├── pyproject.toml                PEP 621 + setuptools, package-data, extras
├── mkdocs.yml                    readthedocs theme, mkdocstrings, api-autonav
├── README.md  CONTRIBUTING.md  CODE_OF_CONDUCT.md  LICENSE  CITATION.cff  CHANGELOG.md
├── .github/workflows/            ci.yml (ruff + pytest), deploy-mkdocs.yml
├── src/softpaws/
│   ├── utils/                    constants, units, data-root lookup
│   ├── data/                     DR2/HESE/KM3NeT/P-ONE/TRIDENT loaders + shipped tables
│   ├── transport/                kernel, exponent, ranges, earth, tau, loss law
│   ├── detectors/                one Site dataclass and a registry (new)
│   ├── fluxes/                   power law, broken power law, atmospheric table (new)
│   ├── response/                 effective area engine, soft-volume and IRF paths, reduced response
│   ├── comparison/               likelihood, MCMC and Feldman-Cousins helpers, event benchmark
│   └── styles/                   beacom_conformal.mplstyle as package data
├── examples/                     01 to ~11, short tutorials, each under a minute
├── paper/                        manuscript sources (gitignored as today)
├── scripts/
│   └── 2026_muon_transport/      one script per paper figure, table and number block
├── tests/                        unit tests + regression fixtures
└── docs/                         mkdocs source: guides, theory notes, reproduction
```

Names of the new library modules follow what the code does, in plain words.
Nothing in the library plots. Figure code stays in `examples/` and `scripts/`.

## 3. Phases

### Phase 0. Freeze and baseline (done 2026-09-03)

Nothing gets refactored until the current outputs are pinned.

1. Done. The snapshot commit is `6bffc38`, tagged `pre-cleanup`.
2. Done. The manuscript moved out of `docs/` into `/paper/`, which
   `.gitignore` excludes whole. `docs/` becomes the docs-site source.
3. Done. Root and `examples/` scratch files removed. The `*.pre-*` backups
   went to `/paper/` with the manuscript; `src/softpaws.egg-info/` stays
   because the editable install uses it, and git ignores it.
4. Done. `tests/regression/baseline.json` holds 25 blocks, each with a
   source and a tolerance, and `test_baseline.py` checks the kernel
   moments, Table D.1, Table D.2 and the Table C.1 rows. Two findings came
   out of writing it, both stale paper numbers, not library bugs:
   - Table C.1's range rows were generated at commit `0ce30cb` (2026-08-07),
     before the running-kernel range. The current library sits about 0.6 km
     away at every energy. The test pins today's library values; the table
     is regenerated from the library before submission.
   - The water Phi'(0) values in Tables C.1 and E.2 are 0.7 to 1.0% below the
     current water table (0.424 vs 0.428 at 10 TeV, 0.484 vs 0.487 at 1 PeV).
     The rock values match, so the water table was regenerated after they
     were typed. Same fix.
5. Done. The `examples/output/*.npz` and `*.json` caches are copied to
   `scripts/2026_muon_transport/cache/` (gitignored, 72 MB).

### Phase 1. Lift the hub examples into the library (the main work, 4 to 6 days)

This is where the 33,500 lines shrink. Work bottom-up through the dependency
graph, one hub at a time. For each hub: move the physics into a library
module, write docstrings as it moves, add a unit test that pins its numbers
against `baseline.json`, then rewrite the consumers to import from the
library and delete the private copy. Never move two hubs in one commit.

| Step | New module | Lifted from | Kills |
|---|---|---|---|
| 1.1 done | `detectors/sites.py`, `detectors/optics.py` | `34`, `35`, `45` (`Site`), `33`, `56` (`Detector`) | three dataclasses, two `build_sites`, ~60 loose constants in 8 files; IceCube 0.564 km prism, Gen2 7.9 km^3 by 1.25 km; example 35 reproduces its log exactly. `33`/`56`'s fit `Detector` waits for 1.8 |
| 1.2 done | `data/icecube.py` | `03`, `04`, `07`, `28`, `29`, `31`, `33` | `icecube_upgoing` x4, `load_ic86_observed_and_livetime` x5, `_canonical_irf_season` x9, `combine_seasons`, `IC86_SEASONS` x6 in 15 scripts; example 03's curve unchanged |
| 1.3 done | `transport/earth.py` | `attenuation.py` (PREM, chord, column) + `30`, `31`, `32`, `33`, `56` (`earth_column_g_cm2`, `upstream_column_km`, `zenith_grid`, `arca_columns`, `water_columns`) | the five-way column duplication; `attenuation.py` re-exports the moved names. The two `column_depth` units are still open |
| 1.4 | `response/effective_area.py` | `30`, `31`, `32`, `45` (`truncated_range_km`, `projected_area_km2`, `fit_reach_law`, `arca_effective_area`, `ic_effective_area_*`, reach derivation) | the engine that `41`, `43`, `44`, `45` reach through `32` for |
| 1.5 | `response/declination.py` | `35`, `46`, `47` (`directional_aeff_cm2`, band-by-band evaluation, point-source ceiling) | the two largest hubs |
| 1.6 done | `fluxes/astrophysical.py`, `fluxes/atmospheric.py` | `06`, `07`, `12`, `14`, `21`, `22`, `49`, `51`, `54`, `57` (`power_law_flux`, `bpl_shape`, the IceCube fits, MCEq builder and interpolator) | `PHI0`/`GAMMA` in 6 files, `bpl_shape` x2, three copies of the atmospheric grid reader; grids and anchors of 49/51/54 unchanged |
| 1.7 done | `transport/loss_ensemble.py` + a `KernelScaling` hook in `coefficients.py` | `69`, `70` (the variant ensemble; `70`'s `sys.modules` monkeypatch becomes `set_kernel_scaling`), `72` (implied scales), `71`/`74` (variant activation) | the patch walker; drift, second moment and range per variant identical to the old route |
| 1.8 | `response/reduced.py` | `77`, `78`, `81` (two-number reduced response, optics-predicted reach, TRIDENT 2025 map) | `78` to `83` collapse to plotting |
| 1.9 | `comparison/mcmc.py`, `comparison/feldman_cousins.py` | `29`, `33`, `51`, `54`, `56`, `72` | `log_probability` x2, `summarize` x5, `fc_calibration` x2, `bayes_factor` x4 |
| 1.10 | `comparison/events.py` | `51`, `76` (upgoing IC86 window, pinned-flux prediction, published-IRF baseline) | the headline benchmark becomes one function call |
| 1.11 done | `comparison/event_energy.py` | `57` (potential density, measurement, energy likelihood, posterior, summary), `68` (two-layer column, survival, unity energy), `15` (parent-energy posterior, quantile); the `TrackEvent` record for KM3-230213A | the three private copies; the tension ladder of `57` Part B stays with `31`'s likelihood engine for step 1.9 |

Not lifted: the BSM scripts `59` to `66` (stau, millicharge). They belong to a
later paper. They move to `scripts/future_bsm/` untouched and ruff-excluded
in Phase 3. Their `load_example` calls keep working there only if the hubs
they load (`51`, `35`, `45`, `46`, `57`, `63`) are copied alongside them, so
the move takes the hub scripts' pre-cleanup versions with it.

Exit criterion: no `load_example` call remains anywhere, and every entry in
`baseline.json` passes.

### Phase 2. Library cleanup and packaging (2 to 3 days)

1. **Split the two large modules.** `transport/soft_volume.py` into
   `geometry`, `closed_form`, `ranges` (the 930-line range block first),
   `target_volume`, `exact`. `response/soft_volume.py` into `flux` (moves to
   `fluxes/`), `soft_volume`, `tau`. Keep the old import paths working
   through the package `__init__` for one release.
2. **Remove the duplicates inside the library.** One `TOTAL_TO_CC_RATIO`
   in `utils/constants.py`. `source.cc_cross_section` becomes a wrapper on
   `PowerLawCrossSection`. The three `expected_counts_*` variants apply
   attenuation through one function.
3. **Delete or move.** `comparison/datasets.py` (unreachable). The
   `extract_flavor2510_fig1.py` digitizer moves to `scripts/tools/`.
   `PointSpreadFunction` is dropped now, before there are users.
   `log_posterior` and `initial_walkers` become private.
4. **Docstrings.** Add the 12 missing `__init__` and dunder docstrings, the 6
   missing parameters in `transport/soft_volume.py`, the 5 `Returns` in
   `comparison/likelihood.py`. Every new module from Phase 1 is written to
   the skill from the start. Then turn on ruff `D` rules with
   `convention = "numpy"` so CI enforces the skill. Adopt the prometheus
   text rules on top: present tense, "create" or "build" not "make", one
   shell command per block.
5. **`transport/__init__.py`** gets the same `__all__` re-exports as the
   other subpackages. The top-level `softpaws/__init__.py` exposes the ten
   entry points a user needs (`Site` registry, `effective_area`,
   `SoftVolumeResponse`, `phi_eigenvalue`, `muon_range_km`, loaders).
6. **Packaging.** Add

   ```toml
   [tool.setuptools.package-data]
   softpaws = ["data/**/*.csv", "data/**/*.json", "data/**/*.geo", "styles/*.mplstyle"]
   ```

   Commit `proposal_muon_rock.csv`, the TRIDENT 2025 map and
   `data/bounds/*.csv`. Add extras `docs`, `atm`, `transport`, `dev`. Verify
   with `python -m build` and `unzip -l dist/*.whl`, then `pip install` into
   a fresh venv from a clean clone and run the tests.
7. **Data root.** Add `softpaws.data.data_root()` that reads
   `SOFTPAWS_DATA_DIR`, falling back to the package directory. Loaders raise
   a clear error naming the DOI and the expected layout when a file is
   missing. No download code. Remove every mention of `neutrino_subhalos`.
8. **`tests/conftest.py`** copied from prometheus: chdir to the repo root and
   `--run-slow` gating. Mark the MCMC and PROPOSAL tests slow.

### Phase 3. Paper scripts (1 to 2 days, after Phase 1)

`scripts/2026_muon_transport/` reproduces the paper and nothing else. Rename
the directory to the arXiv number on submission.

```
scripts/2026_muon_transport/
├── README.md              run order, runtime, what needs MCEq or PROPOSAL
├── run_all.py             runs the steps below in order, skips cached ones
├── cache/                 chains and tables (gitignored), written by the steps
├── _figures.py            style loading and the save-to-pdf-and-png helper
├── 01_kernel_tables.py            22 + 27 + 69: MCEq table, PROPOSAL benchmark, loss ensemble
├── 02_site_fits.py                33 → 56 → 72 → 73 → 77 → 81 → 82: the chains
├── fig02_transport_exponent.py    from 37
├── fig03_four_detector_aeff.py    from 83
├── fig04_instrument_plane.py      from 82
├── fig05_event_spectrum.py        from 76 (also J.1, J.2)
├── fig06_point_source_ceiling.py  from 74 (also H.1, I.1)
├── fig07_km3_event_energy.py      from 75 (also fig08, K.1)
├── tabC1_range_validation.py      from 39, 48
├── tabD1_truncations.py           from 27, 36, 37
├── tabE1_recipe.py                from docs/make_recipe_table.py
└── numbers.py                     every number quoted in the text, printed as JSON
```

Each figure script is plotting plus a few library calls. The two heavy steps
own all caching, so a figure script never runs a chain. `numbers.py` replaces
`collect_numbers.py` and the run-log parsing: it computes the numbers from the
library and writes `numbers.json`, which the paper checks against.

The old `examples/` scripts are deleted in this phase. Git history and the
`pre-cleanup` tag keep them.

### Phase 4. Examples (1 day)

Short tutorials that run in under a minute with no cache, numbered from 01.
Each starts with a module docstring, an `argparse` block built from that
docstring, and writes to `examples/output/`. Proposed set:

| # | Title | Shows | From |
|---|---|---|---|
| 01 | Load the DR2 release | events, effective area, smearing matrix, livetime | 01 to 04 |
| 02 | The transport exponent | Phi(A) for a kernel, the truncations, PROPOSAL quadrature | 05, 36, 37 |
| 03 | Muon range and loss law | first-passage range, the loss distribution, moments | 13, 39 |
| 04 | Earth attenuation | chord, PREM column, transmission with regeneration | 14 |
| 05 | An effective area from first principles | one site from the registry, energy and zenith | 28, 30, 45 |
| 06 | Effective area against declination | band-by-band comparison to DR2, point-source ceiling | 35, 47 |
| 07 | Event rates, two ways | soft-volume path against the IRF path on IC86 | 07, 12, 76 |
| 08 | Fitting the instrument numbers | threshold and reach on one site with emcee, short chain | 33, 77 |
| 09 | The energy of one event | parent-energy posterior for a single track | 15, 17, 57 |
| 10 | Tau-induced tracks | the tau channel and its share of the track rate | 16, 49 |
| 11 | Atmospheric background with MCEq | building the flux table, requires the `atm` extra | 22 |

Add a `tests/test_examples.py` that runs each one with `--quick` so CI catches
drift.

### Phase 5. Documentation site (1 to 2 days)

Copy the prometheus setup: MkDocs, `readthedocs` theme, `mkdocstrings` with
`docstring_style: numpy`, `mkdocs-api-autonav` generating the API reference
from the package, `docs/requirements.txt` pinned separately, and
`deploy-mkdocs.yml` running `mkdocs gh-deploy --force` on push to main.

```
docs/
├── index.md               what softpaws is, in one paragraph shared with the README
├── installation.md        pip install, extras, fetching the DR2 release
├── quickstart.md          five calls: site, exponent, effective area, rate, fit
├── data.md                what ships in the wheel, what is fetched, provenance and DOIs
├── theory/                the four derivation notes, restyled, plus the paper-to-code map
├── examples.md            one paragraph per example
├── reproduce.md           how to rebuild the paper with scripts/2026_muon_transport
├── citation.md            BibTeX for the paper and the dependencies
└── css/, js/              copied from prometheus
```

Prose follows the Microsoft style guide and the prometheus text skill: second
person, present tense, short sentences, one command per code block. Copy the
three prometheus skills into `.claude/skills/` so the rules travel with the
repository, and point `CONTRIBUTING.md` at them.

`ci.yml` runs `ruff check`, `ruff format --check` and `pytest -m "not slow"`
on Python 3.11 to 3.13.

### Phase 6. Release (half a day)

README rewritten to the prometheus structure (badges, link table, summary,
citation, contributing, getting help). `CITATION.cff`. `CHANGELOG.md` with a
`0.1.0` entry. Version in `pyproject.toml` and `__init__.py` from one source.
Tag `v0.1.0`. Clean-clone install and `run_all.py` as the acceptance test.
PyPI is wired but not triggered: `publish.yml` uploads on a version tag
through trusted publishing once the project is registered there, and the
README install line stays `pip install git+https://...` until then.

## 4. Decisions (settled 2026-09-03)

1. **Manuscript location.** `/paper/` at the repository root, ignored by git
   and never synced. Done: the manuscript, its backups, figures, run logs,
   plans, referee reports, style guides and the three paper tools
   (`collect_numbers.py`, `make_recipe_table.py`, `measure_style.py`) now
   live there. The four derivation notes stay in `docs/` and feed the site.
2. **The BSM scripts (59 to 66).** Move to `scripts/future_bsm/` untouched,
   excluded from ruff and CI, with a one-line README saying they wait for the
   next paper.
3. **PyPI.** GitHub-only for now, but everything set up for PyPI: PEP 621
   metadata complete (classifiers, URLs, keywords), `python -m build` clean,
   `twine check` clean, a `publish.yml` workflow that uploads on a `v*` tag
   through trusted publishing, and the name `softpaws` registered on TestPyPI
   first.
4. **The paper-script directory name.** `2026_muon_transport` until the arXiv
   number exists, then rename.
5. **IceCube data.** The user fetches the DR2 release and the HESE 7.5-year
   files themselves. The package ships no fetch command. `installation.md`
   and `data.md` give the DOIs, the expected directory layout
   (`events/`, `irfs/`, `uptime/`), and the `SOFTPAWS_DATA_DIR` variable that
   points at it. The two small `bounds/` CSVs ship in the wheel.

## 5. Order and effort

| Phase | Days | Depends on |
|---|---|---|
| 0 Freeze and baseline | 0.5 | nothing |
| 1 Lift the hubs | 4 to 6 | 0 |
| 2 Library cleanup and packaging | 2 to 3 | 1 (steps 2.6 to 2.8 can start after 0) |
| 3 Paper scripts | 1 to 2 | 1 |
| 4 Examples | 1 | 1 |
| 5 Docs site | 1 to 2 | 2 (the skeleton can start after 0) |
| 6 Release | 0.5 | all |

Around two weeks of focused work. Phase 1 is the risk. It is bounded by the
regression file from Phase 0: if a number moves, the move is a bug or a
discovered inconsistency (the IceCube radius is the first candidate), and
either way it is found before the paper ships.
