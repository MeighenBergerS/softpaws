# Reproducing the paper

Every figure, table and quoted number in *Analytical High-Energy Muon
Transport for Neutrino Telescopes* is produced by a script in
`scripts/2026_muon_transport/`, and the numbers the paper depends on are
pinned in `tests/regression/baseline.json`.

## Running it

```sh
python scripts/2026_muon_transport/run_all.py --dry-run
```

That prints the 27 steps in dependency order and says which are already
cached. Drop the flag to run them. A step whose outputs sit in
`scripts/2026_muon_transport/output/` is skipped unless you pass `--force`,
which matters because several take hours.

The directory's own README maps each paper figure and table to its script.

## What you need

The tabulated inputs ship with the package. Beyond them you need the
IceTracks-DR2 release for anything touching IceCube data (see
[Data](data.md)), MCEq once to tabulate the atmospheric background, and
PROPOSAL for the kernel benchmark and the loss-model ensemble:

```sh
pip install -e ".[atm,transport]"
```

## The pinned numbers

The regression fixture records what the paper quotes, with the source and a
tolerance for each block: the kernel moments, the transport exponent against
its truncations, the loss law, the range to threshold, the four-detector
instrument numbers, the event benchmark, the KM3-230213A energies and
tensions, the declination bands and the point-source ceilings.

Check them against the library:

```sh
pytest -q --run-slow tests/regression
```

A value that moves outside its tolerance is either a bug or an inconsistency
the scripts previously hid. The transport rows of Tables C.1 and E.2 are
machine-written from the library by `make_transport_table.py`, and the
plug-in response of Table E.1 by `make_recipe_table.py`, both in the scripts
directory, so those tables cannot drift from the code.

## Two scripts do not reproduce run to run

Scripts 79 and 82 refit by Markov chain, and the sampler draws its moves from
an unseeded generator. Their parameter values move by a fraction of their own
uncertainty between runs; every deterministic quantity they print, including
the deviance and the residual scatter, is stable. This predates the cleanup.

## Beyond the standard model

`scripts/future_bsm/` holds the stau and millicharged-particle searches. They
belong to a later paper, are not part of the tested surface, and their numbers
are not pinned here.
