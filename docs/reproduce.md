# Reproducing the paper

Every figure, table and quoted number in *Analytical High-Energy Muon
Transport for Neutrino Telescopes* is produced by a script in this
repository, and the numbers the paper depends on are pinned in
`tests/regression/baseline.json`.

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
the scripts previously hid. Two blocks are already known to be stale, and are
regenerated rather than trusted: the range rows of Table C.1 and the water
log-loss rate of Tables C.1 and E.2 predate the current kernel table, and the
two-flavour run log of the reach example predates a column the script now
prints. Both are noted in the fixture.

## What you need

The tabulated inputs ship with the package. Beyond them you need the
IceTracks-DR2 release for anything touching IceCube data (see
[Data](data.md)), MCEq once to tabulate the atmospheric background, and
PROPOSAL for the kernel benchmark and the loss-model ensemble:

```sh
pip install -e ".[atm,transport]"
```

## The order

Several steps are expensive and cache their result under
`examples/output/`, and later steps read those caches. The heavy ones are the
atmospheric table, the loss-model ensemble, and the four-site posterior
chains; each runs in minutes to hours and then costs nothing.

The chain runs: the atmospheric table and the kernel benchmark first, then
the effective-area comparisons, then the site fits, then the loss-model
ensemble, and finally the figures that read all of them.

!!! note

    The paper's figures are being moved out of `examples/` into a directory of
    their own, one script per figure with a `run_all.py` that runs them in
    order and skips whatever is already cached. Until then, the order above is
    the recipe, and the run logs of the previous build are kept alongside the
    manuscript.
