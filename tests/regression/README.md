# Regression baseline

`baseline.json` pins every number the paper quotes, with the script or table
that produced it. It was written on 2026-09-03 from the `pre-cleanup` tag,
before the library refactor started.

Each block carries a `source` line and a tolerance (`rtol` or `atol`). Values
come from three places:

- the paper's tables (Tables C.1, D.1, D.2, E.1, E.2, G.1, K.1),
- the captured run logs of the examples (`paper/run_logs/`),
- the JSON results the examples wrote to `examples/output/`.

`test_baseline.py` checks the blocks the library can compute directly. As the
example hubs move into the library during the cleanup, each new module gets a
test against its block here. A value that moves outside its tolerance is
either a bug or an inconsistency the old scripts hid. Either way, fix the
cause or update the paper. Never edit a baseline value to make a test pass.
