# Examples

The `examples/` directory holds runnable scripts. Each one writes its figures
to `examples/output/` and prints the numbers it computes, so the printed
output is the result and the figure is the illustration.

Run one like this:

```sh
python examples/05_soft_volume_drift.py
```

Most take `--out-dir` for the figures and `--data-dir` for the IceCube
release; pass `--help` to see what a script accepts.

## What they cover

**The transport.** How the loss kernel behaves, what the transport exponent
is, how far a muon travels before it drops below a threshold, and how the
closed forms compare with a direct Monte-Carlo propagation.

**The response.** How an effective area is assembled from the transport,
what the light reach adds, and how the result compares with the published
tables of IceCube, KM3NeT/ARCA, P-ONE and TRIDENT, band by band in
declination.

**The measurements.** The event rate of the IceCube through-going sample
with nothing fitted, the flavour composition it constrains, the energy of a
single track, and the loss-model error on all of it.

## A note on the current numbering

The scripts are being reorganized. Today they are numbered in the order they
were written, several are analysis machinery rather than examples, and some
load one another by file path. The plan is a short set of tutorials numbered
from `01`, with the paper's figures moved to their own directory; see
[Reproducing the paper](reproduce.md). The library API those scripts call is
already stable, and is what this documentation describes.
