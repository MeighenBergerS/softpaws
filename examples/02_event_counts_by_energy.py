"""Example 02 — event counts vs. energy, split by hemisphere.

Loads all IceTracks-DR2 events and histograms the reconstructed-energy proxy
separately for upgoing (``dec > 0``, Earth-traversing) and downgoing
(``dec < 0``) events.

Usage
-----
    python examples/02_event_counts_by_energy.py
    python examples/02_event_counts_by_energy.py --data-dir /path/to/dataverse_files
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data.container import EventSet
from softpaws.data.loader import load_all_seasons

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"

LOG10_E_EDGES = np.linspace(2.0, 8.0, 31)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=_DEFAULT_DATA_DIR,
        help="Root of the DR2 data directory (must contain an 'events/' subfolder).",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR / "02_event_counts_by_energy.pdf",
        help="Output file for the figure.",
    )
    return parser.parse_args()


def make_figure(events: EventSet, out_path: pathlib.Path) -> None:
    upgoing = events.filter_dec(0.0, 90.0)
    downgoing = events.filter_dec(-90.0, 0.0)

    counts_up, _ = np.histogram(upgoing.log10_energy, bins=LOG10_E_EDGES)
    counts_down, _ = np.histogram(downgoing.log10_energy, bins=LOG10_E_EDGES)

    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3, 3))

        ax.stairs(counts_up, LOG10_E_EDGES, label="upgoing")
        ax.stairs(counts_down, LOG10_E_EDGES, label="downgoing")

        ax.set_yscale("log")
        ax.set_xlabel(r"$\log_{10}(E\,/\,\mathrm{GeV})$")
        ax.set_ylabel("events per bin")
        ax.legend()

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def main() -> None:
    args = parse_args()

    print(f"Loading data from: {args.data_dir}")
    raw = load_all_seasons(args.data_dir)
    events = EventSet(raw)
    print(f"Total events: {events.n_events:,}")

    make_figure(events, args.out)


if __name__ == "__main__":
    main()
