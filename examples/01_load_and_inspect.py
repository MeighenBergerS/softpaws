"""Example 01 — load IceTracks-DR2 events and print a summary.

Loads all IceTracks-DR2 season files and prints basic sanity-check
statistics (event count, MJD range, energy range, declination range).

Usage
-----
    python examples/01_load_and_inspect.py
    python examples/01_load_and_inspect.py --data-dir /path/to/dataverse_files
"""

import argparse
import pathlib

from softpaws.data.container import EventSet
from softpaws.data.loader import load_all_seasons

_DEFAULT_DATA_DIR = (
    pathlib.Path(__file__).parent.parent
    / "src" / "softpaws" / "data" / "dataverse_files"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=_DEFAULT_DATA_DIR,
        help="Root of the DR2 data directory (must contain an 'events/' subfolder).",
    )
    return parser.parse_args()


def print_summary(events: EventSet) -> None:
    print(f"Total events     : {events.n_events:>10,}")
    print(f"MJD range        : {events.time.min():.2f} - {events.time.max():.2f}")
    print(f"log10(E/GeV)     : {events.log10_energy.min():.2f} - {events.log10_energy.max():.2f}")
    print(f"Dec range    [deg]: {events.dec.min():.2f} - {events.dec.max():.2f}")


def main() -> None:
    args = parse_args()

    print(f"Loading data from: {args.data_dir}")
    raw = load_all_seasons(args.data_dir)
    events = EventSet(raw)
    print_summary(events)


if __name__ == "__main__":
    main()
