"""Run the paper's scripts in dependency order.

Each step writes to ``output/`` and later steps read those caches, so the
order matters and a rerun of a finished step is wasted work. A step whose
outputs are already present is skipped unless ``--force`` is given.

Usage
-----
    python scripts/2026_muon_transport/run_all.py --dry-run
    python scripts/2026_muon_transport/run_all.py
    python scripts/2026_muon_transport/run_all.py --from 69 --force
"""

import argparse
import pathlib
import subprocess
import sys
import time

_HERE = pathlib.Path(__file__).parent
_OUT = _HERE / "output"

#: The scripts, in dependency order, with a file each one writes. The marker
#: is what ``--skip-done`` looks for; a step with no marker always runs.
STEPS: tuple[tuple[str, str | None], ...] = (
    ("22_atmospheric_background_mceq.py", "22_mceq_atmospheric_flux.npz"),
    ("27_proposal_cross_section_and_loss.py", None),
    ("32_effective_area_comparison.py", None),
    ("35_point_source_effective_area.py", None),
    ("45_first_principles_reach.py", None),
    ("46_declination_resolved_reach.py", None),
    ("47_point_source_derived_reach.py", None),
    ("55_pone_trident_prediction.py", None),
    ("33_two_detector_posterior_corner.py", "33_chains.npz"),
    ("56_four_detector_posterior_corner.py", "56_chains.npz"),
    ("69_loss_model_error_budget.py", "69_ensemble.npz"),
    ("51_dr2_flavor_fit.py", "51_fit_inputs.npz"),
    ("31_flux_contours_effective_area.py", None),
    ("57_km3_event_energy_and_bpl_tension.py", None),
    ("70_aeff_error_bands.py", None),
    ("71_km3_tension_loss_error.py", None),
    ("72_informed_transport_corner.py", None),
    ("73_four_detector_aeff_fit.py", "73_chains.npz"),
    ("77_reduced_response_fit.py", "77_chains_sigma05.npz"),
    ("81_trident_2025_map_fit.py", "81_trident_2025_map_fit.json"),
    ("82_reduced_response_plane_trident2025.py", "82_trident2025_chain.npz"),
    ("83_four_detector_aeff_reduced.py", None),
    ("74_point_source_with_bands.py", None),
    ("84_point_source_background_limited.py", None),
    ("75_km3_figures_with_bands.py", None),
    ("76_dr2_event_benchmark.py", None),
    ("37_moment_convergence.py", None),
)

#: Written after every script: the recipe table from the chains of 77 and 82,
#: and the transport table straight from the library.
TABLE_WRITERS = ("make_recipe_table.py", "make_transport_table.py")


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the order and stop.")
    parser.add_argument("--force", action="store_true",
                        help="Run every step, even one whose cache exists.")
    parser.add_argument("--from", dest="start", default=None,
                        help="Start at the step whose name begins with this, "
                             "for example '69'.")
    parser.add_argument("--only", default=None,
                        help="Run only the step whose name begins with this.")
    parser.add_argument("--recipe-table", action="store_true", default=True,
                        help="Write the paper's machine-written tables at the "
                             "end when the manuscript directory is present.")
    return parser.parse_args()


def select(args: argparse.Namespace) -> list[tuple[str, str | None]]:
    """The steps to run, after the start and only filters."""
    steps = list(STEPS)
    if args.only is not None:
        return [s for s in steps if s[0].startswith(args.only)]
    if args.start is not None:
        for i, (name, _) in enumerate(steps):
            if name.startswith(args.start):
                return steps[i:]
        raise SystemExit(f"No step starts with {args.start!r}.")
    return steps


def main() -> None:
    """Run the steps and report how long each took."""
    args = parse_args()
    steps = select(args)

    print(f"{len(steps)} steps, writing to {_OUT}\n")
    for i, (name, marker) in enumerate(steps, start=1):
        done = marker is not None and (_OUT / marker).exists() and not args.force
        print(f"{i:3d}. {name:48s} {'(cached)' if done else ''}")
    if args.dry_run:
        print("\nDry run; nothing was executed.")
        return

    _OUT.mkdir(parents=True, exist_ok=True)
    for i, (name, marker) in enumerate(steps, start=1):
        if marker is not None and (_OUT / marker).exists() and not args.force:
            print(f"\n[{i}/{len(steps)}] {name}: cached, skipping")
            continue
        print(f"\n[{i}/{len(steps)}] {name}")
        started = time.monotonic()
        result = subprocess.run([sys.executable, str(_HERE / name)], check=False)
        elapsed = time.monotonic() - started
        if result.returncode != 0:
            raise SystemExit(f"{name} exited with {result.returncode} after {elapsed:.0f} s.")
        print(f"    done in {elapsed:.0f} s")

    for writer in (_HERE / name for name in TABLE_WRITERS):
        if args.recipe_table and writer.exists():
            print(f"\nWriting a table with {writer} ...")
            subprocess.run([sys.executable, str(writer)], check=True)
    print("\nAll steps finished.")


if __name__ == "__main__":
    main()
