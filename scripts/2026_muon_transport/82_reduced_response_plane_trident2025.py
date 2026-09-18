"""Example 82 -- example 78's plane with TRIDENT taken from its 2025 map.

Example 78 puts the two instrument numbers of the four published tables on
one plane, and TRIDENT's 2022 sky average returns a negative reach that
examples 79-81 trace to its selection. This example replaces that entry by
a fit to the 2025 effective-area map of Morton-Blake et al.
(arXiv:2510.24395, Fig. 3a), restricted to the cells where their selection
is flat, ``|cos(theta_z)| <= 0.5``, and to the decade the two TRIDENT
tables share, 10^5 to 10^6 GeV. The map is published after trigger and
quality cuts, so the selection normalization ``eps_0`` is left free and
the ``(E_thr, Lambda)`` contour is marginalized over it. The cell error is
the rescaled value of example 81, 0.086 dex, which absorbs the map's
Monte Carlo scatter. The other three sites keep example 77's 5% chains.

Usage
-----
    python scripts/2026_muon_transport/82_reduced_response_plane_trident2025.py
    python scripts/2026_muon_transport/82_reduced_response_plane_trident2025.py --steps 1500
"""

import argparse
import importlib.util
import pathlib

import numpy as np

from softpaws._paper import reduced

_HERE = pathlib.Path(__file__).parent
_DEFAULT_OUT_DIR = _HERE / "output"


def load_example(stem: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EX81 = load_example("81_trident_2025_map_fit.py", "_example_81")
_EX78 = load_example("78_reduced_response_plane.py", "_example_78")
_EX77 = _EX81._EX77
_EX33 = _EX81._EX33

COS_MAX = 0.5
LOG10_E_MIN = 5.0
SIGMA_DEX = 0.086


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--walkers", type=int, default=16)
    parser.add_argument("--chains", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "77_chains_sigma05.npz")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR)
    return parser.parse_args()


def trident_2025_detector():
    """The 2025 map's flat-selection cells as one detector, and the cell count.

    A thin wrapper on
    :func:`softpaws._paper.reduced.trident_2025_detector`.
    """
    return reduced.trident_2025_detector(COS_MAX, LOG10_E_MIN)


def main() -> None:
    args = parse_args()
    detector, n = trident_2025_detector()
    fixed = {"eps_0": 0.7, "log10_e_thr": 2.5, "b_scale": 1.0, "lam": _EX33.LAMBDA_BGR18,
             "reach_km": 0.03}
    sigma_ln = SIGMA_DEX * np.log(10.0)
    print(f"TRIDENT 2025 map: {n} cells, |cos| <= {COS_MAX}, log10 E >= {LOG10_E_MIN}; "
          f"(eps_0, E_thr, Lambda) free, {SIGMA_DEX} dex per cell")
    best, chain = _EX77.fit(detector, _EX77.FREE3, fixed, sigma_ln, args.steps, args.walkers, 41)
    theta = _EX77.full_theta(_EX77.FREE3, best, fixed)
    dev, res = _EX77.deviance(detector, theta, sigma_ln)
    print(f"  deviance {dev:.1f} / {n - 3} dof, rms {np.std(res)/np.log(10):.3f} dex")
    for k, name in enumerate(_EX77.FREE3):
        lo, med, hi = np.percentile(chain[:, k], [16, 50, 84])
        if name == "reach_km":
            print(f"  Lambda: median {1e3*med:+.1f} m [{1e3*lo:+.1f}, {1e3*hi:+.1f}]")
        elif name == "log10_e_thr":
            print(f"  E_thr : median {10**med:.0f} GeV [{10**lo:.0f}, {10**hi:.0f}]")
        else:
            print(f"  eps_0 : median {med:.3f} [{lo:.3f}, {hi:.3f}]")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "82_trident2025_chain.npz", chain=chain, best=theta,
             free=np.array(_EX77.FREE3))

    chains, sigma = _EX78.load_chains(args.chains)
    chains["TRIDENT"] = np.column_stack([chain[:, 1], 1.0e3 * chain[:, 2]])
    _EX78.LEVELS["TRIDENT"] = r"2025 map, $\varepsilon_0$ free"
    _EX78.summarize(chains, sigma)
    _EX78.make_figure(chains, sigma, args.out_dir / "82_reduced_response_plane_trident2025.pdf")


if __name__ == "__main__":
    main()
