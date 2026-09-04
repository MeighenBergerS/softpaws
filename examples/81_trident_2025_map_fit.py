"""Example 81 -- the reduced response against TRIDENT's 2025 effective-area map.

Morton-Blake et al. (arXiv:2510.24395, Fig. 3a) publish the nu_mu CC track
effective area of the reference TRIDENT layout (1000 strings at 100 m, 20
hDOMs at 30 m) on a 12 x 12 grid in cos(theta_z) and log10 E, after trigger,
edge and track-extension cuts. The map was digitized from the embedded
raster (``data/trident/trident_2025_fig3a_log10aeff_m2.csv``). This example
fits it with the physics fixed (``b_scale = 1``, ``lam`` at BGR18) and the
three instrument numbers free: the selection normalization ``eps_0``, the
threshold and the reach. Because the map carries a direction-dependent
selection that no single ``eps_0`` can absorb, the fit is repeated on four
cell selections, and the residual map of the full fit is drawn.

Usage
-----
    python examples/81_trident_2025_map_fit.py
    python examples/81_trident_2025_map_fit.py --sigma-dex 0.043
"""

import argparse
import importlib.util
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.response import reduced

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EX77 = load_example("77_reduced_response_fit.py", "_example_77")
_EX56 = _EX77._EX56
_EX33 = _EX77._EX33

# ---------------------------------------------------------------------------
# The map model now lives in softpaws.response.reduced. The names below are
# re-exported so the sibling examples that load this script by path keep
# resolving; Phase 3 of the cleanup retires that helper and this block.
# ---------------------------------------------------------------------------

RADIUS_KM = reduced.MAP_RADIUS_KM
COS_EDGES = reduced.COS_EDGES
PARAM_BOUNDS = reduced.PARAM_BOUNDS

load_map = reduced.trident_2025_cells
MapModel = reduced.MapModel
fit_cells = reduced.fit_cells


def parse_args() -> argparse.Namespace:
    """Command-line arguments; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sigma-dex", type=float, default=0.043,
                        help="Assumed error per digitized cell [dex] (10%% by default).")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cos_c, log10_e_all, log10_a_all = load_map()
    keep = log10_e_all >= _EX33.ARCA_LOG10_E.min()
    log10_e, log10_a = log10_e_all[keep], log10_a_all[:, keep]
    model = MapModel(log10_e)
    hz = np.abs(cos_c) <= 0.5
    selections = {
        "all cells": np.ones_like(log10_a, bool),
        "|cos| <= 0.5": np.broadcast_to(hz[:, None], log10_a.shape),
        "upgoing, cos < -0.2": np.broadcast_to((cos_c < -0.2)[:, None], log10_a.shape),
        "downgoing, cos > 0.2": np.broadcast_to((cos_c > 0.2)[:, None], log10_a.shape),
    }
    print(f"TRIDENT 2025 map, {log10_a.size} cells over "
          f"10^{log10_e[0]:.2f}-10^{log10_e[-1]:.2f} GeV; "
          f"physics fixed, (eps_0, E_thr, Lambda) free, {args.sigma_dex} dex per cell")
    out, resid_all = {}, None
    for name, cells in selections.items():
        x, chi2, (lo, hi), resid = fit_cells(model, log10_a, cells, args.sigma_dex)
        n = int(cells.sum())
        print(f"\n=== {name}: {n} cells ===")
        print(f"  chi2 {chi2:.1f} / {n - 3} dof; eps_0 {x[0]:.3f}, E_thr {10**x[1]:.0f} GeV, "
              f"Lambda {1e3*x[2]:+.1f} m [{1e3*lo:+.1f}, {1e3*hi:+.1f}]")
        print("  residual (data/model) by cos bin, mean over the fitted energies:")
        for c, row in zip(cos_c, resid):
            print(f"    cos {c:+.3f}: {10**row.mean():.3f}   rms {row.std():.3f} dex")
        out[name] = {"eps_0": x[0], "e_thr_gev": 10**x[1], "reach_m": 1e3*x[2],
                     "reach_68_m": [1e3*lo, 1e3*hi], "chi2": chi2, "dof": n - 3}
        if name == "all cells":
            resid_all = resid
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "81_trident_2025_map_fit.json").write_text(json.dumps(out, indent=2))
    rc = {"xtick.labelsize": 8, "ytick.labelsize": 8, "axes.labelsize": 8, "font.size": 8}
    with plt.style.context(str(_STYLE)), plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(3.6, 3.4))
        xe = np.concatenate([log10_e - 0.125, [log10_e[-1] + 0.125]])
        im = ax.pcolormesh(xe, COS_EDGES, resid_all, vmin=-0.5, vmax=0.5, cmap="RdBu_r",
                           shading="flat")
        ax.set_xlabel(r"$\log_{10}(E_\nu/\mathrm{GeV})$")
        ax.set_ylabel(r"$\cos\theta_z$")
        ax.set_box_aspect(1)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r"$\log_{10}$ published / model")
        for suffix in (".pdf", ".png"):
            path = args.out_dir / f"81_trident_2025_map_residual{suffix}"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


if __name__ == "__main__":
    main()
