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
import dataclasses
import importlib.util
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, minimize_scalar

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DATA = _HERE.parent / "src" / "softpaws" / "data" / "trident"
_DEFAULT_OUT_DIR = _HERE / "output"


def load_example(stem: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EX77 = load_example("77_reduced_response_fit.py", "_example_77")
_EX56 = _EX77._EX56
_EX33 = _EX77._EX33

#: Reference layout of the 2025 study: 1000 strings at 100 m average spacing
#: (~9.6 km^2, radius 1.75 km) with 20 hDOMs at 30 m (0.57 km), at the 2022 depth.
RADIUS_KM = 1.75
COS_EDGES = np.linspace(1.0, -1.0, 13)
PARAM_BOUNDS = {"eps_0": (0.05, 1.5), "log10_e_thr": (1.5, 4.5), "reach_km": (-0.08, 0.40)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sigma-dex", type=float, default=0.043,
                        help="Assumed error per digitized cell [dex] (10%% by default).")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_map():
    raw = np.loadtxt(_DATA / "trident_2025_fig3a_log10aeff_m2.csv", delimiter=",")
    cos_c, log10_a = raw[:, 0], raw[:, 1:]
    log10_e = 3.0 + 0.25 * (np.arange(log10_a.shape[1]) + 0.5)
    return cos_c, log10_e, log10_a


class MapModel:
    """Model effective area on the map's grid, per cos bin and energy [m^2]."""

    def __init__(self, log10_e):
        base = [s for s in _EX56.water_sites() if s.name == "TRIDENT"][0]
        self.site = dataclasses.replace(base, radius_km=RADIUS_KM)
        self.zw, ncol, self.mcol = _EX56.water_columns(self.site)
        self.ladders = _EX56.water_ladders(self.site, ncol)
        theta_deg, _ = _EX33.arca_zenith_grid()
        cz = np.cos(np.deg2rad(theta_deg))
        self.weights = [np.where((cz >= COS_EDGES[i + 1]) & (cz < COS_EDGES[i]), self.zw, 0.0)
                        for i in range(12)]
        self.grid = _EX33.ARCA_LOG10_E
        self.log10_e = log10_e

    def __call__(self, theta):
        out = np.empty((12, self.log10_e.size))
        for i, w in enumerate(self.weights):
            full = _EX56.water_model(theta, self.site, self.ladders, w, self.mcol, None) / 1.0e4
            out[i] = np.interp(self.log10_e, self.grid, np.log10(full))
        return out


def fit_cells(model, log10_a, cells, sigma_dex):
    """Best fit of (eps_0, log10 E_thr, reach) over the selected cells, with a reach profile."""
    def theta_of(x):
        return np.array([x[0], x[1], 1.0, _EX33.LAMBDA_BGR18, x[2]])

    def chi2(x):
        for v, (lo, hi) in zip(x, PARAM_BOUNDS.values()):
            if not lo < v < hi:
                return 1.0e9
        r = log10_a[cells] - model(theta_of(x))[cells]
        return float(np.sum((r / sigma_dex) ** 2))

    starts = ([0.7, 2.5, 0.02], [0.5, 3.0, -0.03], [0.9, 2.2, 0.06], [0.6, 3.4, 0.0])
    best = min((minimize(chi2, x0, method="Nelder-Mead",
                         options={"xatol": 1e-4, "fatol": 1e-3, "maxiter": 4000}) for x0 in starts),
               key=lambda r: r.fun)
    # Profile the reach: refit the other two at fixed reach until chi2 rises by one.
    def profile(reach):
        res = minimize(lambda y: chi2([y[0], y[1], reach]), best.x[:2], method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-3})
        return res.fun
    lo = minimize_scalar(lambda r: (profile(r) - best.fun - 1.0) ** 2,
                         bounds=(PARAM_BOUNDS["reach_km"][0], best.x[2]), method="bounded").x
    hi = minimize_scalar(lambda r: (profile(r) - best.fun - 1.0) ** 2,
                         bounds=(best.x[2], PARAM_BOUNDS["reach_km"][1]), method="bounded").x
    resid = log10_a - model(theta_of(best.x))
    return best.x, best.fun, (lo, hi), resid


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
    print(f"TRIDENT 2025 map, {log10_a.size} cells over 10^{log10_e[0]:.2f}-10^{log10_e[-1]:.2f} GeV; "
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
