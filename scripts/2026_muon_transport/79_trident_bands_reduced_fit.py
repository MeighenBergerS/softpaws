"""Example 79 -- TRIDENT band by band under the reduced response.

Example 77 fits TRIDENT's solid-angle-weighted sky average and returns a
negative reach, the one site whose table grows more slowly than footprint
times range. The average carries the downgoing band at 40% of its
weight, where the 3.1 km overburden truncates the muon and the area grows
with the cross section alone. This example refits each of TRIDENT's three
published ``cos(theta)`` bands on its own, with the same physics fixed
(``b_scale = 1``, ``lam`` at BGR18, ``eps_0 = 1``) and only the threshold
and the reach free, and puts the three posteriors on one plane beside the
sky-average fit. A three-parameter variant frees ``eps_0`` per band, since
the 6-degree cut need not cost every band the same.

Usage
-----
    python scripts/2026_muon_transport/79_trident_bands_reduced_fit.py
    python scripts/2026_muon_transport/79_trident_bands_reduced_fit.py --sigma 0.05 --steps 2000
"""

import argparse
import importlib.util
import json
import pathlib

import corner
import matplotlib.pyplot as plt
import numpy as np

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EX77 = load_example("77_reduced_response_fit.py", "_example_77")
_EX56 = _EX77._EX56
_EX55 = _EX56._EX55
_EX33 = _EX77._EX33

COLORS = {"up": "#e7298a", "horizon": "#1b9e77", "down": "#d95f02", "sky": "0.25"}

#: Detector name -> key of :data:`COLORS`, filled in by
#: :func:`build_band_detectors`. Plotting stays in the script, so the colour
#: is not carried on the ``Detector`` record.
BAND_KEYS: dict[str, str] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--walkers", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR)
    return parser.parse_args()


def build_band_detectors() -> list:
    """TRIDENT's three bands and its sky average as example 33 detectors."""
    site = [s for s in _EX56.water_sites() if s.name == "TRIDENT"][0]
    zenith_weights, neutrino_column, muon_column_km = _EX56.water_columns(site)
    ladders = _EX56.water_ladders(site, neutrino_column)
    theta_deg, _ = _EX33.arca_zenith_grid()
    cos_theta = np.cos(np.deg2rad(theta_deg))
    grid = _EX33.ARCA_LOG10_E
    priors = dict(_EX33.PRIORS["ARCA230"])
    priors["reach_km"] = site.reach_prior
    start = np.array([1.0, 3.0, 1.0, _EX33.LAMBDA_BGR18, 0.03])
    keys = ("up", "horizon", "down")
    detectors = []
    bands = _EX55.trident_bands()
    for key, (lo, hi, log10_e, log10_a) in zip(keys, bands, strict=True):
        observed = _EX56._on_grid(log10_e, log10_a)
        mask = (grid >= site.fit_band[0]) & (grid <= site.fit_band[1]) & np.isfinite(observed)
        weights = np.where((cos_theta >= lo) & (cos_theta < hi), zenith_weights, 0.0)
        BAND_KEYS[f"{key} ({lo:+.1f} < cos < {hi:+.1f})"] = key
        detectors.append(_EX33.Detector(
            name=f"{key} ({lo:+.1f} < cos < {hi:+.1f})", log10_e=grid, observed=observed,
            mask=mask,
            predict=(lambda theta, select=None, s=site, la=ladders, w=weights,
                     m=muon_column_km: _EX56.water_model(theta, s, la, w, m, select)),
            priors=priors, start=start.copy(), selection_level="6 deg cut"))
    BAND_KEYS["sky average"] = "sky"
    observed = _EX56.trident_allsky_cm2()
    mask = (grid >= site.fit_band[0]) & (grid <= site.fit_band[1]) & np.isfinite(observed)
    detectors.append(_EX33.Detector(
        name="sky average", log10_e=grid, observed=observed, mask=mask,
        predict=(lambda theta, select=None, s=site, la=ladders, w=zenith_weights,
                 m=muon_column_km: _EX56.water_model(theta, s, la, w, m, select)),
        priors=priors, start=start.copy(), selection_level="6 deg cut"))
    return detectors


def make_figure(results: dict, sigma: float, out_path: pathlib.Path) -> None:
    rc = {"xtick.labelsize": 8, "ytick.labelsize": 8, "axes.labelsize": 8, "font.size": 8}
    with plt.style.context(str(_STYLE)), plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        handles = []
        for name, (chain, color) in results.items():
            data = np.column_stack([chain[:, 0], 1.0e3 * chain[:, 1]])
            filled = name == "sky average"
            base = np.array(plt.matplotlib.colors.to_rgb(color))
            corner.hist2d(data[:, 0], data[:, 1], ax=ax, levels=(0.68, 0.95), color=color,
                          plot_datapoints=False, plot_density=False, fill_contours=filled,
                          no_fill_contours=not filled, smooth=0.8,
                          contourf_kwargs={"colors": [(*base, 0.0), (*base, 0.12), (*base, 0.28)]}
                          if filled else None,
                          contour_kwargs={"linewidths": 1.1})
            handles.append(plt.Line2D([], [], color=color, lw=1.6, label=name.split(" (")[0]))
        ax.axhline(0.0, color="0.6", lw=0.7, zorder=0)
        ax.axhline(68.0, color="0.35", lw=0.9, ls=(0, (1, 2)), zorder=0)
        ax.text(ax.get_xlim()[1], 68.0, "Water ", va="bottom", ha="right", fontsize=8,
                color="0.35")
        ax.set_xlabel(r"$\log_{10}(E_{\mathrm{thr}}/\mathrm{GeV})$")
        ax.set_ylabel(r"$\Lambda$ [m]")
        ax.set_box_aspect(1)
        pct = r"\%" if plt.rcParams["text.usetex"] else "%"
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8,
                  title=f"TRIDENT bands, {100*sigma:.0f}{pct} per node", title_fontsize=8)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    detectors = build_band_detectors()
    fixed = {"eps_0": 1.0, "log10_e_thr": 3.0, "b_scale": 1.0, "lam": _EX33.LAMBDA_BGR18,
             "reach_km": 0.03}
    print(f"\nTRIDENT band by band; physics fixed (b_scale = 1, lam = {_EX33.LAMBDA_BGR18}), "
          f"eps_0 = 1, error model {100*args.sigma:.0f}% per node")
    out, results = {}, {}
    for seed, d in enumerate(detectors, start=31):
        n = int(d.mask.sum())
        print(f"\n=== {d.name}: {n} nodes ===")
        entry = {"nodes": n}
        for tag, free in (("2-param", _EX77.FREE2), ("3-param", _EX77.FREE3)):
            best, chain = _EX77.fit(d, free, fixed, args.sigma, args.steps, args.walkers, seed)
            theta = _EX77.full_theta(free, best, fixed)
            dev, res = _EX77.deviance(d, theta, args.sigma)
            q = {name: np.percentile(chain[:, k], [16, 50, 84]) for k, name in enumerate(free)}
            print(f"  {tag}: deviance {dev:.2f} / {n - len(free)} dof, "
                  f"rms {np.std(res)/np.log(10):.3f} dex, "
                  f"trend {(res[-1]-res[0])/np.log(10):+.3f} dex, "
                  f"level {np.exp(np.mean(res)):.3f}")
            for name in free:
                lo, med, hi = q[name]
                if name == "reach_km":
                    print(f"      Lambda: median {1e3*med:7.1f} m [{1e3*lo:7.1f}, {1e3*hi:7.1f}]")
                elif name == "log10_e_thr":
                    print(f"      E_thr : median {10**med:7.0f} GeV [{10**lo:7.0f}, {10**hi:7.0f}]")
                else:
                    print(f"      eps_0 : median {med:7.3f} [{lo:7.3f}, {hi:7.3f}]")
            entry[tag] = {"best": theta.tolist(), "deviance": dev,
                          "quantiles": {k: v.tolist() for k, v in q.items()}}
            if tag == "2-param":
                results[d.name] = (chain, COLORS[BAND_KEYS[d.name]])
                # Residual of the published band over the best-fit model, node by node.
                print("      published/model per node: "
                      + " ".join(f"{np.exp(r):.3f}" for r in res))
        out[d.name] = entry
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "79_trident_bands_reduced_fit.json"
    path.write_text(json.dumps(out, indent=2))
    make_figure(results, args.sigma, args.out_dir / "79_trident_bands_plane.pdf")
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
