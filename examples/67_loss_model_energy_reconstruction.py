"""Example 67 -- the reconstructed energy is a loss-model statement.

A track's energy estimate is not a measurement of the muon's energy; it is
a measurement of its light, read through an assumed loss law. The standard
calibration inverts ``dE/dX = a + b E``; at the energies of interest the
radiative term carries the light, so the estimator returns
``E_hat = (dE/dX)_rad / b``. If a new radiative channel switches on toward
extreme energies -- example 59's enhancement
``kappa(E) = 1 + eps (E_mu / PeV)^n`` of the loss-spectrum scale -- the
same light is made by a lower-energy muon and the standard-calibrated
estimate reads ``E_hat = E x kappa(E)`` (example 59's kernel algebra: the
Laplace exponent scales, the potential density scales as ``u / kappa``,
the normalized entry-energy mix is untouched -- brighter tracks, not a
different population).

**Why a below-PeV sample cannot settle this.** Example 59 profiled this
loss against the DR2 tracks and found the sensitivity blocked: a smooth
enhancement is degenerate with the flux freedom. The 95% ceiling at
``n = 0.5`` is ``eps = 2.43``, and at ``n >= 1`` the profile never
crosses at all -- the below-PeV sample is blind (:data:`EPS_DRAWN`). The
place the degeneracy breaks is a single track two decades above the
pivot.

**The event.** KM3-230213A arrives with a standard-loss muon estimate of
120 PeV (90% interval 35-380, example 57's constants). Read through the
loss laws DR2 cannot exclude, the same light is a 5-13 PeV muon -- and the
parent-neutrino median rescales from ~220 PeV by roughly the same factor
(the entry-energy mix is unchanged, so the example 57 fold shifts
multiplicatively; a full re-fold would also update the transmission,
second order here). The figure is the map ``E_hat(E_true)`` with the event
overlaid: one line, the whole argument -- the most energetic track ever
recorded is a lever on loss physics at energies no accelerator reaches,
and the tension it carries against the IceCube flux (example 57's 2.8
sigma) is itself a standard-loss statement.

The counterweight, for honesty: faster losses also shorten the range and
dim the rate, so a global fit sees both effects; example 59 profiles them
together. This figure isolates the energy map, which is the piece that
reinterprets a single event.

Usage
-----
    python examples/67_loss_model_energy_reconstruction.py
    python examples/67_loss_model_energy_reconstruction.py --mu-pev 220
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Example 59 supplies ``kappa`` and the pivot; example 57 the event numbers.
_EX59 = load_example("59_bsm_loss_forecast.py", "_example_59")
_EX57 = load_example("57_km3_event_energy_and_bpl_tension.py", "_example_57")

#: What DR2 allows, per exponent (example 59, IC86 exposure, both effects
#: profiled; rerun 2026-09-01): at ``n = 0.5`` the 95% ceiling is 2.43; at
#: ``n >= 1`` the profile never crosses -- the below-PeV sample is blind and
#: a reference ``eps = 1`` is drawn instead. Ceilings and references, not
#: preferred values.
EPS_DRAWN = {0.5: (2.43, "DR2 95\\%"), 1.0: (1.0, "unconstrained"),
             2.0: (1.0, "unconstrained")}

#: True-energy grid of the map [GeV].
E_TRUE_GEV = np.logspace(5.0, 10.0, 400)

COLORS = {"sm": "0.25", 0.5: "#1b9e77", 1.0: "#7570b3", 2.0: "#e7298a",
          "event": "#d95f02"}


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mu-pev", type=float, default=_EX57.MU_PEV,
                        help="Standard-loss muon estimate to reinterpret [PeV].")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figure, '67a'.")
    return parser.parse_args()


def reconstructed_gev(e_true_gev, eps: float, n: float) -> np.ndarray:
    """Standard-calibrated estimate of a muon losing ``kappa`` times faster."""
    e = np.asarray(e_true_gev, dtype=float)
    return e * _EX59.kappa(e, eps, n)


def true_energy_gev(e_hat_gev: float, eps: float, n: float) -> float:
    """Invert ``E kappa(E) = E_hat`` (monotonic; log bisection)."""
    lo, hi = 1.0e3, e_hat_gev
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if float(reconstructed_gev(mid, eps, n)) < e_hat_gev:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def figure_map(mu_pev: float, out_dir: pathlib.Path) -> None:
    """``E_hat`` against ``E_true`` with KM3-230213A overlaid."""
    e_hat_event = mu_pev * 1.0e6
    band = tuple(v * 1.0e6 for v in _EX57.MU_90_PEV)
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.1))
        ax.axhspan(band[0], band[1], color=COLORS["event"], alpha=0.10, lw=0)
        ax.axhline(e_hat_event, color=COLORS["event"], lw=1.0, ls="-")
        ax.text(1.6e5, band[1] * 1.25, "KM3-230213A\n(standard-loss estimate)",
                fontsize=5.2, color=COLORS["event"], va="bottom")
        ax.plot(E_TRUE_GEV, E_TRUE_GEV, color=COLORS["sm"], lw=1.1,
                label=r"standard losses ($\hat E = E$)")
        for n, (eps, tag) in EPS_DRAWN.items():
            ax.plot(E_TRUE_GEV, reconstructed_gev(E_TRUE_GEV, eps, n),
                    color=COLORS[n], lw=1.2, ls="--",
                    label=rf"$\kappa = 1 + {eps:g}\,(E/\mathrm{{PeV}})^{{{n:g}}}$ ({tag})")
            e_true = true_energy_gev(e_hat_event, eps, n)
            ax.plot([e_true, e_true], [1.0e5, e_hat_event], color=COLORS[n],
                    lw=0.7, ls=":")
            ax.plot([e_true], [e_hat_event], marker="o", ms=3, color=COLORS[n])
        ax.plot([e_hat_event, e_hat_event], [1.0e5, e_hat_event],
                color=COLORS["sm"], lw=0.7, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1.0e5, 1.0e10)
        ax.set_ylim(1.0e5, 1.0e10)
        ax.set_xlabel(r"true muon energy at the detector [GeV]")
        ax.set_ylabel(r"standard-calibrated estimate $\hat E$ [GeV]")
        ax.legend(fontsize=5.2, frameon=False, loc="lower right")
        _save(fig, out_dir, "67a_loss_model_energy_map")


def main() -> None:
    args = parse_args()
    mu_gev = args.mu_pev * 1.0e6
    enu_med = _EX57.KM3NET_ENU_PEV[0]
    print(f"KM3-230213A: standard-loss muon estimate {args.mu_pev:g} PeV "
          f"(90% {_EX57.MU_90_PEV[0]:g}-{_EX57.MU_90_PEV[1]:g}), "
          f"neutrino median ~{enu_med:g} PeV")
    print(f"Extra radiative loss kappa = 1 + eps (E/PeV)^n at what DR2 allows "
          f"(example 59):\n  {'n':>4} {'eps':>6} {'status':>14} {'kappa(120 PeV)':>15} "
          f"{'true E_mu [PeV]':>16} {'implied E_nu [PeV]':>19}")
    for n, (eps, tag) in EPS_DRAWN.items():
        e_true = true_energy_gev(mu_gev, eps, n)
        scale = e_true / mu_gev
        print(f"  {n:4.1f} {eps:6.2f} {tag.replace(chr(92), ''):>14} "
              f"{float(_EX59.kappa(mu_gev, eps, n)):15.1f} "
              f"{e_true / 1.0e6:16.1f} {enu_med * scale:19.0f}")
    print("  (a ceiling and two blind spots, not preferred values; the rate "
          "counterweight is profiled in example 59)")
    figure_map(args.mu_pev, args.out_dir)


if __name__ == "__main__":
    main()
