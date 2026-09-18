"""Example 75 -- the KM3-230213A figures remade with the loss-model bands.

Two remakes of example 57's figures under example 69's ensemble, using
example 71's fold (the light reads ``E kappa``, the occupation tilts
``1/kappa``, the ARCA21 area carries ``1/kappa`` for the range).

**Figure 75a** (example 57a's right panel alone). The neutrino-energy
posterior of the event under the exact kernel, for the three flux
priors: ``E^-2`` -- KM3NeT's own convention, so the one to compare with
their 72-2600 PeV interval -- plus the SPL and IceCube-anchored BPL.
Each curve wears the envelope of the seven ensemble variants: the
posterior's *position* inherits the energy-scale error (the ``E^-2``
median spans 210-311 PeV, example 71), which on the log axis reads as a
sideways band about a tenth of a decade wide.

**Figure 75b** (example 57b remade). The profile-likelihood regions of
the IceCube Asimov and the ARCA21 event term, per flux family, with the
event-term contours drawn for the baseline and for the envelope pair
(Bezrukov-Bugaev, ALLM91). Under the consistent treatment the three
sets of contours nearly coincide -- the visible statement of example
71's finding that the tension carries no loss-model caveat -- and the
panel is annotated with the ensemble span of the tension.

Usage
-----
    python scripts/2026_muon_transport/75_km3_figures_with_bands.py
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EX71 = load_example("71_km3_tension_loss_error.py", "_example_71")
_EX69 = _EX71._EX69
_EX57 = _EX71._EX57

#: Contour variants of figure 75b: the ensemble's envelope pair.
ENVELOPE = ("photo BB", "photo ALLM91")

#: The fitted transport scale (example 72's informed four-site corner,
#: b = 0.994 +- 0.028), composed into every kappa so the figures sit on the
#: fitted configuration; the fitted lambda's <= 4% UHE tilt on ``sigma_CC``
#: is left at BGR18, inside the drawn bands.
B_SCALE_FITTED = 0.994


def fitted_kappa(ratios, label):
    """``kappa_1(E)`` of one variant on top of the fitted transport scale."""
    if label is None:
        return lambda e: np.full_like(np.asarray(e, dtype=float), B_SCALE_FITTED)
    base = _EX71.kappa_interp(ratios, label)
    return lambda e: B_SCALE_FITTED * base(e)

PRIOR_COLORS = {"E^-2": "#e7298a", "SPL": "#7570b3", "BPL": "#1b9e77"}


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--n-gamma", type=int, default=41)
    parser.add_argument("--n-phi0", type=int, default=121)
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figures, '75a' and '75b'.")
    return parser.parse_args()


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def figure_posteriors(u, ratios, out_dir) -> None:
    """75a: the three flux priors, each with its ensemble envelope."""
    energy = 10.0**_EX57.LOG10_ENU
    posteriors = {}
    for label in [None, *_EX69.VARIANTS]:
        kappa = fitted_kappa(ratios, label)
        rates = _EX71.variant_likelihood(u, energy, kappa, normalize=False)
        for prior in ("E^-2", "SPL", "BPL"):
            posteriors[(prior, label)] = _EX57.posterior(rates, prior)
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        shown = {"E^-2": r"$E^{-2}$", "SPL": "SPL", "BPL": "BPL"}
        for prior, color in PRIOR_COLORS.items():
            stack = np.array([posteriors[(prior, label)] for label in _EX69.VARIANTS])
            ax.fill_between(_EX57.LOG10_ENU, stack.min(axis=0), stack.max(axis=0),
                            color=color, alpha=0.25, lw=0)
            ax.plot(_EX57.LOG10_ENU, posteriors[(prior, None)], color=color, lw=1.3,
                    label=shown[prior])
        mode, lo, hi = (np.log10(v * 1.0e6) for v in _EX57.KM3NET_ENU_PEV)
        ax.axvspan(lo, hi, color="0.92", zorder=0)
        ax.axvline(mode, color="0.35", lw=0.9, ls="-.",
                   label="KM3NeT Mean")
        ax.set_xlim(_EX57.LOG10_ENU[0], _EX57.LOG10_ENU[-1])
        ax.set_ylim(0.0, None)
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel("Posterior density")
        ax.set_box_aspect(1)
        ax.legend(loc="upper right", bbox_to_anchor=(0.9, 1.0))
        _save(fig, out_dir, "75a_event_energy_banded")


def figure_tension(u, ratios, args, out_dir) -> None:
    """75b: example 57b with the event-term contours per envelope variant."""
    energy = 10.0**_EX57.LOG10_ENU
    ex31 = _EX57._EX31
    ic_log10_e, ic_aeff, ic_livetime_s = ex31.build_icecube_aeff(
        args.data_dir, ex31.DEFAULT_MUON_THRESHOLD_GEV)
    km_log10_e, km_aeff = ex31.arca21_published_aeff()
    gamma_grid = np.linspace(*ex31.GAMMA_RANGE, args.n_gamma)
    phi0_grid = np.logspace(-1.5, 2.7, args.n_phi0)

    results = {}
    for family in ("SPL", "BPL"):
        shape, index_label, truth = _EX57.flux_family(family)
        ic = _EX57.ic_surface(shape, truth, phi0_grid, gamma_grid, ic_log10_e,
                              ic_aeff, ic_livetime_s)
        a_event, a_band = _EX57.km_event_coefficients(shape, gamma_grid, km_log10_e,
                                                      km_aeff, None)
        window = ex31.km_log_likelihood(phi0_grid[:, None], a_event, a_band,
                                        extended=True)
        one_event = ex31.phi0_for_one_event(a_event)
        km, sigmas = {}, {}
        for label in [None, *ENVELOPE]:
            kappa = fitted_kappa(ratios, label)
            like = _EX71.variant_likelihood(u, energy, kappa)
            aeff = km_aeff / kappa(0.4 * 10.0**km_log10_e)
            a_event_v, a_band_v = _EX57.km_event_coefficients(
                shape, gamma_grid, km_log10_e, aeff, like)
            km[label] = ex31.km_log_likelihood(phi0_grid[:, None], a_event_v,
                                               a_band_v, extended=True)
            _, _, sigmas[label] = ex31.compatibility(ic, km[label])
        results[family] = {"ic": ic, "km": km, "window": window,
                           "one_event": one_event, "truth": truth,
                           "index_label": index_label, "sigmas": sigmas}
        span = ", ".join(f"{label or 'baseline'} {s:.2f}" for label, s in sigmas.items())
        print(f"  {family}: tension [sigma] {span}")

    level68, level95 = ex31.DELTA_LNL_LEVELS
    with plt.style.context(str(_STYLE)):
        for family, stem in (("BPL", "75b_tension_bpl"), ("SPL", "75c_tension_spl")):
            r = results[family]
            color = {"SPL": "#7570b3", "BPL": "#1b9e77"}[family]
            fig, ax = plt.subplots(figsize=(3.4, 3.4))
            ic = r["ic"] - r["ic"].max()
            for lo, hi, alpha in ((0.0, level68, 0.4), (level68, level95, 0.2)):
                ax.contourf(gamma_grid, phi0_grid, -ic, levels=[lo, hi],
                            colors=["#d95f02"], alpha=alpha)
            # One combined ARCA term: the energy likelihood unioned over the
            # ensemble variants, so a single pair of contours carries both.
            combined = np.max(
                np.stack([r["km"][label] - r["km"][label].max()
                          for label in r["km"]]), axis=0)
            combined -= combined.max()
            for lo, hi, alpha in ((0.0, level68, 0.4), (level68, level95, 0.2)):
                ax.contourf(gamma_grid, phi0_grid, -combined, levels=[lo, hi],
                            colors=[color], alpha=alpha)
                ax.contour(gamma_grid, phi0_grid, -combined, levels=[hi],
                           colors=color, linewidths=[0.8], alpha=alpha + 0.3)
            ax.plot(*r["truth"][::-1], marker="+", color="k", ms=7, mew=1.2)
            ax.text(0.04, 0.96, f"{family}\n{r['sigmas'][None]:.2f}$\\sigma$",
                    transform=ax.transAxes, va="top")
            ax.set_yscale("log")
            ax.set_ylim(phi0_grid[0], phi0_grid[-1])
            ax.set_xlim(gamma_grid[0], gamma_grid[-1])
            ax.set_xlabel(rf"${r['index_label']}$")
            ax.set_ylabel(r"$\phi_0$ at 100 TeV "
                          r"[$10^{-18}$ GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
            ax.set_box_aspect(1)
            handles = [Patch(facecolor="#d95f02", alpha=0.4, label="IceCube"),
                       Patch(facecolor=color, alpha=0.4, label="ARCA21")]
            ax.legend(handles=handles, loc="lower right")
            _save(fig, out_dir, stem)


def main() -> None:
    args = parse_args()
    _, ratios = _EX69.load_ensemble(False)
    print("Building the exact-kernel potential density (example 57) ...")
    u = _EX57.potential_density("exact")
    figure_posteriors(u, ratios, args.out_dir)
    figure_tension(u, ratios, args, args.out_dir)


if __name__ == "__main__":
    main()
