"""Example 59 -- what an extra muon energy loss would do, on this model alone.

The scratch feasibility against the DR2 events said an extra loss
``Delta b = eps b (E_mu / PeV)^n`` is invisible there: a smooth response
suppression is degenerate with the flux freedom, and the effect that would
break the degeneracy -- the brighter track -- cannot be modelled through
IceCube's released ``nu_mu`` smearing, which bakes in standard-model light
yield. This example asks the question where both effects are under control:
a forecast built purely on this project's forward model, no released IRFs
and no data.

**The two effects, exactly.** If the new channel shares the loss-spectrum
shape, the Laplace exponent scales, ``Phi -> kappa(E) Phi`` with
``kappa = 1 + eps (E / PeV)^n``, and the kernel algebra collapses:

* the potential density scales as ``u(w) / kappa``, so the *normalized*
  entry-energy distribution of through-going muons is unchanged -- faster
  losses mean fewer muons, not a different energy mix;
* the rate carries the whole range effect through the closed-form running-b
  range, ``A_eff -> A_eff x L_mod / L_SM``;
* any ``dE/dx``-based energy proxy reads the loss rate against the
  standard-model calibration, so the reconstructed energy shifts *up*,
  ``E_hat = E x kappa(E)`` -- events get brighter exactly where they get
  rarer.

The two effects have opposite signs and different shapes, and both act on
the atmospheric muons as well as the astrophysical ones, which is what a
consistent transport treatment buys.

**The forecast.** An Asimov dataset is built at ``eps = 0``: this model's
banded responses (the fitted configuration of example 45, ``nu_mu`` and
tau channels), example 22's atmospheric fluxes, the entry-energy marginal
from the exact kernel's potential density, and a lognormal proxy of
:data:`PROXY_SIGMA_DEX` about the shifted energy. The fit profiles example
51's nuisances -- astrophysical normalization and index under the tracks
anchors, conventional and prompt normalizations under their priors -- and
the sensitivity is where ``2 Delta lnL(eps)`` crosses 3.84. Both effects
are also switched on separately, to show which one carries the
sensitivity, and the whole exercise is run for the IC86 exposure and for
ten years of IceCube-Gen2.

Scope. Sky-averaged over the upgoing declination bands; the entry-energy
marginal takes an unlimited upstream column (through-going upgoing tracks);
the proxy resolution is a stand-in for a real estimator and moves the
absolute numbers, not the story. The new channel is assumed to share the
standard loss-spectrum shape; a harder channel would add to the variance
as well and be easier to see, so the forecast is conservative.

Usage
-----
    python examples/59_bsm_loss_forecast.py
    python examples/59_bsm_loss_forecast.py --proxy-sigma 0.4
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm as gauss

from softpaws.data.loader import compute_livetime_s, load_uptime
from softpaws.transport.coefficients import diffusion_coefficient, drift_coefficient
from softpaws.transport.loss_distribution import loss_density
from softpaws.transport.source import mean_inelasticity

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"

#: Pivot of the extra loss [GeV] and the exponents scanned.
PIVOT_GEV = 1.0e6
N_GRID = (0.5, 1.0, 2.0)

#: eps grid of the profile scans (log-spaced).
EPS_GRID = np.logspace(-2.0, 0.7, 28)

#: Reconstructed-energy grid [log10 GeV] and fit window, example 51's.
RECO_EDGES = np.arange(1.0, 8.51, 0.25)
FIT_RECO = (4.25, 7.5)

#: Muon energy below which a track is not accepted [GeV]; example 31's
#: smearing-pinned threshold.
ACCEPT_GEV = 10.0**2.9

#: Proxy resolution about the (shifted) entry energy [dex].
PROXY_SIGMA_DEX = 0.3

#: Kernel reference energy and the w grid of the potential density; the grid
#: must hold the deepest slab's whole loss law (see example 57).
KERNEL_ENERGY_GEV = 1.0e6
W_GRID = np.linspace(0.0, 24.0, 2401)
X_MAX_KM = 30.0
N_X = 120

#: Gen2 forecast years.
GEN2_YEARS = 10.0

COLORS = {"both": "#e7298a", "rate": "#1b9e77", "proxy": "#7570b3", "gen2": "#d95f02"}


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EX51 = load_example("51_dr2_flavor_fit.py", "_example_51")


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--proxy-sigma", type=float, default=PROXY_SIGMA_DEX,
                        help="Proxy resolution [dex].")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figures, '59a' and '59b'.")
    return parser.parse_args()


def kappa(energy_gev, eps: float, n: float) -> np.ndarray:
    """Loss enhancement ``1 + eps (E / PIVOT)^n``."""
    return 1.0 + eps * (np.asarray(energy_gev, dtype=float) / PIVOT_GEV) ** n


def range_ratio(energy_nu, eps: float, n: float) -> np.ndarray:
    """``L_mod / L_SM`` for the muon of a neutrino of each energy."""
    e_mu = (1.0 - np.squeeze(mean_inelasticity(energy_nu))) * energy_nu
    out = np.ones(np.atleast_1d(e_mu).shape)
    for i, em in enumerate(np.atleast_1d(e_mu)):
        if em <= ACCEPT_GEV:
            continue
        x = np.linspace(np.log(ACCEPT_GEV), np.log(em), 200)
        out[i] = np.trapezoid(1.0 / kappa(np.exp(x), eps, n), x) / (x[-1] - x[0])
    return out


def potential_density() -> np.ndarray:
    """Exact ``u(w)`` at the kernel reference energy (standard model)."""
    b = float(np.squeeze(drift_coefficient(KERNEL_ENERGY_GEV)))
    d = float(np.squeeze(diffusion_coefficient(KERNEL_ENERGY_GEV)))
    x_grid = np.linspace(0.0, X_MAX_KM, N_X + 1)[1:]
    u = np.zeros(W_GRID.size)
    for x in x_grid:
        u += loss_density(W_GRID, float(x), b, d, n_k=2**13)
    return u * (x_grid[1] - x_grid[0])


def build_marginal(u: np.ndarray, enu_centers_log10, eps: float, n: float) -> np.ndarray:
    """``P(reco bin | true bin)`` from the kernel and the shifted proxy.

    The entry-energy density per true energy is the normalized potential
    density (its shape is ``kappa``-independent); the proxy is lognormal
    about ``log10(E kappa(E))``.

    Returns
    -------
    marginal : np.ndarray, shape (n_true, n_reco)
    """
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (u[1:] + u[:-1]) * np.diff(W_GRID))])
    marginal = np.zeros((len(enu_centers_log10), RECO_EDGES.size - 1))
    w = W_GRID
    for i, log10_enu in enumerate(enu_centers_log10):
        e0 = (1.0 - float(np.squeeze(mean_inelasticity(10.0**log10_enu)))) * 10.0**log10_enu
        w_max = np.log(e0 / ACCEPT_GEV)
        if w_max <= 0.0:
            continue
        inside = w <= w_max
        weights = np.where(inside, u, 0.0)
        total = np.interp(w_max, w, cdf)
        entry = e0 * np.exp(-w)
        center = np.log10(entry * kappa(entry, eps, n))
        # Gaussian proxy about each entry energy, accumulated over the w grid.
        cells = (gauss.cdf((RECO_EDGES[None, 1:] - center[:, None]) / PROXY_SIGMA_DEX)
                 - gauss.cdf((RECO_EDGES[None, :-1] - center[:, None]) / PROXY_SIGMA_DEX))
        marginal[i] = np.trapezoid(weights[:, None] * cells, w, axis=0) / total
    return marginal


class Forecast:
    """Asimov likelihood over reco bins, this model's chain only."""

    def __init__(self, responses, atmos, d_omega, livetime_s, u, enu_log10):
        self._responses = responses
        self._atmos = atmos
        self._d_omega = d_omega
        self._livetime = livetime_s
        self._u = u
        self._enu = enu_log10
        self._window = ((RECO_EDGES[:-1] >= FIT_RECO[0] - 1e-9)
                        & (RECO_EDGES[1:] <= FIT_RECO[1] + 1e-9))
        self._sm_marginal = build_marginal(u, enu_log10, 0.0, 1.0)

    def _true_rates(self, ratio):
        """Per-true-bin counts for each component, range ratio applied."""
        energy = 10.0**self._enu
        d_omega = self._d_omega
        out = {}
        for name in ("conv", "prompt"):
            integrand = (self._responses["mu"] * self._atmos[name]
                         * d_omega[None, :]).sum(axis=1) * ratio
            out[name] = self._livetime * integrand * np.gradient(energy)
        summed = {ch: (self._responses[ch] * d_omega[None, :]).sum(axis=1) * ratio
                  for ch in ("mu", "tau")}
        astro = np.zeros((len(_EX51.GAMMA_GRID), energy.size))
        for g, gamma in enumerate(_EX51.GAMMA_GRID):
            spec = 2.0 * _EX51.PIVOT_PHI0 * (energy / 1.0e5) ** (-gamma)
            astro[g] = self._livetime * (summed["mu"] + summed["tau"]) * spec * np.gradient(energy)
        out["astro"] = astro
        return out

    def folded(self, eps: float, n: float):
        """Reco-space components under ``(eps, n)``, three effect switches."""
        ratio = range_ratio(10.0**self._enu, eps, n)
        marginal = (self._sm_marginal if eps == 0.0
                    else build_marginal(self._u, self._enu, eps, n))
        rates = self._true_rates(ratio)
        fold = {"conv": rates["conv"] @ marginal, "prompt": rates["prompt"] @ marginal,
                "astro": rates["astro"] @ marginal}
        return {k: v[..., self._window] for k, v in fold.items()}

    def folded_partial(self, eps, n, effect):
        """Only one effect on: ``"rate"`` or ``"proxy"``."""
        if effect == "rate":
            ratio = range_ratio(10.0**self._enu, eps, n)
            marginal = self._sm_marginal
        else:
            ratio = np.ones(self._enu.size)
            marginal = build_marginal(self._u, self._enu, eps, n)
        rates = self._true_rates(ratio)
        fold = {"conv": rates["conv"] @ marginal, "prompt": rates["prompt"] @ marginal,
                "astro": rates["astro"] @ marginal}
        return {k: v[..., self._window] for k, v in fold.items()}

    @staticmethod
    def _astro_at(astro, gamma):
        grid = _EX51.GAMMA_GRID
        i = int(np.clip(np.searchsorted(grid, gamma) - 1, 0, grid.size - 2))
        t = (gamma - grid[i]) / (grid[i + 1] - grid[i])
        return (1.0 - t) * astro[i] + t * astro[i + 1]

    def deviance(self, components, data) -> callable:
        """Profiled scaled deviance of ``data`` against one component set."""
        anchors = _EX51.ANCHORS

        def objective(p):
            norm, gamma, a_conv, a_prompt = p
            if (norm < 0.0 or a_conv <= 0.0 or a_prompt < 0.0
                    or not _EX51.GAMMA_BOUNDS[0] <= gamma <= _EX51.GAMMA_BOUNDS[1]):
                return 1.0e12
            mu = np.clip(a_conv * components["conv"] + a_prompt * components["prompt"]
                         + norm * self._astro_at(components["astro"], gamma), 1e-12, None)
            with np.errstate(divide="ignore", invalid="ignore"):
                dev = 2.0 * (mu - data + np.where(data > 0, data * np.log(data / mu), 0.0))
            scaled = np.sum(dev / (1.0 + _EX51.MODEL_SYS**2 * mu))
            penalty = (((a_conv - 1.0) / _EX51.CONV_PRIOR[1]) ** 2
                       + ((a_prompt - 1.0) / _EX51.PROMPT_PRIOR[1]) ** 2
                       + ((gamma - anchors["gamma"][0]) / anchors["gamma"][1]) ** 2
                       + ((2.0 * norm - anchors["phi_mu"][0]) / anchors["phi_mu"][1]) ** 2)
            return float(scaled) + penalty

        best = None
        for x0 in ((0.4, 2.4, 1.0, 1.0), (0.6, 2.2, 1.1, 1.0)):
            trial = minimize(objective, x0=np.asarray(x0), method="Nelder-Mead",
                             options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 8000})
            if best is None or trial.fun < best.fun:
                best = trial
        return best.fun

    def scan(self, n: float, data, effect: str = "both"):
        """``2 Delta lnL(eps)`` on :data:`EPS_GRID` for one exponent."""
        base = self.deviance(self.folded(0.0, n), data)
        curve = []
        for eps in EPS_GRID:
            comp = (self.folded(eps, n) if effect == "both"
                    else self.folded_partial(eps, n, effect))
            curve.append(self.deviance(comp, data) - base)
        return np.array(curve)


def limit(curve, level=3.84):
    """First ``eps`` crossing the threshold, log-interpolated."""
    above = curve >= level
    if not above.any():
        return np.inf
    k = int(np.argmax(above))
    if k == 0:
        return float(EPS_GRID[0])
    return float(np.exp(np.interp(level, [curve[k - 1], curve[k]],
                                  np.log([EPS_GRID[k - 1], EPS_GRID[k]]))))


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def figure_scans(scans, out_dir) -> None:
    """``2 Delta lnL(eps)`` per exponent: both effects, and each alone."""
    with plt.style.context(str(_STYLE)):
        fig, axes = plt.subplots(1, len(N_GRID), figsize=(2.6 * len(N_GRID), 2.7),
                                 sharey=True, gridspec_kw={"wspace": 0.08})
        for ax, n in zip(axes, N_GRID):
            for effect, label in (("both", "both effects"), ("rate", "rate only"),
                                  ("proxy", "proxy shift only")):
                ax.plot(EPS_GRID, scans["IC86"][(n, effect)], color=COLORS[effect], lw=1.2,
                        label=label)
            ax.plot(EPS_GRID, scans["Gen2"][(n, "both")], color=COLORS["gen2"], lw=1.2,
                    ls="--", label=f"both, Gen2 {GEN2_YEARS:g} yr")
            ax.axhline(3.84, color="0.7", lw=0.7, ls=":")
            ax.set_xscale("log")
            ax.set_ylim(0.0, 25.0)
            ax.set_xlabel(r"$\varepsilon$")
            ax.text(0.05, 0.95, rf"$n = {n:g}$", transform=ax.transAxes, va="top", fontsize=8)
        axes[0].set_ylabel(r"$2\,\Delta\ln L$ (Asimov)")
        axes[0].legend(fontsize=6, frameon=False, loc="center left")
        _save(fig, out_dir, "59a_bsm_loss_scans")


def figure_spectra(forecast, data, n, eps, out_dir) -> None:
    """Asimov reco spectrum against the ``(eps, n)`` expectation at the limit."""
    centers = 0.5 * (RECO_EDGES[:-1] + RECO_EDGES[1:])[forecast._window]
    sm = forecast.folded(0.0, n)
    mod = forecast.folded(eps, n)
    rate = forecast.folded_partial(eps, n, "rate")

    def total(c):
        return c["conv"] + c["prompt"] + forecast._astro_at(
            c["astro"], _EX51.ANCHORS["gamma"][0]) * 0.5 * _EX51.ANCHORS["phi_mu"][0]

    with plt.style.context(str(_STYLE)):
        fig, (ax, ratio) = plt.subplots(2, 1, figsize=(3.0, 3.6), sharex=True,
                                        gridspec_kw={"height_ratios": (2.6, 1.3),
                                                     "hspace": 0.05})
        ax.step(centers, total(sm), where="mid", color="k", lw=1.2, label="standard model")
        ax.step(centers, total(mod), where="mid", color=COLORS["both"], lw=1.2,
                label=rf"$\varepsilon = {eps:.2f}$, $n = {n:g}$")
        ax.step(centers, total(rate), where="mid", color=COLORS["rate"], lw=1.0, ls=":",
                label="rate effect only")
        ax.set_yscale("log")
        ax.set_ylabel("events per bin")
        ax.legend(fontsize=6, frameon=False, loc="upper right")
        ratio.axhline(1.0, color="0.6", lw=0.7, ls=":")
        ratio.step(centers, total(mod) / total(sm), where="mid", color=COLORS["both"], lw=1.1)
        ratio.step(centers, total(rate) / total(sm), where="mid", color=COLORS["rate"],
                   lw=1.0, ls=":")
        ratio.set_ylim(0.4, 1.8)
        ratio.set_xlabel(r"$\log_{10}(\hat E\,/\,\mathrm{GeV})$")
        ratio.set_ylabel("modified / SM")
        _save(fig, out_dir, "59b_bsm_loss_spectra")


def main() -> None:
    args = parse_args()
    global PROXY_SIGMA_DEX
    PROXY_SIGMA_DEX = args.proxy_sigma

    print("Loading examples 35, 45 and 46; this model's banded responses ...")
    ex35 = load_example("35_point_source_effective_area.py", "_example_35")
    ex45 = load_example("45_first_principles_reach.py", "_example_45")
    ex46 = load_example("46_declination_resolved_reach.py", "_example_46")
    enu_edges, dec_edges, _, responses = _EX51.fit_inputs(ex35, ex45, ex46, args.data_dir,
                                                          False)
    responses_gen2 = _EX51.gen2_responses(ex35, ex45, ex46, dec_edges, False)
    atmos = _EX51.atmospheric_fluxes(dec_edges)
    d_omega = 2.0 * np.pi * np.diff(np.sin(np.deg2rad(dec_edges)))
    livetime = sum(compute_livetime_s(load_uptime(args.data_dir / "uptime" / f"{s}_exp.csv"))
                   for s in _EX51.IC86_SEASONS)
    print(f"  IC86 livetime {livetime / 3.156e7:.1f} yr; proxy sigma {PROXY_SIGMA_DEX:g} dex")

    print("Building the exact-kernel potential density ...")
    u = potential_density()

    scans = {}
    for name, resp, live in (("IC86", responses, livetime),
                             ("Gen2", responses_gen2, GEN2_YEARS * 3.156e7)):
        forecast = Forecast(resp, atmos, d_omega, live, u, _EX51.LOG10_E_GRID)
        sm = forecast.folded(0.0, 1.0)
        data = (sm["conv"] + sm["prompt"]
                + forecast._astro_at(sm["astro"], _EX51.ANCHORS["gamma"][0])
                * 0.5 * _EX51.ANCHORS["phi_mu"][0])
        print(f"\n{name}: {data.sum():,.0f} Asimov events in the window")
        scans[name] = {}
        effects = ("both", "rate", "proxy") if name == "IC86" else ("both",)
        for n in N_GRID:
            for effect in effects:
                curve = forecast.scan(n, data, effect)
                scans[name][(n, effect)] = curve
                print(f"  n {n:g}, {effect:>11}: 95% eps sensitivity "
                      f"{limit(curve):.3g}")
        if name == "IC86":
            ic_forecast, ic_data = forecast, data

    print()
    figure_scans(scans, args.out_dir)
    # The spectra figure uses the flattest exponent, the one with a finite limit.
    eps_star = limit(scans["IC86"][(0.5, "both")])
    if np.isfinite(eps_star):
        figure_spectra(ic_forecast, ic_data, 0.5, eps_star, args.out_dir)


if __name__ == "__main__":
    main()
