"""Example 22 -- MCEq atmospheric background, added to example 21.

Every comparison up to and including ``21_full_icecube_comparison.py`` fits a
bare astrophysical template above 1 PeV, because below that the upgoing
IceTracks-DR2 sample is dominated by atmospheric neutrinos that the model does
not predict. This example computes that background from first principles with
**MCEq** (https://github.com/mceq-project/MCEq), which solves the coupled
cascade equations for the air shower, and folds it through both forward models,
so the comparison can be extended down to 10 TeV.

The calculation has two stages:

1. **Tabulate.** MCEq is run once per zenith angle on a South Pole atmosphere
   and the resulting ``nu_mu + nu_mu_bar`` fluxes (conventional and prompt,
   separately) are cached to an ``.npz`` table. This is the slow step -- of
   order ten minutes, dominated by the near-horizontal angles, whose curved
   atmosphere takes the longest to integrate -- so it is skipped whenever the
   table is already on disk.
2. **Integrate.** The table is interpolated in ``(E, dec)`` and pushed through
   the two response paths: the soft-volume model via the new
   :meth:`~softpaws.response.soft_volume.SoftVolumeResponse.
   expected_counts_from_flux`, and the published IRFs via the new
   :func:`~softpaws.comparison.rates.irf_expected_counts_directional`. Neither
   the flux's spectral shape nor its zenith dependence is a power law, which is
   what those two entry points add over the ones example 21 uses.

Fitting
-------
Both normalizations then float together
(:func:`~softpaws.comparison.rates.fit_component_scales`): the astrophysical
one, as in example 21, and the atmospheric one, because the MCEq prediction
carries its own flux and hadronic-model uncertainty and because the soft-volume
path models no detection efficiency. For reference the fit is also run with the
background frozen at its predicted normalization
(:func:`~softpaws.comparison.rates.fit_scale_factor_with_background`), which
leaves no room for a signal at all -- the raw MCEq prediction already exceeds
the observed upgoing counts between 10 and 100 TeV, by about 50% on the IRF
path and by an order of magnitude on the soft-volume one. The soft-volume path
then needs nearly the same suppression (~0.15) for its atmospheric component as
for its astrophysical one, which is the cleanest statement yet that what
separates the soft-volume ceiling from the data is detection efficiency rather
than spectral shape.

Geometry
--------
Atmospheric neutrinos arriving upgoing at the South Pole are produced in the
atmosphere on the opposite side of the Earth, so a source declination ``dec``
(arrival zenith ``90 deg + dec``, the convention of
:mod:`softpaws.transport.attenuation`) corresponds to an MCEq production zenith
``theta = 90 deg - dec``: vertically upgoing tracks come from showers that were
vertical overhead on the far side, and horizontal ones from horizontal showers.
Their propagation through the Earth is then the same PREM absorption
(:func:`~softpaws.transport.attenuation.prem_column`) applied to the
astrophysical flux in example 14 -- which is why the background dies off above
a few hundred TeV rather than following the atmospheric spectrum forever.

Caveats
-------
* The soft-volume path models no detection efficiency, so its background, like
  its signal, is a geometric ceiling; the IRF path's background is the realistic
  one. The ratio of the two fitted signal normalizations is still the efficiency
  estimate of example 21, now measured over the wider fit range.
* Both paths evaluate the atmospheric flux in the energy variable their own
  model uses -- observed muon energy for the soft-volume path (Eq. 2.23),
  true neutrino energy plus smearing for the IRF path -- exactly as they treat
  the astrophysical flux. See the energy-smearing note in example 21.
* Atmospheric *muons* are not modelled and do not need to be: they cannot cross
  the Earth, and the sample here is upgoing only.

Usage
-----
    python examples/22_atmospheric_background_mceq.py
    python examples/22_atmospheric_background_mceq.py --recompute-table
    python examples/22_atmospheric_background_mceq.py --data-dir /path/to/dataverse_files

Requires the optional MCEq dependency (``pip install -e ".[atm]"``), but only
when the flux table has to be built.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from softpaws.comparison.rates import (
    fit_component_scales,
    fit_scale_factor_with_background,
    irf_expected_counts,
    irf_expected_counts_directional,
    observed_counts,
)
from softpaws.data.container import EventSet
from softpaws.data.loader import compute_livetime_s, load_all_seasons, load_irfs, load_uptime
from softpaws.response.soft_volume import (
    SoftVolumeResponse,
    power_law_flux,
    tau_induced_expected_counts_attenuated,
)

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"
_DEFAULT_TABLE = _DEFAULT_OUT_DIR / "22_mceq_atmospheric_flux.npz"

IC86_SEASONS = (
    "IC86_I", "IC86_II", "IC86_III", "IC86_IV", "IC86_V", "IC86_VI",
    "IC86_VII", "IC86_VIII", "IC86_IX", "IC86_X", "IC86_XI",
)

RADIUS_KM = 0.62  # IceCube-like instrumented sphere
PHI0 = 0.63  # reference flux normalization [1e-18 GeV^-1 cm^-2 s^-1 sr^-1]
GAMMA = 2.38  # reference spectral index
LOG10_E_MIN_FIT = 4.0  # 10 TeV; reachable now that the background is modelled
LOG10_E_EDGES = np.arange(3.0, 8.01, 0.5)
DEC_MIN, DEC_MAX = 0.0, 90.0  # upgoing hemisphere

# MCEq settings. SIBYLL-2.3d and H3a are the standard conventional/prompt
# atmospheric-neutrino baseline; the atmosphere is the South Pole profile MCEq
# ships for IceCube, in January (austral summer, the thinner atmosphere).
INTERACTION_MODEL = "SIBYLL2.3d"
PRIMARY_MODEL = "H3a"
ATMOSPHERE = ("SouthPole", "January")
N_DEC_TABLE = 13  # production zeniths, evenly spaced in declination over the band


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=_DEFAULT_DATA_DIR,
        help="Root of the DR2 data directory.",
    )
    parser.add_argument(
        "--table",
        type=pathlib.Path,
        default=_DEFAULT_TABLE,
        help="Cached MCEq atmospheric-flux table.",
    )
    parser.add_argument(
        "--recompute-table",
        action="store_true",
        help="Rebuild the MCEq table even if it already exists.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Use the conventional atmospheric component only, dropping charm decay.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR / "22_atmospheric_background_mceq.pdf",
        help="Output file for the figure.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Stage 1: tabulate the atmospheric flux per zenith angle with MCEq.
# ---------------------------------------------------------------------------


def build_flux_table(path: pathlib.Path, n_dec: int = N_DEC_TABLE) -> dict[str, np.ndarray]:
    """Run MCEq once per zenith angle and cache the ``nu_mu`` fluxes.

    Parameters
    ----------
    path : pathlib.Path
        Destination ``.npz`` file.
    n_dec : int, optional
        Number of declinations sampled over ``[DEC_MIN, DEC_MAX]``, mapped to
        MCEq production zeniths ``theta = 90 deg - dec``.

    Returns
    -------
    table : dict of np.ndarray
        Keys ``energy_gev`` (n_e,), ``dec_deg`` (n_dec,), and the fluxes
        ``conv`` and ``prompt``, both of shape ``(n_e, n_dec)``, in
        ``GeV^-1 cm^-2 s^-1 sr^-1`` and summed over ``nu_mu`` and ``nu_mu_bar``.
    """
    import importlib.util  # noqa: F401  (MCEq 1.4.1 uses it without importing it)

    import crflux.models as pm
    from MCEq.core import MCEqRun

    dec_deg = np.linspace(DEC_MIN, DEC_MAX, n_dec)
    mceq = MCEqRun(
        interaction_model=INTERACTION_MODEL,
        primary_model=(pm.HillasGaisser2012, PRIMARY_MODEL),
        density_model=("MSIS00_IC", ATMOSPHERE),
        theta_deg=0.0,
    )

    conv = np.empty((mceq.e_grid.size, n_dec))
    prompt = np.empty_like(conv)
    for j, dec in enumerate(dec_deg):
        theta = 90.0 - dec
        print(f"  MCEq: dec = {dec:5.1f} deg (production zenith {theta:5.1f} deg) ...")
        mceq.set_theta_deg(theta)
        mceq.solve()
        conv[:, j] = mceq.get_solution("conv_numu") + mceq.get_solution("conv_antinumu")
        prompt[:, j] = mceq.get_solution("pr_numu") + mceq.get_solution("pr_antinumu")

    table = {
        "energy_gev": np.asarray(mceq.e_grid, dtype=float),
        "dec_deg": dec_deg,
        "conv": conv,
        "prompt": prompt,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        interaction_model=INTERACTION_MODEL,
        primary_model=PRIMARY_MODEL,
        atmosphere="/".join(ATMOSPHERE),
        **table,
    )
    print(f"Flux table saved to: {path.resolve()}")
    return table


def load_flux_table(path: pathlib.Path, recompute: bool = False) -> dict[str, np.ndarray]:
    """Load the cached MCEq table, building it first if needed.

    Parameters
    ----------
    path : pathlib.Path
        Table file.
    recompute : bool, optional
        Rebuild even if ``path`` exists.

    Returns
    -------
    table : dict of np.ndarray
        As returned by :func:`build_flux_table`.
    """
    if recompute or not path.exists():
        print(f"Building MCEq flux table ({N_DEC_TABLE} zenith angles; this takes minutes) ...")
        return build_flux_table(path)

    with np.load(path) as data:
        table = {key: data[key] for key in ("energy_gev", "dec_deg", "conv", "prompt")}
        print(
            f"Loaded MCEq flux table: {path.resolve()}\n"
            f"  {data['interaction_model']} / {data['primary_model']} / {data['atmosphere']}, "
            f"{table['dec_deg'].size} zenith angles"
        )
    return table


class AtmosphericFlux:
    """Interpolated atmospheric ``nu_mu`` flux from an MCEq table.

    Bilinear in ``(log10 E, dec)`` on the logarithm of the flux, which is the
    interpolation both variables are smooth in. Callable in the two-argument
    form the forward models expect.

    Parameters
    ----------
    table : dict of np.ndarray
        Table from :func:`load_flux_table`.
    include_prompt : bool, optional
        Whether to add the prompt (charm) component to the conventional one.
        Defaults to ``True``.

    Attributes
    ----------
    energy_gev : np.ndarray
        Tabulated energies [GeV].
    dec_deg : np.ndarray
        Tabulated declinations [deg].
    flux : np.ndarray, shape (n_e, n_dec)
        Tabulated flux [GeV^-1 cm^-2 s^-1 sr^-1].

    Examples
    --------
    >>> flux = AtmosphericFlux(load_flux_table(_DEFAULT_TABLE))  # doctest: +SKIP
    >>> float(flux(1.0e5, 45.0)) > 0  # doctest: +SKIP
    True
    """

    def __init__(self, table: dict[str, np.ndarray], include_prompt: bool = True) -> None:
        self.energy_gev = np.asarray(table["energy_gev"], dtype=float)
        self.dec_deg = np.asarray(table["dec_deg"], dtype=float)
        flux = np.asarray(table["conv"], dtype=float)
        if include_prompt:
            flux = flux + np.asarray(table["prompt"], dtype=float)
        self.flux = flux
        # A floor keeps the logarithm finite where the cascade solution underflows.
        floor = flux[flux > 0.0].min() * 1e-10
        self._interp = RegularGridInterpolator(
            (np.log10(self.energy_gev), self.dec_deg),
            np.log(np.maximum(flux, floor)),
            bounds_error=False,
            fill_value=None,  # linear extrapolation off the ends of the table
        )

    def __call__(
        self,
        energy_gev: float | np.ndarray,
        dec_deg: float | np.ndarray,
    ) -> np.ndarray:
        """Differential flux at an energy and declination.

        Parameters
        ----------
        energy_gev : float or np.ndarray
            Neutrino energy [GeV].
        dec_deg : float or np.ndarray
            Source declination [deg]; broadcast against ``energy_gev``.

        Returns
        -------
        flux : np.ndarray
            Differential flux [GeV^-1 cm^-2 s^-1 sr^-1].
        """
        energy, dec = np.broadcast_arrays(
            np.asarray(energy_gev, dtype=float), np.asarray(dec_deg, dtype=float)
        )
        points = np.stack([np.log10(energy), dec], axis=-1)
        return np.exp(self._interp(points))


# ---------------------------------------------------------------------------
# Stage 2: integrate the table into background counts (example 21's machinery).
# ---------------------------------------------------------------------------


def load_ic86_observed_and_livetime(data_dir: pathlib.Path) -> tuple[np.ndarray, float]:
    raw = load_all_seasons(data_dir)
    events = EventSet(raw)
    mask = np.zeros(events.n_events, dtype=bool)
    for season in IC86_SEASONS:
        uptime = load_uptime(data_dir / "uptime" / f"{season}_exp.csv")
        for start, stop in uptime:
            mask |= (events.time >= start) & (events.time <= stop)
    ic86_events = EventSet(events.data[mask])

    counts = observed_counts(ic86_events, LOG10_E_EDGES, DEC_MIN, DEC_MAX)
    livetime_s = sum(
        compute_livetime_s(load_uptime(data_dir / "uptime" / f"{season}_exp.csv"))
        for season in IC86_SEASONS
    )
    return counts, livetime_s


def soft_volume_signal(response: SoftVolumeResponse, livetime_s: float) -> np.ndarray:
    """Astrophysical ``numu + nu_tau`` soft-volume template of example 21."""
    numu = response.expected_counts_attenuated(
        LOG10_E_EDGES, PHI0, GAMMA, livetime_s, DEC_MIN, DEC_MAX,
    )
    tau = tau_induced_expected_counts_attenuated(
        response, LOG10_E_EDGES, PHI0, GAMMA, livetime_s, DEC_MIN, DEC_MAX,
    )
    return numu + tau


def make_figure(
    table: dict[str, np.ndarray],
    atm_flux: AtmosphericFlux,
    observed: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: pathlib.Path,
) -> None:
    """Two panels: the tabulated flux, and observed counts vs signal + background.

    Parameters
    ----------
    table : dict of np.ndarray
        MCEq table, for the declinations it was sampled at.
    atm_flux : AtmosphericFlux
        Interpolated flux, drawn at a few representative declinations.
    observed : np.ndarray, shape (n_bins,)
        Observed counts per bin.
    predictions : dict
        Maps a label to ``(fitted signal, fitted background)`` counts per bin.
    out_path : pathlib.Path
        Output file; written with both ``.pdf`` and ``.png`` suffixes.
    """
    energy = table["energy_gev"]
    in_range = (energy >= 1.0e2) & (energy <= 1.0e8)
    # The zenith dependence is all near the horizon, so sample it there.
    dec_samples = (0.0, 7.5, 30.0, 90.0)
    styles = {"soft": ("-", "C3"), "IRF": ("--", "C1")}

    with plt.style.context(str(_STYLE)):
        fig, (ax_flux, ax_counts) = plt.subplots(1, 2, figsize=(6.8, 3.0))

        for dec in dec_samples:
            e = energy[in_range]
            ax_flux.plot(
                np.log10(e),
                e**2 * atm_flux(e, dec),
                label=rf"$\delta = {dec:g}^\circ$",
            )
        ax_flux.set_yscale("log")
        ax_flux.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax_flux.set_ylabel(
            r"$E^2 \phi_{\nu_\mu + \bar{\nu}_\mu}$"
            r"$\,[\mathrm{GeV\,cm^{-2}\,s^{-1}\,sr^{-1}}]$"
        )
        ax_flux.set_title("MCEq atmospheric flux (at production)", fontsize=7)
        ax_flux.legend(fontsize=6)

        ax_counts.stairs(observed, LOG10_E_EDGES, label="observed (IC86)", lw=1.8, color="k")
        for name, (signal, background) in predictions.items():
            ls, color = styles[name]
            ax_counts.stairs(
                signal + background, LOG10_E_EDGES,
                label=f"{name}: astro + atm", ls=ls, color=color,
            )
            ax_counts.stairs(
                background, LOG10_E_EDGES,
                label=f"{name}: atm alone", ls=":", color=color, alpha=0.7,
            )
        ax_counts.axvline(LOG10_E_MIN_FIT, color="0.7", lw=0.6, zorder=0)
        ax_counts.set_yscale("log")
        ax_counts.set_ylim(1.0e-3, 5.0 * max(observed.max(), 1.0))
        ax_counts.set_xlabel(r"$\log_{10}(E\,/\,\mathrm{GeV})$")
        ax_counts.set_ylabel("tracks / bin")
        ax_counts.set_title("Upgoing IC86 tracks", fontsize=7)
        ax_counts.legend(fontsize=6)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def main() -> None:
    args = parse_args()

    table = load_flux_table(args.table, args.recompute_table)
    atm_flux = AtmosphericFlux(table, include_prompt=not args.no_prompt)

    print(f"Loading IC86 events from: {args.data_dir}")
    observed, livetime_s = load_ic86_observed_and_livetime(args.data_dir)
    print(f"  IC86 upgoing events: {observed.sum():,.0f}")
    print(f"  IC86 combined livetime: {livetime_s / 86400.0:.1f} d")

    print("Loading IC86 IRFs ...")
    irfs = load_irfs(args.data_dir / "irfs", "IC86_I")

    def astro_flux(energy_gev):
        return power_law_flux(energy_gev, PHI0, GAMMA)

    print("Computing IRF signal and atmospheric background ...")
    irf_signal = irf_expected_counts(
        irfs["aeff"], irfs["smearing"], LOG10_E_EDGES, DEC_MIN, DEC_MAX, astro_flux, livetime_s,
    )
    irf_background = irf_expected_counts_directional(
        irfs["aeff"], irfs["smearing"], LOG10_E_EDGES, DEC_MIN, DEC_MAX, atm_flux, livetime_s,
    )

    print("Computing soft-volume signal and atmospheric background ...")
    response = SoftVolumeResponse(radius_km=RADIUS_KM, method="exact")
    soft_signal = soft_volume_signal(response, livetime_s)
    soft_background = response.expected_counts_from_flux(
        LOG10_E_EDGES, atm_flux, livetime_s, DEC_MIN, DEC_MAX,
    )

    print("Atmospheric prediction vs. data, per bin (MCEq normalization as computed):")
    for i, (lo, hi) in enumerate(zip(LOG10_E_EDGES[:-1], LOG10_E_EDGES[1:])):
        print(
            f"  {lo:4.1f}-{hi:4.1f}: observed {observed[i]:9,.0f}   "
            f"IRF atm {irf_background[i]:11,.1f}   soft atm {soft_background[i]:11,.1f}"
        )

    print(f"Fitting above 10^{LOG10_E_MIN_FIT:.1f} GeV ...")
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    scales: dict[str, float] = {}
    bkg_scales: dict[str, float] = {}
    for name, signal, background in (
        ("soft", soft_signal, soft_background),
        ("IRF", irf_signal, irf_background),
    ):
        # The MCEq normalization carries its own sizeable flux and hadronic
        # uncertainty, and the soft-volume path has no detection efficiency at
        # all, so the background normalization floats alongside the signal.
        signal_scale, bkg_scale = fit_component_scales(
            observed, [signal, background], LOG10_E_EDGES, LOG10_E_MIN_FIT,
        )
        fixed_bkg_scale = fit_scale_factor_with_background(
            observed, signal, background, LOG10_E_EDGES, LOG10_E_MIN_FIT,
        )
        scales[name] = signal_scale
        bkg_scales[name] = bkg_scale
        predictions[name] = (signal * signal_scale, background * bkg_scale)
        print(
            f"  {name:4s}: best-fit phi0 = {PHI0 * signal_scale:6.3f}, "
            f"atmospheric normalization = {bkg_scale:5.2f}  "
            f"(phi0 = {PHI0 * fixed_bkg_scale:6.3f} with the background held fixed)"
        )
    if scales["IRF"] > 0.0:
        # Two independent handles on the same missing detection efficiency: the
        # astrophysical normalization the soft-volume path needs relative to the
        # IRF path, and the atmospheric one. They should agree.
        print(f"  implied efficiency eps = {scales['soft'] / scales['IRF']:.3f} (astrophysical), "
              f"{bkg_scales['soft'] / bkg_scales['IRF']:.3f} (atmospheric)")
        print("  (paper efficiency eps_IC-TG ~ 0.45)")

    make_figure(table, atm_flux, observed, predictions, args.out)


if __name__ == "__main__":
    main()
