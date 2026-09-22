"""Example 42 -- the starting-event term against the IceCube HESE 7.5-year sample.

Example 41 splits the effective area of Eq. (18) by event topology: ``V_det``
counts **starting** events, whose neutrino vertex lies inside the instrumented
volume, and ``A_proj L`` counts **entering** events. Examples 28, 30 and 32 test
the sum against through-going tables, where the entering term supplies 93 to 97%
of the area. The starting term has therefore never been tested on its own.

It is worth testing on its own because it carries **no transport**. With
``V_det`` fixed, its energy and zenith dependence is the charged-current cross
section and the Earth transmission alone,

.. math:: A_{\\rm eff}^{\\rm start}(E_\\nu, \\Omega) = n_N V_{\\rm det}
    \\sum_k w_k(E_\\nu, \\Omega)\\, \\sigma_{\\rm CC}(E_k),

so a comparison against a starting sample isolates the half of the forward model
that the through-going comparisons integrate over. The regeneration ladder
``w_k`` of Sec. V C is what is on trial here, and nothing about ``Phi``.

The HESE 7.5-year release is the right sample for it. It ships Monte Carlo
rather than a tabulated response, and every simulated event carries its true
flavour, true energy and **true zenith**, so the response can be sliced in
zenith -- which a released direction-averaged table cannot be, since it has
already integrated over the very thing we want to see.

What the comparison can and cannot claim
----------------------------------------
Our curve is a geometric ceiling on the full ``V_det = 1 km^3``, and HESE's
fiducial volume is smaller, being defined by the veto that makes the sample
starting in the first place. The ratio of the two is therefore a fiducial
fraction times a selection efficiency, and its **normalization is not a
prediction**. Its **shape is**. Specifically:

* in energy, the ratio should rise and flatten as the 60 TeV deposited-energy
  cut stops biting, exactly as the implied efficiency does for DR2 in Sec. VI;
* in zenith, the ratio should be flat, because the whole zenith dependence of
  the starting term is Earth transmission and the HESE veto is geometric.

A ratio that drifts between the Earth-crossing band and the downgoing band is a
failure of the transmission model, and it is the one thing here that no free
normalization can absorb. That is the test.

The flavours are free. Charged-current absorption is terminal for ``nu_e`` and
``nu_mu`` alike, so the ladder of Sec. V C serves both, and ``nu_tau`` swaps in
the regenerating transmission already used for the tau channel. Since Eq. (18)'s
starting term has no muon in it, ``nu_e`` costs nothing to add here even though
the rest of the paper carries no ``nu_e``.

Usage
-----
    python scripts/2026_muon_transport/42_hese_starting_events.py
    python scripts/2026_muon_transport/42_hese_starting_events.py --no-deposited-cut
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data.hese import effective_area_cm2, load_hese_mc
from softpaws.transport.attenuation import (
    flavour_transmission,
    prem_column,
    regenerated_transmission,
)
from softpaws.transport.cross_section import bgr18_cross_section
from softpaws.transport.source import nucleon_number_density

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

CROSS_SECTION = bgr18_cross_section()

# IceCube's instrumented volume, the same 1.00 km^3 Table VIII carries.
V_DET_CM3 = 1.0e15

# The fiducial volume the HESE veto leaves, from the exclusions of Ref. [1]
# Sec. II: the topmost 90 m, the bottom 10 m, a 60 m layer below the dust
# region, and 90 m inward from the outer side DOMs. On a hexagonal 1 km^2
# footprint that is 0.840 in height and 0.693 in area. The hexagon and the
# reading of "90 m from the outer layer of DOMs" are both approximations, so
# this carries maybe 10% of its own.
HESE_FIDUCIAL_FRACTION = 0.582
HESE_FIDUCIAL_CM3 = HESE_FIDUCIAL_FRACTION * V_DET_CM3

# Energy bins, 4 per decade over the range the HESE response covers.
ENERGY_BINS_GEV = np.logspace(4.0, 7.0, 13)

# Bands in cos(zenith). IceCube's convention puts +1 straight down through the
# atmosphere and -1 straight up through the full Earth chord, so the first band
# is the most absorbed and the last is unabsorbed.
COS_ZENITH_BANDS = ((-1.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 1.0))
BAND_LABEL = {
    (-1.0, -0.5): r"$-1 < \cos\theta_z < -0.5$",
    (-0.5, 0.0): r"$-0.5 < \cos\theta_z < 0$",
    (0.0, 0.5): r"$0 < \cos\theta_z < 0.5$",
    (0.5, 1.0): r"$0.5 < \cos\theta_z < 1$",
}
# Dark2, ordered so the most Earth-crossing band is the most saturated.
BAND_COLOR = {
    (-1.0, -0.5): "#1b9e77",
    (-0.5, 0.0): "#7570b3",
    (0.0, 0.5): "#e7298a",
    (0.5, 1.0): "#d95f02",
}

FLAVOURS = ("e", "mu", "tau")
FLAVOUR_LABEL = {"e": r"$\nu_e$ CC", "mu": r"$\nu_\mu$ CC", "tau": r"$\nu_\tau$ CC"}

# Zenith samples per band for the solid-angle average. Solid angle is uniform in
# cos(zenith), so a uniform grid in cos(zenith) is already the right measure.
N_ZENITH_PER_BAND = 24


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-deposited-cut",
        action="store_true",
        help="Drop the 60 TeV deposited-energy bound, the release's Fig. 33 convention.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR / "42_hese_starting_events",
        help="Output stem; .pdf and .png are both written.",
    )
    return parser.parse_args()


def band_columns_g_cm2(cos_zenith_range: tuple[float, float]) -> np.ndarray:
    """PREM column depths sampled uniformly across a cos-zenith band.

    IceCube sits at the Pole, so a direction of zenith ``theta_z`` arrives from
    declination ``dec`` with ``cos(theta_z) = -sin(dec)``. Downgoing directions
    return a zero column, since the atmosphere is negligible against a neutrino
    interaction length at these energies.

    Parameters
    ----------
    cos_zenith_range : tuple of float
        Band edges in ``cos(zenith)``.

    Returns
    -------
    columns : np.ndarray
        Column depths [g cm^-2], one per sampled direction.
    """
    low, high = cos_zenith_range
    # Cell centres, so neither edge is sampled twice across adjacent bands.
    edges = np.linspace(low, high, N_ZENITH_PER_BAND + 1)
    cos_zenith = 0.5 * (edges[1:] + edges[:-1])
    declination_deg = np.degrees(np.arcsin(np.clip(-cos_zenith, -1.0, 1.0)))
    return np.array([prem_column(float(d)) for d in declination_deg])


def predicted_starting_area_cm2(
    energy_gev: np.ndarray,
    cos_zenith_range: tuple[float, float],
    flavour: str,
    volume_cm3: float = V_DET_CM3,
) -> np.ndarray:
    """Starting-event effective area, band-averaged [cm^2].

    Each rung of the regeneration ladder carries its own cross section, the same
    convention Eq. (18) uses, because a response tabulated at the surface energy
    credits the event to that energy even when the neutrino scattered on the way
    in.

    Parameters
    ----------
    energy_gev : np.ndarray
        True neutrino energy at the surface [GeV].
    cos_zenith_range : tuple of float
        Band edges in ``cos(zenith)``.
    flavour : {"e", "mu", "tau"}
        Neutrino flavour. Charged current is terminal for ``e`` and ``mu``, so
        both take the neutral-current ladder; ``tau`` regenerates.
    volume_cm3 : float, optional
        Target volume [cm^3]. Defaults to the full instrumented
        :data:`V_DET_CM3`, which makes the result a ceiling; pass
        :data:`HESE_FIDUCIAL_CM3` to compare against the volume the veto
        actually leaves.

    Returns
    -------
    aeff : np.ndarray
        Effective area [cm^2], broadcast to the shape of ``energy_gev``.
    """
    columns = band_columns_g_cm2(cos_zenith_range)
    n_nucleon = nucleon_number_density()
    out = np.empty(energy_gev.size)
    for i, e_nu in enumerate(energy_gev):
        if flavour == "tau":
            rung_energy, rung_weight = flavour_transmission(
                float(e_nu), columns, CROSS_SECTION, flavour="tau"
            )
        else:
            rung_energy, rung_weight = regenerated_transmission(
                float(e_nu), columns, CROSS_SECTION
            )
        # Sum the ladder at each direction, then average over the band.
        per_direction = (rung_weight * CROSS_SECTION.cc(rung_energy)[:, None]).sum(axis=0)
        out[i] = n_nucleon * volume_cm3 * per_direction.mean()
    return out


def build(deposited_cut: bool) -> dict:
    """Compute both sides of the comparison for every flavour and band.

    Parameters
    ----------
    deposited_cut : bool
        Apply the 60 TeV bound on reconstructed deposited energy.

    Returns
    -------
    result : dict
        ``"energy"`` bin centres [GeV], and ``"hese"``/``"ours"`` dicts keyed by
        ``(flavour, band)`` holding effective areas [cm^2].
    """
    mc = load_hese_mc(
        deposited_min_gev=6.0e4 if deposited_cut else None,
        deposited_max_gev=1.0e7 if deposited_cut else None,
    )
    print(f"  HESE Monte Carlo events in use: {mc['primaryEnergy'].size:,}")

    centre = np.sqrt(ENERGY_BINS_GEV[1:] * ENERGY_BINS_GEV[:-1])
    hese, ours = {}, {}
    for flavour in FLAVOURS:
        for band in COS_ZENITH_BANDS:
            area, _ = effective_area_cm2(
                mc, ENERGY_BINS_GEV, flavour=flavour, interaction="cc",
                cos_zenith_range=band,
            )
            hese[(flavour, band)] = area
            ours[(flavour, band)] = predicted_starting_area_cm2(centre, band, flavour)
    return {"energy": centre, "hese": hese, "ours": ours}


def report(result: dict) -> None:
    """Print the ratio, and how much of it is zenith drift."""
    energy = result["energy"]
    print("\n  Ratio of the HESE effective area to the starting-term ceiling")
    print("  (normalization is a fiducial fraction times an efficiency; the "
          "shape is the prediction)\n")
    for flavour in FLAVOURS:
        print(f"  nu_{flavour}:")
        print("    log10(E/GeV)" + "".join(f"{str(b):>16s}" for b in COS_ZENITH_BANDS))
        for i, e in enumerate(energy):
            if i % 2:
                continue
            row = "".join(
                f"{result['hese'][(flavour, b)][i] / result['ours'][(flavour, b)][i]:>16.4f}"
                for b in COS_ZENITH_BANDS
            )
            print(f"    {np.log10(e):>12.2f}{row}")
        # The zenith drift is the statement no free normalization can absorb.
        top = ENERGY_BINS_GEV[:-1] >= 3.0e5
        spread = []
        for i in np.flatnonzero(top):
            vals = np.array(
                [result["hese"][(flavour, b)][i] / result["ours"][(flavour, b)][i]
                 for b in COS_ZENITH_BANDS]
            )
            if np.all(np.isfinite(vals)) and np.all(vals > 0):
                spread.append(np.log10(vals.max() / vals.min()))
        if spread:
            print(f"    band-to-band spread above 300 TeV: "
                  f"{np.mean(spread):.3f} dex mean, {np.max(spread):.3f} dex worst")
        # The all-sky plateau is the implied fiducial fraction times efficiency,
        # and it should agree between flavours where the veto is geometric.
        top = energy >= 3.0e5
        num = sum(result["hese"][(flavour, b)][top].sum() for b in COS_ZENITH_BANDS)
        den = sum(result["ours"][(flavour, b)][top].sum() for b in COS_ZENITH_BANDS)
        print(f"    all-sky plateau above 300 TeV: {num / den:.3f}\n")


def make_figure(result: dict, out_path: pathlib.Path) -> None:
    """Draw the ratio against energy, one panel per flavour, one line per band.

    Parameters
    ----------
    result : dict
        From :func:`build`.
    out_path : pathlib.Path
        Output file; both ``.pdf`` and ``.png`` are written.
    """
    energy = result["energy"]
    with plt.style.context(str(_STYLE)):
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
        for ax, flavour in zip(axes, FLAVOURS, strict=True):
            for band in COS_ZENITH_BANDS:
                ratio = result["hese"][(flavour, band)] / result["ours"][(flavour, band)]
                ax.plot(np.log10(energy), ratio, color=BAND_COLOR[band], lw=1.2,
                        label=BAND_LABEL[band])
            ax.set_yscale("log")
            ax.set_xlim(4.0, 7.0)
            ax.set_ylim(1.0e-3, 2.0)
            ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
            ax.set_title(FLAVOUR_LABEL[flavour], fontsize=7)
        axes[0].set_ylabel(r"$A_{\rm eff}^{\rm HESE}\,/\,A_{\rm eff}^{\rm start}$")
        axes[0].legend(loc="lower right", fontsize=5, frameon=False, handlelength=1.8)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    print("Building the HESE starting-event comparison ...")
    result = build(deposited_cut=not args.no_deposited_cut)
    report(result)
    make_figure(result, args.out)


if __name__ == "__main__":
    main()
