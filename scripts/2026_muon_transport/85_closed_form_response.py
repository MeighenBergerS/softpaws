"""Example 85 -- the closed-form response against the full one, at IceCube.

Section III of the paper takes the textbook effective area in two steps. The
first puts the range to threshold into it and gives a closed form a reader can
evaluate with two loss moments, Eq. (aeffcf): the closed-form range with one
kernel, taken at 100 TeV, in pure water at ``(1 - <y_w>) E_nu``, the survival
factor ``exp(-X / Lambda_nu)`` along the PREM chord, the reach in the projected
area, ``V_det`` for the starting tracks, and the ``nu_mu`` channel alone. The
second restores what the closed form leaves out, the rock below the array, the
regeneration ladder and the tau channel, and gives Eq. (aeff), the response
every number in the paper uses.

This example evaluates both for IceCube at the two instrument numbers of
example 77, per declination, and prints every number the prose of Section III
quotes: the closed form over the full response by energy and by declination
band, the multiplicative decomposition into the three restored pieces, the
one-kernel water range against the running spliced one, the rock factor, and the
worked value of both responses at 1 PeV. The closed form is rebuilt here
switch by switch, and with every switch on it reproduces
:func:`softpaws.response.site_models.icecube_model` to four digits, which the
example asserts.

Usage
-----
    python scripts/2026_muon_transport/85_closed_form_response.py
"""

import argparse
import itertools
import pathlib

import numpy as np

from softpaws.constants import CM_PER_KM
from softpaws.response import reduced
from softpaws.response import site_models as sm
from softpaws.transport.attenuation import flavour_transmission, survival_probability
from softpaws.transport.coefficients import log_loss_moments
from softpaws.transport.earth import prem_column
from softpaws.transport.muon_range import stochastic_muon_range_km, two_medium_range_ratio
from softpaws.transport.soft_volume import light_reach_radius_km
from softpaws.transport.source import MEAN_INELASTICITY, nucleon_number_density

_HERE = pathlib.Path(__file__).parent
_DEFAULT_CHAINS = _HERE / "output" / "77_chains_sigma05.npz"
_DEFAULT_OUT = _HERE / "output" / "85_closed_form_ratio.npz"

#: Energies the prose quotes [log10 GeV]; 7.8 is the top of the IceCube fit band.
QUOTED_LOG10_E = (5.0, 6.0, 7.0, 7.8)

#: Declination bands of the prose [deg], upgoing at the Pole.
BANDS = ((0.0, 10.0), (10.0, 30.0), (30.0, 60.0), (60.0, 90.0), (0.0, 90.0))

ENERGY_GEV = 10.0**sm.IC_LOG10_E
DEC_DEG = np.linspace(0.5, 89.5, sm.IC_N_DEC)
_DEC_RAD = np.deg2rad(DEC_DEG)
SOLID_ANGLE = np.cos(_DEC_RAD)
COS_THETA = np.sin(_DEC_RAD)
SIN_THETA = np.cos(_DEC_RAD)

CHANNELS = {
    "mu": (1.0 - MEAN_INELASTICITY, 1.0),
    "tau": (sm.MEAN_Z * (1.0 - MEAN_INELASTICITY), sm.F_TAU * sm.BR_TAU_TO_MU),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chains", type=pathlib.Path, default=_DEFAULT_CHAINS)
    parser.add_argument("--out", type=pathlib.Path, default=_DEFAULT_OUT)
    return parser.parse_args()


def instrument_numbers(chains: pathlib.Path) -> tuple[float, float, float]:
    """IceCube's held normalization and the posterior-median threshold and reach.

    Returns
    -------
    eps_0, e_thr_gev, reach_km : float
        Selection normalization, threshold [GeV] and reach [km per e-fold].
    """
    chain = np.load(chains)["IceCube_2p_chain"]
    median = np.median(chain, axis=0)
    return float(reduced.EPS_FIXED["IceCube"]), float(10.0 ** median[0]), float(median[1])


#: The one muon energy [GeV] at which the closed-form range takes its two loss
#: moments, for every production energy. The column a muon spends per decade is
#: ``1 / Phi'(0)``, largest at low energy, so every descent to a TeV threshold
#: has an effective loss rate close to the tabulated one a decade above the
#: threshold. One kernel here is closer to the running range than freezing at
#: each muon's own production energy, which is one-sidedly short.
KERNEL_GEV = 1.0e5


def closed_form_range_km(energy_gev: np.ndarray, threshold_gev: float) -> np.ndarray:
    """Eq. (Lclosed) with one loss kernel, at :data:`KERNEL_GEV`, in water."""
    phi1, phi2, _ = log_loss_moments(np.array([KERNEL_GEV]))
    phi1, phi2 = float(np.ravel(phi1)[0]), float(np.ravel(phi2)[0])
    w = np.log(np.asarray(energy_gev, dtype=float) / threshold_gev)
    return np.where(w > 0.0, w / phi1 + phi2 / (2.0 * phi1**2), 0.0)


def ladders() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Regeneration ladders per energy, rung and declination, one per flavour."""
    columns = np.array([prem_column(float(d)) for d in DEC_DEG])
    out = {}
    for flavour in ("mu", "tau"):
        energies = np.empty((ENERGY_GEV.size, sm.IC_N_RUNG))
        weights = np.empty((ENERGY_GEV.size, sm.IC_N_RUNG, DEC_DEG.size))
        for i, energy in enumerate(ENERGY_GEV):
            energies[i], weights[i] = flavour_transmission(
                energy, columns, sm.CROSS_SECTION, flavour=flavour,
                n_grid=sm.IC_N_RUNG, decades=4.0,
            )
        out[flavour] = (energies, weights)
    return out, columns


def response_per_declination(
    eps_0: float, e_thr: float, reach_km: float,
    lad: dict, columns: np.ndarray,
    ladder: bool, medium: bool, tau: bool,
) -> np.ndarray:
    """Effective area per (energy, declination) [cm^2] with three switches.

    ``ladder`` replaces the survival factor by the regeneration ladder,
    ``medium`` replaces the one-kernel water range by the running spliced
    two-medium one, and ``tau`` adds the tau channel. All three on is
    Eq. (aeff); all three off is Eq. (aeffcf).
    """
    n_nucleon = nucleon_number_density()
    total = np.zeros((ENERGY_GEV.size, DEC_DEG.size))
    for flavour in (("mu", "tau") if tau else ("mu",)):
        fraction, weight = CHANNELS[flavour]
        if ladder:
            energies, arrival = lad[flavour]
        else:
            energies = ENERGY_GEV[:, None]
            arrival = survival_probability(
                ENERGY_GEV[:, None], columns[None, :], sm.LAMBDA_BGR18, sm.CROSS_SECTION
            )[:, None, :]
        muon = fraction * energies
        if medium:
            length = stochastic_muon_range_km(muon.ravel(), e_thr).reshape(muon.shape)
            rock = np.stack([
                two_medium_range_ratio(muon[i], 1.0e3, -COS_THETA, sm.IC_ICE_BELOW_KM)
                for i in range(ENERGY_GEV.size)
            ])
        else:
            length = closed_form_range_km(muon, e_thr)
            rock = np.ones((ENERGY_GEV.size, muon.shape[1], DEC_DEG.size))
        radius = light_reach_radius_km(sm.IC_RADIUS_KM, muon, reach_km, sm.REACH_PIVOT_GEV)
        sigma = sm.tilted_cc(energies, sm.LAMBDA_BGR18)
        cap = (np.pi * radius**2 * length)[:, :, None] * rock * COS_THETA[None, None, :]
        side = (sm.IC_SIDE_COEFF * radius * sm.IC_HEIGHT_KM * length)[:, :, None] \
            * rock * SIN_THETA[None, None, :]
        v_det = (np.pi * radius**2 * sm.IC_HEIGHT_KM)[:, :, None] * np.ones((1, 1, DEC_DEG.size))
        rate = (n_nucleon * sigma * CM_PER_KM**3)[:, :, None]
        contribution = rate * arrival * (cap + side + v_det)
        total += weight * contribution.sum(axis=1)
    return eps_0 * total


def band_average(per_dec: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Solid-angle average over a declination band, as the DR2 table averages."""
    mask = (DEC_DEG >= lo) & (DEC_DEG < hi)
    return np.average(per_dec[:, mask], axis=1, weights=SOLID_ANGLE[mask])


def at(values: np.ndarray, log10_e: float) -> float:
    return float(values[int(np.argmin(np.abs(sm.IC_LOG10_E - log10_e)))])


def main() -> None:
    args = parse_args()
    eps_0, e_thr, reach_km = instrument_numbers(args.chains)
    print("IceCube instrument numbers (example 77 medians)")
    print(f"  eps_0 = {eps_0:.3f}, E_thr = {e_thr:.0f} GeV, reach = {1.0e3 * reach_km:.1f} m")

    lad, columns = ladders()
    configs = {}
    for switches in itertools.product((False, True), repeat=3):
        configs[switches] = response_per_declination(
            eps_0, e_thr, reach_km, lad, columns, *switches
        )
    full = configs[(True, True, True)]
    closed = configs[(False, False, False)]

    theta = np.array([eps_0, np.log10(e_thr), 1.0, sm.LAMBDA_BGR18, reach_km])
    library = sm.icecube_model(theta, sm.icecube_ladders())
    rebuilt = band_average(full, 0.0, 90.0)
    check = rebuilt[sm.IC_LOG10_E >= 4.0] / library[sm.IC_LOG10_E >= 4.0]
    assert np.allclose(check, 1.0, atol=2.0e-4), check
    deviation = np.max(np.abs(check - 1.0))
    print(f"  rebuilt full response against the library: max |ratio - 1| = {deviation:.1e}")

    print("\nWorked value at 1 PeV, upgoing hemisphere [m^2]")
    print(f"  closed form, Eq. (aeffcf): {at(band_average(closed, 0, 90), 6.0) / 1.0e4:.0f}")
    print(f"  full response, Eq. (aeff): {at(band_average(full, 0, 90), 6.0) / 1.0e4:.0f}")

    print("\nClosed form over full response, by declination band")
    print("  log10(E_nu/GeV):  " + "  ".join(f"{x:5.1f}" for x in QUOTED_LOG10_E))
    for lo, hi in BANDS:
        ratio = band_average(closed, lo, hi) / band_average(full, lo, hi)
        label = "upgoing sky" if hi - lo == 90.0 else f"{lo:.0f} to {hi:.0f} deg"
        print(f"  {label:>14}: " + "  ".join(f"{at(ratio, x):5.3f}" for x in QUOTED_LOG10_E))

    print("\nRestoring one piece at a time, upgoing sky, ratio to the full response")
    names = {
        (True, False, False): "ladder",
        (False, True, False): "medium",
        (False, False, True): "tau",
    }
    hemi_full = band_average(full, 0, 90)
    for switches, name in names.items():
        ratio = band_average(configs[switches], 0, 90) / hemi_full
        cells = "  ".join(f"{at(ratio, x):5.3f}" for x in QUOTED_LOG10_E)
        print(f"  closed form + {name:6s}: {cells}")
    print("  each piece alone, as a factor on the closed form (full with it removed over full):")
    removed = {
        (True, False, True): "water range, no rock",
        (False, True, True): "pure absorption",
        (True, True, False): "no tau channel",
    }
    product = np.ones(ENERGY_GEV.size)
    for switches, name in removed.items():
        ratio = band_average(configs[switches], 0, 90) / hemi_full
        product *= ratio
        cells = "  ".join(f"{at(ratio, x) - 1.0:+5.0%}" for x in QUOTED_LOG10_E)
        print(f"    {name:22s}: {cells}")
    closed_ratio = band_average(closed, 0, 90) / hemi_full
    cells = "  ".join(f"{at(product, x):5.3f}" for x in QUOTED_LOG10_E)
    print(f"    product of the three   : {cells}")
    cells = "  ".join(f"{at(closed_ratio, x):5.3f}" for x in QUOTED_LOG10_E)
    print(f"    closed form over full  : {cells}")

    print("\nThe range side alone, nu_mu at (1 - <y_w>) E_nu")
    muon = (1.0 - MEAN_INELASTICITY) * ENERGY_GEV
    running = stochastic_muon_range_km(muon, e_thr)
    frozen = closed_form_range_km(muon, e_thr)
    rock = np.stack([
        two_medium_range_ratio(np.array([m]), 1.0e3, -COS_THETA, sm.IC_ICE_BELOW_KM)[0]
        for m in muon
    ])
    rock_mean = np.average(rock, axis=1, weights=SOLID_ANGLE * COS_THETA)
    print(f"  closed form (kernel at {KERNEL_GEV:.0e} GeV) over running spliced, water: "
          + "  ".join(f"{at(frozen / running, x):5.3f}" for x in QUOTED_LOG10_E))
    print("  rock factor, upgoing sky weighted as the cap term: "
          + "  ".join(f"{at(rock_mean, x):5.3f}" for x in QUOTED_LOG10_E))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        log10_e=sm.IC_LOG10_E, dec_deg=DEC_DEG, solid_angle=SOLID_ANGLE,
        eps_0=eps_0, e_thr_gev=e_thr, reach_km=reach_km,
        closed_form=closed, full=full,
        **{"config_" + "".join("1" if s else "0" for s in k): v for k, v in configs.items()},
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
