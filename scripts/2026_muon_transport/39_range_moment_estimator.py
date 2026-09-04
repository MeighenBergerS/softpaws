"""Example 39 -- the range to threshold against a Monte Carlo, by its moments.

Appendix C validates the closed-form range of Eq.~(C4) against a direct
evaluation of the depth integral it came from. Both sides of that check use our
own kernel, so it tests the algebra and not the physics. Table D.2 does compare
against PROPOSAL, but at fixed depth: it asks how much energy a muon has lost
after 1 km. The range asks the inverse question -- how far a muon gets before
falling below a threshold -- and nothing in the paper currently tests it
externally. That matters because a tabulated effective area is proportional to
this length, so the ceiling of Sec. VI, the 0.69 selection efficiency and the
12% shape number all inherit whatever the range gets wrong.

Ref.~[Palmisano:2026sid] supplies the estimator. To calibrate ``b_mu`` and
``d_mu`` they propagate muons until the energy has fallen by a fixed factor,
record the distance ``R``, and match its first two central moments. Applied to
the exact transport the same two observables become a test rather than a fit,
because there is no parameter left to move.

**The predictions.** Write ``w = ln(epsilon / E_stop)`` for the log-loss the muon
must accumulate. The first-passage law of App. C, with the kernel frozen at the
production energy, gives

.. math:: \\langle R \\rangle = \\frac{w}{\\Phi'(0)}
    - \\frac{\\Phi''(0)}{2\\,\\Phi'(0)^2}, \\qquad
    \\mathrm{Var}(R) = \\frac{-\\Phi''(0)\\, w}{\\Phi'(0)^3}
    - \\frac{\\Phi'''(0)}{3\\,\\Phi'(0)^3}
    + \\frac{\\Phi''(0)^2}{4\\,\\Phi'(0)^4},

with the three moments quadratures of PROPOSAL's tabulated spectrum. The mean is
Eq.~(C4). The variance is the companion the paper does not yet quote, and it has
the same structure: a term linear in ``w`` plus a constant belonging to the
crossing. Both constants are moments of the same stationary overshoot -- the
mean overshoot appears in ``<R>``, its variance in ``Var(R)``, the latter with a
minus sign because a muon that overshoots further crossed its level sooner. The
leading term alone runs 8 to 25% high over ``w = 3.5`` to ``9.2``; with the
constant the closed form tracks a direct simulation of the same kernel to better
than 2%.

The coefficients run with energy -- ``Phi'(0)`` falls from 0.51 km^-1 at 10 PeV
to 0.46 at 100 TeV -- and a muon crossing three decades does not see one value
of them. The script therefore also carries the running form,

.. math:: \\langle R \\rangle = \\int_{\\ln E_{\\rm stop}}^{\\ln\\varepsilon}
    \\frac{{\\rm d}\\ln E}{\\Phi'(0; E)}
    - \\frac{\\Phi''(0; E_{\\rm stop})}{2\\,\\Phi'(0; E_{\\rm stop})^2},

which is what App. C's 3% local-evaluation systematic refers to. The two
together bracket the answer, and their separation is the size of that
systematic.

The drift-diffusion truncation replaces the subordinator by a Brownian motion
with drift ``M = b_mu + d_mu / 2`` and variance rate ``d_mu``, whose first
passage is inverse Gaussian: ``<R> = w / M`` exactly, ``Var(R) = w d_mu / M^3``.
No overshoot constant survives, because a continuous path crosses its level
instead of jumping over it -- the difference between the two transports, in one
line. For completeness the script also evaluates the moment formulas as
published in Ref.~[Palmisano:2026sid] Sec. 5.2, ``<R> = w/M + d/M^2`` and
``sigma_R^2 = w d/M^3 + 2 d^2/M^4``, which invert the Gaussian at fixed log-loss
rather than solving the first-passage problem and so keep terms of order
``d/M^2``. The spread among these three is a fair measure of what the
second-order transport leaves undetermined.

**Why the comparison stops at 100 TeV.** The kernel here is radiative only, as
everywhere else in this repository, and ionization enters the paper through the
deterministic splice of Eq.~(C5) instead of through ``Phi``. A radiative-only
muon below about 10 TeV barely loses energy, so its range distribution grows a
heavy tail and neither side of the comparison converges. The scan therefore
holds the stopping energy at 100 TeV, safely radiative, and varies the
production energy to move ``w``. Pushing the stop energy down to the TeV scale
instead makes the sampled mean range drift by tens of percent and tests the
splice rather than the transport.

**The Monte Carlo.** PROPOSAL propagates muons through water with the same three
radiative channels used elsewhere here, with a finite energy cut so that losses
above it are sampled stochastically. A muon is tracked until its energy first
falls below the stopping energy and the distance at that point is recorded.
Distances are converted to km of water equivalent at the repository's reference
density, so no density convention floats between the two sides.

Examples
--------
::

    python examples/39_range_moment_estimator.py
    python examples/39_range_moment_estimator.py --n-muons 8000
    python examples/39_range_moment_estimator.py --stop-energy-gev 3e5
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.transport.coefficients import (
    diffusion_coefficient,
    drift_coefficient,
    loss_spectrum_y_grid,
    proposal_loss_spectrum,
    proposal_parametrizations,
)
from softpaws.utils.constants import CM_PER_KM, RHO_WATER_G_CM3

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

# Stopping energy well above the muon critical energy, so that the whole
# trajectory is radiative and the closed form is exercised without the
# ionization splice of Eq.~(C5). See the module docstring.
STOP_ENERGY_GEV = 1.0e5
LOG10_START_ENERGY_SCAN = (5.5, 6.0, 6.5, 7.0, 7.5)
# Production energy of the histogrammed sample.
REFERENCE_ENERGY_GEV = 1.0e7
# Nodes per decade for the running-coefficient integral.
RUNNING_NODES_PER_DECADE = 12


# ---------------------------------------------------------------------------
# Kernel moments
# ---------------------------------------------------------------------------


def log_loss_moments(energy_gev: float) -> tuple[float, float, float]:
    """The three log-loss moments of PROPOSAL's spectrum that the range needs.

    Quadratures of the tabulated ``dGamma/dy`` on the two-branch grid of
    :func:`~softpaws.transport.coefficients.loss_spectrum_y_grid`, which resolves
    both the soft pile-up and the ``y -> 1`` edge. The integrands are finite at
    both ends: ``-ln(1-y) -> y`` cancels the ``1/y`` bremsstrahlung tail, and
    ``ln^2(1-y)`` diverges only logarithmically where the spectrum is falling.

    Parameters
    ----------
    energy_gev : float
        Muon energy [GeV].

    Returns
    -------
    phi_prime : float
        ``Phi'(0) = <-ln(1-y)>`` [km^-1].
    phi_second : float
        ``-Phi''(0) = <ln^2(1-y)>`` [km^-1], returned positive.
    phi_third : float
        ``Phi'''(0) = <-ln^3(1-y)>`` [km^-1]. Only the variance needs it.
    """
    y = loss_spectrum_y_grid()
    total = proposal_loss_spectrum(energy_gev, y)["total"]
    log_loss = -np.log1p(-y)
    phi_prime = float(np.trapezoid(total * log_loss, y))
    phi_second = float(np.trapezoid(total * log_loss**2, y))
    phi_third = float(np.trapezoid(total * log_loss**3, y))
    return phi_prime, phi_second, phi_third


def overshoot_variance(
    phi_prime: np.ndarray,
    phi_second: np.ndarray,
    phi_third: np.ndarray,
) -> np.ndarray:
    """Variance of the stationary overshoot, converted to depth.

    A first passage crosses its level from above, and the amount by which it
    overshoots has a stationary law with mean ``-Phi''(0) / 2 Phi'(0)`` and
    second moment ``Phi'''(0) / 3 Phi'(0)``. The *mean* overshoot is the constant
    Eq.~(C4) already carries. Its *variance*, divided by ``Phi'(0)^2`` to turn
    log-energy into depth, is the constant the variance needs, and it enters with
    a minus sign: a muon that overshoots further crossed its level sooner.

    Verified against a direct simulation of the same kernel to better than 2%
    over ``w = 3.5`` to ``9.2``, where the leading term alone is 8 to 25% high.

    Parameters
    ----------
    phi_prime, phi_second, phi_third : np.ndarray
        ``Phi'(0)``, ``-Phi''(0)`` and ``Phi'''(0)`` [km^-1].

    Returns
    -------
    variance_km2 : np.ndarray
        Overshoot variance expressed as a depth [km^2 w.e.].
    """
    log_energy_variance = phi_third / (3.0 * phi_prime) - phi_second**2 / (
        4.0 * phi_prime**2
    )
    return log_energy_variance / phi_prime**2


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def exact_moments_frozen(
    w: np.ndarray,
    phi_prime: np.ndarray,
    phi_second: np.ndarray,
    phi_third: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """First-passage moments with the kernel frozen at the production energy.

    Parameters
    ----------
    w : np.ndarray
        Log-loss to accumulate, ``ln(epsilon / E_stop)``.
    phi_prime, phi_second, phi_third : np.ndarray
        ``Phi'(0)``, ``-Phi''(0)`` and ``Phi'''(0)`` [km^-1] at the production
        energy.

    Returns
    -------
    mean_km, sigma_km : np.ndarray
        Mean and standard deviation of the range [km w.e.].
    """
    mean = w / phi_prime + phi_second / (2.0 * phi_prime**2)
    var = phi_second * w / phi_prime**3 - overshoot_variance(
        phi_prime, phi_second, phi_third
    )
    return mean, np.sqrt(var)


def exact_moments_running(
    start_energies_gev: np.ndarray,
    stop_energy_gev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """First-passage moments with the kernel followed down the trajectory.

    The rate at which log-energy is shed is a local quantity, so over a
    trajectory spanning decades the mean depth accumulates as ``d L / d ln E =
    1 / Phi'(0; E)`` and the variance as ``d Var / d ln E = -Phi''(0; E) /
    Phi'(0; E)^3``. The overshoot constant is evaluated at the stopping energy,
    where the crossing happens.

    Parameters
    ----------
    start_energies_gev : np.ndarray
        Muon energies at production [GeV].
    stop_energy_gev : float
        Energy at which the muon is counted as stopped [GeV].

    Returns
    -------
    mean_km, sigma_km : np.ndarray
        Mean and standard deviation of the range [km w.e.].
    """
    start = np.atleast_1d(np.asarray(start_energies_gev, dtype=float))
    log_max = np.log10(start.max())
    log_min = np.log10(stop_energy_gev)
    n_nodes = int(np.ceil((log_max - log_min) * RUNNING_NODES_PER_DECADE)) + 1
    log10_grid = np.linspace(log_min, log_max, n_nodes)
    moments = np.array([log_loss_moments(10.0**value) for value in log10_grid])
    phi_prime, phi_second, phi_third = moments[:, 0], moments[:, 1], moments[:, 2]

    ln_grid = log10_grid * np.log(10.0)
    mean_integrand = 1.0 / phi_prime
    var_integrand = phi_second / phi_prime**3
    mean_cumulative = np.concatenate(
        [[0.0], np.cumsum(np.diff(ln_grid) * 0.5 * (mean_integrand[1:] + mean_integrand[:-1]))]
    )
    var_cumulative = np.concatenate(
        [[0.0], np.cumsum(np.diff(ln_grid) * 0.5 * (var_integrand[1:] + var_integrand[:-1]))]
    )

    # Both constants belong to the crossing, so both are read at the stop energy.
    overshoot = phi_second[0] / (2.0 * phi_prime[0] ** 2)
    overshoot_var = overshoot_variance(phi_prime[0], phi_second[0], phi_third[0])
    mean = np.interp(np.log10(start), log10_grid, mean_cumulative) + overshoot
    var = np.interp(np.log10(start), log10_grid, var_cumulative) - overshoot_var
    return mean, np.sqrt(var)


def drift_diffusion_moments(
    w: np.ndarray,
    b_mu: np.ndarray,
    d_mu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """First-passage moments of the second-order (Brownian) transport.

    The drift-diffusion Green function is Gaussian in ``ln E`` with drift
    ``M = b_mu + d_mu / 2`` and variance rate ``d_mu``, so its first passage to a
    level ``w`` is inverse Gaussian.

    Parameters
    ----------
    w : np.ndarray
        Log-loss to accumulate.
    b_mu, d_mu : np.ndarray
        Drift and diffusion coefficients [km^-1].

    Returns
    -------
    mean_km, sigma_km : np.ndarray
        Mean and standard deviation of the range [km w.e.].
    """
    drift = b_mu + 0.5 * d_mu
    return w / drift, np.sqrt(w * d_mu / drift**3)


def published_moments(
    w: np.ndarray,
    b_mu: np.ndarray,
    d_mu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The range moments as published in Ref.~[Palmisano:2026sid] Sec. 5.2.

    Their estimator inverts the Gaussian at fixed log-loss rather than solving
    the first-passage problem, which leaves terms of order ``d_mu / M^2`` that
    :func:`drift_diffusion_moments` does not have. Reproduced here so that the
    size of that ambiguity can be read off the same table.

    Parameters
    ----------
    w : np.ndarray
        Log-loss to accumulate.
    b_mu, d_mu : np.ndarray
        Drift and diffusion coefficients [km^-1].

    Returns
    -------
    mean_km, sigma_km : np.ndarray
        Mean and standard deviation of the range [km w.e.].
    """
    drift = b_mu + 0.5 * d_mu
    mean = w / drift + d_mu / drift**2
    var = w * d_mu / drift**3 + 2.0 * d_mu**2 / drift**4
    return mean, np.sqrt(var)


# ---------------------------------------------------------------------------
# PROPOSAL Monte Carlo
# ---------------------------------------------------------------------------


def propagate_to_energy(
    start_energy_gev: float,
    stop_energy_gev: float,
    n_muons: int,
    ecut_mev: float,
    seed: int,
) -> np.ndarray:
    """Distance at which a muon first falls below a stopping energy.

    The estimator of Ref.~[Palmisano:2026sid], with the stopping level held
    fixed in energy rather than as a fraction of the production energy, so that
    the whole scan stays in the radiative regime. PROPOSAL reports the
    propagated distance in cm of its own medium, converted here to km of water
    equivalent at :data:`~softpaws.utils.constants.RHO_WATER_G_CM3` so that the
    result is directly comparable to kernel moments in km^-1.

    Parameters
    ----------
    start_energy_gev : float
        Muon energy at production [GeV].
    stop_energy_gev : float
        Energy at which the muon is counted as stopped [GeV].
    n_muons : int
        Number of muons to propagate.
    ecut_mev : float
        Energy cut [MeV] separating stochastic from continuous losses.
    seed : int
        Seed for PROPOSAL's random generator.

    Returns
    -------
    range_km : np.ndarray, shape (n_muons,)
        First-passage distance [km w.e.].
    """
    import proposal as pp

    particle = pp.particle.MuMinusDef()
    medium = pp.medium.Water()
    cuts = pp.EnergyCutSettings(ecut_mev, 1, False)
    cross_sections = [
        pp.crosssection.make_crosssection(param, particle, medium, cuts, True)
        for param in proposal_parametrizations().values()
    ]

    collection = pp.PropagationUtilityCollection()
    collection.displacement = pp.make_displacement(cross_sections, True)
    collection.interaction = pp.make_interaction(cross_sections, True)
    collection.time = pp.make_time(cross_sections, particle, True)
    utility = pp.PropagationUtility(collection=collection)
    geometry = pp.geometry.Sphere(pp.Cartesian3D(0, 0, 0), 1.0e20)
    density = pp.density_distribution.density_homogeneous(medium.mass_density)
    propagator = pp.Propagator(particle, [(geometry, utility, density)])

    pp.RandomGenerator.get().set_seed(seed)
    min_energy_mev = stop_energy_gev * 1.0e3
    # Far beyond any range at these energies; the stop is set by min_energy.
    max_distance_cm = 1.0e3 * CM_PER_KM
    to_kmwe = medium.mass_density / RHO_WATER_G_CM3 / CM_PER_KM

    distances = np.empty(n_muons)
    for index in range(n_muons):
        state = pp.particle.ParticleState()
        state.type = particle.particle_type
        state.position = pp.Cartesian3D(0, 0, 0)
        state.direction = pp.Cartesian3D(0, 0, 1)
        state.energy = start_energy_gev * 1.0e3
        state.propagated_distance = 0.0
        track = propagator.propagate(
            state, max_distance=max_distance_cm, min_energy=min_energy_mev
        )
        distances[index] = track.final_state().propagated_distance * to_kmwe
    return distances


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def run_scan(
    log10_start: tuple[float, ...],
    stop_energy_gev: float,
    n_muons: int,
    ecut_mev: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Propagate at each production energy and assemble every prediction.

    Parameters
    ----------
    log10_start : tuple of float
        ``log10(epsilon / GeV)`` at production.
    stop_energy_gev : float
        Stopping energy [GeV].
    n_muons : int
        Muons per production energy.
    ecut_mev : float
        Energy cut [MeV].
    seed : int
        Base seed; each production energy is offset from it.

    Returns
    -------
    results : dict
        Monte Carlo moments with their errors, and the four model predictions,
        each a ``(mean, sigma)`` pair.
    """
    start = 10.0 ** np.asarray(log10_start, dtype=float)
    w = np.log(start / stop_energy_gev)

    mc_mean = np.empty_like(w)
    mc_sigma = np.empty_like(w)
    for index, energy in enumerate(start):
        sample = propagate_to_energy(energy, stop_energy_gev, n_muons, ecut_mev, seed + index)
        mc_mean[index] = sample.mean()
        mc_sigma[index] = sample.std(ddof=1)

    moments = np.array([log_loss_moments(energy) for energy in start])
    phi_prime, phi_second, phi_third = moments[:, 0], moments[:, 1], moments[:, 2]
    b_mu = np.ravel(drift_coefficient(start))
    d_mu = np.ravel(diffusion_coefficient(start))

    return {
        "start": start,
        "w": w,
        "phi_prime": phi_prime,
        "phi_second": phi_second,
        "b_mu": b_mu,
        "d_mu": d_mu,
        "mc_mean": mc_mean,
        "mc_sigma": mc_sigma,
        "mc_mean_err": mc_sigma / np.sqrt(n_muons),
        "mc_sigma_err": mc_sigma / np.sqrt(2.0 * (n_muons - 1)),
        "frozen": exact_moments_frozen(w, phi_prime, phi_second, phi_third),
        "running": exact_moments_running(start, stop_energy_gev),
        "diffusion": drift_diffusion_moments(w, b_mu, d_mu),
        "published": published_moments(w, b_mu, d_mu),
    }


def report_scan(results: dict[str, np.ndarray], stop_energy_gev: float, n_muons: int) -> None:
    """Print the mean and spread of the range against every prediction.

    Parameters
    ----------
    results : dict
        Output of :func:`run_scan`.
    stop_energy_gev : float
        Stopping energy [GeV].
    n_muons : int
        Muons per production energy, for the header.
    """
    models = ("frozen", "running", "diffusion", "published")
    labels = {"frozen": "exact, frozen", "running": "exact, running",
              "diffusion": "drift-diff", "published": "published"}

    print(f"Range from production energy down to {stop_energy_gev:.3g} GeV in water, "
          f"{n_muons} muons per point")
    print()
    header = (f"{'log10(eps)':>10} {'w':>5} {'<R> MC':>16} "
              + " ".join(f"{labels[name]:>14}" for name in models))
    print(header)
    print("-" * len(header))
    for index in range(results["w"].size):
        row = " ".join(f"{results[name][0][index]:>14.3f}" for name in models)
        print(f"{np.log10(results['start'][index]):>10.1f} {results['w'][index]:>5.2f} "
              f"{results['mc_mean'][index]:>8.3f} +-{results['mc_mean_err'][index]:<5.3f} {row}")
    print()

    header = (f"{'log10(eps)':>10} {'w':>5} {'sigma_R MC':>16} "
              + " ".join(f"{labels[name]:>14}" for name in models))
    print(header)
    print("-" * len(header))
    for index in range(results["w"].size):
        row = " ".join(f"{results[name][1][index]:>14.3f}" for name in models)
        print(f"{np.log10(results['start'][index]):>10.1f} {results['w'][index]:>5.2f} "
              f"{results['mc_sigma'][index]:>8.3f} +-{results['mc_sigma_err'][index]:<5.3f} {row}")
    print()

    print(f"{'model':>16} {'<R> worst':>11} {'<R> rms':>9} {'sigma worst':>12} {'sigma rms':>10}")
    for name in models:
        mean_dev = 100.0 * (results[name][0] / results["mc_mean"] - 1.0)
        sigma_dev = 100.0 * (results[name][1] / results["mc_sigma"] - 1.0)
        print(f"{labels[name]:>16} {np.abs(mean_dev).max():>10.1f}% "
              f"{np.sqrt(np.mean(mean_dev**2)):>8.1f}% "
              f"{np.abs(sigma_dev).max():>11.1f}% {np.sqrt(np.mean(sigma_dev**2)):>9.1f}%")
    print()


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def figure_moments(
    results: dict[str, np.ndarray],
    sample_km: np.ndarray,
    reference_energy_gev: float,
    stop_energy_gev: float,
    stem: pathlib.Path,
) -> None:
    """Draw the sampled range and the spread against the two transports.

    Parameters
    ----------
    results : dict
        Output of :func:`run_scan`.
    sample_km : np.ndarray
        Sampled ranges [km w.e.] at ``reference_energy_gev``.
    reference_energy_gev : float
        Production energy of the histogrammed sample [GeV].
    stop_energy_gev : float
        Stopping energy [GeV].
    stem : pathlib.Path
        Output path without extension; ``.pdf`` and ``.png`` are written.
    """
    plt.style.use(_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    w_single = np.array([np.log(reference_energy_gev / stop_energy_gev)])
    phi_prime, phi_second, phi_third = log_loss_moments(reference_energy_gev)
    b_mu = np.ravel(drift_coefficient(reference_energy_gev))
    d_mu = np.ravel(diffusion_coefficient(reference_energy_gev))

    ax = axes[0]
    ax.hist(sample_km, bins=60, density=True, color="0.82", edgecolor="none",
            label="PROPOSAL")
    for label, moments, colour in (
        ("exact", exact_moments_frozen(w_single, np.array([phi_prime]),
                                       np.array([phi_second]),
                                       np.array([phi_third])), "C0"),
        ("drift-diffusion", drift_diffusion_moments(w_single, b_mu, d_mu), "C3"),
    ):
        mean = float(np.ravel(moments[0])[0])
        sigma = float(np.ravel(moments[1])[0])
        ax.axvline(mean, color=colour, lw=1.4, label=label)
        ax.axvspan(mean - sigma, mean + sigma, color=colour, alpha=0.13, lw=0)
    ax.axvline(sample_km.mean(), color="k", lw=1.4, ls=":", label="MC mean")
    ax.set_xlabel(r"$R$ [km w.e.]")
    ax.set_ylabel(r"$p(R)$")
    ax.set_title(rf"(a) $\varepsilon = 10^{{{np.log10(reference_energy_gev):.0f}}}$ GeV")
    ax.legend(frameon=False, fontsize="small")

    ax = axes[1]
    log10_start = np.log10(results["start"])
    ax.errorbar(log10_start, results["mc_sigma"], yerr=results["mc_sigma_err"], fmt="o",
                color="k", ms=3.5, lw=1.0, label="PROPOSAL")
    ax.plot(log10_start, results["running"][1], color="C0", lw=1.4, label="exact")
    ax.plot(log10_start, results["diffusion"][1], color="C3", lw=1.4, ls="--",
            label="drift-diffusion")
    ax.plot(log10_start, results["published"][1], color="C2", lw=1.2, ls=":",
            label="published estimator")
    ax.set_xlabel(r"$\log_{10}(\varepsilon/\mathrm{GeV})$")
    ax.set_ylabel(r"$\sigma_R$ [km w.e.]")
    ax.set_title("(b) spread of the range")
    ax.legend(frameon=False, fontsize="small")

    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(suffix))
    plt.close(fig)
    print(f"wrote {stem.with_suffix('.pdf')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the output figure.")
    parser.add_argument("--stop-energy-gev", type=float, default=STOP_ENERGY_GEV,
                        help="Energy at which the muon is counted as stopped [GeV]; "
                        "keep it well above the critical energy.")
    parser.add_argument("--reference-energy-gev", type=float, default=REFERENCE_ENERGY_GEV,
                        help="Production energy of the histogrammed sample [GeV].")
    parser.add_argument("--n-muons", type=int, default=2000,
                        help="Muons propagated per production energy.")
    parser.add_argument("--ecut-mev", type=float, default=500.0,
                        help="Energy cut [MeV] above which losses are stochastic.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Base seed for PROPOSAL's random generator.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = run_scan(
        LOG10_START_ENERGY_SCAN, args.stop_energy_gev, args.n_muons,
        args.ecut_mev, args.seed,
    )
    report_scan(results, args.stop_energy_gev, args.n_muons)
    sample = propagate_to_energy(
        args.reference_energy_gev, args.stop_energy_gev, args.n_muons,
        args.ecut_mev, args.seed + 100,
    )
    figure_moments(
        results, sample, args.reference_energy_gev, args.stop_energy_gev,
        args.out_dir / "39_range_moments",
    )


if __name__ == "__main__":
    main()
