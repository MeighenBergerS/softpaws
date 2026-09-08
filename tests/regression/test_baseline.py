"""Regression tests against the numbers the paper quotes.

Each test reads one block of ``baseline.json`` and recomputes it with the
library. The blocks that still need an example hub (effective areas, fits,
event benchmark) get their tests as those hubs move into the library.
"""

import json
import pathlib

import numpy as np
import pytest

from softpaws.transport import coefficients, eigenvalue, loss_distribution, muon_range

BASELINE = json.loads((pathlib.Path(__file__).parent / "baseline.json").read_text())


def _scalar(value):
    return float(np.asarray(value).ravel()[0])


def _kernel(energy_gev, source="proposal"):
    b = _scalar(coefficients.drift_coefficient(energy_gev, source=source))
    d = _scalar(coefficients.diffusion_coefficient(energy_gev, source=source))
    t = _scalar(coefficients.third_moment_coefficient(energy_gev, source=source))
    phi_prime = _scalar(coefficients.log_loss_moments(energy_gev, source=source)[0])
    return b, d, t, phi_prime


@pytest.mark.parametrize(
    "block, energy_gev, source",
    [
        ("kernel_water_1pev", 1.0e6, "proposal"),
        ("kernel_water_100pev", 1.0e8, "proposal"),
        ("kernel_rock_1pev", 1.0e6, "proposal_rock"),
    ],
)
def test_kernel_moments(block, energy_gev, source):
    """Table E.2: drift and log-loss rate of the shipped PROPOSAL tables."""
    want = BASELINE[block]
    b, d, t, phi_prime = _kernel(energy_gev, source)
    # The paper's table, at its (loose) tolerance.
    rtol = want["rtol"]
    assert b == pytest.approx(want["b_mu_per_km"], rel=rtol)
    assert phi_prime == pytest.approx(want["phi_prime_0_per_km"], rel=rtol)
    # The pre-cleanup library, tightly: the refactor must not move these.
    pinned = want["library_pre_cleanup"]
    assert b == pytest.approx(pinned["b_mu_per_km"], rel=1e-3)
    assert d == pytest.approx(pinned["d_mu_per_km"], rel=1e-3)
    assert t == pytest.approx(pinned["t_mu_per_km"], rel=1e-3)
    assert phi_prime == pytest.approx(pinned["phi_prime_0_per_km"], rel=1e-3)


def test_transport_exponent_table():
    """Table D.1: the exponent against its drift and second-order truncations."""
    want = BASELINE["transport_exponent_water_1pev"]
    b, d, t, _ = _kernel(1.0e6)
    a = np.asarray(want["A"])
    atol = want["atol_per_km"]
    np.testing.assert_allclose(
        eigenvalue.phi_eigenvalue_three_moment(a, b, d, t), want["phi_per_km"], atol=atol
    )
    np.testing.assert_allclose(eigenvalue.phi_drift(a, b), want["drift_per_km"], atol=atol)
    np.testing.assert_allclose(
        eigenvalue.phi_fokker_planck(a, b, d), want["phi2_per_km"], atol=atol
    )


def test_loss_law_table():
    """Table D.2: survival above a log loss after 1 km of water at 1 PeV."""
    want = BASELINE["loss_law_1pev_1km_water"]
    b, d, t, _ = _kernel(1.0e6)
    w_grid = np.linspace(-2.0, 12.0, 4001)
    w_query = np.asarray(want["w"])
    rtol = want["rtol"]

    three = loss_distribution.loss_density_three_moment(w_grid, 1.0, b, d, t)
    got = [loss_distribution.survival_from_density(w, w_grid, three) for w in w_query]
    np.testing.assert_allclose(got, want["three_moment"], rtol=rtol)

    # The two-moment and Gaussian tails at w = 3 sit below the resolution of
    # this grid, so the check stops at w = 2 for them.
    two = loss_distribution.loss_density(w_grid, 1.0, b, d)
    got = [loss_distribution.survival_from_density(w, w_grid, two) for w in w_query[:4]]
    np.testing.assert_allclose(got, want["two_moment"][:4], rtol=rtol)

    gauss = loss_distribution.loss_density_gaussian(w_grid, 1.0, b, d)
    got = [loss_distribution.survival_from_density(w, w_grid, gauss) for w in w_query[:4]]
    np.testing.assert_allclose(got, want["gaussian"][:4], rtol=rtol)


def test_range_table_kernel_rows():
    """Table C.1, upper rows: the kernel moments per energy."""
    want = BASELINE["range_to_threshold_water_1tev"]
    pinned = want["library_pre_cleanup"]
    energies = 10.0 ** np.asarray(want["log10_e"], dtype=float)
    b = np.asarray(coefficients.drift_coefficient(energies)).ravel()
    phi_prime = np.asarray(coefficients.log_loss_moments(energies)[0]).ravel()
    np.testing.assert_allclose(b, want["b_mu_per_km"], rtol=want["rtol"])
    np.testing.assert_allclose(phi_prime, want["phi_prime_0_per_km"], rtol=want["rtol"])
    np.testing.assert_allclose(b, pinned["b_mu_per_km"], rtol=1e-3)
    np.testing.assert_allclose(phi_prime, pinned["phi_prime_0_per_km"], rtol=1e-3)


@pytest.mark.parametrize(
    "key, kwargs, function",
    [
        ("csda_km", {}, "muon_range_km"),
        ("csda_frozen_km", {"kernel_evaluation": "frozen"}, "muon_range_km"),
        ("first_passage_running_km", {}, "stochastic_muon_range_km"),
        ("first_passage_frozen_km", {"kernel_evaluation": "frozen"}, "stochastic_muon_range_km"),
        (
            "first_passage_no_ionization_km",
            {"include_ionization": False},
            "stochastic_muon_range_km",
        ),
    ],
)
def test_range_table_pinned(key, kwargs, function):
    """Table C.1, range rows, pinned to the pre-cleanup library.

    The paper's rows are machine-written from the library by
    ``make_transport_table.py`` in the scripts directory, so they agree with these to the printed
    digits and the refactor is held to today's values.
    """
    want = BASELINE["range_to_threshold_water_1tev"]
    energies = 10.0 ** np.asarray(want["log10_e"], dtype=float)
    got = getattr(muon_range, function)(energies, 1.0e3, **kwargs)
    np.testing.assert_allclose(got, want["library_pre_cleanup"][key], rtol=1e-3)
