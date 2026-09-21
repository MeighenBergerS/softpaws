"""Tests for the muon log-loss distribution.

Lock in the exactness identities of the Mellin symbol (``Phi(1) = b_mu``,
``Phi(2) = 2 b_mu - d_mu``), the normalization and heavy tail of the
characteristic-function-inverted density, and -- as a slow Monte-Carlo
cross-check -- that the inversion reproduces the physical jump process it stands
in for.
"""

import numpy as np
import pytest

from softpaws.transport.coefficients import (
    diffusion_coefficient,
    drift_coefficient,
    third_moment_coefficient,
)
from softpaws.transport.eigenvalue import (
    phi_eigenvalue,
    phi_symbol,
    phi_symbol_three_moment,
    three_moment_loss_spectrum,
    two_moment_loss_spectrum,
)
from softpaws.transport.loss_distribution import (
    _three_moment_mean_rate,
    gaussian_survival,
    invert_log_loss_symbol,
    loss_density,
    loss_density_running,
    loss_density_three_moment,
    running_log_loss_symbol,
    survival_from_density,
)

E_100PEV = 1.0e8  # GeV
B_MU = float(drift_coefficient(E_100PEV)[0])
D_MU = float(diffusion_coefficient(E_100PEV)[0])


# ---------------------------------------------------------------------------
# Mellin symbol Phi(s)
# ---------------------------------------------------------------------------


def test_phi_symbol_exactness_identities():
    # Phi(1) = b_mu and Phi(2) = 2 b_mu - d_mu, exactly, for any (b, d).
    kappa, p = two_moment_loss_spectrum(B_MU, D_MU)
    assert phi_symbol(1.0, kappa, p) == pytest.approx(B_MU)
    assert phi_symbol(2.0, kappa, p) == pytest.approx(2.0 * B_MU - D_MU)


def test_phi_symbol_matches_real_eigenvalue():
    kappa, p = two_moment_loss_spectrum(B_MU, D_MU)
    a = np.array([0.5, 0.98, 1.5, 3.0])
    np.testing.assert_allclose(
        phi_symbol(a, kappa, p),
        phi_eigenvalue(a, B_MU, D_MU),
    )


def test_phi_symbol_preserves_complex_dtype():
    # The CF inversion needs Phi at s = -i k; the real eigenvalue would drop Im.
    kappa, p = two_moment_loss_spectrum(B_MU, D_MU)
    value = phi_symbol(-1j * np.array([0.0, 1.0, 2.0]), kappa, p)
    assert np.iscomplexobj(value)
    # At k = 0 the symbol is real and vanishes: Phi(0) = 0.
    assert value[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Exact density by CF inversion
# ---------------------------------------------------------------------------


def _w_grid(ell_km, n=4000):
    mean = (B_MU + D_MU / 2.0) * ell_km
    return np.linspace(1e-3, mean + 9.0 * np.sqrt(D_MU * ell_km), n)


def test_loss_density_normalized():
    ell = 3.0
    w = _w_grid(ell)
    density = loss_density(w, ell, B_MU, D_MU)
    assert np.all(density >= 0.0)
    assert np.trapezoid(density, w) == pytest.approx(1.0, rel=1e-6)


def test_loss_density_tail_is_heavier_than_gaussian():
    # The whole point: the exact tail dominates the Fokker-Planck Gaussian well
    # beyond the peak, where a single event's parent energy is inferred.
    ell = 3.0
    w = _w_grid(ell)
    density = loss_density(w, ell, B_MU, D_MU)
    threshold = (B_MU + D_MU / 2.0) * ell + 4.0 * np.sqrt(D_MU * ell)
    exact = survival_from_density(threshold, w, density)
    gauss = gaussian_survival(threshold, ell, B_MU, D_MU)
    assert exact > 10.0 * gauss


def test_gaussian_survival_half_at_mean():
    ell = 3.0
    mean = (B_MU + D_MU / 2.0) * ell
    assert gaussian_survival(mean, ell, B_MU, D_MU) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Slow Monte-Carlo cross-check of the physical jump process
# ---------------------------------------------------------------------------


def _sample_loss_mc(ell_km, b_mu, d_mu, n_samples, ymin=1e-7, seed=0):
    """Draw ``w`` from the compound-Poisson jump process (no expansion).

    Same two-moment family as :func:`loss_density`, sampled directly: in a column
    ``ell`` the process fires ``Poisson(Gamma ell)`` times, each drawing ``y`` from
    the normalized loss spectrum and adding ``-ln(1 - y)`` to ``w``.
    """
    rng = np.random.default_rng(seed)
    kappa, p = two_moment_loss_spectrum(b_mu, d_mu)
    yy = np.logspace(np.log10(ymin), 0.0, 4000)
    pdf = kappa * (1.0 - yy) ** p / yy
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(yy))])
    total_rate = cdf[-1]  # per km, with the ymin cutoff
    cdf_norm = cdf / total_rate

    n_fire = rng.poisson(total_rate * ell_km, n_samples)
    total = int(n_fire.sum())
    w = np.zeros(n_samples)
    if total:
        ys = np.interp(rng.random(total), cdf_norm, yy)
        np.add.at(w, np.repeat(np.arange(n_samples), n_fire), -np.log1p(-ys))
    return w


@pytest.mark.slow
def test_cf_inversion_matches_monte_carlo_tail():
    ell = 5.0
    samples = _sample_loss_mc(ell, B_MU, D_MU, n_samples=400_000, seed=1)
    w = _w_grid(ell)
    density = loss_density(w, ell, B_MU, D_MU)
    mean = (B_MU + D_MU / 2.0) * ell
    for z in (1.0, 2.0, 3.0):
        threshold = mean + z * np.sqrt(D_MU * ell)
        mc = np.mean(samples > threshold)
        cf = survival_from_density(threshold, w, density)
        # Statistical tolerance loosens as the tail thins; require agreement to
        # 20% (relative) or 3 sigma of the MC estimate, whichever is looser.
        mc_sigma = np.sqrt(max(mc, 1.0 / samples.size) / samples.size)
        assert cf == pytest.approx(mc, rel=0.2, abs=3.0 * mc_sigma)


# ---------------------------------------------------------------------------
# Three-moment family
# ---------------------------------------------------------------------------

T_MU = float(third_moment_coefficient(E_100PEV)[0])


def test_three_moment_density_normalized():
    ell = 3.0
    w = _w_grid(ell)
    density = loss_density_three_moment(w, ell, B_MU, D_MU, T_MU)
    assert np.trapezoid(density, w) == pytest.approx(1.0)
    assert np.all(density >= 0.0)


def test_three_moment_tail_heavier_than_two_moment():
    # The whole point of the third moment: it restores the hard end of
    # dGamma/dy, and with it the catastrophic-loss tail that the two-moment
    # family truncates. Checked well out in the tail, where the gap is orders
    # of magnitude (see examples/27).
    ell = 1.0
    # Its own grid, not _w_grid: that one is sized to the Gaussian scale and
    # would truncate the very tail under test.
    w = np.linspace(1e-3, 12.0, 3_000)
    three = loss_density_three_moment(w, ell, B_MU, D_MU, T_MU)
    two = loss_density(w, ell, B_MU, D_MU)
    threshold = 4.0
    assert survival_from_density(threshold, w, three) > 5.0 * survival_from_density(
        threshold, w, two
    )


def test_three_moment_density_matches_its_own_symbol():
    # The inverted density must carry the moments of the symbol that generated
    # it: <e^-w> = exp(-ell Phi(1)) = exp(-ell b_mu).
    ell = 2.0
    # n_w < n_k, or the inversion aliases; see invert_log_loss_symbol.
    w = np.linspace(1e-4, 40.0, 3_000)
    density = loss_density_three_moment(w, ell, B_MU, D_MU, T_MU)
    laplace = np.trapezoid(np.exp(-w) * density, w)
    assert laplace == pytest.approx(np.exp(-ell * B_MU), rel=2e-3)


def test_invert_symbol_reproduces_two_moment_density():
    # loss_density is a thin wrapper over invert_log_loss_symbol; going through
    # the general entry point by hand must give the identical answer.
    ell = 3.0
    w = _w_grid(ell)
    kappa, p = two_moment_loss_spectrum(B_MU, D_MU)
    direct = invert_log_loss_symbol(w, ell, lambda s: phi_symbol(s, kappa, p))
    np.testing.assert_allclose(direct, loss_density(w, ell, B_MU, D_MU), rtol=1e-12)


def test_inversion_rejects_aliasing_grid():
    # A w grid finer than the k grid silently returns the wrong density, so it
    # must be refused rather than quietly inverted.
    kappa, p = two_moment_loss_spectrum(B_MU, D_MU)
    w = np.linspace(1e-4, 40.0, 20_000)
    with pytest.raises(ValueError, match="aliases the inversion"):
        invert_log_loss_symbol(w, 2.0, lambda s: phi_symbol(s, kappa, p), n_k=2**14)


# ---------------------------------------------------------------------------
# The kernel followed down the descent
# ---------------------------------------------------------------------------


def test_three_moment_mean_rate_is_the_symbol_derivative():
    # Phi'(0) in closed form against a finite difference of the symbol, at an
    # energy where q is well away from its digamma limit.
    kappa, q, p = three_moment_loss_spectrum(B_MU, D_MU, T_MU)
    step = 1.0e-6
    numeric = (
        phi_symbol_three_moment(step, float(kappa), float(q), float(p))
        - phi_symbol_three_moment(0.0, float(kappa), float(q), float(p))
    ) / step
    assert float(_three_moment_mean_rate(kappa, q, p)) == pytest.approx(float(numeric), rel=1e-4)


def test_running_symbol_vanishes_at_zero_depth_and_grows_with_depth():
    psi = running_log_loss_symbol(1.0, [0.0, 1.0, 2.0], E_100PEV)
    assert psi.shape == (3,)
    assert psi[0] == 0.0
    assert 0.0 < psi[1].real < psi[2].real


def test_running_symbol_rejects_bad_input():
    with pytest.raises(ValueError, match="energy_gev"):
        running_log_loss_symbol(1.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="ell_km"):
        running_log_loss_symbol(1.0, -1.0, E_100PEV)
    with pytest.raises(ValueError, match="floor_gev"):
        running_log_loss_symbol(1.0, 1.0, E_100PEV, floor_gev=0.0)


def test_running_density_normalized_and_matches_frozen_on_a_short_path():
    # Over a path where the muon sheds a third of an e-fold the kernel barely
    # moves, so the running law must reproduce the frozen one.
    ell = 0.5
    w = np.linspace(0.0, 30.0, 1_500)
    running = loss_density_running(w, ell, E_100PEV)
    frozen = loss_density_three_moment(w, ell, B_MU, D_MU, T_MU)
    assert np.trapezoid(running, w) == pytest.approx(1.0)
    assert np.all(running >= 0.0)
    assert np.trapezoid(w * running, w) == pytest.approx(np.trapezoid(w * frozen, w), rel=5e-3)


def test_running_density_mean_is_the_symbol_derivative():
    # The mean of the inverted law is -dPsi/ds at zero, i.e. the mean log-loss
    # accumulated along the descent, which ties the density to the symbol.
    ell = 10.0
    step = 1.0e-5
    psi = running_log_loss_symbol(np.array([step, 0.0]), ell, E_100PEV)
    from_symbol = float((psi[0] - psi[1]).real) / step
    w = np.linspace(0.0, 60.0, 3_000)
    density = loss_density_running(w, ell, E_100PEV)
    assert np.trapezoid(w * density, w) == pytest.approx(from_symbol, rel=1e-3)


def test_running_kernel_loses_less_than_frozen_from_the_top_of_the_table():
    # From 1e14 GeV the drift halves on the way down, so freezing it at the
    # top overstates the loss; the running law keeps far more energy.
    energy = 1.0e14
    ell = 34.0
    w = np.linspace(0.0, 120.0, 4_000)
    b, d, t = (
        float(f(energy)[0])
        for f in (drift_coefficient, diffusion_coefficient, third_moment_coefficient)
    )
    frozen = loss_density_three_moment(w, ell, b, d, t)
    running = loss_density_running(w, ell, energy)
    assert np.trapezoid(w * running, w) < 0.5 * np.trapezoid(w * frozen, w)
