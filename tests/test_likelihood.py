"""Tests for the Bayesian inference layer (comparison/likelihood.py).

Cover the building blocks that reproduce the paper's Section 4 figures: the
Poisson likelihood and priors, linearity of the signal in the flux
normalization, the atmospheric background template, a fast MCMC smoke run, and
the Eq. (4.6) cross-section solver -- including the pole behaviour that separates
the FP and exact treatments (docs/exact_soft_volume_notes.md Part 11).
"""

import numpy as np
import pytest

from softpaws.comparison.likelihood import (
    FitConfig,
    atmospheric_template,
    cross_section_enhancement,
    free_param_names,
    log_prior,
    poisson_log_likelihood,
    required_lambda,
    run_mcmc,
    signal_counts,
)

EDGES = np.arange(4.0, 9.01, 0.5)
YEAR_S = 365.25 * 86400.0


def diffusion_config(**kw) -> FitConfig:
    base = dict(radius_km=0.62, log10_e_edges=EDGES, livetime_s=9.5 * YEAR_S,
                solid_angle_sr=2.0 * np.pi, method="diffusion")
    base.update(kw)
    return FitConfig(**base)


# ---------------------------------------------------------------------------
# Poisson likelihood and priors
# ---------------------------------------------------------------------------


def test_poisson_ll_maximized_at_truth():
    truth = np.array([5.0, 10.0, 2.0])
    ll_truth = poisson_log_likelihood(truth, truth)
    ll_off = poisson_log_likelihood(truth, truth * 1.5)
    assert ll_truth > ll_off


def test_poisson_ll_rejects_nonpositive_prediction():
    assert poisson_log_likelihood(np.array([1.0]), np.array([0.0])) == -np.inf


def test_free_param_names_by_method():
    assert free_param_names("drift") == ["phi0", "gamma", "b_scale", "bkg_norm"]
    assert "d_scale" in free_param_names("diffusion")
    assert "d_scale" in free_param_names("exact")


def test_log_prior_rejects_out_of_range():
    # phi0 negative -> -inf
    assert log_prior([-1.0, 2.38, 0.94, 1.5, 1.0], "diffusion") == -np.inf
    # gamma above range -> -inf
    assert log_prior([0.7, 9.0, 0.94, 1.5, 1.0], "diffusion") == -np.inf
    # in-range is finite
    assert np.isfinite(log_prior([0.7, 2.38, 0.94, 1.5, 1.0], "diffusion"))


# ---------------------------------------------------------------------------
# Signal and background templates
# ---------------------------------------------------------------------------


def test_signal_scales_linearly_with_phi0():
    cfg = diffusion_config()
    s1 = signal_counts(cfg, 1.0, 2.38, 0.94, 1.5)
    s3 = signal_counts(cfg, 3.0, 2.38, 0.94, 1.5)
    np.testing.assert_allclose(s3, 3.0 * s1, rtol=1e-12)


def test_generic_origin_adds_to_numu_only_signal():
    # origin="generic" (numu + tau-induced) must exceed the numu-only signal.
    exact_cfg = diffusion_config(method="exact", column_depth_km=1.95)
    generic_cfg = diffusion_config(method="exact", column_depth_km=1.95, origin="generic")
    numu_only = signal_counts(exact_cfg, 0.7, 2.38, 0.94, 1.5)
    generic = signal_counts(generic_cfg, 0.7, 2.38, 0.94, 1.5)
    assert np.all(generic > numu_only)


def test_generic_origin_requires_exact_method():
    cfg = diffusion_config(origin="generic")  # method="diffusion" by default
    with pytest.raises(ValueError):
        signal_counts(cfg, 0.7, 2.38, 0.94, 1.5)


def test_atmospheric_template_normalized():
    t = atmospheric_template(EDGES)
    assert t.sum() == pytest.approx(1.0, rel=1e-12)
    # A steeply falling spectrum puts most weight in the lowest bin.
    assert np.argmax(t) == 0


def test_atmospheric_template_slope_effect():
    steep = atmospheric_template(EDGES, slope=3.7)
    flat = atmospheric_template(EDGES, slope=2.0)
    # Steeper spectrum is more concentrated in the first bin.
    assert steep[0] > flat[0]


# ---------------------------------------------------------------------------
# MCMC smoke test
# ---------------------------------------------------------------------------


def test_mcmc_recovers_injected_truth():
    cfg = diffusion_config()
    template = atmospheric_template(EDGES)
    truth = signal_counts(cfg, 0.7, 2.38, 0.94, 1.5)
    observed = np.round(truth + 20.0 * template)

    samples = run_mcmc(cfg, observed, background=template,
                       n_walkers=16, n_steps=600, n_burn=200, seed=1)
    assert np.median(samples["phi0"]) == pytest.approx(0.7, abs=0.4)
    assert np.median(samples["gamma"]) == pytest.approx(2.38, abs=0.3)
    assert "d_mu" in samples  # diffusion model exposes the physical coefficient


# ---------------------------------------------------------------------------
# Eq. (4.6) cross-section solver and the pole
# ---------------------------------------------------------------------------


def test_required_lambda_reproduces_one_event():
    cfg = diffusion_config()
    lam = required_lambda(cfg, phi0=0.7, gamma=2.38, e_star_gev=1.0e8)
    assert np.isfinite(lam)
    # Re-evaluating Eq. (4.6) at the solved lambda must give exactly one event.
    from softpaws.comparison.likelihood import _events_at_energy

    assert _events_at_energy(cfg, 0.7, 2.38, lam, 1.0e8) == pytest.approx(1.0, rel=1e-3)


def test_fp_required_lambda_stays_below_pole():
    # FP (diffusion) drift form: solution must be below the pole lambda = gamma - 1.
    cfg = diffusion_config()
    gamma = 2.38
    lam = required_lambda(cfg, phi0=0.7, gamma=gamma, e_star_gev=1.0e8)
    assert lam < gamma - 1.0


def test_exact_required_lambda_can_exceed_pole():
    # Exact form with a finite column stays finite and can cross the FP pole.
    cfg = diffusion_config(method="exact", column_depth_km=1.95)
    gamma = 2.38
    lam = required_lambda(cfg, phi0=0.7, gamma=gamma, e_star_gev=1.0e8)
    assert np.isfinite(lam)
    assert lam > gamma - 1.0


def test_cross_section_enhancement_at_100pev():
    # R = (E*/E0)^(lambda - 0.4) with E*/E0 = 100PeV / 10PeV = 10.
    assert cross_section_enhancement(1.4, 1.0e8) == pytest.approx(10.0 ** (1.4 - 0.4), rel=1e-9)
