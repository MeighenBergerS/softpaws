"""Tests for softpaws.comparison.rates.

Uses small synthetic ``EffectiveArea``/``SmearingMatrix`` objects (rather than
the real ~500 MB DR2 tables) so :func:`irf_expected_counts` can be checked
against an exact hand-computed count.
"""

import numpy as np
import pytest

from softpaws.comparison.rates import (
    fit_component_scales,
    fit_scale_factor,
    fit_scale_factor_with_background,
    implied_efficiency,
    irf_expected_counts,
    irf_expected_counts_directional,
    observed_counts,
    soft_volume_smeared_counts,
)
from softpaws.data.container import EventSet
from softpaws.data.schema import EVENTS_DTYPE
from softpaws.response.irfs import EffectiveArea, SmearingMatrix

# ---------------------------------------------------------------------------
# Synthetic single-bin IRF: one true-energy bin (1 PeV-10 PeV), one
# declination bin spanning the downgoing hemisphere (-90 to 0 deg), and full
# efficiency (Fractional_Counts = 1) landing entirely in a known reco bin.
# ---------------------------------------------------------------------------

A_EFF_CM2 = 2.0e4
FLUX_GEV_CM2_S_SR = 3.0e-19  # flat (energy-independent) test flux


@pytest.fixture
def synthetic_aeff() -> EffectiveArea:
    log10_e_edges = np.array([6.0, 7.0])
    sin_dec_edges = np.array([-1.0, 0.0])
    values = np.array([[A_EFF_CM2]])
    return EffectiveArea(log10_e_edges, sin_dec_edges, values)


@pytest.fixture
def synthetic_smearing() -> SmearingMatrix:
    # Columns: Enu_min, Enu_max, dec_min, dec_max, Ereco_min, Ereco_max,
    # PSF_min, PSF_max, AngErr_min, AngErr_max, Fractional_Counts.
    raw = np.array([[6.0, 7.0, -90.0, 0.0, 6.0, 6.01, 0.0, 0.0, 0.0, 0.0, 1.0]])
    return SmearingMatrix(raw)


def flat_flux(energy_gev: np.ndarray) -> np.ndarray:
    return np.full_like(np.asarray(energy_gev, dtype=float), FLUX_GEV_CM2_S_SR)


def test_irf_expected_counts_matches_hand_computed_value(synthetic_aeff, synthetic_smearing):
    log10_e_reco_edges = np.array([5.0, 6.0, 7.0])
    livetime_s = 1.0e7

    counts = irf_expected_counts(
        synthetic_aeff, synthetic_smearing, log10_e_reco_edges,
        dec_min=-90.0, dec_max=0.0, flux_fn=flat_flux, livetime_s=livetime_s,
    )

    # Hand-computed: flux is flat, so the trapezoid-integrated flux over the
    # true-energy bin is exact: F0 * (1e7 - 1e6) GeV.
    flux_integral = FLUX_GEV_CM2_S_SR * (1.0e7 - 1.0e6)
    solid_angle_sr = 2.0 * np.pi * (np.sin(0.0) - np.sin(np.deg2rad(-90.0)))
    expected_total = A_EFF_CM2 * flux_integral * solid_angle_sr * livetime_s

    # All Fractional_Counts (=1) migrate to the reco bin containing E=6.005,
    # i.e. the second bin, [6, 7).
    assert counts[0] == pytest.approx(0.0)
    assert counts[1] == pytest.approx(expected_total, rel=1e-9)


def test_irf_expected_counts_zero_outside_dec_band(synthetic_aeff, synthetic_smearing):
    log10_e_reco_edges = np.array([5.0, 6.0, 7.0])
    counts = irf_expected_counts(
        synthetic_aeff, synthetic_smearing, log10_e_reco_edges,
        dec_min=0.0, dec_max=90.0, flux_fn=flat_flux, livetime_s=1.0e7,
    )
    np.testing.assert_array_equal(counts, 0.0)


def test_irf_expected_counts_scales_linearly_with_livetime(synthetic_aeff, synthetic_smearing):
    log10_e_reco_edges = np.array([5.0, 6.0, 7.0])
    base = irf_expected_counts(
        synthetic_aeff, synthetic_smearing, log10_e_reco_edges,
        dec_min=-90.0, dec_max=0.0, flux_fn=flat_flux, livetime_s=1.0,
    )
    scaled = irf_expected_counts(
        synthetic_aeff, synthetic_smearing, log10_e_reco_edges,
        dec_min=-90.0, dec_max=0.0, flux_fn=flat_flux, livetime_s=5.0,
    )
    np.testing.assert_allclose(scaled, 5.0 * base, rtol=1e-12)


# ---------------------------------------------------------------------------
# irf_expected_counts_directional
# ---------------------------------------------------------------------------


def test_directional_counts_match_isotropic_for_a_flux_ignoring_declination(
    synthetic_aeff, synthetic_smearing
):
    log10_e_reco_edges = np.array([5.0, 6.0, 7.0])
    isotropic = irf_expected_counts(
        synthetic_aeff, synthetic_smearing, log10_e_reco_edges,
        dec_min=-90.0, dec_max=0.0, flux_fn=flat_flux, livetime_s=1.0e7,
    )
    directional = irf_expected_counts_directional(
        synthetic_aeff, synthetic_smearing, log10_e_reco_edges,
        dec_min=-90.0, dec_max=0.0,
        flux_fn=lambda energy, dec: flat_flux(energy), livetime_s=1.0e7,
    )
    np.testing.assert_allclose(directional, isotropic, rtol=1e-12)


def test_directional_counts_use_the_declination_of_each_bin(synthetic_aeff, synthetic_smearing):
    log10_e_reco_edges = np.array([5.0, 6.0, 7.0])
    seen: list[float] = []

    def recording_flux(energy: np.ndarray, dec: float) -> np.ndarray:
        seen.append(dec)
        # Twice the flat flux, so the counts must double as well.
        return 2.0 * flat_flux(energy)

    base = irf_expected_counts(
        synthetic_aeff, synthetic_smearing, log10_e_reco_edges,
        dec_min=-90.0, dec_max=0.0, flux_fn=flat_flux, livetime_s=1.0e7,
    )
    doubled = irf_expected_counts_directional(
        synthetic_aeff, synthetic_smearing, log10_e_reco_edges,
        dec_min=-90.0, dec_max=0.0, flux_fn=recording_flux, livetime_s=1.0e7,
    )

    np.testing.assert_allclose(doubled, 2.0 * base, rtol=1e-12)
    # The one declination bin of the synthetic smearing table spans -90 to 0 deg.
    assert seen == [pytest.approx(-45.0)]


# ---------------------------------------------------------------------------
# soft_volume_smeared_counts
# ---------------------------------------------------------------------------


def test_soft_volume_smeared_counts_matches_hand_computed(synthetic_smearing):
    log10_e_reco_edges = np.array([5.0, 6.0, 7.0])

    def true_counts_fn(enu_edges: np.ndarray) -> np.ndarray:
        np.testing.assert_allclose(enu_edges, [6.0, 7.0])
        return np.array([123.0])

    counts = soft_volume_smeared_counts(
        true_counts_fn, synthetic_smearing, log10_e_reco_edges, dec_min=-90.0, dec_max=0.0,
    )

    # All Fractional_Counts (=1) migrate to the reco bin containing E=6.005,
    # i.e. the second bin, [6, 7) -- same migration as the IRF-path test above.
    assert counts[0] == pytest.approx(0.0)
    assert counts[1] == pytest.approx(123.0, rel=1e-9)


def test_soft_volume_smeared_counts_zero_outside_dec_band(synthetic_smearing):
    log10_e_reco_edges = np.array([5.0, 6.0, 7.0])
    counts = soft_volume_smeared_counts(
        lambda edges: np.array([100.0]), synthetic_smearing, log10_e_reco_edges,
        dec_min=0.0, dec_max=90.0,
    )
    np.testing.assert_array_equal(counts, 0.0)


def test_soft_volume_smeared_counts_scales_linearly_with_true_counts(synthetic_smearing):
    log10_e_reco_edges = np.array([5.0, 6.0, 7.0])
    base = soft_volume_smeared_counts(
        lambda edges: np.array([10.0]), synthetic_smearing, log10_e_reco_edges,
        dec_min=-90.0, dec_max=0.0,
    )
    scaled = soft_volume_smeared_counts(
        lambda edges: np.array([50.0]), synthetic_smearing, log10_e_reco_edges,
        dec_min=-90.0, dec_max=0.0,
    )
    np.testing.assert_allclose(scaled, 5.0 * base, rtol=1e-12)


# ---------------------------------------------------------------------------
# observed_counts
# ---------------------------------------------------------------------------


def _make_event(log10_energy: float, dec: float) -> tuple:
    return (0, 0, 0, 0.0, log10_energy, 1.0, 0.0, dec, 0.0, 0.0)


def test_observed_counts_histograms_within_dec_band():
    rows = [
        _make_event(4.5, dec=-10.0),
        _make_event(5.5, dec=-10.0),
        _make_event(5.5, dec=45.0),  # outside the requested dec band
    ]
    arr = np.array(rows, dtype=EVENTS_DTYPE)
    events = EventSet(arr)

    log10_e_edges = np.array([4.0, 5.0, 6.0])
    counts = observed_counts(events, log10_e_edges, dec_min=-90.0, dec_max=0.0)

    np.testing.assert_array_equal(counts, [1, 1])


# ---------------------------------------------------------------------------
# fit_scale_factor / implied_efficiency
# ---------------------------------------------------------------------------


def test_fit_scale_factor_recovers_exact_scale():
    log10_e_edges = np.array([4.0, 5.0, 6.0, 7.0])
    template = np.array([10.0, 20.0, 30.0])
    observed = 2.5 * template

    scale = fit_scale_factor(observed, template, log10_e_edges, log10_e_min_fit=4.0)
    assert scale == pytest.approx(2.5, rel=1e-12)


def test_fit_scale_factor_respects_fit_range():
    log10_e_edges = np.array([4.0, 5.0, 6.0, 7.0])
    template = np.array([10.0, 20.0, 30.0])
    # Only the last two bins scale by 2.5; the first is deliberately off.
    observed = np.array([999.0, 50.0, 75.0])

    scale = fit_scale_factor(observed, template, log10_e_edges, log10_e_min_fit=5.0)
    assert scale == pytest.approx(2.5, rel=1e-12)


def test_fit_scale_factor_raises_for_zero_template():
    log10_e_edges = np.array([4.0, 5.0])
    with pytest.raises(ValueError):
        fit_scale_factor(np.array([5.0]), np.array([0.0]), log10_e_edges, log10_e_min_fit=4.0)


def test_fit_with_background_recovers_the_injected_scale():
    log10_e_edges = np.array([4.0, 5.0, 6.0, 7.0])
    template = np.array([10.0, 20.0, 30.0])
    background = np.array([100.0, 5.0, 1.0])
    observed = 2.5 * template + background

    scale = fit_scale_factor_with_background(
        observed, template, background, log10_e_edges, log10_e_min_fit=4.0,
    )
    assert scale == pytest.approx(2.5, rel=1e-6)


def test_fit_with_background_agrees_with_closed_form_when_background_is_zero():
    log10_e_edges = np.array([4.0, 5.0, 6.0, 7.0])
    template = np.array([10.0, 20.0, 30.0])
    observed = np.array([12.0, 25.0, 28.0])

    with_bkg = fit_scale_factor_with_background(
        observed, template, np.zeros_like(template), log10_e_edges, log10_e_min_fit=4.0,
    )
    closed_form = fit_scale_factor(observed, template, log10_e_edges, log10_e_min_fit=4.0)
    assert with_bkg == pytest.approx(closed_form, rel=1e-6)


def test_fit_with_background_returns_zero_when_background_already_overshoots():
    log10_e_edges = np.array([4.0, 5.0, 6.0])
    template = np.array([10.0, 10.0])
    background = np.array([50.0, 50.0])
    observed = np.array([5.0, 5.0])

    scale = fit_scale_factor_with_background(
        observed, template, background, log10_e_edges, log10_e_min_fit=4.0,
    )
    assert scale == 0.0


def test_fit_with_background_ignores_bins_below_the_fit_range():
    log10_e_edges = np.array([4.0, 5.0, 6.0, 7.0])
    template = np.array([10.0, 20.0, 30.0])
    background = np.zeros_like(template)
    observed = np.array([9999.0, 50.0, 75.0])

    scale = fit_scale_factor_with_background(
        observed, template, background, log10_e_edges, log10_e_min_fit=5.0,
    )
    assert scale == pytest.approx(2.5, rel=1e-6)


def test_fit_with_background_raises_for_zero_template():
    log10_e_edges = np.array([4.0, 5.0])
    with pytest.raises(ValueError):
        fit_scale_factor_with_background(
            np.array([5.0]), np.array([0.0]), np.array([1.0]), log10_e_edges, 4.0,
        )


def test_component_scales_recover_injected_normalizations():
    log10_e_edges = np.array([4.0, 5.0, 6.0, 7.0])
    signal = np.array([1.0, 5.0, 20.0])  # rising
    background = np.array([200.0, 20.0, 1.0])  # falling
    observed = 3.0 * signal + 0.5 * background

    scales = fit_component_scales(observed, [signal, background], log10_e_edges, 4.0)
    np.testing.assert_allclose(scales, [3.0, 0.5], rtol=1e-4)


def test_component_scales_reduce_to_the_single_template_fit():
    log10_e_edges = np.array([4.0, 5.0, 6.0, 7.0])
    template = np.array([10.0, 20.0, 30.0])
    observed = np.array([12.0, 25.0, 28.0])

    (scale,) = fit_component_scales(observed, [template], log10_e_edges, 4.0)
    assert scale == pytest.approx(
        fit_scale_factor(observed, template, log10_e_edges, 4.0), rel=1e-6
    )


def test_component_scales_conserve_the_total_in_the_fit_range():
    # A property of the Poisson maximum likelihood with free normalizations:
    # the fitted total matches the observed total over the fit range.
    log10_e_edges = np.array([4.0, 5.0, 6.0, 7.0])
    signal = np.array([1.0, 5.0, 20.0])
    background = np.array([200.0, 20.0, 1.0])
    observed = np.array([150.0, 40.0, 12.0])

    scales = fit_component_scales(observed, [signal, background], log10_e_edges, 4.0)
    predicted = scales[0] * signal + scales[1] * background
    assert predicted.sum() == pytest.approx(observed.sum(), rel=1e-6)


def test_component_scales_raise_for_an_empty_template():
    log10_e_edges = np.array([4.0, 5.0, 6.0])
    with pytest.raises(ValueError):
        fit_component_scales(
            np.array([5.0, 5.0]), [np.array([1.0, 1.0]), np.zeros(2)], log10_e_edges, 4.0,
        )


def test_implied_efficiency_ratio_of_scales():
    log10_e_edges = np.array([4.0, 5.0, 6.0])
    soft_template = np.array([10.0, 10.0])
    irf_template = np.array([20.0, 20.0])
    observed = np.array([100.0, 100.0])  # scale_soft = 10, scale_irf = 5

    eff = implied_efficiency(observed, soft_template, irf_template, log10_e_edges)
    assert eff == pytest.approx(2.0, rel=1e-12)


def test_implied_efficiency_below_one_when_soft_overpredicts():
    # A soft-volume template larger than the IRF template at the same
    # reference flux needs a smaller fitted normalization to match the same
    # data, i.e. efficiency < 1 -- the qualitative KM3NeT/IceCube result.
    log10_e_edges = np.array([4.0, 5.0])
    soft_template = np.array([100.0])
    irf_template = np.array([45.0])
    observed = np.array([45.0])

    eff = implied_efficiency(observed, soft_template, irf_template, log10_e_edges)
    assert eff < 1.0
