"""Tests for the drift-limit soft-volume model.

These lock in the reproduction of the reference numbers in arXiv:2607.13143:
Table 1 transport coefficients in water, and the soft-volume figures of merit
for IceCube (~4x) and KM3NeT (~7.5x) from Section 2.3.
"""

import numpy as np
import pytest

from softpaws.response.soft_volume import SoftVolumeResponse
from softpaws.transport.coefficients import (
    critical_energy_gev,
    diffusion_coefficient,
    drift_coefficient,
    ionization_coefficient,
)
from softpaws.transport.soft_volume import (
    DEFAULT_MUON_THRESHOLD_GEV,
    muon_range_km,
    range_target_volume_km3,
    soft_volume_diffusion,
    soft_volume_drift,
    spectral_penalty,
    sphere_radius_from_volume,
    volume_ratio_drift,
)
from softpaws.transport.source import (
    MEAN_INELASTICITY,
    cc_cross_section,
    nucleon_number_density,
)
from softpaws.utils.constants import CM_PER_KM, RHO_ICE_G_CM3, RHO_WATER_G_CM3

E_1PEV = 1.0e6  # GeV
E_100PEV = 1.0e8  # GeV
GAMMA_IC = 2.38  # IceCube 9.5 yr diffuse-flux best fit (Eq. 1.3)


# ---------------------------------------------------------------------------
# Transport coefficients (Table 1, water)
# ---------------------------------------------------------------------------


def test_drift_coefficient_matches_table1():
    assert drift_coefficient(E_1PEV)[0] == pytest.approx(0.35, abs=1e-6)
    assert drift_coefficient(E_100PEV)[0] == pytest.approx(0.40, abs=1e-6)


def test_diffusion_coefficient_matches_table1():
    assert diffusion_coefficient(E_1PEV)[0] == pytest.approx(0.0766, abs=1e-6)
    assert diffusion_coefficient(E_100PEV)[0] == pytest.approx(0.0982, abs=1e-6)


def test_drift_coefficient_interpolates_in_log_energy():
    # 10 PeV = log10 16, the midpoint between the two reference decades.
    mid = drift_coefficient(1.0e7)[0]
    assert mid == pytest.approx(0.5 * (0.35 + 0.40), abs=1e-6)


def test_drift_coefficient_clips_outside_reference_range():
    assert drift_coefficient(1.0e3)[0] == pytest.approx(0.35, abs=1e-6)
    assert drift_coefficient(1.0e12)[0] == pytest.approx(0.40, abs=1e-6)


def test_drift_coefficient_scales_with_density():
    b_water = drift_coefficient(E_1PEV, RHO_WATER_G_CM3)[0]
    b_ice = drift_coefficient(E_1PEV, RHO_ICE_G_CM3)[0]
    assert b_ice == pytest.approx(b_water * RHO_ICE_G_CM3 / RHO_WATER_G_CM3, rel=1e-12)


def test_coefficients_accept_arrays():
    b = drift_coefficient(np.array([E_1PEV, E_100PEV]))
    assert b.shape == (2,)
    np.testing.assert_allclose(b, [0.35, 0.40], atol=1e-6)


# ---------------------------------------------------------------------------
# Spectral penalty
# ---------------------------------------------------------------------------


def test_spectral_penalty_value():
    # gamma = 2.38, lambda = 0.4 -> A = 0.98 ~ 1 for the IceCube flux.
    assert spectral_penalty(GAMMA_IC) == pytest.approx(0.98, abs=1e-9)


def test_spectral_penalty_rejects_nonpositive():
    with pytest.raises(ValueError):
        spectral_penalty(1.2)  # A = 1.2 - 0.4 - 1 = -0.2


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_sphere_radius_from_volume():
    # IceCube instrumented volume ~1 km^3 -> R ~ 0.62 km.
    assert sphere_radius_from_volume(1.0) == pytest.approx(0.6204, abs=1e-3)


# ---------------------------------------------------------------------------
# Soft volume figures of merit (Section 2.3)
# ---------------------------------------------------------------------------


def test_volume_ratio_icecube_about_four():
    r_ic = sphere_radius_from_volume(1.0)
    ratio = volume_ratio_drift(r_ic, E_1PEV, GAMMA_IC)[0]
    assert ratio == pytest.approx(4.5, abs=0.2)


def test_volume_ratio_km3net_about_seven_and_half():
    # KM3NeT is smaller; the paper quotes ~7.5 for R ~ 0.33 km.
    ratio = volume_ratio_drift(0.33, E_1PEV, GAMMA_IC)[0]
    assert ratio == pytest.approx(7.5, abs=0.3)


def test_soft_volume_matches_ratio_definition():
    # V_tot / V_det = 1 + V_soft / V_det, with V_det the sphere volume.
    r = sphere_radius_from_volume(1.0)
    v_det = 4.0 / 3.0 * np.pi * r**3
    v_soft = soft_volume_drift(r, E_1PEV, GAMMA_IC)[0]
    ratio = volume_ratio_drift(r, E_1PEV, GAMMA_IC)[0]
    assert (1.0 + v_soft / v_det) == pytest.approx(ratio, rel=1e-12)


def test_soft_volume_grows_with_shallower_spectrum():
    # Smaller A (shallower flux) -> larger spectral enhancement -> larger V_soft.
    r = sphere_radius_from_volume(1.0)
    steep = soft_volume_drift(r, E_1PEV, 2.6)[0]
    shallow = soft_volume_drift(r, E_1PEV, 2.2)[0]
    assert shallow > steep


# ---------------------------------------------------------------------------
# Diffusion correction (Eq. 2.25) and coefficient nuisance scales
# ---------------------------------------------------------------------------


def test_diffusion_is_drift_times_correction():
    # V_soft|diff = V_soft|drift (1 - d_mu / 2 b_mu).
    r = sphere_radius_from_volume(1.0)
    b = drift_coefficient(E_1PEV)[0]
    d = diffusion_coefficient(E_1PEV)[0]
    v_drift = soft_volume_drift(r, E_1PEV, GAMMA_IC)[0]
    v_diff = soft_volume_diffusion(r, E_1PEV, GAMMA_IC)[0]
    assert v_diff == pytest.approx(v_drift * (1.0 - d / (2.0 * b)), rel=1e-12)
    assert v_diff < v_drift  # the diffusion correction is a reduction


def test_b_scale_rescales_drift_volume():
    # V_soft|drift ~ 1 / b_mu, so doubling b_scale halves the volume.
    r = sphere_radius_from_volume(1.0)
    base = soft_volume_drift(r, E_1PEV, GAMMA_IC)[0]
    scaled = soft_volume_drift(r, E_1PEV, GAMMA_IC, b_scale=2.0)[0]
    assert scaled == pytest.approx(base / 2.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Muon range and the range target volume (the published-A_eff convention)
# ---------------------------------------------------------------------------


def test_critical_energy_in_water_is_a_few_hundred_gev():
    # E_c = a_mu / b_mu with a_mu ~ 2 MeV cm^2/g and b_mu ~ 0.35 km^-1.
    e_crit = critical_energy_gev(E_1PEV)[0]
    assert 400.0 < e_crit < 700.0


def test_critical_energy_is_density_independent():
    # a_mu and b_mu both scale linearly with density, so the ratio does not.
    water = critical_energy_gev(E_1PEV, RHO_WATER_G_CM3)[0]
    ice = critical_energy_gev(E_1PEV, RHO_ICE_G_CM3)[0]
    assert ice == pytest.approx(water, rel=1e-12)


def test_ionization_coefficient_scales_with_density():
    dense = ionization_coefficient(2.0 * RHO_WATER_G_CM3)
    assert dense == pytest.approx(2.0 * ionization_coefficient(RHO_WATER_G_CM3), rel=1e-12)


def test_muon_range_matches_closed_form():
    # R = ln[(E + E_c) / (E_thr + E_c)] / b_mu.
    b = drift_coefficient(E_1PEV)[0]
    e_crit = critical_energy_gev(E_1PEV)[0]
    expected = np.log((E_1PEV + e_crit) / (DEFAULT_MUON_THRESHOLD_GEV + e_crit)) / b
    assert muon_range_km(E_1PEV)[0] == pytest.approx(expected, rel=1e-12)


def test_muon_range_is_tens_of_km_at_uhe():
    # A PeV muon travels ~20 km w.e. before dropping below a TeV.
    assert 15.0 < muon_range_km(E_1PEV)[0] < 25.0
    assert 20.0 < muon_range_km(E_100PEV)[0] < 35.0


def test_muon_range_grows_logarithmically():
    # Two decades in energy lengthen the range by well under a factor of two --
    # the qualitative difference from a spectral length, which is flat.
    ratio = muon_range_km(E_100PEV)[0] / muon_range_km(E_1PEV)[0]
    assert 1.1 < ratio < 1.6


def test_muon_range_vanishes_below_threshold():
    assert muon_range_km(0.5 * DEFAULT_MUON_THRESHOLD_GEV)[0] == 0.0


def test_range_volume_is_column_plus_detector():
    # pi R^2 (4R/3) = V_det exactly, so the column and the sphere add cleanly.
    r = sphere_radius_from_volume(1.0)
    volume = range_target_volume_km3(r, E_1PEV)[0]
    column = np.pi * r**2 * muon_range_km(E_1PEV)[0]
    assert volume == pytest.approx(column + 1.0, rel=1e-12)


def test_range_volume_vanishes_below_threshold():
    # A muon born below threshold is never selected, so not even V_det counts.
    r = sphere_radius_from_volume(1.0)
    assert range_target_volume_km3(r, 0.5 * DEFAULT_MUON_THRESHOLD_GEV)[0] == 0.0


def test_range_volume_exceeds_soft_volume_at_uhe():
    # The two conventions diverge: the soft volume is a flat few km^3 while the
    # range volume keeps growing, which is the whole point of example 20.
    r = sphere_radius_from_volume(1.0)
    soft = soft_volume_drift(r, E_100PEV, GAMMA_IC)[0] + 1.0
    assert range_target_volume_km3(r, E_100PEV)[0] > 5.0 * soft


def test_threshold_effective_area_matches_range_volume():
    # SoftVolumeResponse.threshold_effective_area_cm2 = V_range n_N sigma_CC,
    # with the muon born at (1 - <y_w>) E_nu.
    r = sphere_radius_from_volume(1.0)
    resp = SoftVolumeResponse(radius_km=r)
    e_nu = E_1PEV
    volume_cm3 = (
        range_target_volume_km3(r, (1.0 - MEAN_INELASTICITY) * e_nu) * CM_PER_KM**3
    )
    expected = volume_cm3 * nucleon_number_density() * cc_cross_section(e_nu)
    assert resp.threshold_effective_area_cm2(e_nu)[0] == pytest.approx(expected[0], rel=1e-12)


def test_threshold_effective_area_is_flux_independent():
    # Unlike effective_area_cm2, it takes no gamma at all -- a tabulated
    # effective area should not depend on the flux used to derive it.
    r = sphere_radius_from_volume(1.0)
    resp = SoftVolumeResponse(radius_km=r)
    energies = np.array([1.0e5, 1.0e6, 1.0e7])
    area = resp.threshold_effective_area_cm2(energies)
    assert np.all(np.diff(area) > 0.0)
    assert area.shape == energies.shape
