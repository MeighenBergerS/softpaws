"""Tests for the flux package."""


import numpy as np
import pytest

from softpaws.fluxes import (
    ICECUBE_BPL_2025,
    ICECUBE_TRACKS_2022,
    REFERENCE_SPL,
    SHIPPED_TABLE,
    AtmosphericFlux,
    broken_power_law_flux,
    broken_power_law_shape,
    power_law_flux,
)
from softpaws.response.soft_volume import power_law_flux as reexported

MCEQ_TABLE = SHIPPED_TABLE


def test_power_law_reexport_and_pivot():
    assert reexported is power_law_flux
    assert power_law_flux(1.0e5, 0.63, 2.38).item() == pytest.approx(0.63e-18)
    e = np.array([1e4, 1e6])
    np.testing.assert_allclose(power_law_flux(e, 1.0, 2.0), 1e-18 * (e / 1e5) ** -2.0)
    np.testing.assert_allclose(REFERENCE_SPL.flux(e), power_law_flux(e, 0.63, 2.38))


def test_broken_power_law_matches_the_scripts():
    """The shape example 54 and 57 wrote: continuous at the break, unit at 100 TeV."""
    e = np.logspace(3, 8, 51)
    shape = broken_power_law_shape(e, 2.735)
    e_break = 10.0**4.39
    above = (e / 1e5) ** (-2.735)
    below = (e_break / 1e5) ** (-2.735) * (e / e_break) ** (-1.31)
    np.testing.assert_allclose(shape, np.where(e >= e_break, above, below))
    assert broken_power_law_shape(1e5, 2.735).item() == pytest.approx(1.0)
    edges = np.array([e_break * (1 - 1e-9), e_break * (1 + 1e-9)])
    just_below, just_above = broken_power_law_shape(edges, 2.0)
    assert just_below == pytest.approx(just_above, rel=1e-6)
    np.testing.assert_allclose(
        ICECUBE_BPL_2025.flux(e), broken_power_law_flux(e, 1.77, 2.735, 1.31, 4.39)
    )
    assert ICECUBE_TRACKS_2022.gamma == 2.37 and ICECUBE_TRACKS_2022.phi0 == 1.44


def _synthetic_table():
    energy = np.logspace(2, 8, 25)
    dec = np.linspace(0.0, 90.0, 7)
    conv = np.outer(energy**-3.7, 1.0 + 0.5 * np.cos(np.deg2rad(dec)))
    prompt = np.outer(energy**-2.7 * 1e-3, np.ones_like(dec))
    return {"energy_gev": energy, "dec_deg": dec, "conv": conv, "prompt": prompt}


def test_atmospheric_flux_interpolation():
    table = _synthetic_table()
    flux = AtmosphericFlux(table)
    # On the nodes the interpolation is exact.
    got = flux(table["energy_gev"][5], table["dec_deg"][2])
    assert got.item() == pytest.approx(table["conv"][5, 2] + table["prompt"][5, 2])
    conv_only = AtmosphericFlux(table, include_prompt=False)
    assert conv_only(1e5, 45.0).item() < flux(1e5, 45.0).item()
    # The clamped grid reader reproduces the two-step log interpolation of
    # examples 49 and 51, including the fold to positive declination.
    e = np.logspace(3, 7, 9)
    d = np.array([-60.0, 10.0, 80.0])
    grid = flux.component_on_grid("conv", e, d)
    log_f = np.log10(table["conv"])
    on_dec = np.array([np.interp(np.abs(d), table["dec_deg"], log_f[i]) for i in range(25)])
    log_e, log_e_table = np.log10(e), np.log10(table["energy_gev"])
    want = np.array([10.0 ** np.interp(log_e, log_e_table, on_dec[:, j]) for j in range(3)]).T
    np.testing.assert_allclose(grid, want)


@pytest.mark.skipif(not MCEQ_TABLE.exists(), reason="no cached MCEq table")
def test_cached_table_loads():
    from softpaws.fluxes import load_mceq_table

    table = load_mceq_table(MCEQ_TABLE)
    flux = AtmosphericFlux(table)
    assert flux(1e5, 45.0) > flux(1e6, 45.0) > 0.0
    grid = flux.component_on_grid("prompt", np.array([1e5, 1e6]), np.array([5.0, 45.0]))
    assert grid.shape == (2, 2) and np.all(grid > 0.0)
