"""Tests for the shipped PROPOSAL coefficient tables.

Lock in the energy range of the tables and the behaviour of the moments at
their top, where the photonuclear channel dominates on an extrapolation of the
HERA fit and the LPM suppression of bremsstrahlung first shows.
"""

import numpy as np
import pytest

from softpaws.transport.coefficients import (
    PROPOSAL_ROCK_SOURCE,
    PROPOSAL_SOURCE,
    _load_proposal_table,
    drift_coefficient,
    log_loss_moments,
)


@pytest.mark.parametrize("source", [PROPOSAL_SOURCE, PROPOSAL_ROCK_SOURCE])
def test_table_spans_two_to_sixteen_decades(source):
    log10_e = _load_proposal_table(source)[0]
    assert log10_e[0] == pytest.approx(2.0)
    assert log10_e[-1] == pytest.approx(16.0)


@pytest.mark.parametrize("source", [PROPOSAL_SOURCE, PROPOSAL_ROCK_SOURCE])
def test_drift_keeps_growing_above_ten_eev(source):
    # Bremsstrahlung and pair production saturate; the photonuclear channel
    # does not, so the total drift keeps rising through the top of the table.
    energies = np.logspace(10.0, 16.0, 13)
    b = np.asarray(drift_coefficient(energies, source=source)).ravel()
    assert np.all(np.diff(b) > 0.0)
    assert b[-1] > 3.0 * b[0]


def test_drift_at_ten_eev_unchanged_by_the_extension():
    # The rows below 10 EeV are those of the earlier table, to the precision
    # of PROPOSAL's rebuilt interpolation.
    b = float(np.asarray(drift_coefficient(1.0e10)).ravel()[0])
    assert b == pytest.approx(0.5135, rel=1e-3)


def test_log_loss_moments_finite_to_the_top():
    energies = np.logspace(2.0, 16.0, 29)
    for moment in log_loss_moments(energies):
        moment = np.asarray(moment).ravel()
        assert np.all(np.isfinite(moment))
        assert np.all(moment > 0.0)
