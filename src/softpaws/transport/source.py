"""Weak source ingredients: neutrino CC cross section and target density.

These feed the weak collisional term that sources muons in the transport
equation (arXiv:2607.13143, Eq. 2.2). In the reduced drift-limit picture of
Section 2.3 the weak-rate density at observed muon energy ``E`` is
``n_N * sigma_CC(E) * phi_nu(E)``, which multiplies the target volume to give
the event rate.
"""

from __future__ import annotations

import numpy as np

from ..utils.constants import AVOGADRO_PER_MOL, RHO_WATER_G_CM3

# UHE CC cross-section power law, sigma_CC = sigma0 (E / E0)^lambda
# (arXiv:2607.13143, Eq. 2.5). sigma0 matches the MadGraph result at E0 = 10 PeV
# with the default LHAPDF set; lambda ~ 0.4 follows the small-x PDF behaviour.
SIGMA0_CM2 = 1.48e-33
E0_CROSS_GEV = 1.0e7  # 10 PeV
DEFAULT_LAMBDA = 0.4

# Average CC (DIS) inelasticity <y_w>; near-elastic at UHE (Section 2.3).
MEAN_INELASTICITY = 0.2


def cc_cross_section(
    energy_gev: float | np.ndarray,
    lam: float = DEFAULT_LAMBDA,
) -> np.ndarray:
    """Neutrino-nucleon CC cross section (Eq. 2.5).

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Neutrino energy [GeV]. In the reduced drift limit this is evaluated at
        the observed muon energy, since ``<y_w>`` is small.
    lam : float, optional
        Cross-section slope. Defaults to :data:`DEFAULT_LAMBDA`.

    Returns
    -------
    sigma : np.ndarray
        CC cross section per nucleon [cm^2].

    Notes
    -----
    This single power law is anchored at ``E0 = 10 PeV`` and is intended for the
    UHE regime. Below a few PeV the true cross section flattens relative to the
    ``lambda = 0.4`` extrapolation, so treat sub-PeV values as approximate.
    """
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    return SIGMA0_CM2 * (energy / E0_CROSS_GEV) ** lam


def inelasticity_factor(
    spectral_index_value: float | np.ndarray,
    mean_inelasticity: float = MEAN_INELASTICITY,
) -> np.ndarray:
    """Weak-vertex inelasticity factor ``I(A) = <(1 - y_w)^A>``.

    The muon is born with only a fraction ``1 - y_w`` of the neutrino energy, so
    the source picks up the spectrum-weighted average
    ``I(A) = <(1 - y_w)^A>_P`` (arXiv:2607.13143 companion derivation,
    ``docs/exact_soft_volume_notes.md`` Part 5). This factor multiplies both the
    inside and the soft contributions in the exact master formula.

    Parameters
    ----------
    spectral_index_value : float or np.ndarray
        Source spectral index ``A = gamma - lambda - 1``.
    mean_inelasticity : float, optional
        Mean CC inelasticity ``<y_w>``. Defaults to :data:`MEAN_INELASTICITY`.

    Returns
    -------
    factor : np.ndarray
        ``I(A)``, of order ``0.8``.

    Notes
    -----
    With only the mean ``<y_w>`` available, the inelasticity distribution
    ``P(y_w)`` is modelled as a delta at its mean, giving
    ``I(A) = (1 - <y_w>)^A``. This is exact at ``A = 1`` (``I(1) = 1 - <y_w>``,
    the only value the paper actually uses) and a mild approximation elsewhere;
    supply a full ``P(y_w)`` average to improve it away from ``A = 1``.
    """
    a = np.asarray(spectral_index_value, dtype=float)
    return (1.0 - mean_inelasticity) ** a


def nucleon_number_density(density_g_cm3: float = RHO_WATER_G_CM3) -> float:
    """Target nucleon number density of the medium.

    Parameters
    ----------
    density_g_cm3 : float, optional
        Medium mass density [g cm^-3]. Defaults to water.

    Returns
    -------
    n_nucleon : float
        Nucleon number density [cm^-3].

    Notes
    -----
    Uses ``n = rho * N_A``, i.e. approximately one nucleon per atomic mass unit
    (``A / molar mass ~ 1 g/mol``). The CC cross section is defined per nucleon,
    so this is the matching target density.
    """
    return density_g_cm3 * AVOGADRO_PER_MOL
