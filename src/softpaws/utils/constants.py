"""Physical constants, medium properties, and unit conversions.

All energies are in GeV, lengths in km unless stated otherwise. Values are
collected here so the transport and response subpackages share a single source
of truth. Medium densities and the reference muon transport coefficients follow
Palmisano et al., arXiv:2607.13143 (see ``docs/soft_volume_notes.md``).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Fundamental constants
# ---------------------------------------------------------------------------

AVOGADRO_PER_MOL = 6.02214076e23

# Speed of light, for lab-frame decay lengths (gamma * c * tau0).
C_KM_PER_S = 2.99792458e5

# ---------------------------------------------------------------------------
# Particle masses [GeV]
# ---------------------------------------------------------------------------

M_ELECTRON_GEV = 0.51099895e-3
M_MUON_GEV = 0.1056583755
M_PION_GEV = 0.13957039
M_TAU_GEV = 1.77686

# ---------------------------------------------------------------------------
# Particle lifetimes [s], rest frame
# ---------------------------------------------------------------------------

TAU_LIFETIME_S = 2.903e-13

# ---------------------------------------------------------------------------
# Medium densities [g cm^-3]
# ---------------------------------------------------------------------------
# The paper calibrates the muon transport coefficients in pure water; South
# Pole ice is provided for later reuse. See ``docs/soft_volume_notes.md``.

RHO_WATER_G_CM3 = 1.02
RHO_ICE_G_CM3 = 0.92
#: Fresh water, for Lake Baikal [g cm^-3].
RHO_LAKE_G_CM3 = 1.00

# ---------------------------------------------------------------------------
# Earth
# ---------------------------------------------------------------------------
# Used by the neutrino Earth-attenuation model (``softpaws.transport.attenuation``).
# The mean density feeds the constant-density closed-form column; the layered
# PREM profile (Dziewonski & Anderson 1981) feeds the per-event column.

EARTH_RADIUS_KM = 6371.0
RHO_EARTH_MEAN_G_CM3 = 5.513

# ---------------------------------------------------------------------------
# Length conversions
# ---------------------------------------------------------------------------

CM_PER_KM = 1.0e5
KM_PER_CM = 1.0e-5
CM_PER_M = 1.0e2
M_PER_KM = 1.0e3
