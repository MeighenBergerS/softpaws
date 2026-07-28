"""Soft-volume forward model: neutrino flux to muon track rate (drift limit).

This is the soft-volume counterpart to the IRF path in :mod:`softpaws.response.irfs`.
Both map an incident neutrino flux to a predicted track rate; keeping them
interchangeable is what enables the head-to-head comparison.

In the drift limit with the Section 2.3 approximations, the master formula
(arXiv:2607.13143, Eq. 2.20) factorizes: the differential track rate at observed
muon energy ``E`` is the target volume times the local weak-rate density,

.. math:: \\frac{dN}{dt\\,dE\\,d\\Omega}
    = \\bigl[V_\\mathrm{det} + V_\\mathrm{soft}(E)\\bigr]\\,
      n_N\\,\\sigma_\\mathrm{CC}(E)\\,\\phi_\\nu(E),

where the soft volume ``V_soft`` carries the muon-transport enhancement (Eq. 2.23)
and ``V_det`` is the instrumented sphere. The drift closed form assumes a
power-law neutrino flux, so this model is parametrized directly by ``(phi0, gamma)``.
A tabulated, direction-dependent flux (an atmospheric model, say) goes through
:meth:`SoftVolumeResponse.expected_counts_from_flux` instead, which reads the
spectral index off the flux's local slope.

Two transport methods are available (``method`` argument of
:class:`SoftVolumeResponse`): ``"drift"`` is the paper's leading form above, and
``"exact"`` uses the exact eigenvalue ``Phi(A)`` with the ``I(A)`` normalization and
a finite upstream column depth (``docs/exact_soft_volume_notes.md``). In the exact
master formula ``I(A)`` multiplies both populations, so both the inside and soft
target volumes carry it.

Earth attenuation of the parent neutrino (``D_nu``, Eq. 2.4;
:mod:`softpaws.transport.attenuation`) is off by default (``D_nu = 1``, valid
downgoing). It can be applied in two ways: a closed-form representative column
via ``attenuation_column_g_cm2`` (Form A), or the per-event PREM treatment of
:meth:`SoftVolumeResponse.expected_counts_attenuated` (Form B).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..transport.attenuation import effective_solid_angle, prem_column, survival_probability
from ..transport.eigenvalue import spectral_index
from ..transport.soft_volume import (
    DEFAULT_MUON_THRESHOLD_GEV,
    range_target_volume_km3,
    soft_volume_diffusion,
    soft_volume_drift,
    soft_volume_exact,
)
from ..transport.source import (
    DEFAULT_LAMBDA,
    MEAN_INELASTICITY,
    cc_cross_section,
    inelasticity_factor,
    nucleon_number_density,
)
from ..transport.tau import tau_to_muon_ratio
from ..utils.constants import CM_PER_KM, RHO_WATER_G_CM3

# Flux pivot energy for the power-law parametrization (Eq. 4.1): 100 TeV.
FLUX_PIVOT_GEV = 1.0e5

# Range the local effective spectral index is clipped to in
# :meth:`SoftVolumeResponse.expected_counts_from_flux`. Wherever a tabulated flux
# falls off a cliff -- an Earth-absorbed atmospheric spectrum, or the end of a
# table -- the measured local slope runs away, while the counts it multiplies are
# already negligible; clipping keeps the soft volume in the regime the transport
# expansion was built for. The lower bound sits above the divergence of the soft
# volume at ``gamma = lambda + 1``.
GAMMA_EFF_BOUNDS = (1.5, 8.0)


def power_law_flux(
    energy_gev: float | np.ndarray,
    phi0: float,
    gamma: float,
) -> np.ndarray:
    """Single power-law diffuse neutrino flux (Eq. 4.1).

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Neutrino energy [GeV].
    phi0 : float
        Flux normalization in units of ``1e-18 GeV^-1 cm^-2 s^-1 sr^-1`` at the
        100 TeV pivot; of order unity for typical diffuse fluxes.
    gamma : float
        Spectral index, ``phi_nu ~ E^-gamma``.

    Returns
    -------
    flux : np.ndarray
        Differential flux [GeV^-1 cm^-2 s^-1 sr^-1].
    """
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    return (phi0 * 1.0e-18) * (energy / FLUX_PIVOT_GEV) ** (-gamma)


class SoftVolumeResponse:
    """Soft-volume forward model for a spherical detector.

    Parameters
    ----------
    radius_km : float
        Radius of the spherical instrumented volume [km].
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water.
    method : {"drift", "diffusion", "exact"}, optional
        Transport treatment for the soft volume. ``"drift"`` (default) uses the
        paper's leading form ``V_soft = A_proj / (b_mu A)`` (Eq. 2.23).
        ``"diffusion"`` applies the paper's diffusion correction
        ``1 - d_mu/(2 b_mu)`` (Eq. 2.25). ``"exact"`` uses the exact eigenvalue
        ``Phi(A)`` with the ``I(A)`` normalization and the finite-column
        saturation factor (``docs/exact_soft_volume_notes.md``).
    column_depth_km : float or None, optional
        Available upstream column depth ``x`` [km] for the exact method. ``None``
        (the default) uses the infinite-column limit, which requires
        ``Phi(A) > 0``. Ignored by the drift and diffusion methods.
    attenuation_column_g_cm2 : float or None, optional
        Representative Earth column depth [g cm^-2] for the closed-form neutrino
        attenuation (Form A). When set, the flux is multiplied by the survival
        probability ``D_nu(E) = exp(-N_A sigma_tot(E) X)`` at every energy (see
        :func:`softpaws.transport.attenuation.representative_column`). ``None``
        (the default) leaves the flux unattenuated (``D_nu = 1``). For the
        per-event PREM attenuation (Form B) leave this ``None`` and use
        :meth:`expected_counts_attenuated` instead.
    b_scale, d_scale : float, optional
        Multiplicative rescalings of the Table 1 drift and diffusion coefficients,
        used as transport nuisance parameters in the data fits (Section 3).
        Default to 1 (the theoretical values).

    Attributes
    ----------
    radius_km : float
        Detector radius [km].
    density_g_cm3 : float
        Medium density [g cm^-3].
    method : str
        Selected transport method.
    column_depth_km : float or None
        Upstream column depth for the exact method.
    b_scale, d_scale : float
        Transport-coefficient nuisance rescalings.
    n_nucleon_cm3 : float
        Target nucleon number density [cm^-3].
    v_det_cm3 : float
        Instrumented sphere volume [cm^3].

    Examples
    --------
    >>> resp = SoftVolumeResponse(radius_km=0.62)
    >>> rate = resp.differential_rate(1.0e6, phi0=0.63, gamma=2.38)
    >>> float(rate[0]) > 0
    True
    >>> exact = SoftVolumeResponse(radius_km=0.62, method="exact", column_depth_km=1.95)
    >>> float(exact.differential_rate(1.0e6, phi0=0.63, gamma=2.38)[0]) > 0
    True
    """

    def __init__(
        self,
        radius_km: float,
        density_g_cm3: float = RHO_WATER_G_CM3,
        method: str = "drift",
        column_depth_km: float | None = None,
        attenuation_column_g_cm2: float | None = None,
        b_scale: float = 1.0,
        d_scale: float = 1.0,
    ) -> None:
        if method not in ("drift", "diffusion", "exact"):
            raise ValueError(
                f"method must be 'drift', 'diffusion', or 'exact', got {method!r}."
            )
        self.radius_km = radius_km
        self.density_g_cm3 = density_g_cm3
        self.method = method
        self.column_depth_km = column_depth_km
        self.attenuation_column_g_cm2 = attenuation_column_g_cm2
        self.b_scale = b_scale
        self.d_scale = d_scale
        self.n_nucleon_cm3 = nucleon_number_density(density_g_cm3)
        radius_cm = radius_km * CM_PER_KM
        self.v_det_cm3 = 4.0 / 3.0 * np.pi * radius_cm**3

    def target_volume_cm3(
        self,
        energy_gev: float | np.ndarray,
        gamma: float,
        lam: float = DEFAULT_LAMBDA,
        part: str = "total",
    ) -> np.ndarray:
        """Effective target volume at a given muon energy.

        Parameters
        ----------
        energy_gev : float or np.ndarray
            Observed muon energy [GeV].
        gamma : float
            Neutrino flux spectral index.
        lam : float, optional
            CC cross-section slope. Defaults to :data:`DEFAULT_LAMBDA`.
        part : {"total", "soft", "inside"}, optional
            Which contribution to return. ``"inside"`` is the instrumented
            sphere, ``"soft"`` the transport-enhanced soft volume, ``"total"``
            their sum.

        Returns
        -------
        volume : np.ndarray
            Target volume [cm^3], broadcast to the shape of ``energy_gev``.

        Notes
        -----
        For ``method="exact"`` the inelasticity factor ``I(A)`` multiplies both
        populations, so the reported ``"inside"`` volume is the effective
        ``I(A) V_det`` rather than the bare geometric sphere.
        """
        if self.method == "exact":
            v_soft = soft_volume_exact(
                self.radius_km,
                energy_gev,
                gamma,
                lam,
                self.column_depth_km,
                self.density_g_cm3,
                b_scale=self.b_scale,
                d_scale=self.d_scale,
            )
            a = spectral_index(gamma, lam)
            v_det_cm3 = inelasticity_factor(a) * self.v_det_cm3
        elif self.method == "diffusion":
            v_soft = soft_volume_diffusion(
                self.radius_km, energy_gev, gamma, lam, self.density_g_cm3,
                b_scale=self.b_scale, d_scale=self.d_scale,
            )
            v_det_cm3 = self.v_det_cm3
        else:
            v_soft = soft_volume_drift(
                self.radius_km, energy_gev, gamma, lam, self.density_g_cm3,
                b_scale=self.b_scale,
            )
            v_det_cm3 = self.v_det_cm3

        v_soft_cm3 = v_soft * CM_PER_KM**3
        v_det_cm3 = np.full_like(v_soft_cm3, v_det_cm3)
        if part == "soft":
            return v_soft_cm3
        if part == "inside":
            return v_det_cm3
        if part == "total":
            return v_det_cm3 + v_soft_cm3
        raise ValueError(f"part must be 'total', 'soft', or 'inside', got {part!r}.")

    def effective_area_cm2(
        self,
        energy_gev: float | np.ndarray,
        gamma: float,
        lam: float = DEFAULT_LAMBDA,
        part: str = "total",
    ) -> np.ndarray:
        """Implied effective area ``A_eff(E) = V_target(E) n_N sigma_CC(E)``.

        Recasts the target-volume factorization (Eq. 2.20) in the same units and
        convention as the published :class:`~softpaws.response.irfs.EffectiveArea`
        (``dN/dE = A_eff(E) dphi/dE``), so the two can be overlaid directly. Unlike
        the published table, this is fit-free but not flux-independent: the soft
        volume itself depends on the assumed spectral index ``gamma`` (Eq. 2.23),
        so ``A_eff`` here is only exact at the ``gamma`` it is evaluated with.

        Parameters
        ----------
        energy_gev : float or np.ndarray
            Observed muon energy [GeV].
        gamma : float
            Neutrino flux spectral index, entering only through the soft
            volume's spectral dependence.
        lam : float, optional
            CC cross-section slope. Defaults to :data:`DEFAULT_LAMBDA`.
        part : {"total", "soft", "inside"}, optional
            Which target-volume contribution to use.

        Returns
        -------
        aeff : np.ndarray
            Implied effective area [cm^2], broadcast to the shape of
            ``energy_gev``.
        """
        volume_cm3 = self.target_volume_cm3(energy_gev, gamma, lam, part)
        sigma = cc_cross_section(energy_gev, lam)
        return volume_cm3 * self.n_nucleon_cm3 * sigma

    def threshold_effective_area_cm2(
        self,
        energy_nu_gev: float | np.ndarray,
        threshold_gev: float = DEFAULT_MUON_THRESHOLD_GEV,
        lam: float = DEFAULT_LAMBDA,
        mean_inelasticity: float = MEAN_INELASTICITY,
    ) -> np.ndarray:
        """Effective area in the published-``A_eff`` convention.

        Like-for-like counterpart of :meth:`effective_area_cm2`, built to match
        how the IceCube instrument response is tabulated rather than how the soft
        volume is derived. Two things change:

        * the argument is the **true neutrino energy**, not the observed muon
          energy, with the muon born at ``(1 - <y_w>) E_nu``;
        * the target volume is the threshold muon range
          (:func:`~softpaws.transport.soft_volume.range_target_volume_km3`)
          rather than the spectrally weighted soft volume, because the published
          table integrates over every selected muon energy at fixed ``E_nu``.

        The result carries no ``gamma``: unlike :meth:`effective_area_cm2` it is
        genuinely flux independent, as a tabulated effective area should be.
        Detection efficiency is still not modelled, so this is the geometric
        ceiling of a perfect through-going selection.

        Parameters
        ----------
        energy_nu_gev : float or np.ndarray
            True neutrino energy [GeV].
        threshold_gev : float, optional
            Muon selection threshold [GeV]. Defaults to
            :data:`~softpaws.transport.soft_volume.DEFAULT_MUON_THRESHOLD_GEV`.
        lam : float, optional
            CC cross-section slope. Defaults to :data:`DEFAULT_LAMBDA`.
        mean_inelasticity : float, optional
            Mean CC inelasticity ``<y_w>`` relating the muon energy to the
            neutrino energy. Defaults to
            :data:`~softpaws.transport.source.MEAN_INELASTICITY`.

        Returns
        -------
        aeff : np.ndarray
            Implied effective area [cm^2], broadcast to the shape of
            ``energy_nu_gev``.
        """
        energy_nu = np.atleast_1d(np.asarray(energy_nu_gev, dtype=float))
        energy_mu = (1.0 - mean_inelasticity) * energy_nu
        volume_cm3 = (
            range_target_volume_km3(
                self.radius_km,
                energy_mu,
                threshold_gev,
                self.density_g_cm3,
                self.b_scale,
            )
            * CM_PER_KM**3
        )
        return volume_cm3 * self.n_nucleon_cm3 * cc_cross_section(energy_nu, lam)

    def weak_rate_density(
        self,
        energy_gev: float | np.ndarray,
        phi0: float,
        gamma: float,
        lam: float = DEFAULT_LAMBDA,
    ) -> np.ndarray:
        """Local weak-rate density ``n_N sigma_CC(E) phi_nu(E)``.

        Parameters
        ----------
        energy_gev : float or np.ndarray
            Observed muon energy [GeV] (approximately the neutrino energy).
        phi0, gamma : float
            Power-law flux parameters (see :func:`power_law_flux`).
        lam : float, optional
            CC cross-section slope. Defaults to :data:`DEFAULT_LAMBDA`.

        Returns
        -------
        rate_density : np.ndarray
            Weak-rate density [cm^-3 GeV^-1 s^-1 sr^-1].

        Notes
        -----
        If the response was built with ``attenuation_column_g_cm2``, the flux is
        multiplied by the closed-form Earth survival probability ``D_nu(E)``
        (Form A). This is a single representative column for the band; the
        per-event treatment is :meth:`expected_counts_attenuated`.
        """
        sigma = cc_cross_section(energy_gev, lam)
        flux = power_law_flux(energy_gev, phi0, gamma)
        if self.attenuation_column_g_cm2 is not None:
            flux = flux * survival_probability(
                energy_gev, self.attenuation_column_g_cm2, lam
            )
        return self.n_nucleon_cm3 * sigma * flux

    def differential_rate(
        self,
        energy_gev: float | np.ndarray,
        phi0: float,
        gamma: float,
        lam: float = DEFAULT_LAMBDA,
        part: str = "total",
    ) -> np.ndarray:
        """Differential track rate ``dN / (dt dE dOmega)`` (Eq. 2.20).

        Parameters
        ----------
        energy_gev : float or np.ndarray
            Observed muon energy [GeV].
        phi0, gamma : float
            Power-law flux parameters (see :func:`power_law_flux`).
        lam : float, optional
            CC cross-section slope. Defaults to :data:`DEFAULT_LAMBDA`.
        part : {"total", "soft", "inside"}, optional
            Which target-volume contribution to use.

        Returns
        -------
        rate : np.ndarray
            Differential track rate [GeV^-1 s^-1 sr^-1].
        """
        volume = self.target_volume_cm3(energy_gev, gamma, lam, part)
        return volume * self.weak_rate_density(energy_gev, phi0, gamma, lam)

    def expected_counts(
        self,
        log10_energy_edges: np.ndarray,
        phi0: float,
        gamma: float,
        livetime_s: float,
        solid_angle_sr: float,
        lam: float = DEFAULT_LAMBDA,
        part: str = "total",
        n_subdivisions: int = 64,
    ) -> np.ndarray:
        """Expected track counts per muon-energy bin.

        Integrates the differential rate over each energy bin and over the given
        solid angle and livetime. Angular acceptance, attenuation, and detector
        efficiency are not modelled (see ``docs/soft_volume_notes.md``); the
        result is a geometric through-going estimate.

        Parameters
        ----------
        log10_energy_edges : np.ndarray, shape (n_bins + 1,)
            Muon-energy bin edges in ``log10(E / GeV)``.
        phi0, gamma : float
            Power-law flux parameters (see :func:`power_law_flux`).
        livetime_s : float
            Exposure time [s].
        solid_angle_sr : float
            Solid angle over which the (isotropic) flux is integrated [sr].
        lam : float, optional
            CC cross-section slope. Defaults to :data:`DEFAULT_LAMBDA`.
        part : {"total", "soft", "inside"}, optional
            Which target-volume contribution to use.
        n_subdivisions : int, optional
            Number of log-spaced sample points per bin for the energy integral.

        Returns
        -------
        counts : np.ndarray, shape (n_bins,)
            Expected number of tracks in each energy bin.
        """
        edges = np.asarray(log10_energy_edges, dtype=float)
        counts = np.empty(len(edges) - 1)
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            energy = np.logspace(lo, hi, n_subdivisions)
            rate = self.differential_rate(energy, phi0, gamma, lam, part)
            counts[i] = np.trapezoid(rate, energy)
        return counts * livetime_s * solid_angle_sr

    def expected_counts_attenuated(
        self,
        log10_energy_edges: np.ndarray,
        phi0: float,
        gamma: float,
        livetime_s: float,
        dec_min_deg: float,
        dec_max_deg: float,
        lam: float = DEFAULT_LAMBDA,
        part: str = "total",
        n_subdivisions: int = 64,
        n_dec: int = 64,
    ) -> np.ndarray:
        """Expected track counts with per-event PREM Earth attenuation (Form B).

        Folds the neutrino survival probability into the solid-angle integral
        rather than applying a single representative column. Because the target
        volume, cross section, and (isotropic) flux do not depend on direction,
        the band integral collapses to an energy-dependent effective solid angle
        :func:`softpaws.transport.attenuation.effective_solid_angle`, evaluated
        with the layered PREM column along each declination's Earth chord.

        Unlike :meth:`expected_counts`, the geometric ``solid_angle_sr`` is
        supplied implicitly by the declination band. This method must be called
        on a response *without* the closed-form column
        (``attenuation_column_g_cm2 is None``) so attenuation is not applied
        twice.

        Parameters
        ----------
        log10_energy_edges : np.ndarray, shape (n_bins + 1,)
            Muon-energy bin edges in ``log10(E / GeV)``.
        phi0, gamma : float
            Power-law flux parameters (see :func:`power_law_flux`).
        livetime_s : float
            Exposure time [s].
        dec_min_deg, dec_max_deg : float
            Declination band edges [deg] defining the integrated solid angle.
        lam : float, optional
            CC cross-section slope. Defaults to :data:`DEFAULT_LAMBDA`.
        part : {"total", "soft", "inside"}, optional
            Which target-volume contribution to use.
        n_subdivisions : int, optional
            Number of log-spaced sample points per bin for the energy integral.
        n_dec : int, optional
            Number of declination samples for the effective-solid-angle integral.

        Returns
        -------
        counts : np.ndarray, shape (n_bins,)
            Expected number of tracks in each energy bin.

        Raises
        ------
        ValueError
            Raised if the response also carries a closed-form attenuation column,
            which would double-count the Earth absorption.
        """
        if self.attenuation_column_g_cm2 is not None:
            raise ValueError(
                "expected_counts_attenuated applies per-event attenuation; build the "
                "response with attenuation_column_g_cm2=None to avoid double-counting."
            )
        edges = np.asarray(log10_energy_edges, dtype=float)
        counts = np.empty(len(edges) - 1)
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            energy = np.logspace(lo, hi, n_subdivisions)
            # Differential rate is per steradian; weight by the attenuation-folded
            # effective solid angle at each energy before integrating over energy.
            rate = self.differential_rate(energy, phi0, gamma, lam, part)
            omega_eff = effective_solid_angle(energy, dec_min_deg, dec_max_deg, lam, n_dec)
            counts[i] = np.trapezoid(rate * omega_eff, energy)
        return counts * livetime_s

    def expected_counts_from_flux(
        self,
        log10_energy_edges: np.ndarray,
        flux_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        livetime_s: float,
        dec_min_deg: float,
        dec_max_deg: float,
        lam: float = DEFAULT_LAMBDA,
        part: str = "total",
        attenuate: bool = True,
        n_subdivisions: int = 32,
        n_dec: int = 32,
    ) -> np.ndarray:
        """Expected track counts for an arbitrary, direction-dependent flux.

        Generalizes :meth:`expected_counts_attenuated` from the power law of
        :func:`power_law_flux` to a tabulated flux such as an MCEq atmospheric
        model (``examples/22_atmospheric_background_mceq.py``), which is neither
        a power law nor isotropic. The rate is still Eq. 2.20,

        .. math:: N_i = T \\int_{\\mathrm{bin}\\,i}\\!dE \\int_\\mathrm{band}\\!
            d\\Omega\\; A_\\mathrm{eff}(E)\\,\\phi_\\nu(E, \\mathrm{dec})\\,
            D_\\nu(E, \\mathrm{dec}),

        with ``A_eff = V_target n_N sigma_CC`` from :meth:`effective_area_cm2`.
        The flux no longer factors out of the band integral, so unlike
        :meth:`expected_counts_attenuated` the declination integral is done
        explicitly rather than collapsed into an effective solid angle.

        Because the soft volume is derived for a power-law parent spectrum, the
        spectral index it needs is taken locally, as the log-log slope
        ``gamma_eff(E) = -d ln phi / d ln E`` of the band-integrated arriving
        flux, clipped to :data:`GAMMA_EFF_BOUNDS`. This is exact for a power law
        and a good approximation wherever the flux curves slowly compared with
        the soft volume's own energy dependence.

        Parameters
        ----------
        log10_energy_edges : np.ndarray, shape (n_bins + 1,)
            Muon-energy bin edges in ``log10(E / GeV)``.
        flux_fn : callable
            Differential flux ``flux_fn(E_gev, dec_deg) -> dphi/dE`` in
            ``GeV^-1 cm^-2 s^-1 sr^-1``. Called with a column of energies and a
            row of declinations, so it must broadcast and return shape
            ``(n_energy, n_dec)``.
        livetime_s : float
            Exposure time [s].
        dec_min_deg, dec_max_deg : float
            Declination band edges [deg].
        lam : float, optional
            CC cross-section slope. Defaults to :data:`DEFAULT_LAMBDA`.
        part : {"total", "soft", "inside"}, optional
            Which target-volume contribution to use.
        attenuate : bool, optional
            Whether to apply the per-direction PREM Earth survival probability
            (Form B, as in :meth:`expected_counts_attenuated`). Defaults to
            ``True``. Set ``False`` for a flux that is already attenuated.
        n_subdivisions : int, optional
            Number of log-spaced sample points per bin for the energy integral.
        n_dec : int, optional
            Number of declination samples for the band integral.

        Returns
        -------
        counts : np.ndarray, shape (n_bins,)
            Expected number of tracks in each energy bin.

        Raises
        ------
        ValueError
            Raised if the response also carries a closed-form attenuation column
            while ``attenuate`` is ``True``, which would double-count the Earth
            absorption.
        """
        if attenuate and self.attenuation_column_g_cm2 is not None:
            raise ValueError(
                "expected_counts_from_flux applies per-event attenuation; build the "
                "response with attenuation_column_g_cm2=None to avoid double-counting."
            )
        dec_deg = np.linspace(dec_min_deg, dec_max_deg, n_dec)
        dec_rad = np.deg2rad(dec_deg)
        solid_angle_weight = 2.0 * np.pi * np.cos(dec_rad)  # dOmega / d(dec)
        columns = np.array([prem_column(d) for d in dec_deg]) if attenuate else None

        edges = np.asarray(log10_energy_edges, dtype=float)
        counts = np.empty(len(edges) - 1)
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            energy = np.logspace(lo, hi, n_subdivisions)
            flux = np.asarray(flux_fn(energy[:, None], dec_deg[None, :]), dtype=float)
            if columns is not None:
                flux = flux * survival_probability(energy[:, None], columns[None, :], lam)
            # Band-integrated arriving flux [GeV^-1 cm^-2 s^-1]; the soft volume's
            # spectral weighting is read off its local slope.
            band_flux = np.trapezoid(flux * solid_angle_weight[None, :], dec_rad, axis=1)
            gamma_eff = _local_spectral_index(energy, band_flux)
            aeff = np.array([
                float(self.effective_area_cm2(e, g, lam, part)[0])
                for e, g in zip(energy, gamma_eff)
            ])
            counts[i] = np.trapezoid(aeff * band_flux, energy)
        return counts * livetime_s


def _local_spectral_index(
    energy_gev: np.ndarray,
    flux: np.ndarray,
    bounds: tuple[float, float] = GAMMA_EFF_BOUNDS,
) -> np.ndarray:
    """Local log-log slope ``-d ln phi / d ln E`` of a tabulated flux.

    Parameters
    ----------
    energy_gev : np.ndarray, shape (n,)
        Energies [GeV], strictly increasing.
    flux : np.ndarray, shape (n,)
        Flux at those energies, in any units. Non-positive entries are treated
        as the floor of the positive ones, so the slope stays finite.
    bounds : tuple of float, optional
        ``(min, max)`` the slope is clipped to. Defaults to
        :data:`GAMMA_EFF_BOUNDS`.

    Returns
    -------
    gamma_eff : np.ndarray, shape (n,)
        Effective spectral index at each energy.
    """
    positive = flux[flux > 0.0]
    floor = positive.min() * 1e-6 if positive.size else 1.0
    log_flux = np.log(np.maximum(flux, floor))
    slope = np.gradient(log_flux, np.log(energy_gev))
    return np.clip(-slope, bounds[0], bounds[1])


def tau_induced_differential_rate(
    response: SoftVolumeResponse,
    energy_gev: float | np.ndarray,
    phi0: float,
    gamma: float,
    lam: float = DEFAULT_LAMBDA,
) -> np.ndarray:
    """Muon-track rate from a same-normalization ``nu_tau`` flux.

    Scales the direct ``nu_mu`` differential rate (``response.differential_rate``,
    ``part="total"``) by :func:`softpaws.transport.tau.tau_to_muon_ratio`, giving
    the additional muon-track rate a ``nu_tau`` flux with the *same* ``(phi0,
    gamma)`` would produce via CC tau production and leptonic decay
    (:mod:`softpaws.transport.tau`). This is additive to, not a replacement for,
    the ``nu_mu`` prediction.

    Parameters
    ----------
    response : SoftVolumeResponse
        A response built with ``method="exact"``; the tau ratio needs the exact
        eigenvalue ``Phi(A)``.
    energy_gev : float or np.ndarray
        Observed muon energy [GeV].
    phi0, gamma : float
        Power-law flux parameters (see :func:`power_law_flux`), shared by the
        ``nu_tau`` flux.
    lam : float, optional
        CC cross-section slope. Defaults to :data:`DEFAULT_LAMBDA`.

    Returns
    -------
    rate : np.ndarray
        Tau-induced differential track rate [GeV^-1 s^-1 sr^-1].

    Raises
    ------
    ValueError
        Raised if ``response.method != "exact"``.
    """
    if response.method != "exact":
        raise ValueError(
            f"tau_induced_differential_rate requires method='exact', got "
            f"{response.method!r}."
        )
    numu_rate = response.differential_rate(energy_gev, phi0, gamma, lam, part="total")
    a = spectral_index(gamma, lam)
    ratio = tau_to_muon_ratio(a, energy_gev, response.density_g_cm3)
    return ratio * numu_rate


def tau_induced_expected_counts(
    response: SoftVolumeResponse,
    log10_energy_edges: np.ndarray,
    phi0: float,
    gamma: float,
    livetime_s: float,
    solid_angle_sr: float,
    lam: float = DEFAULT_LAMBDA,
    n_subdivisions: int = 64,
) -> np.ndarray:
    """Expected tau-induced track counts per muon-energy bin.

    Bin-integrated counterpart of :func:`tau_induced_differential_rate`, matching
    the bin-integration convention of :meth:`SoftVolumeResponse.expected_counts`.

    Parameters
    ----------
    response : SoftVolumeResponse
        A response built with ``method="exact"``.
    log10_energy_edges : np.ndarray, shape (n_bins + 1,)
        Muon-energy bin edges in ``log10(E / GeV)``.
    phi0, gamma : float
        Power-law flux parameters shared by the ``nu_tau`` flux.
    livetime_s : float
        Exposure time [s].
    solid_angle_sr : float
        Solid angle over which the (isotropic) flux is integrated [sr].
    lam : float, optional
        CC cross-section slope. Defaults to :data:`DEFAULT_LAMBDA`.
    n_subdivisions : int, optional
        Number of log-spaced sample points per bin for the energy integral.

    Returns
    -------
    counts : np.ndarray, shape (n_bins,)
        Expected number of tau-induced tracks in each energy bin.
    """
    edges = np.asarray(log10_energy_edges, dtype=float)
    counts = np.empty(len(edges) - 1)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        energy = np.logspace(lo, hi, n_subdivisions)
        rate = tau_induced_differential_rate(response, energy, phi0, gamma, lam)
        counts[i] = np.trapezoid(rate, energy)
    return counts * livetime_s * solid_angle_sr


def tau_induced_expected_counts_attenuated(
    response: SoftVolumeResponse,
    log10_energy_edges: np.ndarray,
    phi0: float,
    gamma: float,
    livetime_s: float,
    dec_min_deg: float,
    dec_max_deg: float,
    lam: float = DEFAULT_LAMBDA,
    n_subdivisions: int = 64,
    n_dec: int = 64,
) -> np.ndarray:
    """Expected tau-induced track counts with per-event PREM attenuation (Form B).

    Tau counterpart of :meth:`SoftVolumeResponse.expected_counts_attenuated`,
    reusing the same effective-solid-angle weighting
    (:func:`softpaws.transport.attenuation.effective_solid_angle`). The tau
    module neglects regeneration and uses the plain ``nu_mu`` survival
    probability (:mod:`softpaws.transport.attenuation`), so the same ``D_nu(E)``
    is applied to the ``nu_tau`` flux here.

    Parameters
    ----------
    response : SoftVolumeResponse
        A response built with ``method="exact"`` and
        ``attenuation_column_g_cm2=None`` (per-event attenuation is applied
        here, not via the closed-form column).
    log10_energy_edges : np.ndarray, shape (n_bins + 1,)
        Muon-energy bin edges in ``log10(E / GeV)``.
    phi0, gamma : float
        Power-law flux parameters shared by the ``nu_tau`` flux.
    livetime_s : float
        Exposure time [s].
    dec_min_deg, dec_max_deg : float
        Declination band edges [deg] defining the integrated solid angle.
    lam : float, optional
        CC cross-section slope. Defaults to :data:`DEFAULT_LAMBDA`.
    n_subdivisions : int, optional
        Number of log-spaced sample points per bin for the energy integral.
    n_dec : int, optional
        Number of declination samples for the effective-solid-angle integral.

    Returns
    -------
    counts : np.ndarray, shape (n_bins,)
        Expected number of tau-induced tracks in each energy bin.

    Raises
    ------
    ValueError
        Raised if the response also carries a closed-form attenuation column,
        which would double-count the Earth absorption.
    """
    if response.attenuation_column_g_cm2 is not None:
        raise ValueError(
            "tau_induced_expected_counts_attenuated applies per-event attenuation; "
            "build the response with attenuation_column_g_cm2=None to avoid "
            "double-counting."
        )
    edges = np.asarray(log10_energy_edges, dtype=float)
    counts = np.empty(len(edges) - 1)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        energy = np.logspace(lo, hi, n_subdivisions)
        rate = tau_induced_differential_rate(response, energy, phi0, gamma, lam)
        omega_eff = effective_solid_angle(energy, dec_min_deg, dec_max_deg, lam, n_dec)
        counts[i] = np.trapezoid(rate * omega_eff, energy)
    return counts * livetime_s
