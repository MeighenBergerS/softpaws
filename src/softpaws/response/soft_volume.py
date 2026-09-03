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
downgoing). Three ways to apply it, in increasing physical fidelity:

- **Form A**, a closed-form representative column via
  ``attenuation_column_g_cm2``, multiplying the flux by a single scalar
  ``D_nu``;
- **Form B**, the per-event PREM treatment of
  :meth:`SoftVolumeResponse.expected_counts_attenuated`, still a decoupled
  multiplicative ``D_nu(E)`` but with a declination-dependent column;
- **Form C**, :meth:`SoftVolumeResponse.expected_counts_coupled_attenuation`,
  which folds the attenuation directly into the soft-volume propagator
  (Eq. 11, App. C.2-C.3 of ``docs/2026_softvolume.pdf``;
  :func:`softpaws.transport.soft_volume.soft_volume_attenuated_exact`)
  instead of applying it as a separate factor. Forms A/B are valid
  approximations wherever ``D_nu`` varies slowly over the soft volume's own
  ~few-``Phi(A)^-1`` km-w.e. extent; Form C is the exact treatment needed
  once the two exponentials are comparable (strongly-absorbed upgoing UHE
  tracks).

:meth:`SoftVolumeResponse.differential_rate_with_cutoff` implements App. H's
recommended mitigation for the ``A < 0`` cross-section-pole blowup
(Sec. VII.C): a spectral cutoff ``E0`` on the parent flux, propagated via
real-space convolution against the exact log-loss density
(:mod:`softpaws.transport.cutoff_source`) rather than the ad hoc
``GAMMA_EFF_BOUNDS`` clip below, which remains the cruder safety net for the
tabulated-flux path (:meth:`SoftVolumeResponse.expected_counts_from_flux`),
since a tabulated flux carries no single ``E0``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# The flux pivot and the single power law live in softpaws.fluxes; the names
# stay importable from here.
from ..fluxes.astrophysical import FLUX_PIVOT_GEV, power_law_flux  # noqa: E402, F401
from ..transport.attenuation import (
    effective_solid_angle,
    neutrino_interaction_length_km,
    prem_column,
    survival_probability,
)
from ..transport.coefficients import DEFAULT_SOURCE, diffusion_coefficient, drift_coefficient
from ..transport.cross_section import CrossSection
from ..transport.cutoff_source import cutoff_soft_rate_density
from ..transport.eigenvalue import spectral_index
from ..transport.soft_volume import (
    DEFAULT_MUON_THRESHOLD_GEV,
    range_target_volume_km3,
    soft_volume_attenuated_exact,
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

# Range the local effective spectral index is clipped to in
# :meth:`SoftVolumeResponse.expected_counts_from_flux`. Wherever a tabulated flux
# falls off a cliff -- an Earth-absorbed atmospheric spectrum, or the end of a
# table -- the measured local slope runs away, while the counts it multiplies are
# already negligible; clipping keeps the soft volume in the regime the transport
# expansion was built for. The lower bound sits above the divergence of the soft
# volume at ``gamma = lambda + 1``.
GAMMA_EFF_BOUNDS = (1.5, 8.0)


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
    cross_section : softpaws.transport.cross_section.CrossSection, optional
        Cross-section model. ``None`` (the default) uses the analytic power law
        of :func:`~softpaws.transport.source.cc_cross_section` with the ``lam``
        passed to each method, which is what reproduces the paper. Supplying a
        :class:`~softpaws.transport.cross_section.TabulatedCrossSection` swaps
        in a tabulated calculation *and* makes the spectral index energy
        dependent, ``A(E) = gamma - lambda_eff(E) - 1``, since a tabulated cross
        section has no single slope. Both matter: the power law overshoots a
        modern calculation by ~9x at 1 TeV, and correcting it without also
        correcting ``lambda`` would be inconsistent (the two partly cancel,
        because a steeper ``lambda`` shrinks ``A`` and so enlarges the soft
        volume).
    coefficient_source : {"proposal", "table1"}, optional
        Which tabulation of ``b_mu`` and ``d_mu`` to use; see
        :mod:`softpaws.transport.coefficients`. Defaults to the PROPOSAL table;
        pass ``"table1"`` to reproduce the paper.
    light_yield_length_km : float or None, optional
        If set, replaces the static projected area ``pi R_det^2`` with the
        energy-growing :func:`~softpaws.transport.soft_volume.
        dynamic_projected_area_km2` in every target-volume calculation
        (not part of arXiv:2607.13143; see that function's docstring for the
        physical motivation and ``examples/26_dynamic_response_effective_area.py``
        for a calibrated value). ``None`` (the default) preserves the
        static-radius behaviour used everywhere else in the package.
    beta : float, optional
        Scale-breaking exponent of App. F (``dGamma/dy ~ E^beta``), for
        ``method="exact"`` only. Defaults to ``0`` (off): the package's own
        measurement (:mod:`softpaws.transport.coefficients`) finds the LPM
        suppression negligible for muons in the covered energy range, so
        there is no calibrated nonzero value shipped -- this exposes App. F's
        machinery structurally (:func:`softpaws.transport.soft_volume.
        scale_breaking_saturation_factor`) rather than asserting an unverified
        photonuclear-rise exponent.

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
    cross_section : softpaws.transport.cross_section.CrossSection or None
        Cross-section model, or ``None`` for the analytic power law.
    coefficient_source : str
        Transport-coefficient tabulation in use.
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
        cross_section: CrossSection | None = None,
        coefficient_source: str = DEFAULT_SOURCE,
        light_yield_length_km: float | None = None,
        beta: float = 0.0,
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
        self.cross_section = cross_section
        self.coefficient_source = coefficient_source
        self.light_yield_length_km = light_yield_length_km
        self.beta = beta
        self.n_nucleon_cm3 = nucleon_number_density(density_g_cm3)
        radius_cm = radius_km * CM_PER_KM
        self.v_det_cm3 = 4.0 / 3.0 * np.pi * radius_cm**3

    def spectral_slope(
        self,
        energy_gev: float | np.ndarray,
        lam: float = DEFAULT_LAMBDA,
    ) -> float | np.ndarray:
        """Cross-section slope ``lambda`` to use at each energy.

        Parameters
        ----------
        energy_gev : float or np.ndarray
            Neutrino energy [GeV].
        lam : float, optional
            Slope of the analytic power law, returned unchanged when no
            tabulated cross section is configured.

        Returns
        -------
        lam_eff : float or np.ndarray
            ``lam`` itself for the power-law model, or the tabulated model's
            local slope at each energy.
        """
        if self.cross_section is not None:
            return self.cross_section.local_slope(energy_gev)
        return lam

    def cc_cross_section_cm2(
        self,
        energy_gev: float | np.ndarray,
        lam: float = DEFAULT_LAMBDA,
    ) -> np.ndarray:
        """Charged-current cross section from the configured model.

        Parameters
        ----------
        energy_gev : float or np.ndarray
            Neutrino energy [GeV].
        lam : float, optional
            Slope of the analytic power law; ignored when a tabulated cross
            section is configured.

        Returns
        -------
        sigma : np.ndarray
            CC cross section per nucleon [cm^2].
        """
        if self.cross_section is not None:
            return self.cross_section.cc(energy_gev)
        return cc_cross_section(energy_gev, lam)

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
        lam_eff = self.spectral_slope(energy_gev, lam)
        if self.method == "exact":
            v_soft = soft_volume_exact(
                self.radius_km,
                energy_gev,
                gamma,
                lam_eff,
                self.column_depth_km,
                self.density_g_cm3,
                b_scale=self.b_scale,
                d_scale=self.d_scale,
                source=self.coefficient_source,
                beta=self.beta,
                light_yield_length_km=self.light_yield_length_km,
            )
            a = spectral_index(gamma, lam_eff)
            v_det_cm3 = inelasticity_factor(a) * self.v_det_cm3
        elif self.method == "diffusion":
            v_soft = soft_volume_diffusion(
                self.radius_km, energy_gev, gamma, lam_eff, self.density_g_cm3,
                b_scale=self.b_scale, d_scale=self.d_scale,
                source=self.coefficient_source,
                light_yield_length_km=self.light_yield_length_km,
            )
            v_det_cm3 = self.v_det_cm3
        else:
            v_soft = soft_volume_drift(
                self.radius_km, energy_gev, gamma, lam_eff, self.density_g_cm3,
                b_scale=self.b_scale, source=self.coefficient_source,
                light_yield_length_km=self.light_yield_length_km,
            )
            v_det_cm3 = self.v_det_cm3

        v_soft_cm3 = v_soft * CM_PER_KM**3
        # Broadcast rather than fill: with a tabulated cross section the
        # inelasticity factor is itself energy dependent, so v_det is an array.
        v_det_cm3 = np.zeros_like(v_soft_cm3) + v_det_cm3
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
        sigma = self.cc_cross_section_cm2(energy_gev, lam)
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
                self.coefficient_source,
                light_yield_length_km=self.light_yield_length_km,
            )
            * CM_PER_KM**3
        )
        return volume_cm3 * self.n_nucleon_cm3 * self.cc_cross_section_cm2(energy_nu, lam)

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
        sigma = self.cc_cross_section_cm2(energy_gev, lam)
        flux = power_law_flux(energy_gev, phi0, gamma)
        if self.attenuation_column_g_cm2 is not None:
            flux = flux * survival_probability(
                energy_gev, self.attenuation_column_g_cm2, lam, self.cross_section
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

    def differential_rate_with_cutoff(
        self,
        energy_gev: float | np.ndarray,
        phi0: float,
        gamma: float,
        e0_cutoff_gev: float,
        lam: float = DEFAULT_LAMBDA,
        part: str = "total",
        n_xi: int = 12,
        n_w: int = 512,
        n_k: int = 4096,
    ) -> np.ndarray:
        """Differential track rate for a source with a spectral cutoff (App. H).

        Sec. VII.C recommends this as the physical mitigation for the
        ``A < 0`` cross-section-pole blowup: a spectral cutoff ``E0`` on the
        parent flux, ``phi_nu ~ E^-gamma e^{-E/E0}``, propagated by real-space
        convolution against the exact log-loss density
        (:func:`softpaws.transport.cutoff_source.cutoff_soft_rate_density`)
        rather than clipped by :data:`GAMMA_EFF_BOUNDS`. See that module's
        docstring for why this evaluates the same physical convolution App. H
        derives (Eq. H1-H3) without going through its Cahen-Mellin pole
        series, which turns out to overflow numerically in exactly the
        large-``x``, negative-``A`` regime it exists to fix.

        The two target-volume populations pick up the cutoff differently. The
        *soft* (transported) population needs the full log-loss convolution,
        since a muon observed at ``E`` was produced at ``eps = E e^w`` for a
        whole distribution of ``w``, not a single value. The *inside*
        (untransported) population needs no such convolution: a muon produced
        right at the detector has no propagator to combine the cutoff with,
        so it simply inherits the real-space source shape ``e^{-E/E0}``
        directly and multiplicatively. Both reduce to the plain (uncut)
        treatment identically as ``e0_cutoff_gev -> inf``.

        The drift/diffusion coefficients ``b_mu``, ``d_mu`` feeding the
        log-loss density are evaluated once, at the geometric mean of
        ``energy_gev``, rather than per observed energy -- the same
        percent-level approximation :func:`softpaws.transport.soft_volume.
        muon_range_km` already makes, justified there and here by how slowly
        (logarithmically) the QED coefficients move with energy.

        Only supported for the analytic power-law cross section
        (``self.cross_section is None``): a tabulated cross section makes the
        spectral index energy dependent, which is outside the scope of this
        single-``A`` expansion.

        Parameters
        ----------
        energy_gev : float or np.ndarray
            Observed muon energy [GeV].
        phi0, gamma : float
            Power-law flux parameters (see :func:`power_law_flux`), evaluated
            **without** the cutoff -- the cutoff is applied by this method,
            not beforehand.
        e0_cutoff_gev : float
            Spectral cutoff energy ``E0`` [GeV] of the parent neutrino flux.
        lam : float, optional
            CC cross-section slope. Defaults to :data:`DEFAULT_LAMBDA`.
        part : {"total", "soft", "inside"}, optional
            Which target-volume contribution to use.
        n_xi, n_w, n_k : int, optional
            Quadrature sizes passed to :func:`~softpaws.transport.
            cutoff_source.cutoff_soft_rate_density` (production-depth
            samples, log-loss grid points, and log-loss inversion nodes).

        Returns
        -------
        rate : np.ndarray
            Differential track rate [GeV^-1 s^-1 sr^-1].

        Raises
        ------
        ValueError
            Raised if ``method != "exact"``, if ``column_depth_km`` is
            ``None`` (the convolution needs a finite column, like
            :func:`~softpaws.transport.soft_volume.saturation_factor`), or if
            a tabulated ``cross_section`` is configured.
        """
        if self.method != "exact":
            raise ValueError(
                f"differential_rate_with_cutoff requires method='exact', got "
                f"{self.method!r}."
            )
        if self.column_depth_km is None:
            raise ValueError(
                "differential_rate_with_cutoff requires a finite column_depth_km "
                "(no infinite-column limit is defined for a cutoff source)."
            )
        if self.cross_section is not None:
            raise ValueError(
                "differential_rate_with_cutoff only supports the analytic power-law "
                "cross section (cross_section=None); a tabulated cross section makes "
                "A energy dependent, which this single-A expansion does not support."
            )
        energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
        a = spectral_index(gamma, lam)
        e_ref = float(np.sqrt(energy.min() * energy.max()))
        b_mu = float(
            self.b_scale * drift_coefficient(e_ref, self.density_g_cm3, self.coefficient_source)[0]
        )
        d_mu = float(
            self.d_scale
            * diffusion_coefficient(e_ref, self.density_g_cm3, self.coefficient_source)[0]
        )

        def weak_rate_density_uncut(eps: np.ndarray) -> np.ndarray:
            sigma = self.cc_cross_section_cm2(eps, lam)
            flux = power_law_flux(eps, phi0, gamma)
            return self.n_nucleon_cm3 * sigma * flux

        factor = float(inelasticity_factor(a))
        inside_rate = (
            factor * np.exp(-energy / e0_cutoff_gev) * self.v_det_cm3
            * weak_rate_density_uncut(energy)
        )

        if part == "inside":
            return inside_rate

        # cutoff_soft_rate_density's production-depth integral carries an implicit
        # length unit of km (from d_xi); convert that one factor to cm so it
        # combines cleanly with the cm^2 projected area and cm^-3 rate density.
        proj_area_cm2 = np.pi * (self.radius_km * CM_PER_KM) ** 2
        soft_rate = factor * proj_area_cm2 * CM_PER_KM * cutoff_soft_rate_density(
            energy, e0_cutoff_gev, self.column_depth_km, b_mu, d_mu,
            weak_rate_density_uncut, n_xi=n_xi, n_w=n_w, n_k=n_k,
        )

        if part == "soft":
            return soft_rate
        if part == "total":
            return inside_rate + soft_rate
        raise ValueError(f"part must be 'total', 'soft', or 'inside', got {part!r}.")

    def expected_counts_with_cutoff(
        self,
        log10_energy_edges: np.ndarray,
        phi0: float,
        gamma: float,
        e0_cutoff_gev: float,
        livetime_s: float,
        solid_angle_sr: float,
        lam: float = DEFAULT_LAMBDA,
        part: str = "total",
        n_subdivisions: int = 32,
        **cutoff_kwargs: int,
    ) -> np.ndarray:
        """Expected track counts per muon-energy bin, with a spectral cutoff.

        Bin-integrated counterpart of :meth:`differential_rate_with_cutoff`,
        matching the bin-integration convention of :meth:`expected_counts`.

        Parameters
        ----------
        log10_energy_edges : np.ndarray, shape (n_bins + 1,)
            Muon-energy bin edges in ``log10(E / GeV)``.
        phi0, gamma : float
            Power-law flux parameters (see :func:`power_law_flux`).
        e0_cutoff_gev : float
            Spectral cutoff energy ``E0`` [GeV] of the parent neutrino flux.
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
        **cutoff_kwargs
            Extra quadrature-size keywords (``n_xi``, ``n_w``, ``n_k``)
            forwarded to :meth:`differential_rate_with_cutoff`.

        Returns
        -------
        counts : np.ndarray, shape (n_bins,)
            Expected number of tracks in each energy bin.
        """
        edges = np.asarray(log10_energy_edges, dtype=float)
        counts = np.empty(len(edges) - 1)
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            energy = np.logspace(lo, hi, n_subdivisions)
            rate = self.differential_rate_with_cutoff(
                energy, phi0, gamma, e0_cutoff_gev, lam, part, **cutoff_kwargs,
            )
            counts[i] = np.trapezoid(rate, energy)
        return counts * livetime_s * solid_angle_sr

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
            omega_eff = effective_solid_angle(
                energy, dec_min_deg, dec_max_deg, lam, n_dec, self.cross_section,
            )
            counts[i] = np.trapezoid(rate * omega_eff, energy)
        return counts * livetime_s

    def expected_counts_coupled_attenuation(
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
        """Expected track counts with attenuation coupled into the propagator (Form C).

        Unlike :meth:`expected_counts_attenuated` (Form B), which multiplies the
        soft volume by a decoupled survival probability ``D_nu(E)``, this folds
        the parent neutrino's Earth attenuation directly into the same depth
        integral that produces the soft volume (Eq. 11, App. C.2-C.3 of
        ``docs/2026_softvolume.pdf``; :func:`softpaws.transport.soft_volume.
        soft_volume_attenuated_exact`). The two agree wherever ``D_nu`` varies
        slowly over the ~few-``Phi(A)^-1`` km-w.e. range the soft volume is
        produced in, and diverge for strongly-absorbed upgoing UHE tracks,
        where the paper's whole point in deriving Eq. 11 is that the two
        exponentials are not generally separable.

        Because the column depth ``x`` (and hence the soft volume itself) now
        depends on declination, this cannot collapse to an effective-solid-
        angle trick the way :meth:`expected_counts_attenuated` does: it
        integrates over energy and declination explicitly, evaluating the
        layered PREM column at each declination sample
        (:func:`softpaws.transport.attenuation.prem_column`), expressed as a
        length at ``self.density_g_cm3`` (the same reference density
        ``Phi(A)`` and ``Lambda_nu`` already use).

        Parameters
        ----------
        log10_energy_edges : np.ndarray, shape (n_bins + 1,)
            Muon-energy bin edges in ``log10(E / GeV)``.
        phi0, gamma : float
            Power-law flux parameters (see :func:`power_law_flux`), evaluated
            **unattenuated**: attenuation is applied inside this method, not
            beforehand.
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
            Number of declination samples for the band integral.

        Returns
        -------
        counts : np.ndarray, shape (n_bins,)
            Expected number of tracks in each energy bin.

        Raises
        ------
        ValueError
            Raised if ``method != "exact"``, or if the response also carries a
            closed-form attenuation column, which would double-count the Earth
            absorption.
        """
        if self.method != "exact":
            raise ValueError(
                f"expected_counts_coupled_attenuation requires method='exact', got "
                f"{self.method!r}."
            )
        if self.attenuation_column_g_cm2 is not None:
            raise ValueError(
                "expected_counts_coupled_attenuation applies its own per-direction "
                "attenuation; build the response with attenuation_column_g_cm2=None "
                "to avoid double-counting."
            )
        if part not in ("total", "soft", "inside"):
            raise ValueError(f"part must be 'total', 'soft', or 'inside', got {part!r}.")

        dec_deg = np.linspace(dec_min_deg, dec_max_deg, n_dec)
        dec_rad = np.deg2rad(dec_deg)
        solid_angle_weight = 2.0 * np.pi * np.cos(dec_rad)  # dOmega / d(dec)
        columns_g_cm2 = np.array([prem_column(d) for d in dec_deg])
        # Column depth as a length at the response's reference density, matching
        # the units Phi(A) and Lambda_nu are evaluated in.
        x_km = columns_g_cm2 / self.density_g_cm3 / CM_PER_KM

        edges = np.asarray(log10_energy_edges, dtype=float)
        counts = np.empty(len(edges) - 1)
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            energy = np.logspace(lo, hi, n_subdivisions)
            lam_eff = self.spectral_slope(energy, lam)
            sigma = self.cc_cross_section_cm2(energy, lam)
            flux = power_law_flux(energy, phi0, gamma)
            inv_lambda_nu = 1.0 / neutrino_interaction_length_km(
                energy, self.density_g_cm3, lam, self.cross_section
            )
            volume_grid_cm3 = np.empty((energy.size, dec_deg.size))
            for j, x in enumerate(x_km):
                v_det_eff, v_soft = soft_volume_attenuated_exact(
                    self.radius_km, energy, gamma, float(x), inv_lambda_nu,
                    lam_eff, self.density_g_cm3, self.b_scale, self.d_scale,
                    self.coefficient_source,
                )
                if part == "soft":
                    volume_km3 = v_soft
                elif part == "inside":
                    volume_km3 = v_det_eff
                else:
                    volume_km3 = v_det_eff + v_soft
                volume_grid_cm3[:, j] = volume_km3 * CM_PER_KM**3
            rate_grid = volume_grid_cm3 * (self.n_nucleon_cm3 * sigma * flux)[:, None]
            band_rate = np.trapezoid(rate_grid * solid_angle_weight[None, :], dec_rad, axis=1)
            counts[i] = np.trapezoid(band_rate, energy)
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
                flux = flux * survival_probability(
                    energy[:, None], columns[None, :], lam, self.cross_section,
                )
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
        omega_eff = effective_solid_angle(
            energy, dec_min_deg, dec_max_deg, lam, n_dec, response.cross_section,
        )
        counts[i] = np.trapezoid(rate * omega_eff, energy)
    return counts * livetime_s
