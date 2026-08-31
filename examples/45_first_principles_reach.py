"""Example 45 -- the reach law with nothing fitted.

Example 44 showed that the gap between our effective-area ceiling and the
published tables is a **depth-dependent light reach**: the effective footprint
has to be evaluated at the energy the muon has where it is seen, not at the
energy it was born with. That example still fitted the two numbers the reach law
carries, the growth per e-fold ``Lambda`` and the pivot ``E_piv``. Here both are
derived, and the two-detector comparison of example 32 is repeated with nothing
adjusted against a published curve.

Deriving the reach
------------------
A muon's Cherenkov output per unit length follows its energy loss. The bare
track radiates at the Frank-Tamm rate

.. math:: \\frac{{\\rm d}N_\\gamma}{{\\rm d}x} = 2\\pi\\alpha
    \\left(\\frac{1}{\\lambda_1} - \\frac{1}{\\lambda_2}\\right)
    \\left(1 - n^{-2}\\right),

and each GeV of radiative loss makes an electromagnetic shower carrying
``L_em ~ 4 m`` of charged track, so the yield is

.. math:: N'(E) = \\frac{{\\rm d}N_\\gamma}{{\\rm d}x}\\,
    \\left(1 + L_{\\rm em}\\, b_\\mu E\\right),

the ``a + bE`` of the loss law with its own coefficients. ``L_em b_mu`` is
density independent, since the shower track length scales as ``1 / rho`` and
``b_mu`` as ``rho``, so the ratio sets a light critical energy near 700 GeV
against the 537 GeV of the loss law.

Light leaves a track cylindrically and is attenuated on the effective photon
length ``Lambda``, which is ``sqrt(lambda_abs lambda_scat / 3)`` where scattering
dominates, as in deep ice, and ``lambda_abs`` where it does not, as in sea water.
Summing the collected charge over the modules within one attenuation length of
the track, the module density cancels and the detection condition reduces to a
charge per nearby module,

.. math:: \\frac{A_{\\rm mod} N'(E)}{\\pi \\Lambda}\\, f(x) \\ge q,

with ``A_mod`` the module's angle-averaged effective area and ``f(x)`` the
fraction of the surrounding solid angle the array fills. A track inside the
array is surrounded, so ``f = 1``; a track at distance ``x`` beyond the boundary
sees the array on one side and through the ice, so ``f = e^{-x/Lambda} / 2``.
Two numbers follow from the one condition:

``E_thr``
    The muon threshold, where a crossing track first meets the condition.

``R_eff(E) = R_det + Lambda ln[N'(E) / N'(E_piv)]``
    The effective footprint, where the pivot is set by ``N'(E_piv) =
    2 N'(E_thr)``. **The array responds beyond its own footprint once the muon
    is twice as bright as threshold**, because the array surrounds a track
    inside it and stands on one side of a track outside it.

Two conditions follow, and keeping them apart is what makes the reach derivable.
A *hit* is a local coincidence -- an HLC pair on neighbouring modules at
IceCube, two photomultipliers of one module at KM3NeT -- with each receiver
converting its share of the collected charge into at least one photoelectron at
Poisson probability (:func:`hit_probability`). What a selection then demands is
a **multiplicity**: enough of those hits, and IceCube's simple-majority trigger
asks for eight (:data:`DEFAULT_MIN_MODULES`). The mean count over the in-array
transverse plane (:func:`hit_count`) sets the reach, the Poisson probability of
meeting the multiplicity at that mean makes the threshold a smooth turn-on, and
no charge is left to choose.

Every remaining input is an instrument or medium number: the photocathode area,
the module density, the photon detection efficiency against wavelength and the
medium's absorption and scattering against wavelength. **Wavelength is not a
detail here.** Taking the efficiency flat at its peak across the nominal
300-600 nm band overstates the collected light by about a factor of two, since
the real response is a bump ~120 nm wide, cut off below by the housing glass and
above by the cathode; and deep ice is opaque past 500 nm, so the red half of the
band never reaches a distant module at all. Both integrals are done properly.

What the array counts, and what it does not
-------------------------------------------
The reach says where a track is bright enough. It does not say whether the track
is *reconstructible*, and a published response is built from reconstructed
tracks. A line that clips a corner of the array crosses the instrumented volume
and leaves again with no lever arm, and every such line is counted by the
convex-body projection of :func:`prism_projected_area_km2`. Requiring a minimum
path ``l`` inside the volume removes them, and the removal is not a flat
efficiency: the lines it removes are concentrated at oblique incidence, which is
exactly where the intact prism's cap and side terms *add*.

That single requirement fixes a residual the reach cannot. Without it the two
models sit a factor ~1.2 above both published curves at the low-energy end,
where the reach is inert and the Earth is transparent, and the factor is nearly
the same at two detectors of quite different shape -- because for any roughly
equidimensional convex body the direction-averaged area sits ~1.18 above the
equal-volume sphere's, whatever the shape. See :data:`DEFAULT_MIN_TRACK_KM` and
:func:`~softpaws.transport.soft_volume.eroded_prism_target_km2`.

The two act on different things, which is why both are needed. ``l`` sets the
level and the zenith dependence and is energy independent; ``q`` sets how fast
the footprint grows with energy and so carries the slope.

What the example reports
------------------------
Two curves per detector against its published one. ``First principles`` carries
the whole model with every number above taken from the medium and the module.
``Fitted`` floats the attenuation length ``Lambda`` against the same curve, so
the fitted value is directly comparable with what is known about the ice and the
water. That comparison is the test: an argument that predicts the coefficient is
worth more than one that only predicts the shape. ``Lambda`` and ``q`` are
floated together, which is the same two-parameter freedom example 32 gave its
fitted reach.

An optional scan over ``q`` is available with ``--scan``. It shifts ``R_eff``
rigidly and moves the threshold with it, so ``q`` sets the level and every
*shape* stays a prediction.

Both ``nu_mu`` and ``nu_tau -> tau -> mu`` are summed. The DR2 readme describes
the table as an average over simulated muon neutrino events, which reads as a
``nu_mu`` response, and its own deepest declination bands contradict that: a
``nu_mu`` flux cannot cross the full Earth diameter at 10^8 GeV, because escaping
requires degrading three decades through interactions that are neutral current
only 30% of the time. The tau cascade can, since its charged current is not
terminal. See :data:`DEFAULT_FLAVOURS`.

Usage
-----
    python examples/45_first_principles_reach.py
    python examples/45_first_principles_reach.py --min-modules 4
    python examples/45_first_principles_reach.py --numu-only
    python examples/45_first_principles_reach.py --scan
"""

import argparse
import importlib.util
import pathlib
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammainc

from softpaws.transport.coefficients import drift_coefficient
from softpaws.transport.cross_section import CrossSection, bgr18_cross_section
from softpaws.transport.soft_volume import (
    DEFAULT_MUON_THRESHOLD_GEV,
    eroded_prism_target_km2,
    stochastic_muon_range_km,
    truncated_muon_range_km,
)
from softpaws.transport.source import mean_inelasticity, nucleon_number_density
from softpaws.transport.tau import BR_TAU_TO_MU, MEAN_Z
from softpaws.utils.constants import CM_PER_KM, M_PER_KM, RHO_ICE_G_CM3, RHO_WATER_G_CM3

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"

#: Fine-structure constant, for the Frank-Tamm yield.
FINE_STRUCTURE = 7.2973525693e-3

#: Wavelengths every optical integral runs over [nm]. Wide enough that the
#: photocathode and the medium, not the grid, decide where the band ends.
WAVELENGTH_NM = np.linspace(280.0, 680.0, 201)

#: Wavelength the media's quoted attenuation lengths are anchored at [nm]. Both
#: sites' published absorption and scattering lengths are near-peak values, and
#: the tabulated shapes below are normalized here so those numbers keep their
#: meaning.
ANCHOR_NM = 400.0

#: Charged track length in an electromagnetic shower [m GeV^-1], at water
#: density. Paired with the water-density ``b_mu`` below, so the product that
#: enters the yield is density independent.
EM_TRACK_LENGTH_M_PER_GEV = 4.0

#: Coincidence partners of a single-PMT module: IceCube's HLC accepts the
#: nearest or next-to-nearest neighbour on the same string, up or down. Part of
#: the trigger definition, and a count, so it is not a tunable.
HLC_PARTNERS = 4

#: Modules that must register a coincident hit for the track to count. This
#: replaces the earlier "charge on the nearest module", which needed a number
#: of photoelectrons that could not be argued for and was set against the
#: published curves. A hit is a *local coincidence* -- an HLC pair at IceCube,
#: two photomultipliers of one module at KM3NeT -- with each receiver firing on
#: one photoelectron at Poisson probability (:func:`hit_probability`); what a
#: trigger then demands is a *multiplicity*, and IceCube's simple-majority
#: trigger asks for eight. Everything else the condition needs is a published
#: instrument number: the module density, the photocathode area, the efficiency
#: curve and the medium's optics. See :func:`hit_count`.
DEFAULT_MIN_MODULES = 8.0

#: Minimum path a track must have inside the instrumented volume to be
#: reconstructed [km]. A published response is built from *reconstructed* tracks,
#: and a track that clips a corner of the array leaves again before it has a
#: lever arm. Imposing the requirement erodes the target body
#: (:func:`~softpaws.transport.soft_volume.eroded_prism_target_km2`), most
#: strongly at oblique incidence, which is exactly where the intact prism's cap
#: and side terms add and where a published declination dependence carries no
#: such enhancement. The default takes the direction-averaged projected area down
#: by 1.156 at IceCube and the instrumented volume by 1.43. It is 1.8 horizontal
#: string spacings, which is the scale on which the array resolves a direction at
#: all.
#:
#: **It is per detector, because it belongs to the selection and not to the
#: optics**, and the two published curves are not at the same selection level.
#: The DR2 table is an analysis-level response built from reconstructed tracks,
#: so the requirement applies. The ARCA230 curve is at **trigger** level, where
#: nothing has been reconstructed yet and a track only has to fire the array, so
#: ``ARCA_SITE`` carries zero. Imposing 230 m on it instead drives the model 20%
#: *below* the trigger curve near 10^4.8, which a trigger-level comparison may
#: not do.
DEFAULT_MIN_TRACK_KM = 0.23

#: Band the models are compared over at each site, matching example 32.
IC_BAND = (5.0, 7.8)
ARCA_BAND = (4.0, 7.5)

#: Energy range both figures are drawn over, ``log10(E_nu / GeV)``.
PLOT_BAND = (4.0, 9.0)

#: Energies the ratio is tabulated at, ``log10(E_nu / GeV)``.
ANCHORS = (5.0, 6.0, 7.0)

#: Channels summed. The DR2 readme calls the released table an average over
#: "simulated muon neutrino events", which reads as a ``nu_mu`` response, but the
#: table's own deepest declination bands rule that out: at ``sin(dec) = 0.98`` it
#: stays flat at 2-3e5 cm^2 from 10^5.6 to 10^8.6, and a ``nu_mu``-only model
#: falls to 8.9e1 cm^2 there. Nothing about the neutral-current treatment can
#: close that. A 10^8 GeV neutrino crossing 1.02e10 g/cm^2 must degrade to
#: ~10^5 GeV to escape, which takes ~24 successive scatters, and each interaction
#: is neutral current only 30% of the time -- a suppression of order 0.30^24. For
#: ``nu_tau`` the charged current is not terminal, since the tau decays back to a
#: ``nu_tau`` at ~0.4 of the energy, so that cascade alone survives the full
#: Earth diameter. The deep bands therefore require the tau channel, and the
#: smearing matrix rules out their being noise (~1600 simulated events per bin).
DEFAULT_FLAVOURS = ("mu", "tau")

#: CSMS isoscalar cross sections [pb] (Cooper-Sarkar, Mertsch and Sarkar,
#: arXiv:1106.3723, Tables 1 and 2), on ``CSMS_LOG10_E``. The shipped BGR18
#: tables are ``nu-p`` used as per-nucleon, which under-counts an isoscalar
#: target by 17% at 10^5 GeV falling to ~11% at 10^7 -- the valence-quark
#: difference between protons and neutrons, largest where valence still
#: matters. :class:`IsoscalarCrossSection` pins each channel to these values.
CSMS_LOG10_E = np.array([4.0, np.log10(5.0e4), 5.0, np.log10(2.0e5),
                         np.log10(5.0e5), 6.0, np.log10(2.0e6), np.log10(5.0e6),
                         7.0, np.log10(2.0e7), np.log10(5.0e7), 8.0])
CSMS_PB = {
    "nu": {"cc": (47.0, 140.0, 210.0, 310.0, 490.0, 690.0, 950.0, 1400.0,
                  1900.0, 2600.0, 3700.0, 4800.0),
           "nc": (15.0, 49.0, 75.0, 110.0, 180.0, 260.0, 360.0, 540.0,
                  730.0, 980.0, 1400.0, 1900.0)},
    "nubar": {"cc": (31.0, 110.0, 180.0, 270.0, 460.0, 660.0, 920.0, 1400.0,
                     1900.0, 2500.0, 3700.0, 4800.0),
              "nc": (11.0, 39.0, 64.0, 99.0, 170.0, 240.0, 350.0, 530.0,
                     730.0, 980.0, 1400.0, 1900.0)},
}


class IsoscalarCrossSection(CrossSection):
    """A shipped ``nu-p`` table pinned to the CSMS isoscalar values.

    Scales the base model's charged- and neutral-current channels by the ratio
    of the CSMS isoscalar value to the base value at the CSMS energies,
    interpolated in ``log E`` and held at the ends. Both channels scale, so the
    Earth attenuation the total sets stays consistent with the interaction
    rate.

    Parameters
    ----------
    base : softpaws.transport.cross_section.CrossSection
        The shipped table to correct.
    species : {"nu", "nubar"}
        Which CSMS column to pin to.

    Notes
    -----
    ``local_slope`` delegates to the base model: the correction drifts by ~0.2
    in ``ln sigma`` over nine e-folds of energy, a slope shift of ~0.02,
    below the smoothing spline's own uncertainty.
    """

    def __init__(self, base, species: str):
        self._base = base
        self._log_ratio = {}
        for channel in ("cc", "nc"):
            target = np.asarray(CSMS_PB[species][channel], dtype=float) * 1.0e-36
            ours = np.array([
                float(np.atleast_1d(getattr(base, channel)(10.0**log_e))[0])
                for log_e in CSMS_LOG10_E
            ])
            self._log_ratio[channel] = np.log(target / ours)

    def _scale(self, channel: str, energy_gev) -> np.ndarray:
        log_e = np.log10(np.asarray(energy_gev, dtype=float))
        return np.exp(np.interp(log_e, CSMS_LOG10_E, self._log_ratio[channel]))

    def cc(self, energy_gev):
        """Charged-current cross section [cm^2], pinned to CSMS."""
        return self._base.cc(energy_gev) * self._scale("cc", energy_gev)

    def nc(self, energy_gev):
        """Neutral-current cross section [cm^2], pinned to CSMS."""
        return self._base.nc(energy_gev) * self._scale("nc", energy_gev)

    def local_slope(self, energy_gev):
        """Local slope of the base model; see the class notes."""
        return self._base.local_slope(energy_gev)


#: Both published tables are neutrino/antineutrino averages: KM3NeT writes
#: ``A_eff(nu_i + nubar_i) / 2`` and the DR2 companion paper (arXiv:2605.19040)
#: states the area is "averaged assuming an equal number of neutrinos and
#: antineutrinos". The comparand therefore needs both species, each carrying
#: its own Earth absorption, and each pinned to its own CSMS isoscalar column.
SPECIES = (IsoscalarCrossSection(bgr18_cross_section(), "nu"),
           IsoscalarCrossSection(bgr18_cross_section("BGR18_nubar"), "nubar"))

#: Colour carries which curve it is and line style carries which detector, so
#: the two legends factorize. Shared with example 32.
MODEL_COLOR = {
    "Published": "#e7298a",
    "First principles": "#7570b3",
    "Fitted": "#1b9e77",
}
MODEL_WIDTH = {"Published": 2.4, "First principles": 1.0, "Fitted": 1.0}
MODEL_ZORDER = {"Published": 1, "First principles": 3, "Fitted": 2}
#: The published curve is drawn wide, pale and underneath, so the model curves
#: reading on top of it stay legible exactly where they agree with it.
MODEL_ALPHA = {"Published": 0.5, "First principles": 1.0, "Fitted": 1.0}
MODELS = ("First principles", "Fitted")
DETECTOR_STYLE = {"IceCube": "-", "KM3NeT/ARCA230": "--"}


@dataclass(frozen=True)
class Site:
    """Optical medium and optical module of one detector.

    Every field is a published instrument or medium property. None of them is
    adjusted against an effective-area curve.

    Attributes
    ----------
    name : str
        Detector name.
    refractive_index : float
        Phase refractive index of the medium near 400 nm.
    absorption_m : float
        Photon absorption length [m].
    scattering_m : float
        Effective photon scattering length [m].
    cathode_area_m2 : float
        Total photocathode area of one optical module [m^2].
    quantum_efficiency : float
        Photon detection efficiency of the photocathode.
    density_g_cm3 : float
        Density of the medium [g cm^-3].
    headroom_above_m : float
        Depth of optical medium between the top of the instrumented volume and
        the upper boundary of the medium [m].
    headroom_below_m : float
        Depth of optical medium between the bottom of the instrumented volume
        and the lower boundary of the medium [m].
    min_track_km : float
        Minimum path a track must have inside the instrumented volume to enter
        this detector's published response [km]. It belongs to the *selection*
        the curve was made at and not to the optics, so it is per detector; see
        :data:`DEFAULT_MIN_TRACK_KM`.
    module_density_per_km3 : float
        Optical modules per cubic kilometre of instrumented volume.
    efficiency_nm, efficiency : tuple of float
        Photon detection efficiency against wavelength [nm]: quantum efficiency
        times the transmission of the pressure sphere and the gel.
    absorption_nm, absorption_shape : tuple of float
        Relative absorption length against wavelength [nm], normalized at
        :data:`ANCHOR_NM` so that ``absorption_m`` sets the scale.
    scattering_nm, scattering_shape : tuple of float
        Relative effective scattering length against wavelength [nm], normalized
        the same way against ``scattering_m``.
    attenuation_override_m : float or None, optional
        Attenuation length [m] to use in place of the one the absorption and
        scattering lengths imply. Set by the fit, unset everywhere else.
    illuminated_pmts : int or None, optional
        Photomultipliers of one module that see a distant track, for a
        multi-PMT module whose local coincidence is internal: KM3NeT's L1 is
        any two PMTs of one module, and a track's light reaches roughly the
        facing hemisphere, ~12 of ARCA's 31. ``None`` (the default) is a
        single-PMT module whose coincidence partners are its string
        neighbours, as in IceCube's HLC. See :func:`hit_probability`.
    """

    name: str
    refractive_index: float
    absorption_m: float
    scattering_m: float
    cathode_area_m2: float
    quantum_efficiency: float
    density_g_cm3: float
    headroom_above_m: float
    headroom_below_m: float
    min_track_km: float
    module_density_per_km3: float
    efficiency_nm: tuple[float, ...]
    efficiency: tuple[float, ...]
    absorption_nm: tuple[float, ...]
    absorption_shape: tuple[float, ...]
    scattering_nm: tuple[float, ...]
    scattering_shape: tuple[float, ...]
    attenuation_override_m: float | None = None
    illuminated_pmts: int | None = None


#: Photon detection efficiency against wavelength [nm]: photocathode quantum
#: efficiency times pressure-sphere and gel transmission. **These are smooth
#: parameterizations of the published shapes, not digitized vendor curves**, and
#: swapping in the collaborations' own tables is a drop-in improvement. They
#: carry the two features that matter: a peak near 0.22-0.27 in the near
#: ultraviolet, and a band ~120 nm wide inside the nominal 300 nm, cut below by
#: the housing glass and above by the bialkali cathode. IceCube's sphere cuts
#: near 330 nm, KM3NeT's a little lower, and its smaller tubes reach a slightly
#: higher peak.
IC_EFFICIENCY_NM = (280.0, 300.0, 320.0, 340.0, 360.0, 380.0, 400.0, 420.0,
                    450.0, 500.0, 550.0, 600.0, 650.0)
IC_EFFICIENCY = (0.0, 0.003, 0.035, 0.115, 0.180, 0.212, 0.220, 0.212,
                 0.180, 0.118, 0.058, 0.018, 0.0)
ARCA_EFFICIENCY_NM = (280.0, 300.0, 320.0, 340.0, 360.0, 380.0, 400.0, 420.0,
                      450.0, 500.0, 550.0, 600.0, 650.0)
ARCA_EFFICIENCY = (0.0, 0.020, 0.125, 0.205, 0.250, 0.268, 0.262, 0.242,
                   0.200, 0.130, 0.062, 0.020, 0.0)

#: Relative absorption and effective-scattering lengths against wavelength [nm],
#: normalized at :data:`ANCHOR_NM` so each site's published length sets the
#: scale. Also parameterizations of the published shapes. Deep ice is clearest
#: near 400 nm and its absorption collapses beyond 500 nm as the intrinsic
#: absorption of water takes over, which is why the red half of the nominal band
#: contributes almost nothing at any interesting distance; the dust that
#: dominates below 350 nm closes the other end. Scattering in ice falls smoothly
#: with wavelength, roughly as ``lambda^0.9``.
ICE_ABSORPTION_NM = (280.0, 320.0, 360.0, 400.0, 440.0, 480.0, 520.0, 560.0,
                     600.0, 650.0)
ICE_ABSORPTION_SHAPE = (0.28, 0.55, 0.85, 1.00, 0.98, 0.82, 0.52, 0.27,
                        0.11, 0.04)
ICE_SCATTERING_NM = (280.0, 400.0, 600.0)
ICE_SCATTERING_SHAPE = (0.72, 1.00, 1.43)

#: The same for Capo Passero sea water, whose clarity peaks in the blue near
#: 450-470 nm rather than in the near ultraviolet, and whose scattering is
#: Rayleigh-like and so falls steeply with wavelength.
SEA_ABSORPTION_NM = (280.0, 320.0, 360.0, 400.0, 440.0, 470.0, 500.0, 550.0,
                     600.0, 650.0)
SEA_ABSORPTION_SHAPE = (0.22, 0.50, 0.78, 1.00, 1.18, 1.21, 0.98, 0.44,
                        0.14, 0.05)
SEA_SCATTERING_NM = (280.0, 400.0, 600.0)
SEA_SCATTERING_SHAPE = (0.42, 1.00, 2.25)

#: Deep South Pole ice below the dust layer, and the IceCube digital optical
#: module: one downward-facing 10-inch photomultiplier of 324 cm^2 photocathode.
#: The array is instrumented from 1450 to 2450 m and the ice sheet is ~2820 m
#: thick at the Pole, so the reach has 370 m of ice below the deepest module and
#: bedrock after that. The headroom above is nominally 1450 m and never binds at
#: these reaches; the dust layer near 2000 m makes it optically worse than the
#: single attenuation length here says, which is a reason not to lean on it.
ICECUBE_SITE = Site(
    "IceCube", 1.33, 175.0, 60.0, 0.0324, 0.25, 0.92, 1450.0, 370.0,
    DEFAULT_MIN_TRACK_KM,
    module_density_per_km3=5160.0,
    efficiency_nm=IC_EFFICIENCY_NM, efficiency=IC_EFFICIENCY,
    absorption_nm=ICE_ABSORPTION_NM, absorption_shape=ICE_ABSORPTION_SHAPE,
    scattering_nm=ICE_SCATTERING_NM, scattering_shape=ICE_SCATTERING_SHAPE,
)

#: The Capo Passero site, and the KM3NeT multi-photomultiplier module: 31
#: 3-inch photomultipliers covering the full sphere, 1260 cm^2 in total. Sea
#: water scatters weakly, so its attenuation is absorption limited. The lowest
#: storey sits ~80 m above a 3500 m seabed, which is what the downward reach has
#: to work with; above the highest storey there are ~2800 m of water.
ARCA_SITE = Site(
    "KM3NeT/ARCA230", 1.35, 68.0, 265.0, 0.126, 0.25, 1.04, 2810.0, 80.0,
    0.0,
    module_density_per_km3=3902.0,
    efficiency_nm=ARCA_EFFICIENCY_NM, efficiency=ARCA_EFFICIENCY,
    absorption_nm=SEA_ABSORPTION_NM, absorption_shape=SEA_ABSORPTION_SHAPE,
    scattering_nm=SEA_SCATTERING_NM, scattering_shape=SEA_SCATTERING_SHAPE,
    illuminated_pmts=12,
)

#: Fraction of its photocathode area a module presents to an arriving photon.
#: A sphere uniformly covered with photocathode of area ``A`` presents ``A / 4``
#: from every direction; a single flat photomultiplier facing one hemisphere
#: with cosine acceptance averages to the same quarter over the full sky.
PROJECTED_FRACTION = 0.25


def load_example_32():
    """Import example 32, whose published curves and geometry this reuses."""
    spec = importlib.util.spec_from_file_location(
        "_example_32", _HERE / "32_effective_area_comparison.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR,
                        help="Directory holding the IceTracks-DR2 release.")
    parser.add_argument("--min-modules", type=float, default=DEFAULT_MIN_MODULES,
                        help="Modules that must collect at least one photoelectron.")
    parser.add_argument("--min-track-km", type=float, default=None,
                        help="Override the minimum in-detector path [km] at BOTH sites. "
                             "By default each keeps its own; see DEFAULT_MIN_TRACK_KM.")
    parser.add_argument("--n-energy", type=int, default=40,
                        help="Points in the arrival-energy quadrature.")
    parser.add_argument("--numu-only", action="store_true",
                        help="Drop the nu_tau -> tau -> mu channel. Fails the ceiling in "
                             "the deepest declination bands; see DEFAULT_FLAVOURS.")
    parser.add_argument("--scan", action="store_true",
                        help="Also scan the per-module charge, which sets the level.")
    parser.add_argument("--out", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "45_first_principles_reach",
                        help="Output stem; '_area' and '_ratio' figures are written, "
                             "each as .pdf and .png.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# The light-yield model, resolved in wavelength
# ---------------------------------------------------------------------------


def cherenkov_spectrum_per_m_per_nm(site: Site) -> np.ndarray:
    """Frank-Tamm photon yield of a bare relativistic track [m^-1 nm^-1].

    .. math:: \\frac{{\\rm d}^2N_\\gamma}{{\\rm d}x\\,{\\rm d}\\lambda}
        = \\frac{2\\pi\\alpha}{\\lambda^2}\\left(1 - n^{-2}\\right),

    on :data:`WAVELENGTH_NM`. Integrating it over 300-600 nm returns the ~3.3e4
    photons per metre that a single-number treatment starts from, but the
    ``1 / lambda^2`` is what decides *which* photons those are, and neither the
    photocathode nor the medium treats them alike.

    Parameters
    ----------
    site : Site
        Detector site, for its refractive index.

    Returns
    -------
    spectrum : np.ndarray
        Photons per metre of track per nanometre, on :data:`WAVELENGTH_NM`.
    """
    lam_m = WAVELENGTH_NM * 1.0e-9
    per_m_per_m = (2.0 * np.pi * FINE_STRUCTURE / lam_m**2
                   * (1.0 - site.refractive_index**-2))
    return per_m_per_m * 1.0e-9


def detection_efficiency(site: Site) -> np.ndarray:
    """Probability that a photon reaching the module makes a photoelectron.

    The photocathode's quantum efficiency times the transmission of the pressure
    sphere and the optical gel, tabulated together on :data:`WAVELENGTH_NM`.
    Taking it flat at its peak across the whole band, which is what a single
    ``quantum_efficiency`` does, counts photons the module cannot convert: the
    response is a bump ~120 nm wide sitting inside a 300 nm band, cut off below
    by the glass and above by the cathode.

    Parameters
    ----------
    site : Site
        Detector site, for its tabulated efficiency curve.

    Returns
    -------
    efficiency : np.ndarray
        Photon detection efficiency on :data:`WAVELENGTH_NM`.
    """
    grid, values = np.asarray(site.efficiency_nm), np.asarray(site.efficiency)
    return np.interp(WAVELENGTH_NM, grid, values, left=0.0, right=0.0)


def attenuation_spectrum_m(site: Site) -> np.ndarray:
    """Effective photon attenuation length of the medium, per wavelength [m].

    Where scattering is short against absorption the transport is diffusive and
    the flux falls on ``sqrt(lambda_abs lambda_scat / 3)``; where it is not, the
    light travels ballistically and the length is ``lambda_abs``. The shorter of
    the two selects the applicable limit at each wavelength.

    Both lengths are the site's own single-wavelength values carried by a
    tabulated *shape*, normalized at :data:`ANCHOR_NM`, so ``absorption_m`` and
    ``scattering_m`` keep their published meaning and only the wavelength
    dependence is added. That dependence is not a detail: deep ice is clearest
    near 400 nm and opaque by 600 nm, so the red half of the nominal band is
    gone long before the reach is interesting, and the light that survives to
    large distance is a narrow window near the clarity peak.

    Parameters
    ----------
    site : Site
        Detector site.

    Returns
    -------
    length_m : np.ndarray
        Attenuation length [m] on :data:`WAVELENGTH_NM`, or a flat
        ``attenuation_override_m`` when the fit has set one.
    """
    if site.attenuation_override_m is not None:
        return np.full_like(WAVELENGTH_NM, float(site.attenuation_override_m))
    shape_abs = np.interp(WAVELENGTH_NM, np.asarray(site.absorption_nm),
                          np.asarray(site.absorption_shape))
    shape_scat = np.interp(WAVELENGTH_NM, np.asarray(site.scattering_nm),
                           np.asarray(site.scattering_shape))
    anchor_abs = np.interp(ANCHOR_NM, np.asarray(site.absorption_nm),
                           np.asarray(site.absorption_shape))
    anchor_scat = np.interp(ANCHOR_NM, np.asarray(site.scattering_nm),
                            np.asarray(site.scattering_shape))
    absorption = site.absorption_m * shape_abs / anchor_abs
    scattering = site.scattering_m * shape_scat / anchor_scat
    diffusive = np.sqrt(absorption * scattering / 3.0)
    return np.minimum(diffusive, absorption)


def attenuation_length_m(site: Site) -> float:
    """Attenuation length at the clarity peak [m], for reporting only.

    The model integrates :func:`attenuation_spectrum_m` and never uses a single
    number; this is the value at :data:`ANCHOR_NM`, which is what a data sheet
    quotes and what the fitted length is comparable against.
    """
    if site.attenuation_override_m is not None:
        return float(site.attenuation_override_m)
    return float(np.interp(ANCHOR_NM, WAVELENGTH_NM, attenuation_spectrum_m(site)))


def module_area_m2(site: Site) -> float:
    """Geometric photocathode area a module presents to an arriving photon [m^2].

    The photocathode area times :data:`PROJECTED_FRACTION`. Unlike the earlier
    form this carries **no** efficiency: the conversion probability is inside the
    wavelength integral, where it belongs, since it is the one thing in the chain
    that varies fastest across the band.
    """
    return site.cathode_area_m2 * PROJECTED_FRACTION


def brightness_factor(energy_gev: float | np.ndarray) -> np.ndarray:
    """Track brightness relative to a minimum-ionizing muon.

    ``1 + L_em b_mu E``: the bare track plus the electromagnetic showers of the
    radiative loss, which carry ``L_em`` metres of charged track per GeV. The
    shower light has the same Cherenkov spectrum as the bare track, so this
    factors out of every wavelength integral below.
    """
    energy = np.asarray(energy_gev, dtype=float)
    b_water = drift_coefficient(energy, RHO_WATER_G_CM3) / M_PER_KM
    return 1.0 + EM_TRACK_LENGTH_M_PER_GEV * b_water * energy


def module_charge_pe(
    distance_m: float | np.ndarray, energy_gev: float | np.ndarray, site: Site
) -> np.ndarray:
    """Photoelectrons a module collects from a track passing at a distance.

    Light leaves a long track cylindrically, so the fluence at perpendicular
    distance ``d`` is the yield per metre spread over ``2 pi d`` and attenuated
    on the medium's length. Summing over the band,

    .. math:: Q(d, E) = \\frac{A_{\\rm mod}}{2\\pi d}\\, Y(E) \\int {\\rm d}\\lambda\\;
        \\frac{{\\rm d}^2N_\\gamma}{{\\rm d}x\\,{\\rm d}\\lambda}\\,
        \\eta(\\lambda)\\, {\\rm e}^{-d / \\Lambda(\\lambda)},

    with ``Y`` the brightness of :func:`brightness_factor`. The wavelength
    integral is where the single-number treatment loses: ``eta`` and ``Lambda``
    peak in the same narrow window, and the exponential narrows it further with
    distance, so the *effective* band shrinks as the reach grows.

    Parameters
    ----------
    distance_m : float or np.ndarray
        Perpendicular distance from the track [m].
    energy_gev : float or np.ndarray
        Muon energy [GeV], broadcast against ``distance_m``.
    site : Site
        Detector site.

    Returns
    -------
    charge_pe : np.ndarray
        Collected charge [photoelectrons].
    """
    distance = np.atleast_1d(np.asarray(distance_m, dtype=float))
    weight = (cherenkov_spectrum_per_m_per_nm(site) * detection_efficiency(site))
    exponent = -distance[..., None] / attenuation_spectrum_m(site)
    collected = np.trapezoid(weight * np.exp(exponent), WAVELENGTH_NM, axis=-1)
    geometry = module_area_m2(site) / (2.0 * np.pi * np.maximum(distance, 1.0e-6))
    return geometry * collected * brightness_factor(energy_gev)


#: Distances the one-photoelectron radius is tabulated on [m], log spaced so the
#: inversion stays accurate over the four decades of brightness in play.
_HIT_DISTANCE_M = np.logspace(-1.0, 3.2, 400)


def hit_radius_m(energy_gev: float | np.ndarray, site: Site) -> np.ndarray:
    """Distance at which a module still collects one photoelectron [m].

    A reporting scale only: the counting that enters the model is Poisson over
    :func:`hit_probability`, with no step at any radius. This is the distance
    at which the *mean* charge falls to one photoelectron, which is what a
    single-number summary of the optics can be compared against.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Muon energy [GeV].
    site : Site
        Detector site.

    Returns
    -------
    radius_m : np.ndarray
        Radius of the one-photoelectron cylinder [m]. Zero where even a module
        on the track does not reach one photoelectron.
    """
    charge = module_charge_pe(_HIT_DISTANCE_M, np.atleast_1d(energy_gev)[..., None], site)
    # Q falls monotonically with distance, so the crossing is unique. Locate it
    # with one argmax per row and interpolate in log(charge) against
    # log(distance), where both are close to straight.
    log_d = np.log(_HIT_DISTANCE_M)
    with np.errstate(divide="ignore"):
        log_q = np.log(np.clip(charge, 1.0e-300, None))
    faint = charge < 1.0
    crossed = faint.any(axis=-1)
    hi = np.argmax(faint, axis=-1)
    lo = np.clip(hi - 1, 0, None)
    span = np.take_along_axis(log_q, lo[..., None], -1)[..., 0] - \
        np.take_along_axis(log_q, hi[..., None], -1)[..., 0]
    start = np.take_along_axis(log_q, lo[..., None], -1)[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(span > 0.0, start / span, 0.0)
    radius = np.exp(log_d[lo] + frac * (log_d[hi] - log_d[lo]))
    radius = np.where(hi == 0, 0.0, radius)
    return np.where(crossed, radius, _HIT_DISTANCE_M[-1])


def instrumented_chord_km(radius_km: float, height_km: float, n_sides: int | None) -> float:
    """Mean chord of the instrumented body [km].

    ``<c> = 4V / S`` for any convex body, so this needs no new number. It is the
    length of track the array has to work with, and therefore the length over
    which :func:`hit_count` counts modules.
    """
    volume = np.pi * radius_km**2 * height_km
    if n_sides is None:
        perimeter = 2.0 * np.pi * radius_km
    else:
        perimeter = 2.0 * radius_km * np.sqrt(np.pi * n_sides * np.tan(np.pi / n_sides))
    surface = 2.0 * np.pi * radius_km**2 + perimeter * height_km
    return float(4.0 * volume / surface)


def hit_probability(
    distance_m: float | np.ndarray, energy_gev: float | np.ndarray, site: Site
) -> np.ndarray:
    """Probability that a module at a distance registers a *hit*.

    A hit is a local coincidence, because that is what both instruments count:
    IceCube's simple-majority trigger counts HLC hits, and KM3NeT's L1 is two
    photomultipliers of one module within ~10 ns -- in sea water a single
    photoelectron is indistinguishable from potassium-40 decay. Each receiver
    converts its mean charge into at least one photoelectron with Poisson
    probability ``p = 1 - e^{-Q}``, and the coincidence admits **every
    partner** the definition allows:

    - A single-PMT module (IceCube) pairs with its nearest or next-to-nearest
      neighbours on the same string, up or down, so a firing module counts
      when *any* of :data:`HLC_PARTNERS` partners at essentially the same
      track distance also fires: ``p (1 - (1 - p)^4)``.
    - A multi-PMT module (KM3NeT) splits its collected charge over the
      ``illuminated_pmts`` that face the track, and counts when any two fire:
      ``1 - (1-p)^m - m p (1-p)^(m-1)`` with ``p = 1 - e^{-Q/m}``.

    Both reduce to the module firing outright when the track is bright. The
    partner sum matters in the dim limit, where a fixed-pair rule
    under-counts by the number of partners -- enough to push the IceCube
    turn-on from 1.8 TeV to 3.7 TeV and visibly suppress the effective area
    below 100 TeV. The step-function alternative -- every module inside the
    one-photoelectron radius fires, none outside -- is worse still, putting
    the threshold at 5.3 TeV where the array demonstrably triggers below
    1 TeV.

    Parameters
    ----------
    distance_m : float or np.ndarray
        Perpendicular distance from the track [m].
    energy_gev : float or np.ndarray
        Muon energy [GeV], broadcast against ``distance_m``.
    site : Site
        Detector site.

    Returns
    -------
    probability : np.ndarray
        Probability that the module registers a coincident hit.
    """
    charge = module_charge_pe(distance_m, energy_gev, site)
    if site.illuminated_pmts is None:
        single = 1.0 - np.exp(-charge)
        return single * (1.0 - (1.0 - single) ** HLC_PARTNERS)
    m = float(site.illuminated_pmts)
    single = 1.0 - np.exp(-charge / m)
    return 1.0 - (1.0 - single) ** m - m * single * (1.0 - single) ** (m - 1.0)


def hit_count(
    offset_m: float, energy_gev: float | np.ndarray, site: Site, chord_km: float,
) -> np.ndarray:
    """Mean number of hit modules, for a track at a signed distance from the boundary.

    Each module hits with :func:`hit_probability`, so the mean count is that
    probability integrated over the in-array part of the transverse plane. A
    circle of radius ``d`` around a track at signed offset ``x`` from the
    boundary keeps the fraction ``arccos(x / d) / pi`` of its circumference
    inside, hence

    .. math:: \\bar N(x, E) = \\rho_{\\rm mod}\\,\\langle c\\rangle \\int
        2\\,d\\,\\arccos\\!\\left({\\rm clip}(x / d)\\right) p_{\\rm hit}(d, E)\\,
        {\\rm d}d,

    with the mean chord as the track length in view. For a step ``p_hit`` this
    is the circular-segment count the earlier disc treatment used; the Poisson
    form differs where it matters, in the dim limit, where the count becomes
    linear in the collected charge.

    Parameters
    ----------
    offset_m : float
        Signed distance of the track from the boundary [m]; positive is outside.
    energy_gev : float or np.ndarray
        Muon energy [GeV].
    site : Site
        Detector site.
    chord_km : float
        Mean chord of the instrumented body [km], from
        :func:`instrumented_chord_km`.

    Returns
    -------
    count : np.ndarray
        Mean number of modules registering a coincident hit.
    """
    probability = hit_probability(
        _HIT_DISTANCE_M, np.atleast_1d(np.asarray(energy_gev, dtype=float))[..., None],
        site)
    wedge = 2.0 * _HIT_DISTANCE_M * np.arccos(
        np.clip(float(offset_m) / _HIT_DISTANCE_M, -1.0, 1.0))
    density_per_m3 = site.module_density_per_km3 / M_PER_KM**3
    count = (probability * _D_TRAPZ) @ wedge
    return density_per_m3 * count * chord_km * M_PER_KM


#: Signed offsets the mean hit count is tabulated on [m], for the inversion in
#: :func:`reach_offset_m`. The positive end comfortably exceeds any reach in
#: play; the negative end only has to cover the interpolation edge, since a
#: track the condition wants *inside* the array is handled by the multiplicity
#: weight of :func:`effective_body_km` and the offset is clipped at zero.
_REACH_OFFSET_M = np.linspace(-100.0, 1200.0, 261)

#: The wedge kernel of :func:`hit_count` on that offset grid,
#: ``2 d arccos(clip(x / d))``, tabulated once: the mean count at every offset
#: is then one matrix product with the hit probabilities.
_REACH_KERNEL = 2.0 * _HIT_DISTANCE_M[:, None] * np.arccos(
    np.clip(_REACH_OFFSET_M[None, :] / _HIT_DISTANCE_M[:, None], -1.0, 1.0))

#: Trapezoid quadrature weights of the distance grid, so the count integrals
#: reduce to matrix products against :data:`_REACH_KERNEL`.
_D_STEP = np.diff(_HIT_DISTANCE_M)
_D_TRAPZ = np.concatenate(
    [[0.5 * _D_STEP[0]], 0.5 * (_D_STEP[:-1] + _D_STEP[1:]), [0.5 * _D_STEP[-1]]])


def _mean_counts(
    energy_gev: np.ndarray, site: Site, chord_km: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mean hit counts on :data:`_REACH_OFFSET_M`, and for a central track.

    One evaluation of the optics serves both: the reach inversion needs the
    count against the offset, the multiplicity weight needs the count deep
    inside the array, where the wedge is the full circle.

    Parameters
    ----------
    energy_gev : np.ndarray
        Muon energy [GeV].
    site : Site
        Detector site.
    chord_km : float
        Mean chord of the instrumented body [km].

    Returns
    -------
    counts : np.ndarray, shape (energy, offset)
        Mean hit count at each tabulated offset.
    central : np.ndarray, shape (energy,)
        Mean hit count for a central crossing track.
    """
    probability = hit_probability(_HIT_DISTANCE_M, energy_gev[..., None], site)
    scale = site.module_density_per_km3 / M_PER_KM**3 * chord_km * M_PER_KM
    weighted = probability * _D_TRAPZ
    return (scale * (weighted @ _REACH_KERNEL),
            scale * (weighted @ (2.0 * np.pi * _HIT_DISTANCE_M)))


def _invert_reach_m(counts: np.ndarray, min_modules: float) -> np.ndarray:
    """Offset at which each row of ``counts`` falls to ``min_modules`` [m]."""
    return np.array([
        np.interp(min_modules, row[::-1], _REACH_OFFSET_M[::-1],
                  left=_REACH_OFFSET_M[-1], right=_REACH_OFFSET_M[0])
        for row in counts
    ])


def reach_offset_m(
    energy_gev: np.ndarray, site: Site, chord_km: float, min_modules: float
) -> np.ndarray:
    """How far outside the boundary a track can be and still make ``min_modules``.

    Inverts the mean count of :func:`hit_count` in the signed offset. The count
    falls monotonically with the offset, so the crossing is unique and one
    tabulated count against offset serves every energy at once.

    Parameters
    ----------
    energy_gev : np.ndarray
        Muon energy [GeV].
    site : Site
        Detector site.
    chord_km : float
        Mean chord of the instrumented body [km].
    min_modules : float
        Mean number of hits demanded.

    Returns
    -------
    offset_m : np.ndarray
        Signed distance [m], held at the edges of the tabulated offsets. The
        negative edge does not need to be deep: below it the acceptance is
        carried by the multiplicity weight of :func:`effective_body_km`.
    """
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    counts, _ = _mean_counts(energy, site, chord_km)
    return _invert_reach_m(counts, min_modules)


def muon_threshold_gev(site: Site, min_modules: float, chord_km: float) -> float:
    """Muon energy at which a track through the array first fires ``min_modules``.

    Deep inside the array the segment is the whole disc, so the condition is
    ``rho_mod pi d_1^2 <c> = N_min``: the hit radius has to reach a definite
    value, and that fixes an energy. This is the threshold the *light* sets,
    and with Poisson-thinned hits it is the midpoint of a turn-on: the
    multiplicity weight of :func:`effective_body_km` passes ~55% of central
    tracks here, more above, fewer below. The range still runs to the nominal
    threshold; the weight carries the dimming.

    Parameters
    ----------
    site : Site
        Detector site.
    min_modules : float
        Modules that must fire.
    chord_km : float
        Mean chord of the instrumented body [km].

    Returns
    -------
    threshold_gev : float
        Muon threshold [GeV], held at or above
        :data:`DEFAULT_MUON_THRESHOLD_GEV`.
    """
    def gap(log10_e: float) -> float:
        deep = -1.0e4
        count = np.atleast_1d(hit_count(deep, 10.0**log10_e, site, chord_km))
        return float(count[0]) - min_modules

    if gap(12.0) < 0.0:
        return float("nan")
    if gap(0.0) > 0.0:
        return DEFAULT_MUON_THRESHOLD_GEV
    return max(10.0 ** brentq(gap, 0.0, 12.0, xtol=1.0e-4), DEFAULT_MUON_THRESHOLD_GEV)


def effective_body_km(
    radius_km: float, height_km: float, energy_gev: float | np.ndarray, site: Site,
    min_modules: float, n_sides: int | None = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Instrumented body dilated by the derived light reach, and its weight [km].

    The reach is where a track still makes ``min_modules`` mean hits
    (:func:`reach_offset_m`). It is a distance from a *boundary*, so the same
    reach moves the body in every direction and not only radially. Vertically
    it is capped by the medium: both sites are layered, with an upper boundary at
    the ice or sea surface and a lower one at bedrock or seabed, and a track
    outside the optical medium is neither radiating into it nor visible through
    it. The horizontal directions carry no such cap.

    Dim muons are carried by the third return, a multiplicity weight: the hits
    are Poisson-thinned, so a track whose *mean* count sits below
    ``min_modules`` still meets the selection with probability
    ``P(N >= min_modules)``, evaluated for a central crossing. The weight is
    what makes the threshold a smooth turn-on -- an earlier revision emptied
    the body below the mean-count threshold instead, and that delta-function
    condition collapsed the model at low energy where the published response
    falls smoothly. The weight applies the dimming exactly once: the range
    keeps its nominal lower limit (see :func:`build_model`), and the offset is
    clipped at zero because sub-threshold acceptance belongs to the weight.

    Parameters
    ----------
    radius_km : float
        Instrumented footprint radius [km].
    height_km : float
        Instrumented height [km].
    energy_gev : float or np.ndarray
        Muon energy where the track is seen [GeV].
    site : Site
        Detector site.
    min_modules : float
        Mean number of hits demanded.
    n_sides : int or None, optional
        Cross-section of the instrumented body, for its mean chord.

    Returns
    -------
    radius : np.ndarray
        Effective radius [km].
    height : np.ndarray
        Effective height [km], grown at each end cap and held inside the medium.
    weight : np.ndarray
        Probability that a central crossing track meets the multiplicity.
    """
    chord_km = instrumented_chord_km(radius_km, height_km, n_sides)
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    counts, central = _mean_counts(energy, site, chord_km)
    offset_km = np.clip(_invert_reach_m(counts, min_modules) / M_PER_KM, 0.0, None)
    above_km = np.minimum(offset_km, site.headroom_above_m / M_PER_KM)
    below_km = np.minimum(offset_km, site.headroom_below_m / M_PER_KM)
    # P(N >= k) for a Poisson mean is the regularized lower incomplete gamma.
    weight = gammainc(min_modules, central)
    return (np.clip(radius_km + offset_km, 0.0, None),
            np.clip(height_km + above_km + below_km, 0.0, None),
            weight)


# ---------------------------------------------------------------------------
# The column integral of example 44, with a derived radius
# ---------------------------------------------------------------------------


def column_profile(
    production_gev: float, threshold_gev: float, n_energy: int,
    density_g_cm3: float = RHO_WATER_G_CM3,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Column travelled against the energy the muon has there.

    The column a muon has covered by the time it has fallen to ``E`` is
    ``L(eps -> E_thr) - L(E -> E_thr)``, exact for the mean first-passage depth
    by the tower property, so the validated range function is called in its
    normal convention throughout.

    Parameters
    ----------
    production_gev : float
        Muon energy at production [GeV].
    threshold_gev : float
        Muon selection threshold [GeV].
    n_energy : int
        Points in the quadrature.
    density_g_cm3 : float, optional
        Density of the detector medium [g cm^-3].

    Returns
    -------
    profile : tuple or None
        ``(column [km], arrival energy [GeV], total range [km])``, or ``None``
        when the muon is born below threshold.
    """
    if production_gev <= threshold_gev:
        return None
    total = float(np.atleast_1d(
        stochastic_muon_range_km(production_gev, threshold_gev, density_g_cm3))[0])
    if not np.isfinite(total) or total <= 0.0:
        return None
    energy = np.logspace(np.log10(production_gev), np.log10(threshold_gev), n_energy)
    column = total - stochastic_muon_range_km(energy, threshold_gev, density_g_cm3)
    return column, energy, total


# ---------------------------------------------------------------------------
# IceCube
# ---------------------------------------------------------------------------


def ic_column_volume_km3(
    ex32, production_gev: float, threshold_gev: float, cos_theta: np.ndarray,
    site: Site, min_modules: float | None, n_energy: int,
) -> np.ndarray:
    """Target volume for one production energy [km^3].

    The entering term is the silhouette extruded upstream over the column the
    muon can cover, and the instrumented volume is the same extrusion continued
    to the back face, so the two add to the volume of the body extruded by
    ``L``. Both are carried here, since a light reach dilates the body and so
    moves both of them; returning only the column term and adding a fixed
    ``V_det`` outside would grow one and hold the other. A muon born below
    threshold gets neither.

    Parameters
    ----------
    ex32 : ModuleType
        Example 32, for the instrument constants.
    production_gev : float
        Muon energy at production [GeV].
    threshold_gev : float
        Muon selection threshold [GeV].
    cos_theta : np.ndarray
        Arrival directions.
    site : Site
        Detector site.
    min_modules : float or None
        Modules that must fire. ``None`` holds the
        body at the instrumented one.
    n_energy : int
        Points in the arrival-energy quadrature.

    Returns
    -------
    volume : np.ndarray
        Target volume [km^3], one entry per direction.
    """
    zeros = np.zeros_like(np.asarray(cos_theta, dtype=float))
    profile = column_profile(production_gev, threshold_gev, n_energy, RHO_ICE_G_CM3)
    if profile is None:
        return zeros
    column, energy, total = profile

    if min_modules is None:
        area, volume = eroded_prism_target_km2(
            cos_theta, ex32.IC_RADIUS_KM, ex32.IC_HEIGHT_KM, site.min_track_km,
            ex32.IC_N_SIDES)
        return area * total + volume

    radius, height, weight = effective_body_km(
        ex32.IC_RADIUS_KM, ex32.IC_HEIGHT_KM, energy, site, min_modules,
        ex32.IC_N_SIDES)
    area, volume = eroded_prism_target_km2(
        np.asarray(cos_theta, dtype=float)[None, :], radius[:, None],
        height[:, None], site.min_track_km, ex32.IC_N_SIDES,
    )
    # The instrumented term belongs to a vertex inside the body, which the muon
    # leaves at essentially its production energy, so it is taken at ``energy[0]``.
    return (np.trapezoid(weight[:, None] * area, column, axis=0)
            + weight[0] * volume[0])


def ic_effective_area_cm2(
    ex32, site: Site, threshold_gev: float, min_modules: float | None, n_energy: int,
    flavours: tuple[str, ...] = DEFAULT_FLAVOURS, cross_section=None,
) -> np.ndarray:
    """Upgoing-averaged IceCube effective area, both channels [cm^2].

    Parameters
    ----------
    ex32 : ModuleType
        Example 32.
    site : Site
        Detector site.
    threshold_gev : float
        Muon selection threshold [GeV].
    min_modules : float or None
        Modules that must fire, or ``None`` for the instrumented footprint.
    n_energy : int
        Points in the arrival-energy quadrature.
    flavours : tuple of str, optional
        Parent channels to sum. Defaults to :data:`DEFAULT_FLAVOURS`.
    cross_section : softpaws.transport.cross_section.CrossSection, optional
        Cross section of the incident species, used for both the interaction and
        the Earth absorption. Defaults to example 32's neutrino model.

    Returns
    -------
    aeff : np.ndarray
        Effective area [cm^2] on ``ex32.IC_LOG10_E``.
    """
    xsec = ex32.CROSS_SECTION if cross_section is None else cross_section
    columns, weights, cos_theta = ex32.ic_upgoing_columns()
    # South Pole ice, not the module-level water default. `n L` is exactly
    # density-invariant so the entering term does not care, but `V_det` is a
    # geometric volume and scales with it.
    n_nucleon = nucleon_number_density(RHO_ICE_G_CM3)
    out = np.zeros(ex32.IC_LOG10_E.size)

    for i, e_nu in enumerate(10.0**ex32.IC_LOG10_E):
        for flavour in flavours:
            if flavour == "mu":
                rungs, rung_weight = ex32.regenerated_transmission(
                    float(e_nu), columns, xsec)
                branching = 1.0
                muon_gev = (1.0 - mean_inelasticity(rungs)) * rungs
            else:
                rungs, rung_weight = ex32.flavour_transmission(
                    float(e_nu), columns, xsec, flavour="tau")
                branching = BR_TAU_TO_MU
                muon_gev = MEAN_Z * (1.0 - mean_inelasticity(rungs)) * rungs

            rate = np.zeros((rungs.size, cos_theta.size))
            for k, e_mu in enumerate(muon_gev):
                volume = ic_column_volume_km3(
                    ex32, float(e_mu), threshold_gev, cos_theta, site,
                    min_modules, n_energy,
                )
                rate[k] = volume * CM_PER_KM**3
            rate *= n_nucleon * xsec.cc(rungs)[:, None] * branching
            out[i] += np.average((rung_weight * rate).sum(axis=0), weights=weights)
    return out


# ---------------------------------------------------------------------------
# KM3NeT/ARCA230
# ---------------------------------------------------------------------------


def arca_column_volume_km3(
    ex32, production_gev: float, threshold_gev: float, theta_deg: np.ndarray,
    available_km: np.ndarray, site: Site, min_modules: float | None, n_energy: int,
) -> np.ndarray:
    """Column target volume at ARCA230, truncated at the available column [km^3].

    The site supplies a finite upstream column, so the length is the truncated
    first-passage range ``E[tau ^ X]``. The reach enters as the mean projected
    area over the part of the column the muon can actually have travelled, which
    keeps the validated range intact and reduces to example 44's integral where
    the column is unlimited. As at IceCube, the instrumented volume is carried
    here rather than added outside, so that the reach dilates both terms of the
    can and a sub-threshold muon contributes neither.

    Parameters
    ----------
    ex32 : ModuleType
        Example 32, for the ARCA geometry.
    production_gev : float
        Muon energy at production [GeV].
    threshold_gev : float
        Muon selection threshold [GeV].
    theta_deg : np.ndarray
        Zenith samples [deg].
    available_km : np.ndarray
        Upstream sea-water column available in each direction [km].
    site : Site
        Detector site.
    min_modules : float or None
        Modules that must fire, or ``None`` for the instrumented footprint.
    n_energy : int
        Points in the arrival-energy quadrature.

    Returns
    -------
    volume : np.ndarray
        Column volume [km^3], one entry per zenith.
    """
    radius_km = ex32.ARCA230_RADIUS_KM
    height_km = ex32.BLOCK_HEIGHT_KM
    n_blocks = ex32.N_BLOCKS_FULL
    cos_theta = np.cos(np.deg2rad(np.asarray(theta_deg, dtype=float)))
    if production_gev <= threshold_gev:
        return np.zeros_like(theta_deg)

    truncated = np.atleast_1d(truncated_muon_range_km(
        production_gev, available_km, threshold_gev, kernel_evaluation="running"))
    truncated = np.clip(truncated, 0.0, None)

    if min_modules is None:
        area, volume = eroded_prism_target_km2(
            cos_theta, radius_km, height_km, site.min_track_km, None, n_blocks)
        return area * truncated + volume

    profile = column_profile(production_gev, threshold_gev, n_energy)
    if profile is None:
        return np.zeros_like(theta_deg)
    column, energy, _ = profile

    radius, height, weight = effective_body_km(radius_km, height_km, energy, site,
                                               min_modules, None)
    area, volume = eroded_prism_target_km2(
        cos_theta[None, :], radius[:, None], height[:, None], site.min_track_km,
        None, n_blocks)
    area = weight[:, None] * area
    clipped = np.minimum(column[:, None], available_km[None, :])
    span = clipped[-1]
    mean_area = np.where(
        span > 0.0, np.trapezoid(area, clipped, axis=0) / np.where(span > 0.0, span, 1.0),
        weight[0] * eroded_prism_target_km2(cos_theta, radius_km, height_km,
                                            site.min_track_km, None, n_blocks)[0],
    )
    return mean_area * truncated + weight[0] * volume[0]


def arca_effective_area_cm2(
    ex32, site: Site, threshold_gev: float, min_modules: float | None, n_energy: int,
    flavours: tuple[str, ...] = DEFAULT_FLAVOURS, cross_section=None,
) -> np.ndarray:
    """Sky-averaged ARCA230 effective area, both channels [cm^2].

    Parameters
    ----------
    ex32 : ModuleType
        Example 32.
    site : Site
        Detector site.
    threshold_gev : float
        Muon selection threshold [GeV].
    min_modules : float or None
        Modules that must fire, or ``None`` for the instrumented footprint.
    n_energy : int
        Points in the arrival-energy quadrature.
    flavours : tuple of str, optional
        Parent channels to sum. Defaults to :data:`DEFAULT_FLAVOURS`.
    cross_section : softpaws.transport.cross_section.CrossSection, optional
        Cross section of the incident species, used for both the interaction and
        the Earth absorption. Defaults to example 32's neutrino model.

    Returns
    -------
    aeff : np.ndarray
        Effective area [cm^2] on ``ex32.ARCA_LOG10_E``.
    """
    xsec = ex32.CROSS_SECTION if cross_section is None else cross_section
    theta_deg, weights = ex32.zenith_grid()
    columns = ex32.earth_column_g_cm2(theta_deg, ex32.ARCA_DEPTH_KM)
    available_km = ex32.upstream_column_km(theta_deg, ex32.ARCA_DEPTH_KM)
    n_nucleon = nucleon_number_density(ex32.RHO_SEA_G_CM3)
    out = np.zeros(ex32.ARCA_LOG10_E.size)

    for i, e_nu in enumerate(10.0**ex32.ARCA_LOG10_E):
        for flavour in flavours:
            if flavour == "mu":
                rungs, rung_weight = ex32.regenerated_transmission(
                    float(e_nu), columns, xsec)
                branching = 1.0
                muon_gev = (1.0 - mean_inelasticity(rungs)) * rungs
            else:
                rungs, rung_weight = ex32.flavour_transmission(
                    float(e_nu), columns, xsec, flavour="tau")
                branching = BR_TAU_TO_MU
                muon_gev = MEAN_Z * (1.0 - mean_inelasticity(rungs)) * rungs

            rate = np.zeros((rungs.size, theta_deg.size))
            for k, e_mu in enumerate(muon_gev):
                volume = arca_column_volume_km3(
                    ex32, float(e_mu), threshold_gev, theta_deg,
                    available_km, site, min_modules, n_energy,
                )
                rate[k] = volume * CM_PER_KM**3
            rate *= n_nucleon * xsec.cc(rungs)[:, None] * branching
            out[i] += np.average((rung_weight * rate).sum(axis=0), weights=weights)
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_model(ex32, which: str, site: Site, min_modules: float, n_energy: int,
                flavours: tuple[str, ...] = DEFAULT_FLAVOURS) -> np.ndarray:
    """The full model for one detector, threshold and reach both derived [cm^2].

    Averaged over neutrino and antineutrino, each propagated through the Earth
    with its own cross section, since both published tables are that average.

    The light condition enters once. The range runs to the nominal threshold,
    and every track segment carries the probability that its Poisson-thinned
    hits meet the multiplicity -- the weight of :func:`effective_body_km` -- so
    the threshold is a smooth turn-on and not a cut. Truncating the range at
    :func:`muon_threshold_gev` as well would suppress the same physics twice.

    Parameters
    ----------
    ex32 : ModuleType
        Example 32.
    which : {"IceCube", "ARCA"}
        Which detector to build.
    site : Site
        Detector site.
    min_modules : float
        Modules that must fire.
    n_energy : int
        Points in the arrival-energy quadrature.

    Returns
    -------
    aeff : np.ndarray
        Effective area [cm^2] on that detector's grid.
    """
    build = ic_effective_area_cm2 if which == "IceCube" else arca_effective_area_cm2
    curves = [build(ex32, site, DEFAULT_MUON_THRESHOLD_GEV, min_modules, n_energy,
                    flavours, xsec)
              for xsec in SPECIES]
    return np.mean(curves, axis=0)


def fit_reach(ex32, which, site, min_modules, n_energy, published, band, flavours):
    """Float the attenuation length against one published curve.

    ``Lambda`` sets how fast the footprint grows per e-fold, which is shape
    alone, so it is fitted against the scatter about a free normalization. That
    is the convention example 32 already reports its own residuals in. Fitting
    it against the level instead drives it to the bound, since no attenuation
    length can supply a normalization; the level belongs to ``q`` and to
    whatever else sits in the 1.78, and forcing ``Lambda`` to carry it returns a
    number no medium has.

    The returned curve carries that free normalization, so it lands on the
    published one by construction and only its shape is a statement.

    Parameters
    ----------
    ex32 : ModuleType
        Example 32.
    which : {"IceCube", "ARCA"}
        Which detector to build.
    site : Site
        Detector site, whose derived attenuation length starts the search.
    min_modules : float
        Modules that must fire, held at the derived value.
    n_energy : int
        Points in the arrival-energy quadrature.
    published : np.ndarray
        Published effective area [cm^2].
    band : np.ndarray
        Boolean mask of the energies the fit runs over.

    Returns
    -------
    length_m : float
        Fitted attenuation length [m], to compare against the medium's optics.
    normalization : float
        Factor the fitted shape needs to reach the published level.
    curve : np.ndarray
        Normalized effective area [cm^2].
    """
    def model_at(log10_lambda: float) -> np.ndarray:
        trial = replace(site, attenuation_override_m=10.0**log10_lambda)
        return build_model(ex32, which, trial, min_modules, n_energy, flavours)

    def shape_cost(log10_lambda: float) -> float:
        return residuals(published, model_at(log10_lambda), band)[1]

    opt = minimize_scalar(shape_cost, bounds=(1.0, 2.5), method="bounded",
                          options={"xatol": 0.005})
    curve = model_at(opt.x)
    normalization = residuals(published, curve, band)[0]
    return 10.0**opt.x, normalization, curve * normalization


def detector_curves(ex32, which: str, site: Site, min_modules: float, n_energy: int,
                    published: np.ndarray, band: np.ndarray,
                    flavours: tuple[str, ...] = DEFAULT_FLAVOURS) -> tuple[dict, float]:
    """Published, first-principles and fitted effective areas for one detector [cm^2].

    Parameters
    ----------
    ex32 : ModuleType
        Example 32.
    which : {"IceCube", "ARCA"}
        Which detector to build.
    site : Site
        Detector site.
    min_modules : float
        Modules that must fire.
    n_energy : int
        Points in the arrival-energy quadrature.
    published : np.ndarray
        Published effective area [cm^2] on the detector's grid.
    band : np.ndarray
        Boolean mask of the energies the fit runs over.

    Returns
    -------
    curves : dict of str -> np.ndarray
        Keyed as in :data:`MODEL_COLOR`.
    fit : tuple of float
        ``(attenuation length [m], normalization)`` from the fit.
    """
    first_principles = build_model(ex32, which, site, min_modules, n_energy, flavours)
    length_m, normalization, fitted = fit_reach(
        ex32, which, site, min_modules, n_energy, published, band, flavours)
    return {
        "Published": published,
        "First principles": first_principles,
        "Fitted": fitted,
    }, (length_m, normalization)


def report_site(site: Site, min_modules: float, radius_km: float, height_km: float,
                n_sides: int | None) -> None:
    """Print the derived light-reach numbers for one site."""
    lam = attenuation_length_m(site)
    chord_km = instrumented_chord_km(radius_km, height_km, n_sides)
    threshold = muon_threshold_gev(site, min_modules, chord_km)
    spectrum = cherenkov_spectrum_per_m_per_nm(site)
    efficiency = detection_efficiency(site)
    bare = float(np.trapezoid(spectrum, WAVELENGTH_NM))
    detectable = float(np.trapezoid(spectrum * efficiency, WAVELENGTH_NM))
    print(f"\n  {site.name}")
    print(f"    attenuation length at {ANCHOR_NM:.0f} nm  {lam:6.1f} m "
          f"(band {attenuation_spectrum_m(site).min():.0f} to "
          f"{attenuation_spectrum_m(site).max():.0f})")
    print(f"    photocathode area           {module_area_m2(site) * 1e4:6.1f} cm^2 projected")
    print(f"    bare-track yield            {bare:6.2e} photons/m")
    print(f"    detectable yield            {detectable:6.2e} pe/m "
          f"(mean efficiency {detectable / bare:.3f} against a flat "
          f"{site.quantum_efficiency:.2f})")
    print(f"    module density              {site.module_density_per_km3:6.0f} per km^3")
    print(f"    instrumented mean chord     {chord_km * M_PER_KM:6.0f} m")
    print(f"    one-pe radius at 1 PeV      {float(hit_radius_m(1.0e6, site)[0]):6.0f} m")
    print(f"    reach at 1 PeV              "
          f"{float(reach_offset_m(np.array([1.0e6]), site, chord_km, min_modules)[0]):6.0f} m")
    print(f"    muon threshold              {threshold:10,.0f} GeV")
    # Where the vertical reach stops growing, which is the medium and not the light.
    for label, headroom in (("above", site.headroom_above_m),
                            ("below", site.headroom_below_m)):
        print(f"    headroom {label}              {headroom:6.0f} m "
              f"({headroom / lam:.1f} attenuation lengths)")
    note = "" if site.min_track_km > 0.0 else "  (trigger level; none required)"
    print(f"    minimum in-detector track   {site.min_track_km * M_PER_KM:6.0f} m{note}")


def report_anchors(ex32, detector: str, curves: dict[str, np.ndarray],
                   band: np.ndarray) -> None:
    """Print ``published / model`` at :data:`ANCHORS`, where a flat ratio is legible.

    Parameters
    ----------
    ex32 : ModuleType
        Example 32, for the energy grids.
    detector : str
        Detector name, selecting which grid the curves sit on.
    curves : dict of str -> np.ndarray
        Curves keyed as in :data:`MODEL_COLOR`.
    band : np.ndarray
        Boolean mask of the comparison band. The ceiling test is restricted to
        it, since below threshold the model has no acceptance and the ratio
        there says nothing about a geometric ceiling.
    """
    log10_e = ex32.IC_LOG10_E if detector == "IceCube" else ex32.ARCA_LOG10_E
    for target in ANCHORS:
        i = int(np.argmin(np.abs(log10_e - target)))
        row = "  ".join(
            f"{model} {curves['Published'][i] / curves[model][i]:.3f}" for model in MODELS
        )
        print(f"      log10(E/GeV) = {log10_e[i]:.1f}   {row}")
    with np.errstate(divide="ignore"):
        ratio = curves["Published"] / curves["First principles"]
    valid = band & np.isfinite(ratio)
    j = int(np.nanargmax(np.where(valid, ratio, -np.inf)))
    verdict = "held" if ratio[j] <= 1.0 else "VIOLATED"
    print(f"      ceiling {verdict}: max published/model = {ratio[j]:.3f} "
          f"at log10(E/GeV) = {log10_e[j]:.1f}")


def residuals(published: np.ndarray, model: np.ndarray, band: np.ndarray) -> tuple[float, float]:
    """Geometric-mean ratio and shape scatter of ``published / model`` over a band."""
    valid = band & np.isfinite(published) & np.isfinite(model) & (model > 0.0)
    res = np.log10(published[valid] / model[valid])
    return float(10**np.mean(res)), float(np.std(res))


def scan_charge(ex32, which, site, published, band, multiplicities, n_energy, flavours,
                radius_km, height_km, n_sides):
    """Shape scatter left by the reach model against the required multiplicity.

    Returns
    -------
    table : np.ndarray
        Columns ``(modules, threshold [GeV], mean ratio, scatter [dex])``.
    """
    chord_km = instrumented_chord_km(radius_km, height_km, n_sides)
    rows = []
    for modules in multiplicities:
        model = build_model(ex32, which, site, float(modules), n_energy, flavours)
        mean, scatter = residuals(published, model, band)
        threshold = muon_threshold_gev(site, float(modules), chord_km)
        rows.append((modules, threshold, mean, scatter))
        print(f"    N = {modules:5.1f} modules -> E_thr {threshold:9,.0f} GeV, "
              f"mean {mean:.3f}, {scatter:.4f} dex")
    return np.array(rows)


def main() -> None:
    args = parse_args()
    print("Loading example 32 ...")
    ex32 = load_example_32()

    flavours = ("mu",) if args.numu_only else DEFAULT_FLAVOURS
    icecube_site, arca_site = ICECUBE_SITE, ARCA_SITE
    if args.min_track_km is not None:
        icecube_site = replace(icecube_site, min_track_km=args.min_track_km)
        arca_site = replace(arca_site, min_track_km=args.min_track_km)
    print(f"\nChannels summed: {', '.join('nu_' + f for f in flavours)}")
    print("\nDerived from the medium and the module:")
    report_site(icecube_site, args.min_modules, ex32.IC_RADIUS_KM, ex32.IC_HEIGHT_KM,
                ex32.IC_N_SIDES)
    report_site(arca_site, args.min_modules, ex32.ARCA230_RADIUS_KM,
                ex32.BLOCK_HEIGHT_KM, None)

    ic_published = ex32.icecube_published(args.data_dir)
    arca_published = ex32.arca230_published(ex32.ARCA_LOG10_E)
    ic_band = (ex32.IC_LOG10_E >= IC_BAND[0]) & (ex32.IC_LOG10_E <= IC_BAND[1])
    arca_band = (ex32.ARCA_LOG10_E >= ARCA_BAND[0]) & (ex32.ARCA_LOG10_E <= ARCA_BAND[1])

    print("\nBuilding IceCube ...")
    ic_curves, ic_fit = detector_curves(ex32, "IceCube", icecube_site, args.min_modules,
                                        args.n_energy, ic_published, ic_band, flavours)
    print("Building ARCA230 ...")
    arca_curves, arca_fit = detector_curves(ex32, "ARCA", arca_site, args.min_modules,
                                            args.n_energy, arca_published, arca_band,
                                            flavours)

    print(f"\n  published over model, at N = {args.min_modules:g} modules")
    for name, curves, band in (("IceCube", ic_curves, ic_band),
                               ("KM3NeT/ARCA230", arca_curves, arca_band)):
        print(f"    {name}")
        for model in MODELS:
            mean, scatter = residuals(curves["Published"], curves[model], band)
            print(f"      {model:17s}: mean {mean:.3f}, {scatter:.4f} dex of shape")
        report_anchors(ex32, name, curves, band)

    scan = {}
    if args.scan:
        charges = np.array([2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0])
        print("\nScanning the required multiplicity, which shifts the level and moves")
        print("the threshold with it, so only the shape scatter is a test:")
        for name, which, site, published, band, geometry in (
            ("IceCube", "IceCube", icecube_site, ic_published, ic_band,
             (ex32.IC_RADIUS_KM, ex32.IC_HEIGHT_KM, ex32.IC_N_SIDES)),
            ("KM3NeT/ARCA230", "ARCA", arca_site, arca_published, arca_band,
             (ex32.ARCA230_RADIUS_KM, ex32.BLOCK_HEIGHT_KM, None)),
        ):
            print(f"  {name}")
            scan[name] = scan_charge(ex32, which, site, published, band, charges,
                                     args.n_energy, flavours, *geometry)

    summarize(ic_curves, ic_band, ic_fit, arca_curves, arca_band, arca_fit,
              icecube_site, arca_site)
    detectors = {"IceCube": (ex32.IC_LOG10_E, ic_curves, ic_band),
                 "KM3NeT/ARCA230": (ex32.ARCA_LOG10_E, arca_curves, arca_band)}
    fit_notes = {
        "IceCube": (attenuation_length_m(icecube_site), *ic_fit),
        "KM3NeT/ARCA230": (attenuation_length_m(arca_site), *arca_fit),
    }
    make_area_figure(detectors, args.out.with_name(args.out.name + "_area"), fit_notes)
    make_ratio_figure(detectors, args.out.with_name(args.out.name + "_ratio"), fit_notes)


def summarize(ic_curves, ic_band, ic_fit, arca_curves, arca_band, arca_fit,
              icecube_site=ICECUBE_SITE, arca_site=ARCA_SITE) -> None:
    """State where the first-principles curves land, and what the fit costs.

    The fitted attenuation length is deliberately *not* reported against the
    derived one. ``attenuation_override_m`` replaces the whole wavelength
    spectrum by a single flat value, while the derived number is the length at
    :data:`ANCHOR_NM`, the clarity peak. Those are different quantities -- the
    derived spectrum spans 7 to 61 m at IceCube -- so their ratio measures the
    substitution and not the medium, and quoting it invited a comparison the
    model cannot support.
    """
    print("\n  Free normalization the fitted curve needs:")
    for site, (_, norm) in ((icecube_site, ic_fit), (arca_site, arca_fit)):
        print(f"    {site.name:16s} {norm:.3f}")

    ic_mean, ic_shape = residuals(ic_curves["Published"], ic_curves["First principles"], ic_band)
    arca_mean, arca_shape = residuals(
        arca_curves["Published"], arca_curves["First principles"], arca_band)
    print(f"\n  Level: {ic_mean:.3f} at IceCube, {arca_mean:.3f} at ARCA230, with no number")
    print("  in the optical chain set against either published curve. A hit is a")
    print("  local coincidence and the multiplicity is met at Poisson probability,")
    print("  so the threshold is a smooth turn-on, and the cross sections are the")
    print("  shipped nu-p tables pinned to the CSMS isoscalar values. Against the")
    print("  DR2 table use --numu-only; that release is built from simulated")
    print("  nu_mu events.")
    print(f"  Shape left over: {ic_shape:.4f} dex at IceCube, {arca_shape:.4f} dex at ARCA230.")


def _save(fig, out_path: pathlib.Path) -> None:
    """Write both the vector and the raster copy of one figure."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_path.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def _annotate_fit(ax, fit_notes: dict[str, tuple[float, float, float]] | None) -> None:
    """Write what the ``Fitted`` curve floated, in the top-right corner.

    One line per detector: the attenuation length from the derived value at
    :data:`ANCHOR_NM` to the fitted flat override, and the free normalization.
    The two lengths are different quantities -- a spectrum's anchor against a
    flat substitute, see :func:`summarize` -- so the arrow states what the fit
    did, not a measurement of the medium.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to annotate.
    fit_notes : dict of str -> tuple of float, or None
        Per detector ``(derived length [m], fitted length [m], normalization)``.
    """
    if not fit_notes:
        return
    short = {"IceCube": "IC", "KM3NeT/ARCA230": "ARCA"}
    lines = []
    for detector, (derived_m, fitted_m, norm) in fit_notes.items():
        lines.append(f"{short.get(detector, detector)}: "
                     f"$\\Lambda$ {derived_m:.0f}$\\to${fitted_m:.0f} m")
        lines.append(f"norm 1$\\to${norm:.2f}")
    ax.text(0.97, 0.975, "\n".join(lines), transform=ax.transAxes,
            ha="right", va="top", fontsize=8, linespacing=1.4,
            multialignment="right", color=MODEL_COLOR["Fitted"], zorder=5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                  "edgecolor": "none", "alpha": 0.85})


def make_area_figure(detectors, out_path: pathlib.Path, fit_notes=None) -> None:
    """Both detectors' effective areas on one panel, as in example 32.

    Colour carries which curve it is and line style carries which detector, so
    the six lines need three plus two legend entries. ``fit_notes`` states in
    the corner what the ``Fitted`` curve floated; see :func:`_annotate_fit`.
    """
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        for detector, (log10_e, curves, _) in detectors.items():
            for name, curve in curves.items():
                ax.plot(log10_e, curve, color=MODEL_COLOR[name],
                        ls=DETECTOR_STYLE[detector], lw=MODEL_WIDTH[name],
                        alpha=MODEL_ALPHA[name], zorder=MODEL_ZORDER[name])
        ax.set_yscale("log")
        ax.set_xlim(*PLOT_BAND)
        ax.set_ylim(1.0e4, 1.0e9)
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        _annotate_fit(ax, fit_notes)

        curve_legend = ax.legend(
            handles=[plt.Line2D([], [], color=c, lw=MODEL_WIDTH[n], alpha=MODEL_ALPHA[n],
                                label=n)
                     for n, c in MODEL_COLOR.items()],
            loc="upper left", fontsize=8, frameon=False, handlelength=2.0,
        )
        ax.add_artist(curve_legend)
        ax.legend(
            handles=[plt.Line2D([], [], color="k", ls=s, lw=1.0, label=d)
                     for d, s in DETECTOR_STYLE.items()],
            loc="lower right", fontsize=8, frameon=False, handlelength=2.4,
        )
        _save(fig, out_path)


def make_ratio_figure(detectors, out_path: pathlib.Path, fit_notes=None) -> None:
    """Published over model, which is where the shape agreement is legible.

    ``fit_notes`` states in the corner what the ``Fitted`` curve floated; see
    :func:`_annotate_fit`.
    """
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        ax.axhline(1.0, color="0.6", lw=0.8, ls=":", zorder=1)
        for detector, (log10_e, curves, _) in detectors.items():
            for name in MODELS:
                # Drawn over the whole plotted range; the fit still uses its own
                # narrower band, so the curve extends past what was fitted. The
                # model is zero below the light threshold, so the ratio is left
                # infinite there and matplotlib drops those points.
                with np.errstate(divide="ignore"):
                    ratio = curves["Published"] / curves[name]
                ax.plot(log10_e, ratio, color=MODEL_COLOR[name],
                        ls=DETECTOR_STYLE[detector], lw=1.3, zorder=3)
        ax.set_xlim(*PLOT_BAND)
        ax.set_ylim(0.0, 2.0)
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"published $/$ model")
        _annotate_fit(ax, fit_notes)

        curve_legend = ax.legend(
            handles=[plt.Line2D([], [], color=MODEL_COLOR[n], lw=1.3, label=n)
                     for n in MODELS],
            loc="upper left", fontsize=8, frameon=False, handlelength=2.0,
        )
        ax.add_artist(curve_legend)
        ax.legend(
            handles=[plt.Line2D([], [], color="k", ls=s, lw=1.0, label=d)
                     for d, s in DETECTOR_STYLE.items()],
            loc="lower right", fontsize=8, frameon=False, handlelength=2.4,
        )
        _save(fig, out_path)


if __name__ == "__main__":
    main()
