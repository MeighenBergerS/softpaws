"""Optical medium and optical module of a detector.

An :class:`Optics` record carries the published properties from which the
light reach of a site is predicted: the clarity of the medium, the size and
efficiency of the module, how densely the modules fill the instrumented
volume, and how much medium lies beyond it. The two records shipped here are
deep South Pole ice with the IceCube digital optical module and Capo Passero
sea water with the KM3NeT multi-photomultiplier module.

The wavelength shapes are smooth parameterizations of the published curves,
not digitized vendor tables. Swapping in a collaboration's own table is a
drop-in improvement.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ANCHOR_NM",
    "ARCA_OPTICS",
    "DEFAULT_MIN_TRACK_KM",
    "ICECUBE_OPTICS",
    "Optics",
]

#: Wavelength at which the absorption and scattering shapes are normalized [nm].
ANCHOR_NM = 400.0

#: Minimum path a track must have inside the instrumented volume to enter a
#: published response [km]. It belongs to the selection the response was made
#: at, not to the optics, so each site carries its own; this is IceCube's.
DEFAULT_MIN_TRACK_KM = 0.23


@dataclass(frozen=True)
class Optics:
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
        this detector's published response [km]; see
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
        scattering lengths imply. Set by a fit, unset everywhere else.
    illuminated_pmts : int or None, optional
        Photomultipliers of one module that see a distant track, for a
        multi-PMT module whose local coincidence is internal: KM3NeT's L1 is
        any two PMTs of one module, and a track's light reaches roughly the
        facing hemisphere, ~12 of ARCA's 31. ``None`` (the default) is a
        single-PMT module whose coincidence partners are its string
        neighbours, as in IceCube's HLC.
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
#: efficiency times pressure-sphere and gel transmission. The curves carry the
#: two features that matter: a peak near 0.22 to 0.27 in the near ultraviolet,
#: and a band ~120 nm wide inside the nominal 300 nm, cut below by the housing
#: glass and above by the bialkali cathode. IceCube's sphere cuts near 330 nm,
#: KM3NeT's a little lower, and its smaller tubes reach a slightly higher peak.
IC_EFFICIENCY_NM = (
    280.0, 300.0, 320.0, 340.0, 360.0, 380.0, 400.0, 420.0, 450.0, 500.0, 550.0, 600.0, 650.0,
)
IC_EFFICIENCY = (
    0.0, 0.003, 0.035, 0.115, 0.180, 0.212, 0.220, 0.212, 0.180, 0.118, 0.058, 0.018, 0.0,
)
ARCA_EFFICIENCY_NM = (
    280.0, 300.0, 320.0, 340.0, 360.0, 380.0, 400.0, 420.0, 450.0, 500.0, 550.0, 600.0, 650.0,
)
ARCA_EFFICIENCY = (
    0.0, 0.020, 0.125, 0.205, 0.250, 0.268, 0.262, 0.242, 0.200, 0.130, 0.062, 0.020, 0.0,
)

#: Relative absorption and effective-scattering lengths against wavelength [nm],
#: normalized at :data:`ANCHOR_NM` so each site's published length sets the
#: scale. Deep ice is clearest near 400 nm and its absorption collapses beyond
#: 500 nm as the intrinsic absorption of water takes over; the dust that
#: dominates below 350 nm closes the other end. Scattering in ice falls
#: smoothly with wavelength, roughly as ``lambda^0.9``.
ICE_ABSORPTION_NM = (280.0, 320.0, 360.0, 400.0, 440.0, 480.0, 520.0, 560.0, 600.0, 650.0)
ICE_ABSORPTION_SHAPE = (0.28, 0.55, 0.85, 1.00, 0.98, 0.82, 0.52, 0.27, 0.11, 0.04)
ICE_SCATTERING_NM = (280.0, 400.0, 600.0)
ICE_SCATTERING_SHAPE = (0.72, 1.00, 1.43)

#: The same for Capo Passero sea water, whose clarity peaks in the blue near
#: 450 to 470 nm and whose scattering is Rayleigh-like, falling steeply with
#: wavelength.
SEA_ABSORPTION_NM = (280.0, 320.0, 360.0, 400.0, 440.0, 470.0, 500.0, 550.0, 600.0, 650.0)
SEA_ABSORPTION_SHAPE = (0.22, 0.50, 0.78, 1.00, 1.18, 1.21, 0.98, 0.44, 0.14, 0.05)
SEA_SCATTERING_NM = (280.0, 400.0, 600.0)
SEA_SCATTERING_SHAPE = (0.42, 1.00, 2.25)

#: Deep South Pole ice below the dust layer, and the IceCube digital optical
#: module: one downward-facing 10-inch photomultiplier of 324 cm^2 photocathode.
#: The array is instrumented from 1450 to 2450 m and the ice sheet is ~2820 m
#: thick at the Pole, so the reach has 370 m of ice below the deepest module
#: and bedrock after that. The headroom above never binds at these reaches.
ICECUBE_OPTICS = Optics(
    name="IceCube",
    refractive_index=1.33,
    absorption_m=175.0,
    scattering_m=60.0,
    cathode_area_m2=0.0324,
    quantum_efficiency=0.25,
    density_g_cm3=0.92,
    headroom_above_m=1450.0,
    headroom_below_m=370.0,
    min_track_km=DEFAULT_MIN_TRACK_KM,
    module_density_per_km3=5160.0,
    efficiency_nm=IC_EFFICIENCY_NM,
    efficiency=IC_EFFICIENCY,
    absorption_nm=ICE_ABSORPTION_NM,
    absorption_shape=ICE_ABSORPTION_SHAPE,
    scattering_nm=ICE_SCATTERING_NM,
    scattering_shape=ICE_SCATTERING_SHAPE,
)

#: The Capo Passero site, and the KM3NeT multi-photomultiplier module: 31
#: 3-inch photomultipliers covering the full sphere, 1260 cm^2 in total. Sea
#: water scatters weakly, so its attenuation is absorption limited. The lowest
#: storey sits ~80 m above a 3500 m seabed; above the highest there are ~2800 m
#: of water.
ARCA_OPTICS = Optics(
    name="KM3NeT/ARCA230",
    refractive_index=1.35,
    absorption_m=68.0,
    scattering_m=265.0,
    cathode_area_m2=0.126,
    quantum_efficiency=0.25,
    density_g_cm3=1.04,
    headroom_above_m=2810.0,
    headroom_below_m=80.0,
    min_track_km=0.0,
    module_density_per_km3=3902.0,
    efficiency_nm=ARCA_EFFICIENCY_NM,
    efficiency=ARCA_EFFICIENCY,
    absorption_nm=SEA_ABSORPTION_NM,
    absorption_shape=SEA_ABSORPTION_SHAPE,
    scattering_nm=SEA_SCATTERING_NM,
    scattering_shape=SEA_SCATTERING_SHAPE,
    illuminated_pmts=12,
)
