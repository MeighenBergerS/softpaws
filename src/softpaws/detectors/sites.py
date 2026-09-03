"""Detector sites: geometry, medium, and where on Earth each one sits.

One :class:`Site` describes the instrumented body of a neutrino telescope as
a convex solid (a sphere, an upright cylinder, or a prism with a regular
polygon cross-section), the medium it is built in, its depth, and its
latitude. The module-level constants are the published layouts of the
detectors the paper compares, and :data:`SITES` collects them by name.

Every number here is a published layout value. None is adjusted against an
effective-area curve; the two instrument numbers that are fitted (the
selection threshold and the light reach) live with the response, not here.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from softpaws.transport.earth import MAX_UPSTREAM_KM, neutrino_column_g_cm2, overburden_km
from softpaws.transport.soft_volume import prism_projected_area_km2
from softpaws.utils.constants import RHO_ICE_G_CM3, RHO_LAKE_G_CM3, RHO_WATER_G_CM3

from .optics import ARCA_OPTICS, ICECUBE_OPTICS, Optics

__all__ = [
    "ARCA21",
    "ARCA230",
    "GEN2",
    "GVD",
    "ICECUBE",
    "MAX_UPSTREAM_KM",
    "PONE",
    "SITES",
    "Site",
    "TRIDENT",
    "TRIDENT_2025",
    "get_site",
]


@dataclass(frozen=True)
class Site:
    """One detector: its geometry, its medium, and where on Earth it sits.

    Attributes
    ----------
    name : str
        Detector name, as printed in tables and figures.
    shape : {"sphere", "cylinder", "prism"}
        Body used for the projected area and the instrumented volume. A prism
        is a cylinder whose cross-section is a regular ``n_sides``-gon of the
        same area, which lengthens the perimeter and so the side projection.
    latitude_deg : float
        Geographic latitude [deg]. Sets how a declination maps to a zenith.
    depth_km : float
        Depth of the instrumented centre below the surface of the medium [km].
        This is the column a downgoing muon has to be born in.
    density_g_cm3 : float
        Density of the detector medium [g cm^-3].
    radius_km : float
        Instrumented radius, per block for a cylinder [km].
    height_km : float, optional
        Instrumented height of one cylinder or prism [km]. Ignored for a sphere.
    n_blocks : int, optional
        Number of identical blocks. Ignored for a sphere.
    n_sides : int, optional
        Number of sides of the prism cross-section. Ignored unless ``shape``
        is ``"prism"``.
    below_km : float, optional
        Optical medium between the bottom of the instrumented volume and the
        rock beneath it [km]. An upgoing muon crosses only this and half the
        height before it is seen, and is in rock for the rest of its range.
    optics : Optics or None, optional
        Optical medium and module properties, when published. ``None`` for a
        site whose light reach is only ever fitted.
    """

    name: str
    shape: str
    latitude_deg: float
    depth_km: float
    density_g_cm3: float
    radius_km: float
    height_km: float = 0.0
    n_blocks: int = 1
    n_sides: int = 6
    below_km: float = 0.0
    optics: Optics | None = None

    def __post_init__(self) -> None:
        if self.shape not in ("sphere", "cylinder", "prism"):
            raise ValueError(
                f"shape must be 'sphere', 'cylinder' or 'prism', got {self.shape!r}."
            )
        if self.shape != "sphere" and self.height_km <= 0.0:
            raise ValueError(f"A {self.shape} needs a positive height_km.")

    def replace(self, **changes) -> Site:
        """Copy of this site with some fields changed.

        Parameters
        ----------
        **changes
            Field values to override, as for :func:`dataclasses.replace`.

        Returns
        -------
        site : Site
            The modified copy.

        Examples
        --------
        >>> TRIDENT.replace(radius_km=1.75).radius_km
        1.75
        """
        return dataclasses.replace(self, **changes)

    def projected_area_km2(
        self,
        cos_theta: float | np.ndarray,
        radius_km: float | np.ndarray | None = None,
        height_km: float | np.ndarray | None = None,
    ) -> np.ndarray:
        """Projected area presented to a given arrival zenith [km^2].

        A sphere presents ``pi R^2`` from every direction. An upright cylinder
        presents ``pi R^2`` overhead and ``2 R h`` at the horizon, and the
        convex-body projection interpolates between them. A prism replaces the
        ``2 R h`` by ``(P / pi) h`` for the perimeter ``P`` of its regular
        cross-section, 5% larger than the circle of equal area for a hexagon.
        The two terms add, so the oblique projection exceeds both face-on
        values.

        Parameters
        ----------
        cos_theta : float or np.ndarray
            Cosine of the arrival zenith; ``+1`` is overhead.
        radius_km : float or np.ndarray, optional
            Effective radius [km], broadcast against ``cos_theta``. ``None``
            (the default) uses the instrumented one; a light reach that
            dilates the body passes its own.
        height_km : float or np.ndarray, optional
            Effective height [km]. ``None`` (the default) uses the instrumented
            one.

        Returns
        -------
        area : np.ndarray
            Projected area [km^2].
        """
        radius = np.asarray(self.radius_km if radius_km is None else radius_km, dtype=float)
        cos_theta = np.asarray(cos_theta, dtype=float)
        if self.shape == "sphere":
            return np.pi * radius**2 * np.ones_like(cos_theta)
        return prism_projected_area_km2(
            cos_theta,
            radius,
            self.height_km if height_km is None else height_km,
            n_sides=self.n_sides if self.shape == "prism" else None,
            n_blocks=self.n_blocks,
        )

    def mean_projected_area_km2(
        self,
        radius_km: float | np.ndarray | None = None,
        height_km: float | np.ndarray | None = None,
        n_cos_theta: int = 2001,
    ) -> float:
        """Projected area averaged over the full sphere of arrival directions [km^2].

        Parameters
        ----------
        radius_km : float or np.ndarray, optional
            Effective radius [km]; see :meth:`projected_area_km2`.
        height_km : float or np.ndarray, optional
            Effective height [km]; see :meth:`projected_area_km2`.
        n_cos_theta : int, optional
            Number of nodes of the uniform ``cos(theta)`` grid the average
            runs over.

        Returns
        -------
        area : float
            Solid-angle mean of the projected area [km^2].
        """
        cos_theta = np.linspace(-1.0, 1.0, n_cos_theta)
        area = self.projected_area_km2(cos_theta, radius_km, height_km)
        return float(np.trapezoid(area, cos_theta) / 2.0)

    def detector_volume_km3(
        self,
        radius_km: float | np.ndarray | None = None,
        height_km: float | np.ndarray | None = None,
    ) -> np.ndarray:
        """Volume of the instrumented body itself [km^3].

        Parameters
        ----------
        radius_km : float or np.ndarray, optional
            Effective radius [km]; ``None`` uses the instrumented one.
        height_km : float or np.ndarray, optional
            Effective height [km]; ``None`` uses the instrumented one.

        Returns
        -------
        volume : np.ndarray
            Instrumented volume [km^3].
        """
        radius = np.asarray(self.radius_km if radius_km is None else radius_km, dtype=float)
        if self.shape == "sphere":
            return 4.0 / 3.0 * np.pi * radius**3
        height = np.asarray(self.height_km if height_km is None else height_km, dtype=float)
        return self.n_blocks * np.pi * radius**2 * height

    def columns(self, cos_theta: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Neutrino column and available muon column for each arrival direction.

        A source above the horizon sends its neutrino down through the
        overburden only, which is negligible except within a degree of the
        horizon, and confines the muon to that same overburden. A source below
        the horizon sends its neutrino through the layered-PREM Earth chord and
        gives the muon effectively unlimited rock upstream.

        Parameters
        ----------
        cos_theta : float or np.ndarray
            Cosine of the arrival zenith; ``+1`` is overhead, ``-1`` the nadir.

        Returns
        -------
        neutrino_column : np.ndarray
            Column traversed before reaching the detector [g cm^-2].
        muon_column_km : np.ndarray
            Column available upstream of the detector, as a length of the
            detector medium [km].
        """
        cos_theta = np.asarray(cos_theta, dtype=float)
        neutrino_column = neutrino_column_g_cm2(cos_theta, self.depth_km, self.density_g_cm3)
        muon_column_km = overburden_km(cos_theta, self.depth_km)
        return neutrino_column, muon_column_km


#: IceCube: 1 km^2 of hexagonal footprint by 1 km of instrumented height,
#: which reproduces the 1 km^3 instrumented volume exactly, centred 1.95 km
#: below the surface. The array is instrumented from 1450 to 2450 m and the ice
#: sheet is ~2820 m thick at the Pole, so 370 m of ice lie below the deepest
#: module and bedrock after that.
ICECUBE = Site(
    name="IceCube",
    shape="prism",
    latitude_deg=-90.0,
    depth_km=1.95,
    density_g_cm3=RHO_ICE_G_CM3,
    radius_km=float(np.sqrt(1.0 / np.pi)),
    height_km=1.0,
    n_sides=6,
    below_km=0.37,
    optics=ICECUBE_OPTICS,
)

#: IceCube-Gen2: the 7.9 km^3 optical array of the technical design report,
#: as a hexagonal prism 1.25 km tall at the IceCube depth.
GEN2 = Site(
    name="IceCube-Gen2",
    shape="prism",
    latitude_deg=-90.0,
    depth_km=1.95,
    density_g_cm3=RHO_ICE_G_CM3,
    radius_km=float(np.sqrt(7.9 / (np.pi * 1.25))),
    height_km=1.25,
    n_sides=6,
    below_km=0.37,
    optics=ICECUBE_OPTICS,
)

#: One KM3NeT/ARCA building block: 115 detection units of 18 modules at 36.8 m
#: vertical spacing, a cylinder of 0.517 km radius and 0.632 km height.
_ARCA_BLOCK_RADIUS_KM = 0.517
_ARCA_BLOCK_HEIGHT_KM = 0.632
#: The seabed at Capo Passero is 3.5 km down and the lowest storey sits ~80 m
#: above it, so the block centre is at 3.5 km minus half its height.
_ARCA_DEPTH_KM = 3.5 - 0.5 * _ARCA_BLOCK_HEIGHT_KM

#: KM3NeT/ARCA with both building blocks.
ARCA230 = Site(
    name="ARCA230",
    shape="cylinder",
    latitude_deg=36.27,
    depth_km=_ARCA_DEPTH_KM,
    density_g_cm3=RHO_WATER_G_CM3,
    radius_km=_ARCA_BLOCK_RADIUS_KM,
    height_km=_ARCA_BLOCK_HEIGHT_KM,
    n_blocks=2,
    below_km=0.08,
    optics=ARCA_OPTICS,
)

#: The 21-line partial block that recorded KM3-230213A, as one cylinder of the
#: same height and the footprint radius of 21 of the 115 units.
ARCA21 = Site(
    name="ARCA21",
    shape="cylinder",
    latitude_deg=36.27,
    depth_km=_ARCA_DEPTH_KM,
    density_g_cm3=RHO_WATER_G_CM3,
    radius_km=0.221,
    height_km=_ARCA_BLOCK_HEIGHT_KM,
    n_blocks=1,
    below_km=0.08,
    optics=ARCA_OPTICS,
)

#: TRIDENT as proposed in 2022: one block of 2 km radius and 0.57 km height in
#: the South China Sea, centred between 2.8 and 3.4 km depth, about 100 m
#: above the seabed at its 3.5 km site.
TRIDENT = Site(
    name="TRIDENT",
    shape="cylinder",
    latitude_deg=17.4,
    depth_km=0.5 * (2.800 + 3.400),
    density_g_cm3=RHO_WATER_G_CM3,
    radius_km=2.0,
    height_km=0.570,
    n_blocks=1,
    below_km=0.1,
)

#: The reference layout of the 2025 TRIDENT simulation study: 1000 strings at
#: 100 m average spacing (~9.6 km^2, radius 1.75 km) with 20 hybrid modules at
#: 30 m spacing, at the 2022 depth.
TRIDENT_2025 = TRIDENT.replace(name="TRIDENT", radius_km=1.75)

#: P-ONE: seven clusters of 0.12 km radius and 1 km height standing on the
#: Cascadia Basin floor at 2.66 km.
PONE = Site(
    name="P-ONE",
    shape="cylinder",
    latitude_deg=47.75,
    depth_km=2.66 - 0.5 * 1.0,
    density_g_cm3=RHO_WATER_G_CM3,
    radius_km=0.120,
    height_km=1.0,
    n_blocks=7,
)

#: Baikal-GVD: fourteen clusters of 0.06 km radius and 0.525 km height in
#: fresh water, instrumented between 0.75 and 1.275 km depth.
GVD = Site(
    name="Baikal-GVD",
    shape="cylinder",
    latitude_deg=51.77,
    depth_km=0.5 * (0.750 + 1.275),
    density_g_cm3=RHO_LAKE_G_CM3,
    radius_km=0.060,
    height_km=0.525,
    n_blocks=14,
)

#: Every site above, keyed by name. ``TRIDENT_2025`` is stored under
#: ``"TRIDENT-2025"`` so that both layouts stay reachable.
SITES: dict[str, Site] = {
    ICECUBE.name: ICECUBE,
    GEN2.name: GEN2,
    ARCA230.name: ARCA230,
    ARCA21.name: ARCA21,
    TRIDENT.name: TRIDENT,
    "TRIDENT-2025": TRIDENT_2025,
    PONE.name: PONE,
    GVD.name: GVD,
}


def get_site(name: str) -> Site:
    """Look a site up by name.

    Parameters
    ----------
    name : str
        Key of :data:`SITES`, matched without regard to case.

    Returns
    -------
    site : Site
        The published layout.

    Raises
    ------
    KeyError
        Raised if ``name`` is not a known site.

    Examples
    --------
    >>> get_site("icecube").radius_km
    0.5641895835477563
    """
    for key, site in SITES.items():
        if key.lower() == name.lower():
            return site
    raise KeyError(f"Unknown site {name!r}; known sites are {sorted(SITES)}.")
