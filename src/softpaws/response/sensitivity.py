"""What a search would exclude, for three shapes of injected signal.

An effective area answers one question: how many events a flux delivers. Turn
that around and it answers the one an observer asks instead, which is how
large a signal has to be before a telescope sees it. The count is linear in
the flux normalization, so every sensitivity here is one quadrature and one
division, and the three functions differ only in the shape injected.

A single event is the weakest statement a telescope can make, and the flux it
corresponds to is differential: one event per decade of neutrino energy,
which is :func:`single_event_sensitivity`. A line is a flux that arrives at
one energy, the shape a decaying or annihilating particle of fixed mass
gives, and :func:`line_sensitivity` is its counterpart. A power law spreads
the same normalization over the whole band, and
:func:`power_law_sensitivity` integrates the response against it.

Each takes the effective area of a point source or of the diffuse sky. Pass
``solid_angle_sr`` and the result is per steradian, which is the only
difference between the two cases: a point source is a direction and a diffuse
flux is that direction times the sky it covers.

Each also takes ``n_events``, the number of events a search can exclude, and
that number is where a background enters. Left alone it is
:data:`N_EVENTS_LIMIT`, the Feldman-Cousins limit on zero observed events over
zero expected background, which is the right statement at the highest energies
where a track selection expects no atmospheric event at all. Below a few
hundred TeV it is optimistic, because the background is what the search has to
climb out of. :func:`atmospheric_background_counts` folds an MCEq flux through
the same response to say how many events that is, and
:func:`sensitivity_upper_limit` turns the answer back into an ``n_events`` to
pass in. At zero background that route returns :data:`N_EVENTS_LIMIT` again, so
the background-free ceiling is the zero-background limit of the same formula
rather than a separate one.

Notes
-----
The background here is the atmospheric neutrino floor alone. Atmospheric muons
are the larger background wherever they reach the detector at all, which is the
downgoing sky, so a limit computed here for a downgoing source is still a
ceiling and not a forecast.
"""

from __future__ import annotations

import functools

import numpy as np
from scipy.special import gammaln

from ..fluxes.atmospheric import AtmosphericFlux, table_declination_deg

__all__ = [
    "CONFIDENCE_LEVEL",
    "DEFAULT_BIN_RADIUS_DEG",
    "EXACT_BACKGROUND_MAX",
    "N_EVENTS_LIMIT",
    "OPTIMAL_CONTAINMENT",
    "PIVOT_ENERGY_GEV",
    "atmospheric_background_counts",
    "atmospheric_background_density",
    "feldman_cousins_upper_limit",
    "line_sensitivity",
    "optimized_window_sensitivity",
    "point_source_bin_sr",
    "power_law_sensitivity",
    "psf_bin_radius_deg",
    "sensitivity_upper_limit",
    "single_event_sensitivity",
]

#: Events a background-free search excludes at 90% confidence (Feldman-Cousins,
#: zero observed on zero background).
N_EVENTS_LIMIT = 2.44

#: Pivot energy of a quoted power-law flux [GeV].
PIVOT_ENERGY_GEV = 1.0e5

#: Natural logarithm of ten, the width of a decade in ``ln(E)``.
LN10 = float(np.log(10.0))

#: Confidence level the Feldman-Cousins construction is built at.
CONFIDENCE_LEVEL = 0.9

#: Radius of the bin a point-source background is counted in [deg]. About the
#: angular resolution a through-going track is reconstructed to above a TeV,
#: which is what sets how much sky an unbinned search has to look through.
#: A measured point-spread function replaces it wherever one is available.
DEFAULT_BIN_RADIUS_DEG = 1.0

#: Background above which the exact construction is replaced by its large-count
#: form. The construction below costs memory linear in the background, because
#: it holds every count a Poisson of that mean can deliver, so an unbounded one
#: is not merely slow. By here the average upper limit is a straight line in
#: ``sqrt(b)`` to better than a per cent, and :func:`sensitivity_upper_limit`
#: continues along that line instead.
EXACT_BACKGROUND_MAX = 200.0

#: Containment of the point-spread function that makes the best counting bin.
#: For a Gaussian the radius maximizing signal over the root of the background
#: is ``1.585 sigma``, and the 68% containment radius is ``1.510 sigma``, so
#: taking the published containment as the bin is the optimum to 5% with
#: nothing fitted.
OPTIMAL_CONTAINMENT = 0.68


def _exposure(livetime_s: float, solid_angle_sr: float | None) -> float:
    """Exposure the count is proportional to, per unit effective area.

    Parameters
    ----------
    livetime_s : float
        Exposure [s].
    solid_angle_sr : float or None
        Solid angle the diffuse flux covers [sr]. ``None`` is a point source.

    Returns
    -------
    exposure : float
        ``livetime_s``, times the solid angle when there is one.
    """
    return livetime_s if solid_angle_sr is None else livetime_s * solid_angle_sr


def _energy_axis(log10_e: np.ndarray, ndim: int) -> np.ndarray:
    """Energies [GeV] shaped to broadcast against an effective area.

    Parameters
    ----------
    log10_e : np.ndarray
        Neutrino energies [log10 GeV].
    ndim : int
        Dimensions of the effective-area array, energy first.

    Returns
    -------
    energy : np.ndarray
        ``10 ** log10_e`` [GeV], with trailing axes of length one.
    """
    energy = 10.0 ** np.asarray(log10_e, dtype=float)
    return energy.reshape((-1,) + (1,) * (ndim - 1))


def _events_axis(n_events: float | np.ndarray, ndim: int) -> np.ndarray:
    """Excludable events shaped to broadcast against an effective area.

    ``n_events`` is a scalar when the background is ignored and an array once
    it is not, since the background differs from one energy or one direction
    to the next. A one-dimensional array is read as the energy axis and given
    the trailing axes an effective area needs.

    Parameters
    ----------
    n_events : float or np.ndarray
        Events the search excludes.
    ndim : int
        Dimensions of the effective-area array, energy first.

    Returns
    -------
    n_events : np.ndarray
        The same values, reshaped to broadcast.
    """
    events = np.asarray(n_events, dtype=float)
    if events.ndim != 1 or ndim <= 1:
        return events
    return events.reshape((-1,) + (1,) * (ndim - 1))


def single_event_sensitivity(
    aeff_cm2: np.ndarray,
    livetime_s: float,
    log10_e: np.ndarray,
    n_events: float | np.ndarray = 1.0,
    solid_angle_sr: float | None = None,
) -> np.ndarray:
    """Flux delivering one event per decade of neutrino energy.

    A decade of a flux ``phi`` delivers ``T A_eff E ln(10) phi`` events, so the
    flux that delivers ``n_events`` of them is

    .. math:: E^2 \\phi = \\frac{n\\, E}{\\ln(10)\\, T\\, \\Delta\\Omega\\,
        A_{\\rm eff}(E)}.

    This is the differential sensitivity a telescope is usually quoted by, and
    the weakest signal it can report: one event, at the energy where it landed,
    with no assumption about the spectrum it came from.

    Parameters
    ----------
    aeff_cm2 : np.ndarray, shape (log10_e.size, ...)
        Effective area [cm^2] on the energy grid, energy first.
    livetime_s : float
        Exposure [s].
    log10_e : np.ndarray
        Neutrino energies [log10 GeV].
    n_events : float or np.ndarray, optional
        Events the flux delivers per decade. One, by default. An array of one
        dimension is read as the energy axis, which is how a background that
        varies across the band enters; see :func:`sensitivity_upper_limit`.
    solid_angle_sr : float or None, optional
        Solid angle of a diffuse signal [sr]. ``None``, the default, gives a
        point-source flux.

    Returns
    -------
    e2_flux : np.ndarray
        ``E^2 phi`` at each energy [GeV cm^-2 s^-1], per steradian when
        ``solid_angle_sr`` is given.

    Examples
    --------
    >>> import numpy as np
    >>> log10_e = np.array([5.0, 6.0])
    >>> aeff = np.array([1.0e4, 1.0e5])
    >>> single_event_sensitivity(aeff, 1.0e8, log10_e)
    array([4.34294482e-08, 4.34294482e-08])
    """
    aeff = np.asarray(aeff_cm2, dtype=float)
    energy = _energy_axis(log10_e, aeff.ndim)
    events = _events_axis(n_events, aeff.ndim)
    with np.errstate(divide="ignore", invalid="ignore"):
        return events * energy / (LN10 * _exposure(livetime_s, solid_angle_sr) * aeff)


def line_sensitivity(
    aeff_cm2: np.ndarray,
    livetime_s: float,
    n_events: float | np.ndarray = N_EVENTS_LIMIT,
    solid_angle_sr: float | None = None,
) -> np.ndarray:
    """Normalization of a monoenergetic flux a search would exclude.

    For ``phi(E) = Phi delta(E - E_0)`` the count is ``T Phi A_eff(E_0)`` with
    no quadrature left to do, so the limit is the effective area inverted at
    each energy in turn. Multiply the result by ``E_0`` to put it on the axis a
    continuum ``E^2 phi`` is drawn on.

    Parameters
    ----------
    aeff_cm2 : np.ndarray
        Effective area [cm^2] at each line energy.
    livetime_s : float
        Exposure [s].
    n_events : float or np.ndarray, optional
        Events the search excludes; see :data:`N_EVENTS_LIMIT`. An array of
        one dimension is read as the energy axis.
    solid_angle_sr : float or None, optional
        Solid angle of a diffuse signal [sr]. ``None``, the default, gives a
        point-source flux.

    Returns
    -------
    flux : np.ndarray
        Line normalization [cm^-2 s^-1], per steradian when
        ``solid_angle_sr`` is given.
    """
    aeff = np.asarray(aeff_cm2, dtype=float)
    events = _events_axis(n_events, aeff.ndim)
    with np.errstate(divide="ignore", invalid="ignore"):
        return events / (_exposure(livetime_s, solid_angle_sr) * aeff)


def power_law_sensitivity(
    aeff_cm2: np.ndarray,
    livetime_s: float,
    gamma: float,
    log10_e: np.ndarray,
    emin_gev: float | None = None,
    pivot_gev: float = PIVOT_ENERGY_GEV,
    n_events: float | np.ndarray = N_EVENTS_LIMIT,
    solid_angle_sr: float | None = None,
) -> np.ndarray:
    """Normalization of a power law a search would exclude.

    For ``phi(E) = phi_0 (E / E_piv)^-gamma`` the expected count is linear in
    ``phi_0``, so the limit is one quadrature,

    .. math:: \\phi_0^{\\rm lim} = \\frac{N_{\\rm lim}}
        {T\\, \\Delta\\Omega \\int_{E_{\\min}} \\dd E\\, A_{\\rm eff}(E)
        (E/E_{\\rm piv})^{-\\gamma}}.

    Parameters
    ----------
    aeff_cm2 : np.ndarray, shape (log10_e.size, ...)
        Effective area [cm^2] on the energy grid, energy first.
    livetime_s : float
        Exposure [s].
    gamma : float
        Spectral index of the injected flux.
    log10_e : np.ndarray
        Neutrino energies [log10 GeV].
    emin_gev : float or None, optional
        Bottom of the analysis window [GeV]. ``None``, the default, uses the
        whole grid.
    pivot_gev : float, optional
        Pivot energy of the quoted flux [GeV].
    n_events : float or np.ndarray, optional
        Events the search excludes; see :data:`N_EVENTS_LIMIT`. The whole
        band is one number here, so an array is read as one entry per column
        of ``aeff_cm2`` and not as an energy axis.
    solid_angle_sr : float or None, optional
        Solid angle of a diffuse signal [sr]. ``None``, the default, gives a
        point-source flux.

    Returns
    -------
    e2_flux : np.ndarray
        ``E^2 phi`` at ``pivot_gev`` [GeV cm^-2 s^-1], per steradian when
        ``solid_angle_sr`` is given.
    """
    aeff = np.asarray(aeff_cm2, dtype=float)
    energy = 10.0 ** np.asarray(log10_e, dtype=float)
    window = np.ones_like(energy, dtype=bool) if emin_gev is None else energy >= emin_gev
    weight = (energy / pivot_gev) ** (-gamma)
    shape = (-1,) + (1,) * (aeff.ndim - 1)
    integral = np.trapezoid((aeff * weight.reshape(shape))[window], energy[window], axis=0)
    phi_0 = n_events / (_exposure(livetime_s, solid_angle_sr) * integral)
    return pivot_gev**2 * phi_0


# ---------------------------------------------------------------------------
# The atmospheric background a point-source search sits on
# ---------------------------------------------------------------------------


def point_source_bin_sr(radius_deg: float | np.ndarray = DEFAULT_BIN_RADIUS_DEG) -> np.ndarray:
    """Solid angle of the bin a point-source background is counted in [sr].

    A point source is a direction and the background is a flux per steradian,
    so turning one into the other needs the sky the search has to look
    through. That is set by how well a track is reconstructed rather than by
    anything about the source, which is why the radius is an instrument
    number and not a physical one.

    Parameters
    ----------
    radius_deg : float or np.ndarray, optional
        Angular radius of the bin [deg]. See :data:`DEFAULT_BIN_RADIUS_DEG`.

    Returns
    -------
    solid_angle_sr : np.ndarray
        ``2 pi (1 - cos r)`` [sr].

    Examples
    --------
    >>> round(float(point_source_bin_sr(1.0)), 6)
    0.000957
    """
    radius = np.deg2rad(np.asarray(radius_deg, dtype=float))
    return 2.0 * np.pi * (1.0 - np.cos(radius))


def psf_bin_radius_deg(
    psf_table: dict | None,
    cos_theta: np.ndarray,
    log10_e: np.ndarray,
    fallback_deg: float = DEFAULT_BIN_RADIUS_DEG,
) -> np.ndarray | float:
    """Bin radius against energy and arrival direction, from a released point spread.

    A fixed bin is the wrong shape twice over. A track is reconstructed better
    the more light it leaves, so the bin closes as the energy rises, and the
    Earth takes the high-energy end away where the column is deepest, so it
    opens again there. Both are in the table and both run the way a published
    point-source sensitivity needs.

    Only IceCube has released one, and it is tabulated against declination at
    a site where a declination is one arrival zenith all day, which makes it a
    function of arrival direction. That is the form every other site can read
    it in. It stays IceCube's optics, IceCube's spacing and IceCube's
    selection, so for the others it is a stand-in and not a measurement, and
    the downgoing end of it is a selection rather than a resolution.

    Parameters
    ----------
    psf_table : dict or None
        Cached table from :func:`softpaws.data.load_psf_table`. ``None``
        returns ``fallback_deg``.
    cos_theta : np.ndarray, shape (n_dir,)
        Cosine of the arrival zenith of each direction; ``+1`` is overhead.
    log10_e : np.ndarray
        Neutrino energies the grid is wanted on [log10 GeV].
    fallback_deg : float, optional
        Radius used when there is no table [deg].

    Returns
    -------
    radius : np.ndarray, shape (log10_e.size, n_dir), or float
        Bin radius at each energy and direction [deg].
    """
    if psf_table is None:
        return fallback_deg
    grid = np.asarray(psf_table["containment_deg"], dtype=float)
    # Seen from the Pole a source at declination dec arrives at
    # cos_theta = -sin(dec), so the table's declination axis is an arrival
    # direction and every site can be evaluated on it.
    table_cos = -np.sin(np.deg2rad(np.asarray(psf_table["dec_deg"], dtype=float)))
    order = np.argsort(table_cos)
    cos_theta = np.atleast_1d(np.asarray(cos_theta, dtype=float))
    on_direction = np.array([np.interp(cos_theta, table_cos[order], row[order])
                             for row in grid])
    return np.array([np.interp(log10_e, psf_table["log10_e"], on_direction[:, j])
                     for j in range(cos_theta.size)]).T


def atmospheric_background_density(
    flux: AtmosphericFlux,
    aeff_cm2: np.ndarray,
    cos_theta: np.ndarray,
    weights: np.ndarray,
    livetime_s: float,
    log10_e: np.ndarray,
    bin_radius_deg: float | np.ndarray = DEFAULT_BIN_RADIUS_DEG,
) -> np.ndarray:
    """Atmospheric events per decade in a point-source bin.

    The background rides the same directions the signal does. A source at a
    fixed declination sweeps a fixed set of zeniths as the Earth turns, and
    both the atmospheric flux and the effective area vary along that sweep, so
    the two are multiplied direction by direction before the time average
    rather than after it. Doing it the other way rounds a flux that changes by
    an order of magnitude between the horizon and the vertical.

    Parameters
    ----------
    flux : AtmosphericFlux
        Interpolated MCEq table, summed over ``nu_mu`` and ``nu_mu_bar``.
    aeff_cm2 : np.ndarray, shape (log10_e.size, n_dir)
        Effective area per arrival direction [cm^2].
    cos_theta : np.ndarray, shape (n_dir,)
        Cosine of the arrival zenith of each direction; ``+1`` is overhead.
    weights : np.ndarray, shape (n_dir,) or (n_dec, n_dir)
        Time fraction each direction carries. One row per source declination
        gives one column of the result per declination.
    livetime_s : float
        Exposure [s].
    log10_e : np.ndarray
        Neutrino energies [log10 GeV].
    bin_radius_deg : float or np.ndarray, optional
        Angular radius of the bin [deg]: a scalar, one value per energy, or a
        grid shaped like ``aeff_cm2``. A real bin is not a scalar, because a
        track is reconstructed better the more light it leaves, so the sky a
        search looks through closes as the energy rises and opens again wherever
        the Earth has taken the high-energy end away. See
        :meth:`~softpaws.response.irfs.SmearingMatrix.psf_containment_deg`.

    Returns
    -------
    density : np.ndarray, shape (log10_e.size,) or (log10_e.size, n_dec)
        Atmospheric events per decade of neutrino energy at each node.

    Notes
    -----
    The table is ``nu_mu + nu_mu_bar`` while the default response is built on
    the ``nu`` cross section, which is about half the ``nu_bar`` one at a TeV
    and equal to it above a PeV. Pass a response averaged over the two to
    remove the resulting overestimate at the bottom of the band.
    """
    aeff = np.asarray(aeff_cm2, dtype=float)
    weights = np.asarray(weights, dtype=float)
    energy = 10.0 ** np.asarray(log10_e, dtype=float)
    table_dec = table_declination_deg(cos_theta)
    # (n_e, n_dir): the flux each direction of the sweep actually sees.
    per_direction = flux(energy[:, None], table_dec[None, :]) * aeff
    # The bin belongs inside the direction sum, not outside it. A measured
    # point spread varies with arrival direction as much as the flux does, and
    # a source that sweeps the sky sees both changing together.
    solid_angle = _events_axis(np.asarray(point_source_bin_sr(bin_radius_deg)), 2)
    density = (per_direction * solid_angle) @ weights.T
    return density * LN10 * livetime_s * _energy_axis(log10_e, density.ndim)


def atmospheric_background_counts(
    flux: AtmosphericFlux,
    aeff_cm2: np.ndarray,
    cos_theta: np.ndarray,
    weights: np.ndarray,
    livetime_s: float,
    log10_e: np.ndarray,
    bin_radius_deg: float | np.ndarray = DEFAULT_BIN_RADIUS_DEG,
    emin_gev: float | None = None,
) -> np.ndarray:
    """Atmospheric events in a point-source bin over a window.

    :func:`atmospheric_background_density` integrated over ``log10 E``, which
    is the number that pairs with :func:`power_law_sensitivity`: one
    background for the whole band, against one normalization for the whole
    band. Use the same ``emin_gev`` in both.

    Parameters
    ----------
    flux : AtmosphericFlux
        Interpolated MCEq table.
    aeff_cm2 : np.ndarray, shape (log10_e.size, n_dir)
        Effective area per arrival direction [cm^2].
    cos_theta : np.ndarray, shape (n_dir,)
        Cosine of the arrival zenith of each direction.
    weights : np.ndarray, shape (n_dir,) or (n_dec, n_dir)
        Time fraction each direction carries.
    livetime_s : float
        Exposure [s].
    log10_e : np.ndarray
        Neutrino energies [log10 GeV].
    bin_radius_deg : float or np.ndarray, optional
        Angular radius of the bin [deg].
    emin_gev : float or None, optional
        Bottom of the analysis window [GeV]. ``None``, the default, uses the
        whole grid.

    Returns
    -------
    counts : np.ndarray
        Expected atmospheric events, a scalar or one per declination.
    """
    density = atmospheric_background_density(
        flux, aeff_cm2, cos_theta, weights, livetime_s, log10_e, bin_radius_deg
    )
    grid = np.asarray(log10_e, dtype=float)
    window = np.ones_like(grid, dtype=bool) if emin_gev is None else 10.0**grid >= emin_gev
    return np.trapezoid(density[window], grid[window], axis=0)


# ---------------------------------------------------------------------------
# Feldman-Cousins with a background
# ---------------------------------------------------------------------------


def _acceptance(background: float, confidence_level: float, mu_step: float):
    """Feldman-Cousins acceptance of every observed count, on a grid of signals.

    The unified construction ranks the possible counts by the likelihood ratio
    ``P(n | mu + b) / P(n | mu_best + b)`` with ``mu_best = max(n - b, 0)``,
    then takes them in that order until the coverage reaches the confidence
    level. Doing it once for every count at once costs no more than doing it
    for one, because the ranking is a sort of the same matrix.

    Parameters
    ----------
    background : float
        Expected background events.
    confidence_level : float
        Coverage of the construction.
    mu_step : float
        Spacing of the signal grid [events].

    Returns
    -------
    mu : np.ndarray, shape (n_mu,)
        The signal grid.
    accepted : np.ndarray, shape (n_mu, n_max)
        Whether each count falls in the acceptance interval of each signal.
    """
    if background > EXACT_BACKGROUND_MAX:
        raise ValueError(
            f"The exact construction holds every count a Poisson of mean {background:g} "
            f"can deliver, which does not fit in memory above "
            f"{EXACT_BACKGROUND_MAX:g}. Use sensitivity_upper_limit, which continues "
            "along the large-count form instead."
        )
    # The signal grid has to reach past any limit the construction can return,
    # and the counts past anything either the signal or the background can
    # deliver, or the coverage never reaches the confidence level and every
    # signal is accepted. A cap on the grid keeps the matrix below in hand.
    mu_max = background + 8.0 * np.sqrt(background) + 20.0
    mu = np.arange(0.0, mu_max, max(mu_step, mu_max / 4000.0))
    reach = mu_max + background
    counts = np.arange(int(np.ceil(reach + 10.0 * np.sqrt(reach) + 20.0)))

    log_factorial = gammaln(counts + 1.0)
    total = mu[:, None] + background
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(total > 0.0, counts * np.log(total), -np.inf) - total - log_factorial
    log_p[:, 0] = np.where(total[:, 0] > 0.0, -total[:, 0], 0.0)

    best = np.maximum(counts - background, 0.0) + background
    with np.errstate(divide="ignore", invalid="ignore"):
        log_best = np.where(best > 0.0, counts * np.log(best), 0.0) - best - log_factorial
    log_best[0] = -best[0] if best[0] > 0.0 else 0.0

    probability = np.exp(log_p)
    order = np.argsort(log_best - log_p, axis=1, kind="stable")
    coverage = np.cumsum(np.take_along_axis(probability, order, axis=1), axis=1)
    # Rank at which the coverage first reaches the confidence level, and the
    # rank each count sits at, so membership is one comparison.
    reached = coverage >= confidence_level
    last = np.where(reached.any(axis=1), np.argmax(reached, axis=1), coverage.shape[1] - 1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(counts, order.shape), axis=1)
    return mu, rank <= last[:, None]


def feldman_cousins_upper_limit(
    n_obs: int,
    background: float = 0.0,
    confidence_level: float = CONFIDENCE_LEVEL,
    mu_step: float = 0.005,
) -> float:
    """Upper limit on a Poisson signal over a known background.

    Parameters
    ----------
    n_obs : int
        Events observed.
    background : float, optional
        Expected background events. Zero, by default.
    confidence_level : float, optional
        Coverage of the construction. See :data:`CONFIDENCE_LEVEL`.
    mu_step : float, optional
        Spacing of the signal grid [events], which sets the resolution of the
        answer.

    Returns
    -------
    upper_limit : float
        Largest signal the observation is consistent with, in events.

    Examples
    --------
    Zero observed on zero expected is the number every background-free limit
    in this module is quoted at.

    >>> round(feldman_cousins_upper_limit(0), 2)
    2.44
    """
    mu, accepted = _acceptance(float(background), confidence_level, mu_step)
    column = accepted[:, int(n_obs)]
    if not column.any():
        return 0.0
    return float(mu[len(column) - 1 - int(np.argmax(column[::-1]))])


def sensitivity_upper_limit(
    background: float | np.ndarray,
    confidence_level: float = CONFIDENCE_LEVEL,
    mu_step: float = 0.005,
) -> np.ndarray:
    """Events a search excludes on average, before it looks.

    A limit is a statement about one observation, and a sensitivity is what a
    search expects to be able to say before it has made one. Feldman and
    Cousins define that as the upper limit averaged over background-only
    outcomes, which is the number this returns, and it reduces to
    :data:`N_EVENTS_LIMIT` when there is no background to average over.

    Pass the result as ``n_events`` to any of the three converters and the
    ceiling they return stops being background free.

    Parameters
    ----------
    background : float or np.ndarray
        Expected background events, from
        :func:`atmospheric_background_counts`.
    confidence_level : float, optional
        Coverage of the construction. See :data:`CONFIDENCE_LEVEL`.
    mu_step : float, optional
        Spacing of the signal grid [events].

    Returns
    -------
    n_events : np.ndarray
        Events the search excludes, with the shape of ``background``.

    Notes
    -----
    Above :data:`EXACT_BACKGROUND_MAX` the answer continues along the straight
    line in ``sqrt(b)`` that the exact construction is already following, since
    holding every count a Poisson of that mean can deliver stops fitting in
    memory. The two branches meet, and the second is good to about a per cent.

    IceCube quotes a point-source sensitivity on a slightly different
    criterion, the flux at which 90% of signal trials beat the median
    background test statistic. The two agree exactly with no background, where
    both are :data:`N_EVENTS_LIMIT`, and differ by a few per cent with one.

    Examples
    --------
    >>> round(float(sensitivity_upper_limit(0.0)), 2)
    2.44
    """
    values = np.asarray(background, dtype=float)
    out = np.empty(values.size)
    slope, intercept = _large_count_line(confidence_level, mu_step)
    for i, b in enumerate(values.ravel()):
        out[i] = (
            slope * np.sqrt(b) + intercept
            if b > EXACT_BACKGROUND_MAX
            else _exact_sensitivity(round(float(b), 6), confidence_level, mu_step)
        )
    return out.reshape(values.shape) if values.ndim else out[0]


@functools.lru_cache(maxsize=4096)
def _exact_sensitivity(background: float, confidence_level: float, mu_step: float) -> float:
    """Average Feldman-Cousins upper limit over background-only outcomes.

    Cached, because a window scan asks for the same background many times.

    Parameters
    ----------
    background : float
        Expected background events, at or below :data:`EXACT_BACKGROUND_MAX`.
    confidence_level : float
        Coverage of the construction.
    mu_step : float
        Spacing of the signal grid [events].

    Returns
    -------
    n_events : float
        Events the search excludes.
    """
    mu, accepted = _acceptance(background, confidence_level, mu_step)
    # One construction gives the limit for every count, so the average over
    # background-only outcomes is a dot product and not a loop.
    index = accepted.shape[0] - 1 - np.argmax(accepted[::-1, :], axis=0)
    limit = np.where(accepted.any(axis=0), mu[index], 0.0)
    if background <= 0.0:
        return float(limit[0])
    counts = np.arange(accepted.shape[1])
    weight = np.exp(counts * np.log(background) - background - gammaln(counts + 1.0))
    return float(limit @ weight / weight.sum())


@functools.lru_cache(maxsize=32)
def _large_count_line(confidence_level: float, mu_step: float) -> tuple[float, float]:
    """Straight line in ``sqrt(b)`` the average upper limit follows at large counts.

    Once the Poisson is wide the limit is set by its width, so it grows as the
    root of the background. The two constants are read off the exact
    construction at the top of its range rather than assumed, so the two
    branches meet and the answer stays the same one.

    Parameters
    ----------
    confidence_level : float
        Coverage of the construction.
    mu_step : float
        Spacing of the signal grid [events].

    Returns
    -------
    slope : float
        Growth per unit ``sqrt(b)`` [events].
    intercept : float
        Offset the finite counts leave [events].
    """
    low, high = 0.25 * EXACT_BACKGROUND_MAX, EXACT_BACKGROUND_MAX
    n_low = _exact_sensitivity(low, confidence_level, mu_step)
    n_high = _exact_sensitivity(high, confidence_level, mu_step)
    slope = (n_high - n_low) / (np.sqrt(high) - np.sqrt(low))
    return slope, n_high - slope * np.sqrt(high)


def optimized_window_sensitivity(
    aeff_cm2: np.ndarray,
    background_density: np.ndarray,
    livetime_s: float,
    gamma: float,
    log10_e: np.ndarray,
    pivot_gev: float = PIVOT_ENERGY_GEV,
    confidence_level: float = CONFIDENCE_LEVEL,
    solid_angle_sr: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Best power-law limit over the energy the search is allowed to start at.

    Counting every event in the band is the wrong statistic, and the shape of a
    published sensitivity is what says so. An atmospheric spectrum falls like
    ``E^-3.7`` and an astrophysical one like ``E^-2``, so the events near the
    threshold are almost all background and carry almost no signal. A search
    that counts them anyway pays a penalty largest where its effective area is
    largest, which is the horizon, and the declination dependence it should
    have comes out flattened away.

    Choosing where to start recovers it. Where the response reaches high
    energy the window can be cut above the background entirely and the limit
    returns to the background-free ceiling; where the Earth has absorbed that
    end there is nothing to cut to, and the background is paid in full. Both
    are properties of the response, so the shape is predicted rather than
    tuned. A real unbinned search weights each event instead of cutting, so it
    does better again, and this stays a bound on it.

    Parameters
    ----------
    aeff_cm2 : np.ndarray, shape (log10_e.size, ...)
        Effective area [cm^2] on the energy grid, energy first.
    background_density : np.ndarray
        Atmospheric events per decade, from
        :func:`atmospheric_background_density`, shaped like ``aeff_cm2``.
    livetime_s : float
        Exposure [s].
    gamma : float
        Spectral index of the injected flux.
    log10_e : np.ndarray
        Neutrino energies [log10 GeV].
    pivot_gev : float, optional
        Pivot energy of the quoted flux [GeV].
    confidence_level : float, optional
        Coverage of the construction. See :data:`CONFIDENCE_LEVEL`.
    solid_angle_sr : float or None, optional
        Solid angle of a diffuse signal [sr]. ``None``, the default, gives a
        point-source flux.

    Returns
    -------
    e2_flux : np.ndarray
        ``E^2 phi`` at ``pivot_gev`` [GeV cm^-2 s^-1] at the best window.
    emin_gev : np.ndarray
        Energy that window starts at [GeV].
    """
    aeff = np.asarray(aeff_cm2, dtype=float)
    density = np.asarray(background_density, dtype=float)
    grid = np.asarray(log10_e, dtype=float)
    starts = np.arange(grid.size - 1)

    # The background above every candidate start, for every column at once, so
    # the Feldman-Cousins construction is run once over the whole scan.
    counts = np.stack([np.trapezoid(density[k:], grid[k:], axis=0) for k in starts])
    n_events = sensitivity_upper_limit(counts, confidence_level=confidence_level)
    limits = np.stack(
        [
            power_law_sensitivity(
                aeff[k:], livetime_s, gamma, grid[k:], pivot_gev=pivot_gev,
                n_events=n_events[k], solid_angle_sr=solid_angle_sr,
            )
            for k in starts
        ]
    )
    best = np.argmin(limits, axis=0)
    return np.take_along_axis(limits, best[None], axis=0)[0], 10.0 ** grid[best]
