"""Current best-fit values, so a user never has to run a fit first.

Each site has two numbers its instrument sets: the muon threshold of its
selection and its light reach, given as a median with a 68% range. They ship
in ``data/standards.json`` with the fit they come from. :func:`make_standards`
rebuilds such a file from a fit result, and :func:`use` switches to one. The
``SOFTPAWS_STANDARDS`` environment variable does the same at import.

>>> from softpaws import standards
>>> from softpaws.constants import m
>>> round(standards.REACH["ARCA230"] / m, 1)
44.5
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib

from .constants import GeV, km, m

__all__ = [
    "EFFICIENCY",
    "FIT_PARAMETERS",
    "FIXED",
    "REACH",
    "REACH_68",
    "SHIPPED",
    "SOURCE",
    "THRESHOLD",
    "THRESHOLD_68",
    "make_standards",
    "use",
]

#: The standards file that ships with softpaws.
SHIPPED = pathlib.Path(__file__).parent / "data" / "standards.json"

#: Order of the parameters in a fit result's ``best`` vector.
FIT_PARAMETERS = ("eps_0", "log10_e_thr", "b_scale", "lam", "reach_km")

#: Key of the two-number fit in a fit result, per site.
TWO_NUMBER_FIT = "2-param (E_thr, Lambda)"

#: Muon threshold of each site's selection, median [energy].
THRESHOLD: dict[str, float] = {}

#: 68% range of the threshold, ``(low, high)`` [energy].
THRESHOLD_68: dict[str, tuple[float, float]] = {}

#: Light reach of each site, median [length].
REACH: dict[str, float] = {}

#: 68% range of the reach, ``(low, high)`` [length].
REACH_68: dict[str, tuple[float, float]] = {}

#: Selection efficiency of each site, held fixed in the fit.
EFFICIENCY: dict[str, float] = {}

#: Physics the fit held fixed: ``b_scale`` and ``cross_section_slope``.
FIXED: dict[str, float] = {}

#: Where the loaded values come from: the fit result, the date, the file.
SOURCE: dict[str, str] = {}


def use(path: str | os.PathLike | None = None) -> None:
    """Load the standards from ``path``, or the shipped ones for ``None``.

    The module's dictionaries are updated in place, so names imported from
    here before the call see the new values.

    Parameters
    ----------
    path : str or os.PathLike or None, optional
        A file written by :func:`make_standards`. ``None`` loads :data:`SHIPPED`.

    Raises
    ------
    ValueError
        Raised if the file is not a standards file.

    Examples
    --------
    >>> use("my_standards.json")  # doctest: +SKIP
    >>> use()  # back to the shipped values
    """
    path = pathlib.Path(path) if path is not None else SHIPPED
    data = json.loads(path.read_text())
    try:
        sites, fixed, source = data["sites"], data["fixed"], data["source"]
        loaded = {
            name: (site["threshold_gev"], site["reach_m"], site["efficiency"])
            for name, site in sites.items()
        }
    except (KeyError, TypeError) as missing:
        raise ValueError(f"{path} is not a standards file: no {missing}.") from None
    for table in (THRESHOLD, THRESHOLD_68, REACH, REACH_68, EFFICIENCY, FIXED, SOURCE):
        table.clear()
    for name, ((t_lo, t_mid, t_hi), (r_lo, r_mid, r_hi), efficiency) in loaded.items():
        THRESHOLD[name], THRESHOLD_68[name] = t_mid * GeV, (t_lo * GeV, t_hi * GeV)
        REACH[name], REACH_68[name] = r_mid * m, (r_lo * m, r_hi * m)
        EFFICIENCY[name] = efficiency
    FIXED.update(fixed)
    SOURCE.update(source, file=str(path))


def make_standards(
    fit_result: str | os.PathLike, out: str | os.PathLike, script: str = ""
) -> dict:
    """Write a standards file from a two-number fit result, and return it.

    The fit result is the JSON the two-number response fit writes: per site,
    a ``best`` vector ordered as :data:`FIT_PARAMETERS` and the 16th, 50th and
    84th percentiles of ``log10_e_thr`` and ``reach_km``.

    Parameters
    ----------
    fit_result : str or os.PathLike
        The fit's JSON output.
    out : str or os.PathLike
        Where to write the standards file.
    script : str, optional
        The command that produced the fit, recorded in the file's source.

    Returns
    -------
    standards : dict
        What was written.

    Raises
    ------
    ValueError
        Raised if a site has no two-number fit, or if the sites were fitted
        with different fixed physics.

    Examples
    --------
    >>> make_standards("77_reduced_fit_sigma05.json", "mine.json")  # doctest: +SKIP
    >>> use("mine.json")  # doctest: +SKIP
    """
    fit_result = pathlib.Path(fit_result)
    sites, fixed = {}, None
    for name, entry in json.loads(fit_result.read_text()).items():
        if TWO_NUMBER_FIT not in entry:
            raise ValueError(f"{fit_result}: {name} has no {TWO_NUMBER_FIT!r} result.")
        best = dict(zip(FIT_PARAMETERS, entry[TWO_NUMBER_FIT]["best"]))
        quantiles = entry[TWO_NUMBER_FIT]["quantiles"]
        site_fixed = {"b_scale": best["b_scale"], "cross_section_slope": best["lam"]}
        if fixed not in (None, site_fixed):
            raise ValueError(f"{fit_result}: {name} was fitted with other fixed physics.")
        fixed = site_fixed
        sites[name] = {
            "efficiency": best["eps_0"],
            "threshold_gev": [10.0**q for q in quantiles["log10_e_thr"]],
            "reach_m": [q * km / m for q in quantiles["reach_km"]],
        }
    standards = {
        "source": {
            "fit": fit_result.name,
            "script": script,
            "made": datetime.date.today().isoformat(),
        },
        "fixed": fixed,
        "sites": sites,
    }
    pathlib.Path(out).write_text(json.dumps(standards, indent=2) + "\n")
    return standards


use(os.environ.get("SOFTPAWS_STANDARDS"))
