r"""Loader for the IceCube HESE 7.5-year data release.

The release (`doi:10.21234/4EQJ-BB17
<https://icecube.wisc.edu/data-releases/2021/12/hese-7-5-year-data/>`_, described
in Ref. [1]_) is not a tabulated instrument response. It ships the Monte Carlo
events behind the analysis, one row per simulated event, and the effective area
is something the user computes from them. That is an advantage here: because
each row carries its own true flavour, true energy and **true zenith**, an
effective area can be built resolved in any of the three, where a released table
would already have integrated over them.

The recipe is the one in the release's own ``plot_effective_areas.py``:

.. math:: A_{\\rm eff}(E) = \\frac{1}{\\Delta E \\, \\Delta\\Omega}
    \\sum_{i \\,\\in\\, \\rm bin} w_i,

with ``w_i = weightOverFluxOverLivetime`` in GeV sr cm^2. The sum runs over
whichever subset of simulated events the question asks for, so selecting on
flavour, interaction type and zenith is exactly how the response is sliced.

Two conventions carried from the release, both of which change numbers:

* The HESE selection bound is a cut on **reconstructed deposited energy**,
  ``60 TeV`` in the analysis, applied in the release's ``data_loader.py`` and
  not in its effective-area script. :func:`load_hese_mc` applies it by default,
  so the areas here describe the analysis sample. Pass
  ``deposited_min_gev=None`` to reproduce the release's Fig. 33 convention.
* An area quoted for a flavour averages neutrino and antineutrino, which is the
  factor of ``0.5`` in :func:`effective_area_cm2`. The published Fig. 33 plots
  their sum instead.

The raw json is bulk release data and is not committed; see the module
docstring of :mod:`softpaws.data.loader` for the same convention on DR2.

References
----------
.. [1] IceCube Collaboration, "The IceCube high-energy starting event sample:
   Description and flux characterization with 7.5 years of data",
   Phys. Rev. D 104 (2021) 022002, arXiv:2011.03545.
"""

import json
import pathlib

import numpy as np

from .paths import hese_dir, require

#: PDG codes of the neutrino flavours, keyed as the rest of the package keys them.
FLAVOUR_PDG = {"e": 12, "mu": 14, "tau": 16}

#: Interaction type codes used by the release.
INTERACTION_CODE = {"cc": 1, "nc": 2, "gr": 3}

#: Fields taken from each json file.
_TRUTH_FIELDS = (
    "primaryType",
    "primaryZenith",
    "primaryEnergy",
    "weightOverFluxOverLivetime",
    "interactionType",
)
_OBSERVABLE_FIELDS = ("recoDepositedEnergy",)


def load_hese_mc(
    data_dir: pathlib.Path | None = None,
    deposited_min_gev: float | None = 6.0e4,
    deposited_max_gev: float | None = 1.0e7,
    use_cache: bool = True,
) -> dict[str, np.ndarray]:
    """Load the HESE Monte Carlo events.

    Parameters
    ----------
    data_dir : pathlib.Path or None, optional
        Directory holding ``HESE_mc_truth.json`` and
        ``HESE_mc_observable.json``. Defaults to ``softpaws/data/hese``.
    deposited_min_gev : float or None, optional
        Lower bound on reconstructed deposited energy [GeV]. Defaults to the
        analysis value of ``6e4``. ``None`` applies no cut.
    deposited_max_gev : float or None, optional
        Upper bound on reconstructed deposited energy [GeV]. Defaults to
        ``1e7``. ``None`` applies no cut.
    use_cache : bool, optional
        Write and reuse a compressed ``.npz`` beside the json, since parsing
        88 MB of json takes far longer than the calculation that follows.
        Defaults to ``True``.

    Returns
    -------
    mc : dict of str -> np.ndarray
        Keyed by the release's field names, each of length equal to the number
        of surviving events.

    Raises
    ------
    FileNotFoundError
        If the release json is not present.
    """
    directory = pathlib.Path(data_dir) if data_dir is not None else hese_dir()
    require(directory, "HESE 7.5-year release")
    cache = directory / "_hese_mc_cache.npz"

    if use_cache and cache.is_file():
        with np.load(cache) as handle:
            arrays = {key: handle[key] for key in handle.files}
    else:
        arrays = {}
        for name, fields in (
            ("HESE_mc_truth.json", _TRUTH_FIELDS),
            ("HESE_mc_observable.json", _OBSERVABLE_FIELDS),
        ):
            path = directory / name
            if not path.is_file():
                raise FileNotFoundError(
                    f"HESE release file not found: {path}. Download "
                    "https://icecube.wisc.edu/data-releases/20211217_HESE-7-5-year-data.zip "
                    f"and place its resources/data/*.json in {directory}."
                )
            with path.open() as handle:
                contents = json.load(handle)
            for field in fields:
                arrays[field] = np.asarray(contents[field], dtype=float)
        if use_cache:
            np.savez_compressed(cache, **arrays)

    keep = np.ones(arrays["primaryEnergy"].shape, dtype=bool)
    if deposited_min_gev is not None:
        keep &= arrays["recoDepositedEnergy"] >= deposited_min_gev
    if deposited_max_gev is not None:
        keep &= arrays["recoDepositedEnergy"] <= deposited_max_gev
    return {key: value[keep] for key, value in arrays.items()}


def effective_area_cm2(
    mc: dict[str, np.ndarray],
    energy_bins_gev: np.ndarray,
    flavour: str | None = None,
    interaction: str | None = "cc",
    cos_zenith_range: tuple[float, float] = (-1.0, 1.0),
    average_nu_nubar: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Effective area from the Monte Carlo, sliced as asked.

    Parameters
    ----------
    mc : dict of str -> np.ndarray
        From :func:`load_hese_mc`.
    energy_bins_gev : np.ndarray
        Bin edges in true neutrino energy [GeV].
    flavour : {"e", "mu", "tau"} or None, optional
        Neutrino flavour, taken on ``|primaryType|`` so both signs are kept.
        ``None`` sums every flavour. Defaults to ``None``.
    interaction : {"cc", "nc", "gr"} or None, optional
        Interaction channel. ``None`` sums every channel. Defaults to ``"cc"``.
    cos_zenith_range : tuple of float, optional
        Half-open band in ``cos(primaryZenith)``. IceCube's convention puts
        ``+1`` straight down through the atmosphere and ``-1`` straight up
        through the Earth. Defaults to the whole sky.
    average_nu_nubar : bool, optional
        Divide by two, giving the area per neutrino averaged over neutrino and
        antineutrino. This is the release's own script's convention, and the
        published Fig. 33 uses the sum instead. Defaults to ``True``.

    Returns
    -------
    aeff_cm2 : np.ndarray
        Effective area [cm^2], one entry per energy bin.
    aeff_error_cm2 : np.ndarray
        Monte Carlo statistical error on the same [cm^2].
    """
    cos_zenith = np.cos(mc["primaryZenith"])
    low, high = cos_zenith_range
    keep = (cos_zenith >= low) & (cos_zenith < high)
    if flavour is not None:
        keep &= np.abs(mc["primaryType"]) == FLAVOUR_PDG[flavour]
    if interaction is not None:
        keep &= mc["interactionType"] == INTERACTION_CODE[interaction]

    weight = mc["weightOverFluxOverLivetime"][keep]
    energy = mc["primaryEnergy"][keep]

    # A bin's solid angle is the band's, since the release weights already carry
    # the sr in their units.
    solid_angle_sr = 2.0 * np.pi * (high - low)
    bin_widths = np.diff(energy_bins_gev) * solid_angle_sr

    index = np.digitize(energy, energy_bins_gev) - 1
    valid = (index >= 0) & (index < bin_widths.size)
    n_bins = bin_widths.size
    total = np.bincount(index[valid], weights=weight[valid], minlength=n_bins)
    total_sq = np.bincount(index[valid], weights=weight[valid] ** 2, minlength=n_bins)

    factor = 0.5 if average_nu_nubar else 1.0
    return total / bin_widths * factor, np.sqrt(total_sq) / bin_widths * factor


def load_hese_data(
    data_dir: pathlib.Path | None = None,
    deposited_min_gev: float | None = 6.0e4,
    deposited_max_gev: float | None = 1.0e7,
) -> dict[str, np.ndarray]:
    """Load the 102 observed HESE events.

    Parameters
    ----------
    data_dir : pathlib.Path or None, optional
        Directory holding ``HESE_data.json``. Defaults to
        ``softpaws/data/hese``.
    deposited_min_gev : float or None, optional
        Lower bound on reconstructed deposited energy [GeV]. Defaults to
        ``6e4``.
    deposited_max_gev : float or None, optional
        Upper bound on reconstructed deposited energy [GeV]. Defaults to
        ``1e7``.

    Returns
    -------
    events : dict of str -> np.ndarray
        Keyed ``recoDepositedEnergy`` [GeV], ``recoMorphology`` (0 cascade,
        1 track, 2 double cascade), ``recoZenith`` [rad] and ``recoLength`` [m].
    """
    directory = pathlib.Path(data_dir) if data_dir is not None else hese_dir()
    require(directory, "HESE 7.5-year release")
    with (directory / "HESE_data.json").open() as handle:
        contents = json.load(handle)
    arrays = {key: np.asarray(value, dtype=float) for key, value in contents.items()}

    keep = np.ones(arrays["recoDepositedEnergy"].shape, dtype=bool)
    if deposited_min_gev is not None:
        keep &= arrays["recoDepositedEnergy"] >= deposited_min_gev
    if deposited_max_gev is not None:
        keep &= arrays["recoDepositedEnergy"] <= deposited_max_gev
    return {key: value[keep] for key, value in arrays.items()}
