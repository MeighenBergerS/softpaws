"""Where the large releases live.

The tables that ship with the package are found relative to this file. The
IceCube releases are not shipped: the IceTracks-DR2 release is 2.5 GB and
the HESE 7.5-year Monte Carlo 90 MB. Download them yourself and either place
them under the package's ``data`` directory or point the environment
variable ``SOFTPAWS_DATA_DIR`` at a directory with the same layout::

    <SOFTPAWS_DATA_DIR>/
        dataverse_files/      IceTracks-DR2, DOI 10.7910/DVN/MMIIZA
            events/           <season>_exp.csv
            irfs/             <season>_effectiveArea.csv, <season>_smearing.csv
            uptime/           <season>_exp.csv
        hese/                 HESE 7.5-year release
            HESE_data.json, HESE_mc_observable.json, HESE_mc_truth.json
"""

from __future__ import annotations

import os
import pathlib

__all__ = ["DR2_DOI", "DATA_DIR_VARIABLE", "data_root", "dr2_dir", "hese_dir", "require"]

#: Environment variable that overrides the data root.
DATA_DIR_VARIABLE = "SOFTPAWS_DATA_DIR"

#: DOI of the IceTracks-DR2 release.
DR2_DOI = "10.7910/DVN/MMIIZA"

_PACKAGE_DATA = pathlib.Path(__file__).parent


def data_root() -> pathlib.Path:
    """Directory the releases are read from.

    Returns
    -------
    root : pathlib.Path
        ``$SOFTPAWS_DATA_DIR`` when set, else the package's ``data``
        directory.
    """
    override = os.environ.get(DATA_DIR_VARIABLE)
    return pathlib.Path(override).expanduser() if override else _PACKAGE_DATA


def dr2_dir() -> pathlib.Path:
    """Root of the IceTracks-DR2 release, ``<data_root>/dataverse_files``."""
    return data_root() / "dataverse_files"


def hese_dir() -> pathlib.Path:
    """Directory of the HESE 7.5-year release, ``<data_root>/hese``."""
    return data_root() / "hese"


def require(path: str | pathlib.Path, what: str) -> pathlib.Path:
    """Return ``path`` if it exists, else raise an error that says where to get it.

    Parameters
    ----------
    path : str or pathlib.Path
        File or directory that must exist.
    what : str
        Short name of the release, for the message.

    Returns
    -------
    path : pathlib.Path
        The same path.

    Raises
    ------
    FileNotFoundError
        Raised if ``path`` does not exist, with the DOI and the expected
        layout in the message.
    """
    path = pathlib.Path(path)
    if path.exists():
        return path
    raise FileNotFoundError(
        f"{path} not found. The {what} is not shipped with softpaws: download it "
        f"(IceTracks-DR2: DOI {DR2_DOI}; HESE 7.5-year: the IceCube data releases page) "
        f"and place it under {data_root()} or set {DATA_DIR_VARIABLE} to a directory with "
        "the layout described in softpaws.data.paths."
    )
