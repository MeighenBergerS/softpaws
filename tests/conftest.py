"""Shared test configuration.

Tests run from the repository root so that the relative paths in the example
scripts and the shipped data resolve, and the slow markers are opt-in.
"""

import os
import pathlib

import pytest

ROOT = pathlib.Path(__file__).parents[1]


def pytest_configure(config):
    """Run from the repository root and keep the numeric libraries single-threaded.

    Several tests build small matrices in a loop, where a threaded BLAS spends
    more time synchronizing than computing.
    """
    os.chdir(ROOT)
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ.setdefault(variable, "1")


def pytest_addoption(parser):
    """Add ``--run-slow``, which enables the tests marked ``slow``."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run the slow tests: the Monte-Carlo cross-checks and the pinned "
        "effective areas.",
    )


def pytest_collection_modifyitems(config, items):
    """Skip the ``slow`` tests unless ``--run-slow`` is given."""
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="slow test; pass --run-slow to enable")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
