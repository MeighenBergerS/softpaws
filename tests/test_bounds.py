"""Tests for the published-bounds loader in softpaws.data.loader.

The bounds curves are digitized from published figures, so the loader's job is
mostly to impose the ordering the raw files do not have.
"""

import numpy as np
import pytest

from softpaws.data.loader import BOUNDS_DIR, load_dm_line_bounds


def test_load_dm_line_bounds_returns_sorted_columns():
    mass, sigma_v = load_dm_line_bounds("icecubegen2")
    assert mass.shape == sigma_v.shape
    assert np.all(np.diff(mass) > 0)
    assert np.all(sigma_v > 0)


def test_load_dm_line_bounds_sorts_an_unordered_file():
    """The shipped Gen2 file is not in mass order; the loader must fix that."""
    raw = np.loadtxt(BOUNDS_DIR / "icecubegen2_dm_lines.csv", delimiter=",")
    assert np.any(np.diff(raw[:, 0]) < 0), "fixture no longer exercises the sort"
    mass, _ = load_dm_line_bounds("icecubegen2")
    assert mass.size == raw.shape[0]


def test_load_dm_line_bounds_spans_the_expected_mass_range():
    mass, sigma_v = load_dm_line_bounds("icecubegen2")
    # The Gen2 line projection runs from tens of TeV into the EeV decade.
    assert mass[0] < 1.0e5
    assert mass[-1] > 1.0e9
    # A sensitivity curve degrades monotonically with mass.
    assert np.all(np.diff(sigma_v) > 0)


def test_load_dm_line_bounds_reports_a_missing_experiment():
    with pytest.raises(FileNotFoundError):
        load_dm_line_bounds("nosuchexperiment")
