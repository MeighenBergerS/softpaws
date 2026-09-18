"""Tests for the shipped best-fit standards and for rebuilding them."""

import json
import pathlib

import pytest

from softpaws import standards
from softpaws.constants import GeV, m

SITES = {"IceCube", "ARCA230", "P-ONE", "TRIDENT"}
FIT = (
    pathlib.Path(__file__).parents[1]
    / "scripts" / "2026_muon_transport" / "output" / "77_reduced_fit_sigma05.json"
)


@pytest.fixture(autouse=True)
def shipped():
    """Every test starts and ends on the shipped standards."""
    standards.use()
    yield
    standards.use()


def test_every_site_has_both_numbers():
    assert set(standards.THRESHOLD) == set(standards.REACH) == SITES


def test_medians_sit_inside_their_68_range():
    for site in SITES:
        lo, hi = standards.THRESHOLD_68[site]
        assert lo <= standards.THRESHOLD[site] <= hi
        lo, hi = standards.REACH_68[site]
        assert lo <= standards.REACH[site] <= hi


def test_values_are_in_base_units():
    assert standards.REACH["ARCA230"] / m == pytest.approx(44.5, abs=0.05)
    assert standards.THRESHOLD["IceCube"] / GeV == pytest.approx(10**3.550, rel=1e-3)


def test_source_names_the_fit():
    assert standards.SOURCE["fit"] == "77_reduced_fit_sigma05.json"
    assert standards.FIXED == {"b_scale": 1.0, "cross_section_slope": 0.4538}


@pytest.mark.skipif(not FIT.exists(), reason="the paper's fit output is not in this checkout")
def test_rebuild_reproduces_the_shipped_file(tmp_path):
    out = tmp_path / "standards.json"
    rebuilt = standards.make_standards(FIT, out)
    shipped = json.loads(standards.SHIPPED.read_text())
    assert rebuilt["sites"] == shipped["sites"]
    assert rebuilt["fixed"] == shipped["fixed"]


def test_use_switches_and_names_imported_earlier_follow(tmp_path):
    from softpaws.standards import REACH

    data = json.loads(standards.SHIPPED.read_text())
    data["sites"]["ARCA230"]["reach_m"] = [1.0, 2.0, 3.0]
    mine = tmp_path / "mine.json"
    mine.write_text(json.dumps(data))
    standards.use(mine)
    assert REACH["ARCA230"] == pytest.approx(2.0 * m)
    assert standards.SOURCE["file"] == str(mine)


def test_use_rejects_a_file_that_is_not_standards(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"sites": {}}))
    with pytest.raises(ValueError, match="not a standards file"):
        standards.use(bad)


def test_make_standards_rejects_a_fit_without_the_two_number_result(tmp_path):
    fit = tmp_path / "fit.json"
    fit.write_text(json.dumps({"IceCube": {"five_param": {}}}))
    with pytest.raises(ValueError, match="no '2-param"):
        standards.make_standards(fit, tmp_path / "out.json")
