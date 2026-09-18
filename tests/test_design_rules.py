"""Checks for the design rules in ``.claude/skills/design-*`` that code can enforce.

Rules that existing code still breaks are ratchets: the known offenders are
listed, a new offender fails, and a fixed one must be removed from its list.
"""

import importlib
import inspect

import pytest

import softpaws

SUBPACKAGES = ("transport", "detectors", "fluxes", "response", "comparison", "data")

#: A1: the top level holds the names a typical analysis needs, and no more.
MAX_TOP_LEVEL = 20

#: D1: first docstring line at most this long.
MAX_SUMMARY_CHARS = 75

#: N5: project jargon that does not belong in a public name.
JARGON = ("ladder", "halo", "ceiling", "rung")

#: Known D1 offenders. Shorten one, then delete it here.
LONG_SUMMARY = {
    "softpaws.data.arca230_angle_dependent_aeff",
    "softpaws.data.arca230_quoted_fit",
    "softpaws.response.detector_curves",
    "softpaws.response.hit_count",
    "softpaws.response.psf_bin_radius_deg",
    "softpaws.response.required_footprint_radius_km",
    "softpaws.response.residuals",
    "softpaws.response.truncated_range_km",
    "softpaws.transport.PowerLawCrossSection",
}

#: Known N5 offenders. Rename or make private, then delete it here.
JARGON_NAMES = {
    "softpaws.response.arca_ladders",
    "softpaws.response.icecube_ladders",
    "softpaws.response.water_ladders",
}


def public_objects():
    """Every public function and class, keyed by its dotted name."""
    found = {}
    for sub in SUBPACKAGES:
        module = importlib.import_module(f"softpaws.{sub}")
        for name in module.__all__:
            obj = getattr(module, name)
            if inspect.isfunction(obj) or inspect.isclass(obj):
                found[f"softpaws.{sub}.{name}"] = obj
    return found


def check_ratchet(offenders, known):
    """Fail on a new offender, and on a known one that has been fixed."""
    assert not offenders - known, f"New offenders: {sorted(offenders - known)}"
    assert not known - offenders, f"Fixed, remove from the list: {sorted(known - offenders)}"


def test_top_level_is_small():
    """A1: the top-level namespace stays a short list."""
    assert len(getattr(softpaws, "__all__", [])) <= MAX_TOP_LEVEL


@pytest.mark.parametrize("name, obj", sorted(public_objects().items()))
def test_public_names_have_a_summary(name, obj):
    """D1: every public name has a one-line summary."""
    assert (inspect.getdoc(obj) or "").strip(), f"{name} has no docstring"


def test_summaries_are_short():
    """D1: summary lines fit in MAX_SUMMARY_CHARS."""
    offenders = {
        name for name, obj in public_objects().items()
        if len((inspect.getdoc(obj) or "").split("\n")[0]) > MAX_SUMMARY_CHARS
    }
    check_ratchet(offenders, LONG_SUMMARY)


def test_no_jargon_in_public_names():
    """N5: public names avoid project jargon."""
    offenders = {
        name for name in public_objects()
        if any(word in name.rsplit(".", 1)[1].lower() for word in JARGON)
    }
    check_ratchet(offenders, JARGON_NAMES)
