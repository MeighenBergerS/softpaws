"""Checks for the design rules in ``.claude/skills/design-*`` that code can enforce.

Rules that existing code still breaks are ratchets: the known offenders are
listed, a new offender fails, and a fixed one must be removed from its list.
Where the list would be long, the ratchet holds a count instead, which must
be lowered as offenders are fixed.
"""

import ast
import importlib
import inspect
import pathlib
import re

import pytest

import softpaws

SUBPACKAGES = ("transport", "detectors", "fluxes", "response", "comparison", "data")
PACKAGE_DIR = pathlib.Path(softpaws.__file__).parent

#: Where public names are looked for: the top level (where ``Detector`` is
#: exported), the two top-level modules, and the subpackages.
PUBLIC_MODULES = (
    "softpaws",
    "softpaws.constants",
    "softpaws.standards",
    *(f"softpaws.{sub}" for sub in SUBPACKAGES),
)

#: N4: a unit suffix at the end of a name or argument.
UNIT_SUFFIX = re.compile(
    r"_(gev|tev|pev|km|km2|km3|m|m2|m3|cm|cm2|cm3|g_cm2|g_cm3|s|sr|deg|rad|nm|pe|per_km3)$"
)

#: Known N4 offenders: public names plus ``name(argument)`` pairs. Lower it as they go.
SUFFIXED_COUNT = 296

#: A6: where literal numbers may live besides ``constants.py``. The paper's
#: tuning under ``_paper/`` is exempt too.
A6_EXEMPT = {
    "constants.py",
    "standards.py",
    "detectors/sites.py",
    "detectors/optics.py",
}

#: Known A6 offenders: literal numbers defined anywhere else. Lower it as they go.
LOOSE_NUMBER_COUNT = 0

#: A1: the top level holds the names a typical analysis needs, and no more.
MAX_TOP_LEVEL = 20

#: D1: first docstring line at most this long.
MAX_SUMMARY_CHARS = 75

#: N5: project jargon that does not belong in a public name.
JARGON = ("ladder", "halo", "ceiling", "rung")

#: Known D1 offenders. Shorten one, then delete it here.
LONG_SUMMARY = {
    "softpaws.data.published.arca230_angle_dependent_aeff",
    "softpaws.data.published.arca230_quoted_fit",
    "softpaws.fluxes.astrophysical.BrokenPowerLawFit.shape",
    "softpaws.response.effective_area.required_footprint_radius_km",
    "softpaws.response.effective_area.truncated_range_km",
    "softpaws.response.first_principles.detector_curves",
    "softpaws.response.first_principles.residuals",
    "softpaws.response.irfs.SmearingMatrix.psf_containment_deg",
    "softpaws.response.light_reach.hit_count",
    "softpaws.response.sensitivity.psf_bin_radius_deg",
    "softpaws.response.soft_volume.SoftVolumeResponse.expected_counts_coupled_attenuation",
    "softpaws.transport.cross_section.PowerLawCrossSection",
}

#: Known N5 offenders. Rename or make private, then delete it here.
JARGON_NAMES: set[str] = set()

#: N6: a detector or dataset name as a word of a public name.
SITE_WORD = re.compile(
    r"(^|_)(icecube|ic|ic86|arca|arca21|arca230|trident|pone|gvd|gen2|km3net|hese|dr2)(_|$)"
    r"|^km3_"
)

#: Known N6 offenders outside ``data``. Lower it as they collapse into ``Detector``.
SITE_NAME_COUNT = 14


def public_objects():
    """Every public function, class and method, keyed by where it is defined.

    A name exported from several places is counted once, and the public
    methods of a public class count as public names of their own.
    """
    found = {}
    for path in PUBLIC_MODULES:
        module = importlib.import_module(path)
        names = getattr(module, "__all__", None)
        if names is None:
            names = [n for n in dir(module) if not n.startswith("_")]
        for name in names:
            obj = getattr(module, name, None)
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            if not obj.__module__.startswith("softpaws"):
                continue
            key = f"{obj.__module__}.{obj.__qualname__}"
            found[key] = obj
            if inspect.isclass(obj):
                for attr, member in vars(obj).items():
                    if not attr.startswith("_") and inspect.isfunction(member):
                        found[f"{key}.{attr}"] = member
    return found


def check_ratchet(offenders, known):
    """Fail on a new offender, and on a known one that has been fixed."""
    assert not offenders - known, f"New offenders: {sorted(offenders - known)}"
    assert not known - offenders, f"Fixed, remove from the list: {sorted(known - offenders)}"


def check_count(found, known):
    """Fail if the count grew, and if it shrank without the baseline being lowered."""
    assert found <= known, f"{found - known} new offenders; see the rule"
    assert found == known, f"Fixed some, lower the baseline from {known} to {found}"


def is_literal(node):
    """Whether a value is a literal number, or arithmetic on literals and constants.

    Values computed by package code (a call into softpaws, an attribute of a
    record, an array) are results, not constants, and are left out (A6).
    """
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.Name):
        return node.id.isupper()
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name) and node.value.id == "np"
    if isinstance(node, ast.UnaryOp):
        return is_literal(node.operand)
    if isinstance(node, ast.BinOp):
        return is_literal(node.left) and is_literal(node.right)
    if isinstance(node, ast.Call) and not node.keywords:
        func = node.func
        known = (isinstance(func, ast.Name) and func.id in ("float", "int")) or (
            isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
            and func.value.id == "np"
        )
        return known and all(is_literal(arg) for arg in node.args)
    return False


def has_literal(node):
    """A literal that contains at least one number, so a bare alias is not counted."""
    return is_literal(node) and any(
        isinstance(n, ast.Constant) or isinstance(n, ast.Attribute) for n in ast.walk(node)
    )


def test_no_unit_suffixes():
    """N4: public names and their arguments carry no unit suffix."""
    offenders = set()
    for name, obj in public_objects().items():
        if UNIT_SUFFIX.search(name.lower()):
            offenders.add(name)
        try:
            arguments = inspect.signature(obj).parameters
        except (TypeError, ValueError):
            continue
        offenders |= {f"{name}({arg})" for arg in arguments if UNIT_SUFFIX.search(arg)}
    check_count(len(offenders), SUFFIXED_COUNT)


def test_numbers_live_in_constants():
    """A6: literal named numbers are defined in ``constants.py`` only."""
    found = 0
    for path in PACKAGE_DIR.rglob("*.py"):
        relative = path.relative_to(PACKAGE_DIR).as_posix()
        if relative in A6_EXEMPT or relative.startswith("_paper/"):
            continue
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign):
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                target, value = node.target, node.value
            else:
                continue
            if isinstance(target, ast.Name) and target.id.isupper() and has_literal(value):
                found += 1
    check_count(found, LOOSE_NUMBER_COUNT)


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


def test_no_detector_names_outside_data():
    """N6: outside ``data``, the detector is an argument, never part of a name."""
    offenders = [
        name for name in public_objects()
        if not name.startswith("softpaws.data.")
        and SITE_WORD.search(name.rsplit(".", 1)[1].lower())
    ]
    check_count(len(offenders), SITE_NAME_COUNT)
