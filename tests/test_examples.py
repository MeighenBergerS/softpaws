"""Every tutorial in ``examples/`` runs to completion.

The tutorials are the first code a new user runs, so they are checked end to
end: each is executed as a script with a headless plotting backend and must
exit cleanly. The tests are marked slow because together they take about a
minute; pass ``--run-slow`` to include them.
"""

import os
import pathlib
import subprocess
import sys

import pytest

from softpaws.data import dr2_dir

EXAMPLES = sorted((pathlib.Path(__file__).parents[1] / "examples").glob("[0-9][0-9]_*.py"))

#: Tutorials that cannot run without the IceTracks-DR2 release on disk.
NEEDS_DR2 = {"01_load_the_release.py", "06_declination_and_point_sources.py",
             "09_fitting_the_light_reach.py"}


@pytest.mark.slow
@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.stem)
def test_example_runs(script, tmp_path):
    """The tutorial exits with status zero."""
    if script.name in NEEDS_DR2 and not dr2_dir().exists():
        pytest.skip("needs the IceTracks-DR2 release")
    env = {**os.environ, "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=600
    )
    assert result.returncode == 0, result.stderr[-2000:]
