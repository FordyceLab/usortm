"""The live page's script has to parse, or the page shows nothing at all.

A redeclared identifier is a parse error, and a parse error takes down the
whole script block rather than the one line that caused it.  The page then sits
on its placeholder text with no indication that anything is wrong -- the run
proceeds normally and only the dashboard is dead, which is a hard failure to
attribute.  ``var at`` inside the histogram axis hoisted to the top of
``render()`` and collided with a ``const at`` further down, exactly that way.
"""
import re
import shutil
import subprocess

import pytest

from usortm.demux.live import _PAGE


def _script_body(page: str) -> str:
    """The page's inline script, concatenated."""
    return "\n".join(re.findall(r"<script>(.*?)</script>", page, re.S))


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_script_parses():
    """Parse the real thing, when a JS engine is available to do it."""
    body = _script_body(_PAGE)
    proc = subprocess.run(
        ["node", "--check", "-"], input=body, capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        f"live page script does not parse:\n{proc.stderr.strip()}"
    )


def test_the_page_defines_every_element_the_script_writes_to():
    """A missing id throws at run time, which stops render() part-written."""
    page = _PAGE
    body = _script_body(page)
    wanted = set(re.findall(r'getElementById\("([^"]+)"\)', body))
    present = set(re.findall(r'id="([^"]+)"', page))
    missing = wanted - present
    assert not missing, f"script writes to absent element(s): {sorted(missing)}"
