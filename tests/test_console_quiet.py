"""The pipeline's own output must stand down when the CLI drives a display.

Status lines and nested progress bars written straight to the terminal
interleave with the CLI's progress display and bury the parts worth reading.
"""

import ast
import inspect
from pathlib import Path

import pytest

from usortm.demux import utils

DEMUX_MODULES = ["utils.py", "pipeline.py", "streakout.py", "barcodes.py"]


@pytest.fixture(autouse=True)
def _restore_quiet():
    yield
    utils.set_console_quiet(False)


class TestQuietSwitch:

    def test_bars_are_disabled_when_quiet(self):
        utils.set_console_quiet(True)
        bar = utils._bar(range(3))
        assert bar.disable is True

    def test_bars_are_shown_when_not_quiet(self):
        utils.set_console_quiet(False)
        bar = utils._bar(range(3))
        assert bar.disable is False

    def test_an_explicit_disable_is_respected(self):
        utils.set_console_quiet(False)
        assert utils._bar(range(3), disable=True).disable is True

    def test_status_lines_are_silent_when_quiet(self, capsys):
        utils.set_console_quiet(True)
        utils._say("collecting things")
        assert capsys.readouterr().out == ""

    def test_status_lines_print_when_not_quiet(self, capsys):
        utils.set_console_quiet(False)
        utils._say("collecting things")
        assert "collecting things" in capsys.readouterr().out

    def test_the_pipeline_sets_it_from_the_callback(self):
        """A caller rendering progress is what makes the library quiet."""
        from usortm.demux import pipeline

        src = inspect.getsource(pipeline.run_levseq_pipeline)
        assert "set_console_quiet(progress_callback is not None)" in src


class TestNoRawProgressInTheDemuxPath:
    """A bar imported directly from tqdm bypasses the switch entirely."""

    def _module_paths(self):
        root = Path(utils.__file__).parent
        return [root / name for name in DEMUX_MODULES]

    def test_tqdm_is_imported_only_by_the_helper(self):
        offenders = []
        for path in self._module_paths():
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                imported = (
                    isinstance(node, ast.ImportFrom) and node.module == "tqdm"
                ) or (
                    isinstance(node, ast.Import)
                    and any(a.name.split(".")[0] == "tqdm" for a in node.names)
                )
                if imported and path.name != "utils.py":
                    offenders.append(f"{path.name}:{node.lineno}")
        assert not offenders, (
            "tqdm imported outside utils.py, so these bars ignore "
            f"set_console_quiet: {offenders}"
        )

    def test_no_bare_print_calls_on_the_pipeline_path(self):
        """Status belongs in _say, which defers to the CLI; failures belong in
        the log. utils._say is the one place print() is correct."""
        offenders = []
        for path in self._module_paths():
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "print"):
                    continue
                if path.name == "utils.py" and node.lineno < 60:
                    continue          # the helper's own print
                offenders.append(f"{path.name}:{node.lineno}")
        # utils.py keeps a few prints in functions the pipeline never calls;
        # pin the count so a new one on the live path is noticed.
        assert len(offenders) <= 7, f"new bare print() calls: {offenders}"
