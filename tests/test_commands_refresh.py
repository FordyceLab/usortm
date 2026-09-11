"""commands.txt is rewritten by the steps that record commands.

It used to be written only by ``usortm methods``.  A project went a full day
of demuxes, picks and a merge with the file still dated the day before, and
the commands a methods section needs were in the state file where nobody
reads them.  Each recording step now refreshes the file as it saves.
"""
import inspect

from usortm import provenance
from usortm.cli import demux_cmd, merge, pick, reorder


def _project():
    return {
        "created": "2026-09-10T00:00:00",
        "workflow_steps": {
            "demux": {"completed": True, "timestamp": "2026-09-10T15:30:08",
                      "command": "usortm demux proj/ --workers 8"},
            "pick": {"completed": True, "timestamp": "2026-09-10T16:57:00",
                     "command": "usortm pick proj --tier C --max-disagreement 0.10"},
        },
    }


def test_refresh_writes_the_file_with_the_recorded_commands(tmp_path):
    path = provenance.refresh_commands(_project(), tmp_path)
    assert path == tmp_path / provenance.COMMANDS_FILE
    text = path.read_text()
    assert "usortm demux proj/ --workers 8" in text
    assert "usortm pick proj --tier C --max-disagreement 0.10" in text


def test_refresh_overwrites_a_stale_file(tmp_path):
    stale = tmp_path / provenance.COMMANDS_FILE
    stale.write_text("# yesterday\n")
    provenance.refresh_commands(_project(), tmp_path)
    assert "yesterday" not in stale.read_text()
    assert "usortm pick" in stale.read_text()


def test_a_failure_to_write_does_not_raise(tmp_path):
    """The state file is the record and has already been saved; the
    rendering must not take the step down with it."""
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("")
    assert provenance.refresh_commands(_project(), blocked) is None


def test_every_recording_step_refreshes_the_file():
    """Each module that records a command calls the refresh after saving.

    A source check rather than a run of each command: the commands need a
    demultiplexed project, and what is being asserted is only that the call
    is there.
    """
    for mod in (demux_cmd, pick, merge, reorder):
        src = inspect.getsource(mod)
        assert "_provenance.current_command()" in src, mod.__name__
        assert "_provenance.refresh_commands(" in src, (
            f"{mod.__name__} records a command but does not refresh "
            f"{provenance.COMMANDS_FILE}")
