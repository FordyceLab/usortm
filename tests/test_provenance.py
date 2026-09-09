"""The commands a project was built by, kept so they can be reported.

A parameter table has to be read back into an invocation by whoever writes the
methods section, and doing that from memory is where a methods section stops
matching the run.  Each step records its own command instead.
"""
import json

import pytest

from usortm import provenance


def test_the_command_is_recorded_as_it_would_be_typed():
    """The interpreter that happened to reach the CLI is not part of it."""
    assert provenance.current_command(
        ["/usr/bin/usortm", "demux", "proj/", "--round", "2"]
    ) == "usortm demux proj/ --round 2"
    # Reached through the interpreter, the same command comes back the same.
    assert provenance.current_command(
        ["/opt/py/bin/python", "-m", "usortm", "demux", "proj/"]
    ) == "usortm demux proj/"


def test_an_argument_that_needs_quoting_gets_it():
    """The file is meant to be re-run, so it has to survive a copy and paste."""
    got = provenance.current_command(
        ["usortm", "demux", "/a path/with spaces", "--tier", "C"])
    assert got == "usortm demux '/a path/with spaces' --tier C"


def test_recording_a_step_stamps_it():
    state = provenance.record({"completed": True},
                              argv=["usortm", "pick", "proj/"])
    assert state["command"] == "usortm pick proj/"
    assert state["timestamp"]


def test_an_existing_timestamp_is_left_alone():
    """A step that already dated itself is not re-dated by being recorded."""
    state = provenance.record({"completed": True, "timestamp": "2026-01-01"},
                              argv=["usortm", "pick", "proj/"])
    assert state["timestamp"] == "2026-01-01"


def _project(**steps):
    return {"workflow_steps": steps}


def test_the_steps_come_back_in_the_order_they_ran():
    project = _project(
        pick={"completed": True, "timestamp": "2026-03-02", "command": "b"},
        plan={"completed": True, "timestamp": "2026-03-01", "command": "a"},
    )
    got = provenance.render_commands(project, "proj")
    assert got.index("· plan") < got.index("· pick")


def test_a_round_keeps_its_own_steps():
    """A re-order round's demux is reported beside the first round's."""
    project = _project(
        demux={"completed": True, "timestamp": "2026-03-01",
               "command": "usortm demux proj/"})
    project["rounds"] = {"2": {"workflow_steps": {
        "demux": {"completed": True, "timestamp": "2026-03-05",
                  "command": "usortm demux proj/ --round 2"}}}}

    got = provenance.render_commands(project, "proj")
    assert "usortm demux proj/" in got
    assert "usortm demux proj/ --round 2" in got
    assert "round 2 · demux" in got


def test_a_step_with_no_command_is_said_to_have_none():
    """Rather than reconstructed into an invocation nobody typed."""
    project = _project(pick={"completed": True, "timestamp": "2026-03-02",
                             "tier": "C", "total_hits": 338})
    got = provenance.render_commands(project, "proj")

    assert "command not recorded" in got
    assert "tier: C" in got
    assert "total_hits: 338" in got
    # Nothing that looks like a command it could be mistaken for.
    assert "usortm pick" not in got


def test_an_unfinished_step_is_not_reported():
    """report writes 'completed': False before it runs; that is not a step."""
    project = _project(report={"completed": False},
                       plan={"completed": True, "timestamp": "2026-03-01",
                             "command": "usortm plan v.csv"})
    got = provenance.render_commands(project, "proj")
    assert "· plan" in got
    assert "· report" not in got


def test_a_project_with_nothing_recorded_says_so():
    got = provenance.render_commands({}, "proj")
    assert "No completed steps" in got


def test_the_file_lands_at_the_top_of_the_project(tmp_path):
    project = _project(plan={"completed": True, "timestamp": "2026-03-01",
                             "command": "usortm plan v.csv"})
    path = provenance.write_commands(project, tmp_path)

    assert path == tmp_path / "commands.txt"
    assert "usortm plan v.csv" in path.read_text()


def test_the_command_writes_the_file(tmp_path):
    from typer.testing import CliRunner

    from usortm.cli import app

    (tmp_path / "usortm_project.json").write_text(json.dumps(_project(
        plan={"completed": True, "timestamp": "2026-03-01",
              "command": "usortm plan v.csv"})))

    result = CliRunner().invoke(app, ["methods", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "commands.txt").exists()
    assert "usortm plan v.csv" in (tmp_path / "commands.txt").read_text()


def test_a_directory_that_is_not_a_project_is_refused(tmp_path):
    from typer.testing import CliRunner

    from usortm.cli import app

    result = CliRunner().invoke(app, ["methods", str(tmp_path)])
    assert result.exit_code == 1
    assert not (tmp_path / "commands.txt").exists()
