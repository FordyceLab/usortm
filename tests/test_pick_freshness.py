"""Sequencing the pick plate does not send the summary back to the old pick.

The pick a summary draws is the newest one that still describes the current
wells, so a pick left behind by an earlier demux is not drawn as current.  A
pick-plate round breaks that rule: its demux is the newest in the project and
describes the plate the merged pick built, so counting it as a well source
dated the merge behind it and the plate was drawn from the single-round pick
instead -- every variant the re-order round recovered showing as not
recovered, G3F among them, while the merge and the robot's own worklist had
placed it.
"""
import json
import os

from usortm.report.summary import _current_pick


def _project(tmp_path, *, rounds):
    """A project with a round-1 demux, a pick, a merged pick and more rounds."""
    root = tmp_path / "proj"
    (root / "demux_output").mkdir(parents=True)
    (root / "demux_output" / "well_assignments.csv").write_text("plate,well\n1,A1\n")

    (root / "pick").mkdir()
    (root / "pick" / "pick_list.json").write_text(
        json.dumps([{"variant": "G3F", "empty": True}]))
    (root / "merged").mkdir()
    (root / "merged" / "pick_list.json").write_text(
        json.dumps([{"variant": "G3F", "source_plate": "R2_1", "source_well": "A2"}]))

    project = {"rounds": {}}
    for number, kind in rounds.items():
        rdir = root / "rounds" / str(number)
        (rdir / "demux_output").mkdir(parents=True)
        (rdir / "demux_output" / "well_assignments.csv").write_text("plate,well\n1,A1\n")
        block = {"kind": kind} if kind else {}
        project["rounds"][str(number)] = block
        (rdir / "usortm_round.json").write_text(json.dumps(block))
    (root / "usortm_project.json").write_text(json.dumps(project))

    # Ages: round 1 oldest, then the pick, then the merge, then the later
    # rounds -- the order a real project is built in.
    stamps = {
        root / "demux_output" / "well_assignments.csv": 1000,
        root / "pick" / "pick_list.json": 2000,
        root / "merged" / "pick_list.json": 3000,
    }
    for number in rounds:
        stamps[root / "rounds" / str(number) / "demux_output"
               / "well_assignments.csv"] = 4000
    for path, when in stamps.items():
        os.utime(path, (when, when))
    return root


def test_a_pick_plate_round_does_not_age_out_the_merged_pick(tmp_path):
    root = _project(tmp_path, rounds={2: None, 3: "pick_plate"})
    # round 2 is older than the merge; round 3 is newer but only reads it back
    os.utime(root / "rounds" / "2" / "demux_output" / "well_assignments.csv",
             (2500, 2500))
    pick = _current_pick(root)
    assert pick is not None
    assert pick[0]["source_plate"] == "R2_1", "drew the single-round pick"


def test_a_sorting_round_newer_than_the_merge_still_ages_it_out(tmp_path):
    """The rule itself stays: wells sorted after a merge make it old.

    The merged pick is then dropped, and the summary falls back to the
    single-round pick, which the round-1 demux it was written from still
    describes.
    """
    root = _project(tmp_path, rounds={2: None})
    pick = _current_pick(root)
    assert pick is not None and pick[0].get("empty") is True


def test_the_kind_is_read_from_the_round_when_the_project_lacks_it(tmp_path):
    """A round planned before the master file tracked kinds still counts."""
    root = _project(tmp_path, rounds={3: "pick_plate"})
    (root / "usortm_project.json").write_text(json.dumps({"rounds": {}}))
    assert _current_pick(root)[0]["source_plate"] == "R2_1"
