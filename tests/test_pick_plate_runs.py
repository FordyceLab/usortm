"""Pick-plate runs live in 7_pick_plate/<run>/, not under rounds/.

A pick-plate run sequences the plate the merge built.  It samples nothing
new, so it is recorded under its own key, named by its sequencing run, and
judged against what the liquid-handler worklists moved into each well --
the worklists are what the robot ran, and outlive any re-run of the pick.
"""
import json

import pytest
import typer

from usortm import pickplate as pp
from usortm import provenance
from usortm.integra import read_integra_files, write_integra_files


def _pick():
    return [
        {"variant": "G3F", "source_plate": "R2_1", "source_well": "A2",
         "bench_plate": "R2_2", "bench_well": "A1", "source_round": 2,
         "reads": 181, "target_plate": "0", "target_well": "B1"},
        {"variant": "K16*", "source_plate": "R1_4", "source_well": "C7",
         "source_round": 1, "reads": 900, "target_plate": "0", "target_well": "A1"},
        {"variant": "N4A", "source_plate": "", "source_well": "", "empty": True,
         "target_plate": "0", "target_well": "A3"},
    ]


def test_worklists_read_back_as_the_transfers_they_describe(tmp_path):
    write_integra_files(_pick(), tmp_path, volume=3)
    rows = {r["target_well"]: r for r in read_integra_files(tmp_path)}
    assert set(rows) == {"A1", "B1"}                    # the empty well moves nothing
    assert rows["A1"]["variant"] == "K16*"              # 'tag' read back as '*'
    assert rows["B1"]["round"] == 2                     # round from the file name
    assert (rows["B1"]["source_plate"], rows["B1"]["source_well"]) == ("2", "A1")


def test_layout_follows_the_worklists_and_takes_provenance_from_the_pick(tmp_path):
    write_integra_files(_pick(), tmp_path, volume=3)
    rows, problems = pp.layout_from_worklists(read_integra_files(tmp_path), _pick())
    assert problems == []
    assert rows == pp.expected_layout_from_pick(_pick())


def test_a_pick_that_disagrees_with_the_worklists_is_reported_and_loses(tmp_path):
    write_integra_files(_pick(), tmp_path, volume=3)
    changed = _pick()
    changed[1] = dict(changed[1], variant="K16A")     # the pick was re-run later
    rows, problems = pp.layout_from_worklists(read_integra_files(tmp_path), changed)
    assert any(p.startswith("A1:") for p in problems)
    assert {r["well"]: r["variant"] for r in rows}["A1"] == "K16*"


def test_run_names_must_be_usable_as_directories(tmp_path):
    assert pp.run_paths(tmp_path, "G3Y8KW").root == tmp_path / "7_pick_plate" / "G3Y8KW"
    for bad in ("", "../x", "a/b", " G3Y8KW"):
        with pytest.raises(ValueError):
            pp.run_paths(tmp_path, bad)


def test_newest_run_is_the_last_started():
    project = {pp.STATE_KEY: {"G3Y8KW": {"created": "2026-09-25T13:00"},
                              "GQKLWM": {"created": "2026-09-15T09:00"}}}
    assert [n for n, _ in pp.list_runs(project)] == ["GQKLWM", "G3Y8KW"]
    assert pp.newest_run(project) == "G3Y8KW"
    assert pp.newest_run({}) is None


def test_commands_name_a_run_where_a_round_would_stand(tmp_path):
    project = {
        "workflow_steps": {"demux": {"completed": True, "timestamp": "2026-09-10T15:30",
                                     "command": "usortm demux p/"}},
        pp.STATE_KEY: {"G3Y8KW": {"workflow_steps": {
            "demux": {"completed": True, "timestamp": "2026-09-25T13:10",
                      "command": "usortm demux p/ --pick-plate G3Y8KW"}}}},
    }
    text = provenance.render_commands(project, tmp_path)
    assert "· pick plate G3Y8KW · demux" in text
    assert text.index("round 1 · demux") < text.index("pick plate G3Y8KW")


def _project(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    state = {"workflow_steps": {"demux": {"completed": True, "n": "round 1's"}},
             "n_plates": 14, "rounds": {}}
    (root / "usortm_project.json").write_text(json.dumps(state))
    (root / "merged").mkdir()
    (root / "merged" / "pick_list.json").write_text(json.dumps(_pick()))
    write_integra_files(_pick(), root / "integra_assist_input", volume=3)
    return root, state


def test_starting_a_run_writes_its_layout_and_leaves_round_1_alone(tmp_path):
    from usortm.cli.demux_cmd import _start_pick_plate_run

    root, state = _project(tmp_path)
    run, expected = _start_pick_plate_run(root, state, "G3Y8KW")
    assert expected == {"1A1": "K16*", "1B1": "G3F"}
    assert run.layout.exists()
    saved = json.loads((root / "usortm_project.json").read_text())
    assert saved["n_plates"] == 14                              # the sort's count
    assert saved["workflow_steps"] == {"demux": {"completed": True, "n": "round 1's"}}
    assert saved["rounds"] == {}                               # not a round
    block = saved[pp.STATE_KEY]["G3Y8KW"]
    assert block["n_expected"] == 2 and block["layout_source"] == "integra_assist_input"


def test_a_recorded_layout_is_kept_on_a_rerun(tmp_path):
    """A re-run is judged against what was recorded before its reads were
    seen, not against a pick re-run since."""
    from usortm.cli.demux_cmd import _start_pick_plate_run

    root, state = _project(tmp_path)
    _start_pick_plate_run(root, state, "G3Y8KW")
    write_integra_files([dict(_pick()[1], variant="K16A")],
                        root / "integra_assist_input", volume=3)
    _, expected = _start_pick_plate_run(root, state, "G3Y8KW")
    assert expected["1A1"] == "K16*"


def test_a_bad_run_name_stops_the_demux(tmp_path):
    from usortm.cli.demux_cmd import _start_pick_plate_run

    root, state = _project(tmp_path)
    with pytest.raises(typer.Exit):
        _start_pick_plate_run(root, state, "../elsewhere")
