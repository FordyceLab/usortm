"""The sequenced pick plate is judged against the layout the merge placed.

After the merge the destination plate is built, barcoded and sequenced.  A
pick-plate round fixes the intended layout when it is planned and judges each
well -- confirmed, wrong, empty -- once the demux has run, with the same
machinery the re-order round uses.
"""
import csv
import json

from usortm.pickplate import (CONFIRMED, EMPTY, WRONG, expected_layout_from_pick,
                              expected_wells_from_layout, judge, load_well_rows,
                              read_expected_layout, verdict_rows,
                              write_expected_layout)
from usortm.report.plates import verdict_plate


PICK = [
    {"variant": "G3A", "target_well": "A1", "source_plate": "R1_8", "source_well": "G15",
     "source_round": 1, "reads": 900},
    {"variant": "G3F", "target_well": "B1", "source_plate": "R2_1", "source_well": "A2",
     "source_round": 2, "bench_plate": "R2_2", "bench_well": "A1", "reads": 223},
    {"variant": "N4A", "target_well": "C1", "source_plate": "", "source_well": "", "empty": True},
    {"variant": "V47M", "target_well": "D1", "source_plate": "R1_2", "source_well": "D1",
     "tier_override": "Streakout", "reads": 272},
    {"variant": "K16*", "target_well": "E1", "source_plate": "R2_1", "source_well": "A10",
     "source_round": 2, "bench_plate": "R2_2", "bench_well": "A5", "reads": 36},
]


def _row(well, variant, reads, mmf=0.03, cons="Perfect Match", flank="OK"):
    return {"plate": "1", "well": well, "variant": variant, "reads": reads,
            "consensus_fraction": 1.0, "cons_check": cons, "flank_check": flank,
            "max_mismatch_frac": mmf}


def test_the_layout_is_the_filled_wells_of_the_pick_with_bench_positions():
    rows = expected_layout_from_pick(PICK)
    assert [r["well"] for r in rows] == ["A1", "B1", "E1"]     # no placeholder, no streak-out
    assert rows[1]["bench_plate"] == "R2_2" and rows[1]["bench_well"] == "A1"
    assert rows[0]["source_plate"] == "R1_8"


def test_the_layout_round_trips_through_its_file(tmp_path):
    rows = expected_layout_from_pick(PICK)
    path = write_expected_layout(rows, tmp_path / "rounds" / "3" / "expected_layout.csv")
    back = read_expected_layout(path)
    assert [(r["well"], r["variant"]) for r in back] == [("A1", "G3A"), ("B1", "G3F"), ("E1", "K16*")]
    exp = expected_wells_from_layout(back)
    assert exp[(1, "B1")].variant == "G3F" and exp[(1, "B1")].order_well == "B1"


def test_each_well_is_confirmed_wrong_or_empty():
    layout = expected_layout_from_pick(PICK)
    wells = [_row("A1", "G3A", 500),                 # holds it
             _row("B1", "G15A", 400),                # holds something else
             _row("E1", "K16*", 8)]                  # too few reads
    verdicts, summary = judge(wells, layout, designed={"G3A", "G3F", "G15A", "K16*"})
    by = {v.well: v.status for v in verdicts}
    assert by == {"A1": CONFIRMED, "B1": WRONG, "E1": EMPTY}
    assert summary["wells"] == {CONFIRMED: 1, WRONG: 1, EMPTY: 1}
    assert summary["not_confirmed"] == {"G3F", "K16*"}


def test_a_well_holding_its_variant_unclean_is_not_confirmed():
    layout = expected_layout_from_pick(PICK)
    wells = [_row("A1", "G3A", 500, mmf=0.40)]       # mixed template
    verdicts, _ = judge(wells, layout, designed={"G3A"})
    assert {v.well: v.status for v in verdicts}["A1"] == WRONG
    table = verdict_rows(verdicts, wells, layout)
    a1 = next(r for r in table if r["well"] == "A1")
    assert a1["reason"] == "mixed template" and a1["source_plate"] == "R1_8"


def test_the_verdict_table_carries_where_each_well_was_picked_from():
    layout = expected_layout_from_pick(PICK)
    wells = [_row("A1", "G3A", 500), _row("B1", "G3F", 300)]
    verdicts, _ = judge(wells, layout, designed={"G3A", "G3F"})
    table = {r["well"]: r for r in verdict_rows(verdicts, wells, layout)}
    assert table["B1"]["bench_plate"] == "R2_2" and table["B1"]["bench_well"] == "A1"
    assert table["E1"]["status"] == EMPTY and table["E1"]["observed"] == ""


def test_the_plate_is_drawn_by_verdict_with_the_source_on_the_hover():
    layout = expected_layout_from_pick(PICK)
    wells = [_row("A1", "G3A", 500), _row("B1", "G15A", 400)]
    verdicts, _ = judge(wells, layout, designed={"G3A", "G3F", "G15A"})
    out = verdict_plate(verdicts, {f'{w["plate"]}_{w["well"]}': w for w in wells},
                        {"1_A1": "x/well_1_A1.html"}, {r["well"]: r for r in layout})
    assert out["counts"] == {CONFIRMED: 1, WRONG: 1, EMPTY: 1}
    assert 'href="x/well_1_A1.html"' in out["grid"]
    assert "holds it" in out["grid"] and "read as G15A" in out["grid"]
    assert "colony plate R2_2 A1" in out["grid"]
    assert out["grid"].count('class="w blank"') == 384 - 3


def test_load_well_rows_types_the_fields(tmp_path):
    p = tmp_path / "well_assignments.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["plate", "well", "reads", "variant", "consensus_fraction", "cons_check",
                    "flank_check", "n_flagged_positions", "max_mismatch_frac"])
        w.writerow(["1", "A1", "512", "G3A", "1.0", "Perfect Match", "OK", "0", "0.0312"])
        w.writerow(["1", "A2", "3", "unassigned", "0", "", "", "", ""])
    rows = load_well_rows(p)
    assert rows[0]["reads"] == 512 and rows[0]["max_mismatch_frac"] == 0.0312
    assert rows[1]["max_mismatch_frac"] is None


def test_the_demux_holds_a_laid_out_well_to_the_variant_placed_in_it(tmp_path):
    """The free assignment is the wrong question for a plate built to a
    layout: the reference becomes the placed variant, the free assignment is
    kept beside it, and a well whose expected variant has no reference is
    left as it was."""
    import pandas as pd

    from usortm.demux.pipeline import _hold_to_expected

    ref_dir = tmp_path / "refs"
    (ref_dir / "single_ref_fastas").mkdir(parents=True)
    for v in ("G3A", "G3F"):
        (ref_dir / "single_ref_fastas" / f"{v}.fasta").write_text(f">{v}\nACGT\n")
    well_df = pd.DataFrame({
        "global_well": ["1A1", "1B1", "1C1", "1D1"],
        "major_ref": ["fwd:G15A", "unassigned", "fwd:G3F", "fwd:K16*"],
        "assignment_confidence": [0.6, 0.0, 0.9, 0.8],
    })
    said = []
    out = _hold_to_expected(well_df, {"1A1": "G3A", "1B1": "G3F", "1C1": "G3F",
                                      "1D1": "Q52*"}, ref_dir, said.append)
    got = dict(zip(out["global_well"], out["major_ref"]))
    assert got == {"1A1": "G3A", "1B1": "G3F", "1C1": "G3F", "1D1": "fwd:K16*"}
    kept = dict(zip(out["global_well"], out["assigned_variant"]))
    assert kept == {"1A1": "G15A", "1B1": "unassigned", "1C1": "G3F", "1D1": "K16*"}
    assert list(out["assignment_confidence"])[:3] == [1.0, 1.0, 1.0]
    assert "3 well(s)" in said[0] and "1 expected variant(s) have no reference" in said[0]


def test_a_wrong_well_is_reported_as_what_it_was_read_as():
    """After the demux held the well to its expected variant, the row's
    variant is the expectation; the verdict must name the free assignment."""
    layout = expected_layout_from_pick(PICK)
    wells = [dict(_row("A1", "G3A", 300, cons="Error"), assigned_variant="G15A"),
             dict(_row("B1", "G3F", 300), assigned_variant="G3F")]
    verdicts, _ = judge(wells, layout, designed={"G3A", "G3F", "G15A"})
    by = {v.well: v for v in verdicts}
    assert by["A1"].status == WRONG and by["A1"].observed == "G15A"
    assert by["B1"].status == CONFIRMED and by["B1"].observed == "G3F"


def test_verify_renders_pileups_for_the_intended_wells():
    """The report's plate links each well to its reads; a pick-plate round has
    no pick step to render them, so verify must."""
    import inspect

    from usortm.cli import verify_cmd

    src = inspect.getsource(verify_cmd.verify)
    assert "_pileups(" in src and "render_pileups" in src
    assert 'well=",".join(wanted)' in src, "named wells are rendered whatever their depth"


def test_plan_writes_the_layout_and_marks_the_round(tmp_path):
    """The planner fixes the layout from the merged pick before reads exist."""
    from usortm.cli.plan import _plan_round_n

    proj = tmp_path / "proj"
    (proj / "merged").mkdir(parents=True)
    json.dump({"created": "x", "n_plates": 1, "rounds": {}}, open(proj / "usortm_project.json", "w"))
    json.dump(PICK, open(proj / "merged" / "pick_list.json", "w"))
    variants = [{"name": "G3A", "sequence": "ATG"}, {"name": "G3F", "sequence": "ATG"},
                {"name": "K16*", "sequence": "ATG"}]
    _plan_round_n(variants, proj, 3, "levseq", 294, pick_plate=True)
    state = json.load(open(proj / "rounds" / "3" / "usortm_round.json"))
    assert state["kind"] == "pick_plate" and state["n_expected"] == 3
    assert state["workflow_steps"]["verify"] == {"completed": False}
    layout = read_expected_layout(proj / "rounds" / "3" / "expected_layout.csv")
    assert [r["well"] for r in layout] == ["A1", "B1", "E1"]
    project = json.load(open(proj / "usortm_project.json"))
    assert project["rounds"]["3"]["kind"] == "pick_plate"
