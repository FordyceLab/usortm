"""A re-order round is picked from the plates the bench worked in.

Re-ordered constructs are assembled in 96-well plates; their picked colonies
are consolidated into quadrants of a 384-well plate for sequencing.  The pick
wrote the sequenced position (plate 1, well A2), which names a well nobody
handles.  The robot needs the colony plate and the 96-well position -- the
translation the report's re-order page already draws from.
"""
import csv

from usortm.cli.merge import _bench_fields
from usortm.cli.pick import _attach_bench_positions
from usortm.integra import write_integra_files
from usortm.verify import OrderedWell, Replicate, bench_positions

ORDER = [OrderedWell(1, "A1", "G3F"), OrderedWell(1, "B7", "K16*"),
         OrderedWell(1, "H12", "Q52*")]
REPS = [Replicate(n=1, plate=1, quadrant="A1"),
        Replicate(n=2, plate=1, quadrant="A2"),
        Replicate(n=3, plate=1, quadrant="B1")]


def test_sequenced_wells_map_back_to_colony_plate_and_order_well():
    bench = bench_positions(ORDER, REPS)
    # replicate 1 sits in quadrant A1: order A1 -> seq A1, B7 -> C13, H12 -> O23
    assert bench[(1, "A1")] == (1, "A1")
    assert bench[(1, "C13")] == (1, "B7")
    assert bench[(1, "O23")] == (1, "H12")
    # replicate 2 in quadrant A2: columns shift by one
    assert bench[(1, "A2")] == (2, "A1")
    assert bench[(1, "C14")] == (2, "B7")
    # replicate 3 in quadrant B1: rows shift by one
    assert bench[(1, "B1")] == (3, "A1")
    assert bench[(1, "D13")] == (3, "B7")
    assert len(bench) == 9


def test_hits_are_given_their_bench_position_and_keep_the_sequenced_one():
    bench = bench_positions(ORDER, REPS)
    hits = [{"variant": "G3F", "source_plate": "1", "source_well": "A2",
             "target_plate": "0", "target_well": "B1"},
            {"variant": "K16*", "source_plate": "1", "source_well": "D13",
             "target_plate": "0", "target_well": "D4"},
            {"variant": "N4A", "source_plate": "", "source_well": "", "empty": True}]
    plates = _attach_bench_positions(hits, bench)
    assert plates == {"1", "2", "3"}
    assert (hits[0]["bench_plate"], hits[0]["bench_well"]) == ("2", "A1")
    assert (hits[1]["bench_plate"], hits[1]["bench_well"]) == ("3", "B7")
    assert hits[0]["source_well"] == "A2", "the sequenced position stays"
    assert "bench_plate" not in hits[2]


def test_a_round_without_an_arraying_attaches_nothing():
    hits = [{"variant": "G3A", "source_plate": "13", "source_well": "O12",
             "target_plate": "0", "target_well": "A1"}]
    assert _attach_bench_positions(hits, {}) is None
    assert "bench_plate" not in hits[0]


def test_the_robot_files_are_written_per_colony_plate_in_96_well_positions(tmp_path):
    bench = bench_positions(ORDER, REPS)
    hits = [{"variant": "G3F", "source_plate": "1", "source_well": "A2",
             "target_plate": "0", "target_well": "B1"},
            {"variant": "K16*", "source_plate": "1", "source_well": "D13",
             "target_plate": "0", "target_well": "D4"}]
    plates = _attach_bench_positions(hits, bench)
    files = write_integra_files(hits, tmp_path, 5.0, source_plates=plates)
    assert [f.name for f in files] == ["integra_assist_plate1.csv",
                                       "integra_assist_plate2.csv",
                                       "integra_assist_plate3.csv"]
    rows = list(csv.reader(open(tmp_path / "integra_assist_plate2.csv"), delimiter=";"))
    assert rows[1] == ["G3F", "2", "A1", "0", "B1", "5"]
    rows = list(csv.reader(open(tmp_path / "integra_assist_plate3.csv"), delimiter=";"))
    assert rows[1] == ["K16*", "3", "B7", "0", "D4", "5"]
    rows = list(csv.reader(open(tmp_path / "integra_assist_plate1.csv"), delimiter=";"))
    assert rows[1:] == []                       # colony plate 1 had nothing picked


def test_the_merge_names_the_colony_plate_by_round():
    bench = {2: bench_positions(ORDER, REPS)}
    well = {"plate": "1", "well": "C14"}
    assert _bench_fields(2, well, bench) == {"bench_plate": "R2_2", "bench_well": "B7"}
    assert _bench_fields(1, {"plate": "3", "well": "N23"}, bench) == {}
    assert _bench_fields(2, {"plate": "1", "well": "P24"}, bench) == {}
