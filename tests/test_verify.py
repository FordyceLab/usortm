"""A re-ordered plate is checked against what was ordered for it.

A first round asks what a well holds.  A re-order round already knows, so it
asks whether the assembly produced it, and the three answers -- the ordered
construct, something else, or nothing that grew -- fail for different reasons
and are counted apart.
"""
import csv

import pytest

from usortm.verify import (CONFIRMED, EMPTY, FAILURE_REASONS, LAYOUTS,
                           QUADRANTS, WRONG, failure_reason,
                           OrderedWell, Replicate, ReplicateMapError,
                           expected_wells, infer_layout, parse_replicate_map,
                           read_order_layout, read_replicate_map,
                           single_replicate, summarise, to_seq_well, verify)


def _clean(row, designed):
    """Stand-in for the plate maps' rule, so these tests fix only their own."""
    return row.get("variant") in designed and not row.get("dirty")


def _well(plate, well, variant, reads=200, **kw):
    return {"plate": plate, "well": well, "variant": variant, "reads": reads,
            **kw}


def _order(*wells):
    return [OrderedWell(1, w, v) for w, v in wells]


# --- where a 96-well plate lands -----------------------------------------

def test_block_layout_is_a_straight_copy():
    """A 96-well plate moved one-for-one keeps its own well labels."""
    assert to_seq_well("A1", "block") == "A1"
    assert to_seq_well("H12", "block") == "H12"
    assert to_seq_well("C7", "block") == "C7"


def test_a_quadrant_is_named_by_the_well_its_own_a1_lands_on():
    """Four 96-well plates interleave into one 384, offset by one each way."""
    assert to_seq_well("A1", "A1") == "A1"
    assert to_seq_well("A1", "A2") == "A2"
    assert to_seq_well("A1", "B1") == "B1"
    assert to_seq_well("A1", "B2") == "B2"
    # Interleaved, so the order plate's second column is the run's third.
    assert to_seq_well("A2", "A1") == "A3"
    assert to_seq_well("B1", "A1") == "C1"
    # The far corner still lands inside the plate.
    assert to_seq_well("H12", "B2") == "P24"


def test_the_quadrants_never_share_a_well():
    """Which is what makes three replicates on one plate possible."""
    seen = {}
    for quadrant in QUADRANTS:
        for row in "ABCDEFGH":
            for col in range(1, 13):
                landed = to_seq_well(f"{row}{col}", quadrant)
                assert landed not in seen, f"{quadrant} collides with {seen.get(landed)}"
                seen[landed] = quadrant
    assert len(seen) == 384


def test_a_well_outside_the_order_plate_is_refused():
    """96 wells are A1-H12; anything further is not a position on it."""
    with pytest.raises(ValueError, match="outside a 96-well plate"):
        to_seq_well("I1", "block")
    with pytest.raises(ValueError, match="outside a 96-well plate"):
        to_seq_well("A13", "block")
    with pytest.raises(ValueError, match="not a well label"):
        to_seq_well("", "block")
    with pytest.raises(ValueError, match="unknown layout"):
        to_seq_well("A1", "sideways")


# --- reading the order back ----------------------------------------------

def _order_csv(path, plates):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        for i, plate in enumerate(plates):
            if i:
                w.writerow([])
                w.writerow([f"# Plate {i + 1}"])
            w.writerow(["Well Position", "Name", "Sequence"])
            for well, name in plate:
                w.writerow([well, name, "ACGT"])


def test_the_order_plate_is_the_record_of_what_went_where(tmp_path):
    """The vendor upload doubles as the layout, so it is read back as one."""
    path = tmp_path / "order.csv"
    _order_csv(path, [[("A1", "G3F"), ("A2", "S8M")]])

    order = read_order_layout(path)
    assert order == [OrderedWell(1, "A1", "G3F"), OrderedWell(1, "A2", "S8M")]


def test_each_order_plate_keeps_its_own_number(tmp_path):
    """Plates are separated by a marker, and each carries its own header."""
    path = tmp_path / "order.csv"
    _order_csv(path, [[("A1", "one")], [("A1", "two")], [("A1", "three")]])

    order = read_order_layout(path)
    assert [(o.plate, o.variant) for o in order] == [
        (1, "one"), (2, "two"), (3, "three")]


def test_a_file_with_no_constructs_is_refused(tmp_path):
    """Reading on would invent a layout out of a file that has none."""
    path = tmp_path / "empty.csv"
    path.write_text("Well Position,Name,Sequence\n")
    with pytest.raises(ValueError, match="no ordered constructs"):
        read_order_layout(path)


# --- the replicate map ----------------------------------------------------

def test_the_replicate_map_places_each_colony():
    """Three colonies of every construct, one per quadrant of one plate."""
    reps = parse_replicate_map({"replicate": [
        {"n": 1, "plate": 1, "quadrant": "A1"},
        {"n": 2, "plate": 1, "quadrant": "A2"},
        {"n": 3, "plate": 1, "quadrant": "B1"},
    ]})
    assert reps == [Replicate(1, 1, "A1"), Replicate(2, 1, "A2"),
                    Replicate(3, 1, "B1")]


def test_two_replicates_cannot_share_a_quadrant():
    """They would occupy the same wells, so one colony would be invisible."""
    with pytest.raises(ReplicateMapError, match="already taken"):
        parse_replicate_map({"replicate": [
            {"n": 1, "plate": 1, "quadrant": "A1"},
            {"n": 2, "plate": 1, "quadrant": "A1"},
        ]})


def test_a_replicate_number_cannot_repeat():
    with pytest.raises(ReplicateMapError, match="already defined"):
        parse_replicate_map({"replicate": [
            {"n": 1, "plate": 1, "quadrant": "A1"},
            {"n": 1, "plate": 1, "quadrant": "A2"},
        ]})


def test_the_same_quadrant_on_another_plate_is_fine():
    """Replicates spread over plates are a different arraying, not a clash."""
    reps = parse_replicate_map({"replicate": [
        {"n": 1, "plate": 1, "quadrant": "A1"},
        {"n": 2, "plate": 2, "quadrant": "A1"},
    ]})
    assert [r.plate for r in reps] == [1, 2]


def test_a_malformed_replicate_map_says_what_is_wrong():
    with pytest.raises(ReplicateMapError, match="No \\[\\[replicate\\]\\]"):
        parse_replicate_map({})
    with pytest.raises(ReplicateMapError, match="missing 'plate'"):
        parse_replicate_map({"replicate": [{"n": 1, "quadrant": "A1"}]})
    with pytest.raises(ReplicateMapError, match="not one of"):
        parse_replicate_map({"replicate": [
            {"n": 1, "plate": 1, "quadrant": "C3"}]})


def test_the_replicate_map_reads_from_toml(tmp_path):
    path = tmp_path / "replicates.toml"
    path.write_text(
        '[[replicate]]\nn = 1\nplate = 1\nquadrant = "A1"\n\n'
        '[[replicate]]\nn = 2\nplate = 1\nquadrant = "B1"\n'
    )
    assert read_replicate_map(path) == [Replicate(1, 1, "A1"),
                                        Replicate(2, 1, "B1")]


def test_a_plate_picked_once_needs_no_map():
    assert single_replicate(plate=3, quadrant="A2") == [Replicate(1, 3, "A2")]


# --- what each well should hold -------------------------------------------

def test_every_construct_appears_once_per_replicate():
    order = _order(("A1", "G3F"), ("A2", "S8M"))
    reps = [Replicate(1, 1, "A1"), Replicate(2, 1, "A2"), Replicate(3, 1, "B1")]

    wanted = expected_wells(order, reps)
    assert len(wanted) == 6
    # G3F's three colonies, one per quadrant.
    assert wanted[(1, "A1")].variant == "G3F" and wanted[(1, "A1")].replicate == 1
    assert wanted[(1, "A2")].variant == "G3F" and wanted[(1, "A2")].replicate == 2
    assert wanted[(1, "B1")].variant == "G3F" and wanted[(1, "B1")].replicate == 3
    # S8M sits one order-column over, which is two 384 columns.
    assert wanted[(1, "A3")].variant == "S8M"


def test_a_map_that_puts_two_constructs_in_one_well_is_refused():
    """Which is what a block layout beside a quadrant one would do."""
    order = _order(("A1", "G3F"))
    with pytest.raises(ReplicateMapError, match="claimed by both"):
        expected_wells(order, [Replicate(1, 1, "block"),
                               Replicate(2, 1, "A1")])


# --- reading the arraying off the data -----------------------------------

def test_the_layout_is_inferred_from_where_the_reads_landed():
    """Only the right arraying puts the constructs on wells that grew."""
    order = _order(("A1", "v0"), ("A2", "v1"), ("B1", "v2"), ("B2", "v3"))
    # Arrayed into the B2 quadrant: A1 -> B2, A2 -> B4, B1 -> D2, B2 -> D4.
    grown = [_well(1, w, "x") for w in ("B2", "B4", "D2", "D4")]

    got = infer_layout(order, grown)
    assert got["layout"] == "B2"
    assert got["score"] == 1.0
    assert got["margin"] > 0
    assert set(got["scores"]) == set(LAYOUTS)


def test_a_plate_that_all_grew_cannot_say_which_quadrant_it_is():
    """With three replicates every quadrant is filled, so the margin goes."""
    order = _order(("A1", "v0"), ("A2", "v1"), ("B1", "v2"))
    everywhere = [_well(1, f"{r}{c}", "x")
                  for r in "ABCDEFGHIJKLMNOP" for c in range(1, 25)]

    got = infer_layout(order, everywhere)
    assert got["score"] == 1.0
    assert got["margin"] == 0.0      # the reading means nothing here


def test_wells_below_the_read_floor_are_not_grown():
    """A well with a handful of reads did not grow, whatever it was called."""
    order = _order(("A1", "v0"))
    assert infer_layout(order, [_well(1, "A1", "v0", reads=3)])["score"] == 0.0
    assert infer_layout(order, [_well(1, "A1", "v0", reads=20)])["score"] == 1.0


# --- the verdicts ---------------------------------------------------------

def test_the_three_outcomes_are_separated():
    """Ordered and got it, ordered and got something else, ordered and empty."""
    order = _order(("A1", "G3F"), ("A2", "S8M"), ("A3", "Y11*"))
    wanted = expected_wells(order, single_replicate(1, "block"))
    designed = {"G3F", "S8M", "Y11*"}
    data = [
        _well(1, "A1", "G3F"),                 # what was ordered
        _well(1, "A2", "Parent"),              # assembly gave the backbone
        _well(1, "A3", "Y11*", reads=4),       # never grew
    ]

    got = {v.well: v for v in verify(data, wanted, designed, is_clean=_clean)}
    assert got["A1"].status == CONFIRMED
    assert got["A2"].status == WRONG and got["A2"].observed == "Parent"
    assert got["A3"].status == EMPTY and got["A3"].observed is None


def test_a_well_read_uncleanly_is_not_confirmed():
    """The call has to stand on its own before it can confirm an order."""
    wanted = expected_wells(_order(("A1", "G3F")), single_replicate(1, "block"))
    got = verify([_well(1, "A1", "G3F", dirty=True)], wanted, {"G3F"},
                 is_clean=_clean)
    assert got[0].status == WRONG


def test_a_well_never_sequenced_is_empty_not_missing():
    """An intended well absent from the run still has to be accounted for."""
    wanted = expected_wells(_order(("A1", "G3F")), single_replicate(1, "block"))
    got = verify([], wanted, {"G3F"}, is_clean=_clean)
    assert [v.status for v in got] == [EMPTY]
    assert got[0].reads == 0


def test_wells_that_were_never_ordered_are_not_judged():
    """The run covers a whole plate; the question is only about the order."""
    wanted = expected_wells(_order(("A1", "G3F")), single_replicate(1, "block"))
    data = [_well(1, "A1", "G3F"), _well(1, "H9", "something else")]
    assert [v.well for v in verify(data, wanted, {"G3F"}, is_clean=_clean)] == ["A1"]


def test_the_verdict_carries_its_replicate():
    order = _order(("A1", "G3F"))
    reps = [Replicate(1, 1, "A1"), Replicate(2, 1, "A2")]
    got = verify([], expected_wells(order, reps), set(), is_clean=_clean)
    assert [(v.replicate, v.well) for v in got] == [(1, "A1"), (2, "A2")]


# --- counting -------------------------------------------------------------

def test_a_variant_is_confirmed_by_any_of_its_colonies():
    """Three colonies recover a construct once, not three times."""
    order = _order(("A1", "G3F"), ("A2", "S8M"))
    reps = [Replicate(1, 1, "A1"), Replicate(2, 1, "A2"), Replicate(3, 1, "B1")]
    wanted = expected_wells(order, reps)
    data = [
        _well(1, "A1", "Parent"),     # G3F replicate 1 failed
        _well(1, "A2", "G3F"),        # replicate 2 worked
        _well(1, "B1", "G3F"),        # so did 3
        _well(1, "A3", "S8M"), _well(1, "A4", "S8M"), _well(1, "B3", "S8M"),
    ]

    got = summarise(verify(data, wanted, {"G3F", "S8M"}, is_clean=_clean))
    assert got["n_wells"] == 6
    assert got["wells"] == {CONFIRMED: 5, WRONG: 1, EMPTY: 0}
    # Two constructs ordered, both recovered.
    assert got["n_variants"] == 2
    assert got["confirmed"] == {"G3F", "S8M"}
    assert got["not_confirmed"] == set()


def test_a_construct_that_failed_in_every_colony_is_not_confirmed():
    order = _order(("A1", "G3F"))
    reps = [Replicate(1, 1, "A1"), Replicate(2, 1, "A2")]
    data = [_well(1, "A1", "Parent"), _well(1, "A2", "G3F", reads=2)]

    got = summarise(verify(data, expected_wells(order, reps), {"G3F"},
                           is_clean=_clean))
    assert got["wells"] == {CONFIRMED: 0, WRONG: 1, EMPTY: 1}
    assert got["not_confirmed"] == {"G3F"}


def test_replicates_are_counted_apart():
    """A replicate failing far more than the others points at the picking."""
    order = _order(("A1", "G3F"), ("A2", "S8M"))
    reps = [Replicate(1, 1, "A1"), Replicate(2, 1, "A2")]
    data = [
        _well(1, "A1", "G3F"), _well(1, "A3", "S8M"),     # replicate 1, both
        _well(1, "A2", "G3F", reads=1),                   # replicate 2, empty
        _well(1, "A4", "S8M", reads=1),
    ]

    got = summarise(verify(data, expected_wells(order, reps), {"G3F", "S8M"},
                           is_clean=_clean))
    assert got["by_replicate"][1] == {CONFIRMED: 2, WRONG: 0, EMPTY: 0}
    assert got["by_replicate"][2] == {CONFIRMED: 0, WRONG: 0, EMPTY: 2}
    # The constructs are still recovered, on replicate 1 alone.
    assert got["confirmed"] == {"G3F", "S8M"}


# --- why a well failed ----------------------------------------------------

def _verdict(status, expected="G3F"):
    from usortm.verify import WellVerdict
    return WellVerdict(1, "A1", expected, None, 100, status, 1)


def test_a_confirmed_well_has_no_reason():
    assert failure_reason({"variant": "G3F"}, _verdict(CONFIRMED)) is None


def test_the_reasons_send_you_to_different_places():
    """A plate of parents is an assembly problem, of flanks a cloning one."""
    def reason(row):
        return failure_reason(row, _verdict(WRONG))

    base = {"variant": "G3F", "consensus_fraction": 1.0, "flank_check": "OK",
            "max_mismatch_frac": 0.01}

    assert reason({**base, "variant": "Parent"}) == "parent"
    assert reason({**base, "max_mismatch_frac": 0.40}) == "mixed template"
    assert reason({**base, "flank_check": "3' mismatch"}) == "flank mismatch"
    assert reason({**base, "variant": "S8M"}) == "another variant"
    assert reason({**base, "flank_check": "No alignment"}) == "no alignment"
    assert reason({**base, "consensus_fraction": 0.0}) == "no alignment"
    assert reason(base) == "sequence differs"


def test_a_well_that_never_grew_reads_as_no_reads():
    assert failure_reason(None, _verdict(EMPTY)) == "no reads"
    assert failure_reason({"variant": "G3F"}, _verdict(EMPTY)) == "no reads"


def test_the_most_decisive_reason_wins():
    """An empty vector also reads as another variant; it is not reported so."""
    row = {"variant": "Parent", "consensus_fraction": 1.0,
           "flank_check": "3' mismatch", "max_mismatch_frac": 0.40}
    assert failure_reason(row, _verdict(WRONG)) == "parent"
    # Without the backbone, two templates outrank a bad junction.
    assert failure_reason({**row, "variant": "G3F"},
                          _verdict(WRONG)) == "mixed template"


def test_a_missing_mismatch_figure_is_not_a_mixed_template():
    """The column is absent on runs that never measured it."""
    row = {"variant": "G3F", "consensus_fraction": 1.0, "flank_check": "OK",
           "max_mismatch_frac": float("nan")}
    assert failure_reason(row, _verdict(WRONG)) == "sequence differs"


def test_every_reason_is_declared():
    assert set(FAILURE_REASONS) == {
        "no reads", "no alignment", "parent", "mixed template",
        "flank mismatch", "another variant", "sequence differs"}
