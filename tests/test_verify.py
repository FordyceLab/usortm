"""A re-ordered plate is checked against what was ordered for it.

A first round asks what a well holds.  A re-order round already knows, so it
asks whether the assembly produced it, and the three answers -- the ordered
construct, something else, or nothing that grew -- fail for different reasons
and are counted apart.
"""
import csv

import pytest

from usortm.verify import (CONFIRMED, EMPTY, LAYOUTS, WRONG, OrderedWell,
                           expected_wells, infer_layout, read_order_layout,
                           summarise, to_seq_well, verify)


def _clean(row, designed):
    """Stand-in for the plate maps' rule, so these tests fix only their own."""
    return row.get("variant") in designed and not row.get("dirty")


def _well(plate, well, variant, reads=200, **kw):
    return {"plate": plate, "well": well, "variant": variant, "reads": reads,
            **kw}


# --- where a 96-well plate lands -----------------------------------------

def test_block_layout_is_a_straight_copy():
    """A 96-well plate moved one-for-one keeps its own well labels."""
    assert to_seq_well("A1", "block") == "A1"
    assert to_seq_well("H12", "block") == "H12"
    assert to_seq_well("C7", "block") == "C7"


def test_quadrants_interleave():
    """Four 96-well plates make one 384, so each quadrant is offset by one."""
    assert to_seq_well("A1", "q1") == "A1"
    assert to_seq_well("A1", "q2") == "A2"
    assert to_seq_well("A1", "q3") == "B1"
    assert to_seq_well("A1", "q4") == "B2"
    # Second column of the order plate is the third of the sequenced one.
    assert to_seq_well("A2", "q1") == "A3"
    # The far corner still lands inside the plate.
    assert to_seq_well("H12", "q4") == "P24"


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
    # The same well on two plates is two different constructs.
    wanted = expected_wells(order, "block")
    assert wanted[(1, "A1")] == "one"
    assert wanted[(2, "A1")] == "two"


def test_a_file_with_no_constructs_is_refused(tmp_path):
    """Reading on would invent a layout out of a file that has none."""
    path = tmp_path / "empty.csv"
    path.write_text("Well Position,Name,Sequence\n")
    with pytest.raises(ValueError, match="no ordered constructs"):
        read_order_layout(path)


def test_order_plates_can_be_sequenced_as_other_plates(tmp_path):
    """The order's plate 1 need not be the run's plate 1."""
    path = tmp_path / "order.csv"
    _order_csv(path, [[("A1", "one")], [("A1", "two")]])
    order = read_order_layout(path)

    wanted = expected_wells(order, "block", plate_of={1: 16, 2: 17})
    assert wanted == {(16, "A1"): "one", (17, "A1"): "two"}


# --- reading the arraying off the data -----------------------------------

def test_the_layout_is_inferred_from_where_the_reads_landed():
    """Only the right arraying puts the constructs on wells that grew."""
    order = [OrderedWell(1, w, f"v{i}")
             for i, w in enumerate(["A1", "A2", "B1", "B2"])]
    # Cultures arrayed into quadrant 4: A1 -> B2, A2 -> B4, B1 -> D2, B2 -> D4.
    grown = [_well(1, w, "x") for w in ("B2", "B4", "D2", "D4")]

    got = infer_layout(order, grown)
    assert got["layout"] == "q4"
    assert got["score"] == 1.0
    assert got["margin"] > 0
    assert set(got["scores"]) == set(LAYOUTS)


def test_a_plate_that_all_grew_cannot_say_which_layout_it_is():
    """Every candidate lands on a filled well, so the margin goes to zero."""
    order = [OrderedWell(1, w, f"v{i}")
             for i, w in enumerate(["A1", "A2", "B1"])]
    everywhere = [_well(1, f"{r}{c}", "x")
                  for r in "ABCDEFGH" for c in range(1, 13)]

    got = infer_layout(order, everywhere)
    assert got["score"] == 1.0
    assert got["margin"] == 0.0      # the reading means nothing here


def test_wells_below_the_read_floor_are_not_grown():
    """A well with a handful of reads did not grow, whatever it was called."""
    order = [OrderedWell(1, "A1", "v0")]
    assert infer_layout(order, [_well(1, "A1", "v0", reads=3)])["score"] == 0.0
    assert infer_layout(order, [_well(1, "A1", "v0", reads=20)])["score"] == 1.0


# --- the verdicts ---------------------------------------------------------

def test_the_three_outcomes_are_separated():
    """Ordered and got it, ordered and got something else, ordered and empty."""
    expected = {(1, "A1"): "G3F", (1, "A2"): "S8M", (1, "A3"): "Y11*"}
    designed = {"G3F", "S8M", "Y11*"}
    data = [
        _well(1, "A1", "G3F"),                 # what was ordered
        _well(1, "A2", "Parent"),              # assembly gave the backbone
        _well(1, "A3", "Y11*", reads=4),       # never grew
    ]

    verdicts = {v.well: v for v in verify(data, expected, designed,
                                          is_clean=_clean)}
    assert verdicts["A1"].status == CONFIRMED
    assert verdicts["A2"].status == WRONG
    assert verdicts["A2"].observed == "Parent"
    assert verdicts["A3"].status == EMPTY
    assert verdicts["A3"].observed is None


def test_a_well_read_uncleanly_is_not_confirmed():
    """The call has to stand on its own before it can confirm an order."""
    expected = {(1, "A1"): "G3F"}
    data = [_well(1, "A1", "G3F", dirty=True)]

    got = verify(data, expected, {"G3F"}, is_clean=_clean)
    assert got[0].status == WRONG


def test_a_well_never_sequenced_is_empty_not_missing():
    """An intended well absent from the run still has to be accounted for."""
    got = verify([], {(1, "A1"): "G3F"}, {"G3F"}, is_clean=_clean)
    assert [v.status for v in got] == [EMPTY]
    assert got[0].reads == 0


def test_wells_that_were_never_ordered_are_not_judged():
    """The run covers a whole plate; the question is only about the order."""
    data = [_well(1, "A1", "G3F"), _well(1, "H9", "something else")]
    got = verify(data, {(1, "A1"): "G3F"}, {"G3F"}, is_clean=_clean)
    assert [v.well for v in got] == ["A1"]


def test_verdicts_come_back_in_plate_order():
    """A10 sorts after A9, which string order gets wrong."""
    expected = {(1, "A10"): "a", (1, "A9"): "b", (2, "A1"): "c"}
    got = verify([], expected, set(), is_clean=_clean)
    assert [(v.plate, v.well) for v in got] == [(1, "A9"), (1, "A10"), (2, "A1")]


# --- counting -------------------------------------------------------------

def test_a_variant_is_confirmed_by_any_of_its_wells():
    """Several colonies of one construct recover it once, not twice."""
    expected = {(1, "A1"): "G3F", (1, "A2"): "G3F", (1, "A3"): "S8M"}
    data = [
        _well(1, "A1", "Parent"),     # this colony failed
        _well(1, "A2", "G3F"),        # this one did not
        _well(1, "A3", "S8M"),
    ]

    got = summarise(verify(data, expected, {"G3F", "S8M"}, is_clean=_clean))
    assert got["n_wells"] == 3
    assert got["wells"] == {CONFIRMED: 2, WRONG: 1, EMPTY: 0}
    # Two constructs ordered, both recovered, though one well of one failed.
    assert got["n_variants"] == 2
    assert got["confirmed"] == {"G3F", "S8M"}
    assert got["not_confirmed"] == set()


def test_a_construct_that_failed_in_every_well_is_not_confirmed():
    expected = {(1, "A1"): "G3F", (1, "A2"): "G3F"}
    data = [_well(1, "A1", "Parent"), _well(1, "A2", "G3F", reads=2)]

    got = summarise(verify(data, expected, {"G3F"}, is_clean=_clean))
    assert got["wells"] == {CONFIRMED: 0, WRONG: 1, EMPTY: 1}
    assert got["confirmed"] == set()
    assert got["not_confirmed"] == {"G3F"}
