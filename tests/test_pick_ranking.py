"""The pick prefers a well whose reads agree over a deeper well whose reads split.

Depth used to decide first.  On the AFMtag round 1 pick that chose 2K7 for
G45A -- 516 reads, 86 of them a second clone carrying T at 823, 857 and 858
together -- and 14C1 for T26F, 1,543 reads with 18% disagreeing at one
position, over 1B1 and 3E12, both clean at 5%.  The column scan had already
measured all of this; the pick did not read it.
"""
from usortm.cli.pick import _generate_pick_list


def _well(plate, well, variant, reads, mmf, cons="Perfect Match", conf=0.9):
    return {
        "plate": str(plate), "well": well, "variant": variant, "reads": reads,
        "consensus_fraction": 1.0, "cons_check": cons,
        "assignment_confidence": conf, "max_mismatch_frac": mmf,
        "flank_check": "OK",
    }


def _pick(wells, **kw):
    picks = _generate_pick_list(wells, None, True, 384, "row", **kw)
    return {p["variant"]: (p["source_plate"], p["source_well"]) for p in picks
            if not p.get("empty")}


def test_a_clean_well_beats_a_deeper_watch_well():
    wells = [_well(14, "C1", "T26F", 1543, 0.179),   # watch, deep
             _well(1, "B1", "T26F", 381, 0.05)]      # clean, shallower
    assert _pick(wells)["T26F"] == ("1", "B1")


def test_depth_still_decides_between_two_clean_wells():
    wells = [_well(1, "B1", "T26F", 381, 0.05),
             _well(3, "E12", "T26F", 684, 0.05)]
    assert _pick(wells)["T26F"] == ("3", "E12")


def test_an_unmeasured_well_ranks_behind_clean_and_ahead_of_watch():
    wells = [_well(2, "K7", "G45A", 516, 0.192),     # watch
             _well(9, "Z9", "G45A", 900, None)]      # not measured
    assert _pick(wells)["G45A"] == ("9", "Z9")
    wells.append(_well(4, "A1", "G45A", 40, 0.03))   # clean, shallow
    assert _pick(wells)["G45A"] == ("4", "A1")


def test_consensus_category_still_comes_first():
    """A Silent Mutation well that is clean does not beat a Perfect Match
    well that is only watch: the consensus test outranks the column scan."""
    wells = [_well(2, "K7", "G45A", 516, 0.192),
             _well(5, "A2", "G45A", 800, 0.02, cons="Silent Mutation")]
    assert _pick(wells)["G45A"] == ("2", "K7")


def test_a_mixed_well_is_still_excluded():
    wells = [_well(1, "A1", "I13A", 900, 0.40)]
    assert "I13A" not in _pick(wells)


# --- --max-disagreement -----------------------------------------------------

def test_a_stated_limit_excludes_the_watch_band():
    """At 0.10 a watch well is out, even when it is the only well."""
    wells = [_well(2, "K7", "G45A", 516, 0.192)]
    assert "G45A" in _pick(wells)
    assert "G45A" not in _pick(wells, max_disagreement=0.10)


def test_the_limit_is_inclusive_and_read_on_the_fraction():
    wells = [_well(1, "A1", "X1A", 100, 0.10), _well(1, "A2", "X2A", 100, 0.1001)]
    picked = _pick(wells, max_disagreement=0.10)
    assert "X1A" in picked and "X2A" not in picked


def test_an_unmeasured_well_is_not_excluded_by_the_limit():
    wells = [_well(9, "Z9", "G45A", 900, None)]
    assert _pick(wells, max_disagreement=0.10)["G45A"] == ("9", "Z9")


def test_a_limit_above_the_mixed_threshold_admits_mixed_wells():
    """The limit replaces the default rather than stacking on it, so what
    the command reports is the criterion that was applied."""
    wells = [_well(1, "A1", "I13A", 900, 0.40)]
    assert "I13A" in _pick(wells, max_disagreement=0.50)
