"""The round merge applies the policy the round's pick applied.

The merge chose each variant's well by read depth alone and excluded only
mixed templates.  A strict round 1 pick (--max-disagreement 0.10) removed
ten watch-band wells; the merge that followed put every one of them back,
so the merged plate and summary.html disagreed with the pick plate they were
built from.
"""
from usortm.cli.merge import (_build_merged_pick_list, _round_limits,
                              _well_rank, _within_limit)


def _well(plate, well, variant, reads, mmf, cons="Perfect Match"):
    return {"plate": str(plate), "well": well, "variant": variant,
            "reads": reads, "consensus_fraction": 1.0, "cons_check": cons,
            "flank_check": "OK", "max_mismatch_frac": mmf}


def _picked(pick_list):
    return {p["variant"]: (p["source_plate"], p["source_well"])
            for p in pick_list if not p.get("empty")}


ORDER = {"G45A": 0, "T26F": 1}


def test_excluded_wells_parse_and_are_skipped():
    from usortm.cli.merge import _parse_excluded_wells
    import pytest

    assert _parse_excluded_wells("R1_13N18, R2_1A9") == {(1, "13", "N18"), (2, "1", "A9")}
    assert _parse_excluded_wells(None) == set()
    with pytest.raises(ValueError):
        _parse_excluded_wells("13N18")
    # the merge picks the next-best well once the named one is gone
    all_wells = {1: [_well(13, "N18", "N79*", 746, 0.0965),
                     _well(2, "M6", "N79*", 692, 0.0319)]}
    excluded = {(1, "13", "N18")}
    kept = {r: [w for w in ws if (r, str(w["plate"]), w["well"]) not in excluded]
            for r, ws in all_wells.items()}
    picked = _picked(_build_merged_pick_list(kept, {"N79*": 0}, None))
    assert picked["N79*"] == ("R1_2", "M6")


def test_limits_come_from_each_rounds_recorded_pick():
    project = {
        "workflow_steps": {"pick": {"max_disagreement": 0.10}},
        "rounds": {"2": {"workflow_steps": {"pick": {"max_disagreement": None}}}},
    }
    assert _round_limits(project, [1, 2], None) == {1: 0.10, 2: None}


def test_an_override_applies_to_every_round():
    project = {"workflow_steps": {"pick": {"max_disagreement": 0.10}}}
    assert _round_limits(project, [1, 2], 0.05) == {1: 0.05, 2: 0.05}


def test_a_round_that_recorded_nothing_keeps_the_mixed_threshold():
    assert _round_limits({}, [1], None) == {1: None}
    assert _within_limit(_well(1, "A1", "X", 100, 0.20), None)        # watch: kept
    assert not _within_limit(_well(1, "A1", "X", 100, 0.40), None)    # mixed: out


def test_the_limit_excludes_the_watch_band_and_keeps_the_unmeasured():
    assert not _within_limit(_well(2, "K7", "G45A", 516, 0.192), 0.10)
    assert _within_limit(_well(1, "A1", "G45A", 30, 0.10), 0.10)
    assert _within_limit(_well(1, "A1", "G45A", 30, None), 0.10)


def test_reads_that_agree_outrank_more_reads():
    assert _well_rank(_well(1, "B1", "T26F", 381, 0.05)) < \
        _well_rank(_well(14, "C1", "T26F", 1543, 0.179))
    assert _well_rank(_well(3, "E12", "T26F", 684, 0.05)) < \
        _well_rank(_well(1, "B1", "T26F", 381, 0.05))


def test_the_merged_plate_follows_the_strict_pick():
    """Round 1 picked strictly; the merge must not put its watch wells back."""
    all_wells = {
        1: [_well(2, "K7", "G45A", 516, 0.192),      # watch, round 1's only G45A
            _well(14, "C1", "T26F", 1543, 0.179),    # watch, deep
            _well(1, "B1", "T26F", 381, 0.05)],      # clean
        2: [],
    }
    picked = _picked(_build_merged_pick_list(all_wells, ORDER, None,
                                             limits={1: 0.10, 2: None}))
    assert "G45A" not in picked
    assert picked["T26F"] == ("R1_1", "B1")


def test_a_later_round_fills_what_a_strict_earlier_round_dropped():
    all_wells = {
        1: [_well(2, "K7", "G45A", 516, 0.192)],
        2: [_well(1, "C10", "G45A", 223, 0.036)],
    }
    picked = _picked(_build_merged_pick_list(all_wells, ORDER, None,
                                             limits={1: 0.10, 2: None}))
    assert picked["G45A"] == ("R2_1", "C10")


def test_without_limits_the_old_behaviour_holds_except_for_the_ranking():
    all_wells = {1: [_well(2, "K7", "G45A", 516, 0.192),
                     _well(4, "A1", "G45A", 40, 0.03)]}
    picked = _picked(_build_merged_pick_list(all_wells, ORDER, None))
    # watch is allowed with no limit, but the clean well now wins on agreement
    assert picked["G45A"] == ("R1_4", "A1")
