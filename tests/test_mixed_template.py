"""The mixed-template criterion, and that it is actually reached.

The filter reads ``max_mismatch_frac``.  When the loader did not carry that
column the test could not fail a well, so every well passed and the tier counts
doubled without anything reporting a problem.  The first test here is aimed at
that: it goes through the loader rather than hand-building well records.
"""
import csv

from usortm.cli.report import _compute_quality_bins, _load_well_assignments
from usortm.demux.utils import MIXED_TEMPLATE_CLEAR, MIXED_TEMPLATE_THRESHOLD


HEADER = ["plate", "well", "reads", "variant", "consensus_fraction",
          "cons_check", "flank_check", "n_flagged_positions",
          "max_mismatch_frac"]


def _write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(rows)
    return path


def _well(plate, well, variant, mismatch, reads=200):
    return [plate, well, reads, variant, 1.0, "Perfect Match", "OK", 0,
            mismatch]


def test_loader_carries_the_column_the_filter_reads(tmp_path):
    path = _write(tmp_path / "wa.csv", [_well(1, "A1", "V1", 0.5)])
    loaded = _load_well_assignments(path)
    assert loaded[0]["max_mismatch_frac"] == 0.5


def test_a_well_at_the_threshold_still_counts(tmp_path):
    path = _write(tmp_path / "wa.csv",
                  [_well(1, "A1", "V1", MIXED_TEMPLATE_THRESHOLD)])
    bins = _compute_quality_bins(_load_well_assignments(path), 1)
    assert bins["recovery_tiers"]["C"]["count"] == 1


def test_a_well_past_the_threshold_does_not(tmp_path):
    path = _write(tmp_path / "wa.csv",
                  [_well(1, "A1", "V1", MIXED_TEMPLATE_THRESHOLD + 0.01)])
    bins = _compute_quality_bins(_load_well_assignments(path), 1)
    assert bins["recovery_tiers"]["C"]["count"] == 0


def test_base_calling_noise_is_not_a_mixed_template(tmp_path):
    """A column at 10-15% is where the noise sits, and must survive.

    This is the case the old 10% threshold rejected: across one run it was the
    mode of the distribution, not its tail.
    """
    path = _write(tmp_path / "wa.csv", [
        _well(1, "A1", "V1", 0.104),
        _well(1, "A2", "V2", 0.118),
        _well(1, "A3", "V3", 0.145),
    ])
    bins = _compute_quality_bins(_load_well_assignments(path), 3)
    assert bins["recovery_tiers"]["C"]["count"] == 3


def test_an_even_split_is_rejected(tmp_path):
    """Two templates in one well land near 50% and must not count."""
    path = _write(tmp_path / "wa.csv", [_well(1, "A1", "V1", 0.51)])
    bins = _compute_quality_bins(_load_well_assignments(path), 1)
    assert bins["recovery_tiers"]["C"]["count"] == 0


def test_the_threshold_sits_between_the_two_populations():
    assert MIXED_TEMPLATE_THRESHOLD > 0.15    # above the noise mode
    assert MIXED_TEMPLATE_THRESHOLD < MIXED_TEMPLATE_CLEAR
    assert MIXED_TEMPLATE_CLEAR <= 0.5        # at or below an even split


def test_a_run_without_the_column_is_not_silently_filtered(tmp_path):
    """An older run has no such column; every well should still be judged."""
    path = tmp_path / "wa.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER[:7])
        w.writerow([1, "A1", 200, "V1", 1.0, "Perfect Match", "OK"])
    bins = _compute_quality_bins(_load_well_assignments(path), 1)
    assert bins["recovery_tiers"]["C"]["count"] == 1


def test_classes_partition_the_range():
    """Every well falls in exactly one band, with no gap at a boundary."""
    from usortm.demux.utils import (MIXED_TEMPLATE_WATCH,
                                    column_agreement_class)
    assert column_agreement_class(MIXED_TEMPLATE_WATCH) == "clean"
    assert column_agreement_class(MIXED_TEMPLATE_WATCH + 0.001) == "watch"
    assert column_agreement_class(MIXED_TEMPLATE_THRESHOLD) == "watch"
    assert column_agreement_class(MIXED_TEMPLATE_THRESHOLD + 0.001) == "mixed"


def test_a_missing_value_is_not_treated_as_clean():
    """An unjudgeable well must be distinguishable from one measured clean."""
    from usortm.demux.utils import column_agreement_class
    assert column_agreement_class(None) == "unknown"
    assert column_agreement_class("") == "unknown"
    assert column_agreement_class("not a number") == "unknown"


def test_watch_band_is_kept_by_the_tiers(tmp_path):
    """A watch well is marked, not rejected."""
    from usortm.demux.utils import column_agreement_class
    path = _write(tmp_path / "wa.csv", [_well(1, "A1", "V1", 0.15)])
    loaded = _load_well_assignments(path)
    assert column_agreement_class(loaded[0]["max_mismatch_frac"]) == "watch"
    bins = _compute_quality_bins(loaded, 1)
    assert bins["recovery_tiers"]["C"]["count"] == 1


def test_pick_loader_carries_the_column(tmp_path):
    """pick has its own reader; it must carry what its filter reads."""
    from usortm.cli.pick import _load_well_assignments as pick_load
    path = _write(tmp_path / "wa.csv", [_well(1, "A1", "V1", 0.5)])
    assert pick_load(path)[0]["max_mismatch_frac"] == 0.5


def test_pick_carries_the_band_onto_each_pick(tmp_path):
    """A pick keeps the fraction, so the count reported is of picks.

    Counted over eligible wells instead the number is several times larger --
    one variant is picked from however many wells hold it -- and reads as if
    most of the plate needed checking.
    """
    from usortm.cli.pick import _generate_pick_list
    from usortm.cli.pick import _load_well_assignments as pick_load
    from usortm.demux.utils import column_agreement_class

    path = _write(tmp_path / "wa.csv", [
        _well(1, "A1", "V1", 0.15, reads=300),
        _well(1, "A2", "V1", 0.16, reads=200),   # same variant, not picked
        _well(1, "A3", "V1", 0.17, reads=100),   # same variant, not picked
        _well(1, "B1", "V2", 0.02, reads=300),
    ])
    picks = _generate_pick_list(
        pick_load(path), None, True, 384, "row",
        library_order={"V1": 0, "V2": 1},
    )
    filled = [p for p in picks if not p.get("empty")]
    assert len(filled) == 2, "one well per variant"
    assert all("max_mismatch_frac" in p for p in filled)
    watch = [p for p in filled
             if column_agreement_class(p["max_mismatch_frac"]) == "watch"]
    assert len(watch) == 1, "V1 is picked once, not three times"
