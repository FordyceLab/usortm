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
