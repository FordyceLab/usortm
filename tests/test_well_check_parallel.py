"""Tests for running the per-well variant checks across processes.

Threads bought nothing: the work is pysam walking every aligned pair and
building the result in Python, with no external tool to release the
interpreter lock. Measured on real wells, eight threads came out at 0.95x of
one and eight processes at 3.26x.

What is worth pinning is not the speed but the two things that make it safe:
the same answers whatever the worker count, and only the fields the check
reads crossing the process boundary.
"""

import pandas as pd
import pytest

from usortm.demux.utils import (
    _extract_matches_one,
    _map_well_checks,
    _well_check_task,
    extract_matches,
)

FIELDS = ("global_well", "ref_len", "ref_seq", "CIGAR", "cons_seq")


def _row(well="1A1", cons="ACGT" * 24, ref="ACGT" * 24):
    return {"global_well": well, "ref_len": len(ref), "ref_seq": ref,
            "CIGAR": f"{len(cons)}M", "cons_seq": cons}


class TestTheWorker:

    def test_it_is_picklable(self):
        """It has to cross a process boundary, so it cannot be a closure."""
        import pickle

        pickle.dumps(_well_check_task)

    def test_it_returns_the_index_it_was_given(self):
        """Results come back out of order, so each carries its own row."""
        index, result = _well_check_task((7, _row()), (0, 0, None, 0, False, ""))
        assert index == 7
        assert result is not None

    def test_a_failing_well_does_not_take_the_others_with_it(self):
        bad = (3, {"global_well": "1A1"})     # missing everything else
        index, result = _well_check_task(bad, (0, 0, None, 0, False, ""))
        assert index == 3
        assert result is None


class TestWhatCrossesTheBoundary:
    """ref_seq and cons_seq run to kilobytes each; sending whole rows would
    pickle every column of every well for nothing."""

    def test_only_the_fields_the_check_reads(self):
        big = pd.DataFrame([{
            **_row(),
            "depth": 500, "major_ref": "V1", "major_freq": 0.9,
            "notes": "x" * 10_000,
        }])
        out = extract_matches(big, workers=1)
        # The check ran and wrote its columns back onto the full frame.
        assert "cons_check" in out.columns
        assert out["notes"].iloc[0] == "x" * 10_000

    def test_the_worker_accepts_a_plain_mapping(self):
        """No pandas object needs to be sent, only a dict of five fields."""
        fields = _row()
        assert set(fields) == set(FIELDS)
        out = _extract_matches_one(fields, 0, 0, None, 0, False, "")
        assert "cons_check" in out


class TestSameAnswersEitherWay:

    def _frame(self, n=6):
        return pd.DataFrame([_row(well=f"1A{i + 1}") for i in range(n)])

    def test_one_worker_and_many_agree(self):
        serial = extract_matches(self._frame(), workers=1)
        parallel = extract_matches(self._frame(), workers=4)
        cols = [c for c in ("cons_check", "protein_check") if c in serial]
        assert serial[cols].equals(parallel[cols])

    def test_every_row_is_written_back(self):
        out = extract_matches(self._frame(n=5), workers=4)
        assert out["cons_check"].notna().all()

    def test_an_empty_frame_is_returned_unchanged(self):
        empty = pd.DataFrame(columns=list(FIELDS))
        assert extract_matches(empty, workers=4).empty


class TestMapping:

    def test_a_single_task_skips_the_pool(self):
        """Starting a pool for one well costs more than the well does."""
        got = list(_map_well_checks([(0, _row())], (0, 0, None, 0, False, ""),
                                    workers=8))
        assert len(got) == 1
        assert got[0][0] == 0

    def test_every_task_comes_back(self):
        tasks = [(i, _row(well=f"1A{i + 1}")) for i in range(5)]
        got = list(_map_well_checks(tasks, (0, 0, None, 0, False, ""),
                                    workers=4))
        assert sorted(i for i, _ in got) == [0, 1, 2, 3, 4]
