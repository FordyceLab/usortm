"""Tests for the read-length histogram's axis.

A nanopore run produces a few concatemers hundreds of times the amplicon
length. Scaling the axis to the longest read put every real read in the first
bin of fifty: on a real run the axis ran to 375,350 bp against a median of
2,054, and the chart showed one bar.
"""

import pytest

from usortm.demux.utils import _hist_from_length_counts


def _amplicon(mode=2000, spread=200, depth=5000):
    """A realistic run: one mode, many reads."""
    return {mode + i: depth for i in range(spread)}


class TestTheAxisIgnoresTheLongTail:

    def test_one_concatemer_does_not_set_the_scale(self):
        clean = _hist_from_length_counts(_amplicon())
        with_tail = _hist_from_length_counts({**_amplicon(), 375_350: 1})
        assert with_tail["bin_size"] == clean["bin_size"]

    def test_the_axis_stays_near_the_reads(self):
        h = _hist_from_length_counts({**_amplicon(), 375_350: 1})
        assert h["bin_size"] * 50 < 10_000

    def test_the_distribution_is_still_spread_across_bins(self):
        """The failure was everything landing in bin zero."""
        h = _hist_from_length_counts({**_amplicon(), 375_350: 1})
        assert sum(1 for c in h["counts"] if c) > 1


class TestNothingIsLost:

    def test_every_read_is_counted(self):
        counts = {**_amplicon(), 375_350: 1, 120_000: 2}
        h = _hist_from_length_counts(counts)
        assert sum(h["counts"]) == sum(counts.values()) == h["n_reads"]

    def test_reads_past_the_cap_land_in_the_last_bin(self):
        h = _hist_from_length_counts({**_amplicon(), 375_350: 1, 120_000: 2})
        assert h["n_over"] == 3
        assert h["counts"][-1] >= 3

    def test_the_longest_is_reported_even_though_it_is_off_scale(self):
        """So a reader can tell one concatemer from thousands."""
        h = _hist_from_length_counts({**_amplicon(), 375_350: 1})
        assert h["longest"] == 375_350

    def test_a_run_with_no_tail_reports_none_over(self):
        assert _hist_from_length_counts(_amplicon())["n_over"] == 0


class TestTheMedian:

    def test_it_is_taken_over_every_read(self):
        """Robust already, so the tail needs no trimming for it."""
        h = _hist_from_length_counts({**_amplicon(), 375_350: 1})
        assert 2000 <= h["median"] < 2200

    def test_a_genuinely_long_run_is_not_squashed(self):
        """A run whose reads really are long must scale to them."""
        h = _hist_from_length_counts(_amplicon(mode=40_000, spread=2_000))
        assert h["bin_size"] * 50 > 40_000


class TestEdges:

    def test_no_reads(self):
        assert _hist_from_length_counts({}) == {}

    def test_a_single_length(self):
        h = _hist_from_length_counts({500: 10})
        assert h["n_reads"] == 10
        assert h["median"] == 500
        assert sum(h["counts"]) == 10

    def test_fifty_bins_always(self):
        assert len(_hist_from_length_counts(_amplicon())["counts"]) == 50
