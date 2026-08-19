"""Tests for how the resolved plate mapping is shown.

This table is what the user is asked to approve before a run, so it has to
state the mapping that will actually be applied. Listing barcode plates and
sort plates as two independently sorted columns loses the correspondence:
7->7, 8->8, 1->9, 2->10 reads across as 1->7, 2->8, 7->9, 8->10.
"""

import re

import pytest

from usortm.cli.demux_cmd import _describe_segments
from usortm.demux.plate_map import Segment


def _rendered(segments, capsys):
    _describe_segments(segments)
    out = capsys.readouterr().out
    # Strip styling and collapse wrapping so assertions do not depend on width.
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", out).split())


REUSED = Segment(name="run2", path="b", plates={7: 7, 8: 8, 1: 9, 2: 10})
IDENTITY = Segment(name="run1", path="a", plates={i: i for i in range(1, 7)})


class TestPairsAreExplicit:

    def test_each_pair_is_shown(self, capsys):
        out = _rendered([REUSED], capsys)
        for bc, sort in ((7, 7), (8, 8), (1, 9), (2, 10)):
            assert f"{bc}→{sort}" in out, f"missing {bc}->{sort}"

    def test_the_transposed_reading_does_not_appear(self, capsys):
        """The regression: sorting both columns implied 1->7 and 7->9."""
        out = _rendered([REUSED], capsys)
        for wrong in ("1→7", "2→8", "7→9", "8→10"):
            assert wrong not in out, f"table implies {wrong}"

    def test_pairs_are_ordered_by_sort_plate(self, capsys):
        """Sort plate is what the bench cares about, so lead with its order."""
        out = _rendered([REUSED], capsys)
        positions = [out.index(f"{bc}→{s}")
                     for bc, s in ((7, 7), (8, 8), (1, 9), (2, 10))]
        assert positions == sorted(positions)

    def test_identity_mapping_reads_plainly(self, capsys):
        out = _rendered([IDENTITY], capsys)
        assert "1→1" in out and "6→6" in out

    def test_every_segment_appears(self, capsys):
        out = _rendered([IDENTITY, REUSED], capsys)
        assert "run1" in out and "run2" in out

    def test_no_segments_still_renders(self, capsys):
        out = _rendered([], capsys)
        assert "FASTQ" in out
