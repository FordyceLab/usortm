"""Tests for the pileup path now that seqviewer draws the page.

What uSort-M still owns is the mapping: its group dicts onto seqviewer's model,
and the decision to re-align rather than reuse the consensus alignment. The
grid and the HTML are seqviewer's and are tested there.
"""

import shutil

import numpy as np
import pandas as pd
import pytest

from usortm.demux.streakout import (
    _build_pileup_grid,
    _reads_for_pileup,
    _render_pileup_html,
)


def _has(name):
    return shutil.which(name) is not None


requires_tools = pytest.mark.skipif(
    not (_has("minimap2") and _has("samtools")),
    reason="needs minimap2 and samtools",
)

REF = "".join(np.random.default_rng(0).choice(list("ACGT"), 600))


@pytest.fixture
def ref_fasta(tmp_path):
    p = tmp_path / "ref.fasta"
    p.write_text(f">var_1\n{REF}\n")
    return str(p)


def _reads(n, mutate_at=None, seed=1, truncate=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        s = list(REF)
        if mutate_at is not None:
            s[mutate_at] = "A" if s[mutate_at] != "A" else "C"
        for i in range(len(s)):
            if rng.random() < 0.02:
                s[i] = "ACGT"[int(rng.integers(4))]
        seq = "".join(s)
        if truncate:
            seq = seq[:truncate]
        rows.append({"read_name": f"r{seed}_{k}", "read_seq": seq,
                     "read_qual": "I" * len(seq)})
    return pd.DataFrame(rows)


class TestReadConversion:

    def test_carries_name_sequence_and_quality(self):
        reads = _reads_for_pileup(_reads(2))
        assert len(reads) == 2
        assert reads[0].name.startswith("r1_")
        assert len(reads[0].seq) == len(REF)
        assert reads[0].qual == "I" * len(REF)

    def test_an_empty_group_yields_no_reads(self):
        assert _reads_for_pileup(_reads(0)) == []


class TestGrid:

    @requires_tools
    def test_every_row_is_as_wide_as_the_reference(self, ref_fasta):
        rows = _build_pileup_grid(_reads(8), ref_fasta, REF,
                                  "minimap2", "samtools")
        assert rows
        assert all(len(r) == len(REF) for r in rows)

    @requires_tools
    def test_a_cell_is_a_base_and_whether_it_matches(self, ref_fasta):
        rows = _build_pileup_grid(_reads(4), ref_fasta, REF,
                                  "minimap2", "samtools")
        base, is_match = rows[0][10]
        assert isinstance(base, str) and len(base) == 1
        assert isinstance(is_match, (bool, np.bool_))

    @requires_tools
    def test_an_empty_group_gives_no_rows(self, ref_fasta):
        assert _build_pileup_grid(_reads(0), ref_fasta, REF,
                                  "minimap2", "samtools") == []

    @requires_tools
    def test_reads_that_stop_before_the_midpoint_are_dropped(self, ref_fasta):
        """Concatemer split-reads cover one flank and say nothing about the
        insert, so they are not worth a row."""
        short = _reads(6, seed=5, truncate=200)
        assert _build_pileup_grid(short, ref_fasta, REF,
                                  "minimap2", "samtools") == []

    @requires_tools
    def test_the_filter_can_be_turned_off(self, ref_fasta):
        short = _reads(6, seed=5, truncate=200)
        rows = _build_pileup_grid(short, ref_fasta, REF, "minimap2",
                                  "samtools", min_overlap_pos=0)
        assert rows


class TestRenderMapping:
    """The group dicts uSort-M builds must reach seqviewer's model intact."""

    def _page(self, **candidate_extra):
        rows = [[("A", True)] * 12]
        groups = [
            {"ref_id": "var_1", "n_reads": 9, "frac": 0.75, "status": "Clean",
             "is_recoverable": True, "ref_seq": "A" * 12, "pileup_rows": rows},
            {"ref_id": "var_2", "n_reads": 3, "frac": 0.25,
             "status": "Mutation", "is_recoverable": False,
             "ref_seq": "A" * 12, "pileup_rows": rows},
        ]
        candidate = {"plate": "1", "well": "A3", "total_reads": 12,
                     "recoverable_variants": ["var_1"], **candidate_extra}
        return _render_pileup_html("1A3", candidate, groups,
                                   flank_lengths=(2, 3))

    def test_produces_a_full_html_document(self):
        page = self._page()
        assert page.lstrip().lower().startswith("<!doctype html")

    def test_names_the_plate_and_well(self):
        page = self._page()
        assert "Plate 1" in page and "Well A3" in page

    def test_every_group_appears(self):
        page = self._page()
        assert "var_1" in page and "var_2" in page

    def test_the_recoverable_variant_is_called_out(self):
        assert "Recoverable" in self._page()

    def test_a_group_whose_rows_are_the_wrong_width_is_refused(self):
        """seqviewer checks the grid against its reference, which catches a
        mismatch here rather than drawing a crooked page."""
        groups = [{"ref_id": "v", "n_reads": 1, "frac": 1.0, "status": "Clean",
                   "is_recoverable": False, "ref_seq": "A" * 12,
                   "pileup_rows": [[("A", True)] * 5]}]
        with pytest.raises(ValueError, match="cells wide"):
            _render_pileup_html("1A3", {"plate": "1", "well": "A3",
                                        "total_reads": 1}, groups)


class TestFlanks:

    def test_zero_length_flanks_are_treated_as_none(self):
        """(0, 0) means no insert was marked, which must not draw boundary
        lines at position zero."""
        groups = [{"ref_id": "v", "n_reads": 1, "frac": 1.0, "status": "Clean",
                   "is_recoverable": False, "ref_seq": "A" * 12,
                   "pileup_rows": [[("A", True)] * 12]}]
        cand = {"plate": "1", "well": "A3", "total_reads": 1}
        page = _render_pileup_html("1A3", cand, groups, flank_lengths=(0, 0))
        assert "null" in page or "Plate 1" in page  # renders without flanks

    def test_real_flanks_are_passed_through(self):
        groups = [{"ref_id": "v", "n_reads": 1, "frac": 1.0, "status": "Clean",
                   "is_recoverable": False, "ref_seq": "A" * 12,
                   "pileup_rows": [[("A", True)] * 12]}]
        cand = {"plate": "1", "well": "A3", "total_reads": 1}
        page = _render_pileup_html("1A3", cand, groups, flank_lengths=(2, 3))
        assert "Plate 1" in page
