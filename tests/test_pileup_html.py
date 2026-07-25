"""Tests for per-well pileup HTML generation.

These cover the metrics layered on top of the alignment — read identity,
consensus translation, read counts, and problem-position flagging — which are
what a user actually reads off the page when judging whether a well is
recoverable.  They exercise :func:`_render_pileup_html` directly with synthetic
rows, so no minimap2/samtools/BAM is required.
"""

import os
import re
import tempfile

import pytest

from usortm.demux.streakout import (
    PILEUP_DEL,
    PILEUP_MIN_FLAG_DEPTH,
    PILEUP_NOCOV,
    _render_pileup_html,
    _run_pileup_alignment,
)

# 66 bp: ATG, then 30 bp of A, 30 bp of T, then GGG.
REF = "ATG" + "AAA" * 10 + "TTT" * 10 + "GGG"


def make_row(spec, ref=REF):
    """Build a pileup row from a spec string.

    Characters: a base = observed base, ``*`` = deletion within the aligned
    span, ``-`` = position not covered by the read.
    """
    assert len(spec) == len(ref), "spec must be reference-length"
    row = []
    for i, c in enumerate(spec):
        if c == PILEUP_DEL:
            row.append((PILEUP_DEL, False))
        elif c == PILEUP_NOCOV:
            row.append((PILEUP_NOCOV, False))
        else:
            row.append((c, c.upper() == ref[i].upper()))
    return row


def make_section(**overrides):
    section = {
        "ref_id": "variant_1",
        "n_reads": 1,
        "n_aligned": 1,
        "frac": 1.0,
        "status": "Mutation",
        "is_recoverable": False,
        "ref_seq": REF,
        "pileup_rows": [],
    }
    section.update(overrides)
    return section


CANDIDATE = {
    "plate": "1",
    "well": "A1",
    "recoverable_variants": [],
    "total_reads": 40,
    "top_frac": 0.75,
}


def render(sections, **kwargs):
    return _render_pileup_html("1A1", CANDIDATE, sections, **kwargs)


def identity_of(html):
    m = re.search(r"Identity \(all reads\): ([\d.]+)%", html)
    return float(m.group(1)) if m else None


def flagged_of(html):
    return re.search(r"var flaggedCols=(\[[^\]]*\])", html).group(1)


def group_meta_of(html):
    m = re.search(r'<span class="group-meta">(.*?)</span></div>', html, re.S)
    return re.sub(r"<[^>]+>", "", m.group(1))


def js_string(html, var):
    m = re.search(r'var %s=("(?:[^"\\]|\\.)*")' % var, html)
    return m.group(1).strip('"') if m else None


class TestReadIdentity:
    """Deletions are disagreements with the reference, not missing data."""

    def test_deletion_is_not_scored_as_a_perfect_read(self):
        row = make_row(REF[:20] + PILEUP_DEL * 30 + REF[50:])
        html = render([make_section(pileup_rows=[row])])
        assert identity_of(html) < 100.0

    def test_deletion_scores_worse_than_a_few_substitutions(self):
        deleted = make_row(REF[:20] + PILEUP_DEL * 30 + REF[50:])
        subbed = list(REF)
        for i in (21, 25, 31, 37, 43):
            subbed[i] = "C" if REF[i] != "C" else "A"
        substituted = make_row("".join(subbed))

        id_del = identity_of(render([make_section(pileup_rows=[deleted])]))
        id_sub = identity_of(render([make_section(pileup_rows=[substituted])]))
        assert id_del < id_sub

    def test_uncovered_positions_do_not_lower_identity(self):
        """A short read that matches everywhere it reaches is still 100%."""
        row = make_row(PILEUP_NOCOV * 20 + REF[20:46] + PILEUP_NOCOV * 20)
        assert identity_of(render([make_section(pileup_rows=[row])])) == 100.0

    def test_perfect_full_length_read_is_100_percent(self):
        assert identity_of(render([make_section(pileup_rows=[make_row(REF)])])) == 100.0


class TestConsensusTranslation:
    """A frameshift must shift the frame rather than being padded with N."""

    def test_single_base_deletion_shortens_the_consensus_protein(self):
        row = make_row(REF[:10] + PILEUP_DEL + REF[11:])
        html = render([make_section(pileup_rows=[row])], flank_lengths=(3, 3))
        assert len(js_string(html, "consAA")) < len(js_string(html, "refAA"))

    def test_single_base_deletion_diverges_downstream(self):
        row = make_row(REF[:10] + PILEUP_DEL + REF[11:])
        html = render([make_section(pileup_rows=[row])], flank_lengths=(3, 3))
        ref_aa, cons_aa = js_string(html, "refAA"), js_string(html, "consAA")
        assert ref_aa[3:] != cons_aa[3:]

    def test_frameshift_is_labelled(self):
        row = make_row(REF[:10] + PILEUP_DEL + REF[11:])
        html = render([make_section(pileup_rows=[row])], flank_lengths=(3, 3))
        assert "-1 bp frameshift" in html

    def test_in_frame_deletion_is_not_called_a_frameshift(self):
        row = make_row(REF[:12] + PILEUP_DEL * 3 + REF[15:])
        html = render([make_section(pileup_rows=[row])], flank_lengths=(3, 3))
        assert "-3 bp in-frame indel" in html
        assert "frameshift" not in html

    def test_clean_read_reports_no_indel(self):
        html = render([make_section(pileup_rows=[make_row(REF)])],
                      flank_lengths=(3, 3))
        assert "bp frameshift" not in html
        assert "in-frame indel" not in html

    def test_substitution_only_keeps_reference_length_protein(self):
        subbed = list(REF)
        subbed[12] = "C" if REF[12] != "C" else "A"
        html = render([make_section(pileup_rows=[make_row("".join(subbed))])],
                      flank_lengths=(3, 3))
        assert len(js_string(html, "consAA")) == len(js_string(html, "refAA"))


class TestReadCounts:
    """Group size and aligned-row count are different numbers; report both."""

    def test_empty_group_reports_its_real_size(self):
        html = render([make_section(n_reads=17, n_aligned=0, pileup_rows=[])])
        empty = re.search(r'<div class="pileup-empty">(.*?)</div>', html, re.S).group(1)
        assert "17" in empty

    def test_header_shows_aligned_of_group_when_they_differ(self):
        rows = [make_row(REF)] * 12
        html = render([make_section(n_reads=30, n_aligned=12, frac=0.45,
                                   pileup_rows=rows)])
        assert "12 of 30 reads aligned" in group_meta_of(html)

    def test_header_is_unqualified_when_all_reads_aligned(self):
        rows = [make_row(REF)] * 5
        html = render([make_section(n_reads=5, n_aligned=5, pileup_rows=rows)])
        meta = group_meta_of(html)
        assert "5 reads" in meta and "of" not in meta.split("&middot;")[0]


class TestFlaggedColumns:
    """Problem positions need a depth floor to mean anything."""

    def test_single_read_column_is_not_flagged(self):
        row = make_row(PILEUP_NOCOV * 30 + "C" + PILEUP_NOCOV * 35)
        assert flagged_of(render([make_section(pileup_rows=[row])])) == "[]"

    def test_column_at_min_depth_is_flagged(self):
        rows = [make_row(PILEUP_NOCOV * 30 + "C" + PILEUP_NOCOV * 35)
                for _ in range(PILEUP_MIN_FLAG_DEPTH)]
        html = render([make_section(n_reads=len(rows), n_aligned=len(rows),
                                   pileup_rows=rows)])
        assert flagged_of(html) == "[30]"

    def test_deletions_count_toward_flagging(self):
        """A column deleted in most reads is a problem position."""
        rows = [make_row(REF[:30] + PILEUP_DEL + REF[31:]) for _ in range(4)]
        html = render([make_section(n_reads=4, n_aligned=4, pileup_rows=rows)])
        assert "30" in flagged_of(html)

    def test_clean_deep_pileup_flags_nothing(self):
        rows = [make_row(REF) for _ in range(10)]
        html = render([make_section(n_reads=10, n_aligned=10, pileup_rows=rows)])
        assert flagged_of(html) == "[]"


class TestPayloadEncoding:
    def test_deletion_and_no_coverage_are_distinct_symbols(self):
        row = make_row(REF[:20] + PILEUP_DEL * 5 + PILEUP_NOCOV * 5 + REF[30:])
        html = render([make_section(pileup_rows=[row])])
        encoded = re.search(r'var rows=\["([^"]*)"\]', html).group(1)
        assert PILEUP_DEL in encoded and PILEUP_NOCOV in encoded

    def test_majority_deletion_appears_in_consensus(self):
        rows = [make_row(REF[:30] + PILEUP_DEL + REF[31:]) for _ in range(5)]
        html = render([make_section(n_reads=5, n_aligned=5, pileup_rows=rows)])
        assert js_string(html, "cons")[30] == PILEUP_DEL

    def test_script_close_cannot_escape_the_inline_block(self):
        html = render([make_section(pileup_rows=[make_row(REF)])])
        for var in ("ref", "cons"):
            assert "</" not in js_string(html, var)

    def test_page_documents_the_insertion_limitation(self):
        html = render([make_section(pileup_rows=[make_row(REF)])])
        assert "Insertions are not shown" in html

    def test_canvas_area_is_clamped(self):
        html = render([make_section(pileup_rows=[make_row(REF)])])
        assert "MAX_CANVAS_AREA" in html


class TestRendersAtAll:
    @pytest.mark.parametrize("flanks", [None, (3, 3)])
    def test_produces_a_complete_document(self, flanks):
        rows = [make_row(REF), make_row(REF[:40] + PILEUP_DEL * 2 + REF[42:])]
        html = render([make_section(n_reads=2, n_aligned=2, pileup_rows=rows)],
                      flank_lengths=flanks)
        assert html.startswith("<!DOCTYPE html>")
        assert html.rstrip().endswith("</html>")
        assert "drawPileup(" in html

    def test_multiple_groups_each_get_a_section(self):
        sections = [
            make_section(ref_id="var_A", frac=0.7, pileup_rows=[make_row(REF)]),
            make_section(ref_id="var_B", frac=0.3, pileup_rows=[make_row(REF)]),
        ]
        html = render(sections)
        assert "var_A" in html and "var_B" in html
        # One call site per group (the shared function definition also matches
        # a bare "drawPileup(", so count invocations by their first argument).
        assert html.count('drawPileup("pileup-') == 2

    def test_zero_length_reference_does_not_raise(self):
        html = render([make_section(ref_seq="", n_reads=3, n_aligned=0,
                                   pileup_rows=[])])
        assert html.startswith("<!DOCTYPE html>")


class TestAlignmentFailureHandling:
    """A broken toolchain must fail fast, not block on the minimap2 pipe.

    The parent has to drop its copy of minimap2's stdout once samtools owns it,
    or minimap2 never receives EPIPE when samtools dies and ``wait()`` blocks
    forever.  These assert the failure is reported instead.
    """

    @pytest.fixture
    def alignment_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            fq = os.path.join(td, "reads.fastq")
            fa = os.path.join(td, "ref.fasta")
            with open(fq, "w") as fh:
                fh.write("@r1\nACGTACGTACGT\n+\nIIIIIIIIIIII\n")
            with open(fa, "w") as fh:
                fh.write(">ref\nACGTACGTACGTACGTACGT\n")
            yield fq, fa, os.path.join(td, "out.bam")

    def test_missing_minimap2_returns_false(self, alignment_inputs):
        fq, fa, out = alignment_inputs
        assert _run_pileup_alignment(
            fq, fa, out, "/nonexistent/minimap2", "/nonexistent/samtools"
        ) is False

    def test_missing_samtools_returns_false(self, alignment_inputs):
        """The case that could previously hang: samtools exits, minimap2 waits."""
        fq, fa, out = alignment_inputs
        mm2 = pytest.importorskip("usortm.demux.deps").find_minimap2()
        if not mm2:
            pytest.skip("minimap2 not installed")
        assert _run_pileup_alignment(
            fq, fa, out, mm2, "/nonexistent/samtools"
        ) is False

    def test_missing_reference_returns_false(self, alignment_inputs):
        fq, _fa, out = alignment_inputs
        deps = pytest.importorskip("usortm.demux.deps")
        mm2, st = deps.find_minimap2(), deps.find_samtools()
        if not (mm2 and st):
            pytest.skip("minimap2/samtools not installed")
        assert _run_pileup_alignment(
            fq, out + ".missing.fasta", out, mm2, st
        ) is False

    def test_successful_alignment_returns_true(self, alignment_inputs):
        fq, fa, out = alignment_inputs
        deps = pytest.importorskip("usortm.demux.deps")
        mm2, st = deps.find_minimap2(), deps.find_samtools()
        if not (mm2 and st):
            pytest.skip("minimap2/samtools not installed")
        assert _run_pileup_alignment(fq, fa, out, mm2, st) is True
        assert os.path.exists(out)
