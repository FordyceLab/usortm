"""Tests for working out a read template from a parent vector.

Each case builds a plasmid whose layout is known -- primer sites, flanks and
variable region at fixed places in a random backbone -- and LevSeq reads of
its amplicon, then checks that detection recovers that layout exactly and
that the template it writes parses back to the same flanks and masks.
"""

import shutil

import numpy as np
import pytest

from usortm.demux.barcodes import LEVSEQ_FBC, LEVSEQ_RBC
from usortm.demux.read_template import parse_read_template
from usortm.demux.vector_layout import (
    LayoutError,
    _catalog,
    _place_barcode,
    _rc,
    detect_layout,
    load_vector,
    locate_inserts,
    write_read_template,
)
from usortm.qc.synthetic import _apply_errors, _random_seq

requires_minimap2 = pytest.mark.skipif(
    shutil.which("minimap2") is None, reason="minimap2 not installed"
)

OUTER_5P = "AATATAAATT"
OUTER_3P = "ATAATTTATA"


class Plasmid:
    """A backbone with a barcoded amplicon at a known place.

    Layout: ``up | primer_5 | flank_5 | parent | flank_3 | primer_3 | down``.
    The amplicon runs from ``primer_5`` through ``primer_3``.
    """

    def __init__(self, seed=1, parent_len=300):
        self.rng = np.random.default_rng(seed)
        r = self.rng
        self.up, self.down = _random_seq(r, 1500), _random_seq(r, 1500)
        self.primer_5, self.primer_3 = _random_seq(r, 22), _random_seq(r, 26)
        self.flank_5, self.flank_3 = _random_seq(r, 150), _random_seq(r, 200)
        self.parent = _random_seq(r, parent_len)

    @property
    def sequence(self):
        return (self.up + self.primer_5 + self.flank_5 + self.parent
                + self.flank_3 + self.primer_3 + self.down)

    @property
    def template_flank_5(self):
        return self.primer_5 + self.flank_5

    @property
    def template_flank_3(self):
        return self.flank_3 + self.primer_3

    def scan(self, n):
        """Single-codon substitutions of the parent."""
        out = []
        for _ in range(n):
            pos = 3 * int(self.rng.integers(0, len(self.parent) // 3))
            out.append(self.parent[:pos] + _random_seq(self.rng, 3)
                       + self.parent[pos + 3:])
        return out

    def reads(self, inserts, n=1500, error_rate=0.03, flipped=False,
              rbc_range=(0, 4)):
        out = []
        r = self.rng
        for _ in range(n):
            insert = inserts[int(r.integers(len(inserts)))]
            fbc = LEVSEQ_FBC[int(r.integers(96))]
            rbc = LEVSEQ_RBC[int(r.integers(*rbc_range))]
            amplicon = self.primer_5 + self.flank_5 + insert + self.flank_3 + self.primer_3
            if flipped:
                # Forward barcode 3' of the insert: the primers swapped ends.
                amplicon = _rc(amplicon)
            read = OUTER_5P + fbc + amplicon + _rc(rbc) + OUTER_3P
            if r.random() < 0.5:
                read = _rc(read)
            out.append(_apply_errors(r, read, error_rate))
        return out


def _write(tmp_path, vector, reads):
    vec = tmp_path / "vector.fa"
    vec.write_text(f">parent\n{vector}\n")
    fq = tmp_path / "reads.fastq"
    with open(fq, "w") as fh:
        for i, s in enumerate(reads):
            fh.write(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n")
    return vec, fq


def _detect(tmp_path, vector, inserts, reads):
    vec, fq = _write(tmp_path, vector, reads)
    return detect_layout(vec, inserts, fq, shutil.which("minimap2"),
                         tmp_path / "work", threads=2)


def _assert_recovers(layout, plasmid, tmp_path, variable_length):
    template = parse_read_template(write_read_template(layout, tmp_path / "t.fa"))
    assert template.flank_5p == plasmid.template_flank_5
    assert template.flank_3p == plasmid.template_flank_3
    assert template.variable_length == variable_length
    assert template.masks["mask1_front"] == OUTER_5P
    assert template.masks["mask2_rear"] == OUTER_3P
    assert template.masks["mask1_rear"] == plasmid.template_flank_5[:22]
    assert template.masks["mask2_front"] == plasmid.template_flank_3[-22:]


class TestLoadVector:

    def test_reads_one_fasta_record(self, tmp_path):
        p = tmp_path / "v.fa"
        p.write_text(">v\nacgtACGT\n")
        assert load_vector(p) == "ACGTACGT"

    def test_masked_vector_is_refused(self, tmp_path):
        """A vector with X's is a --vector-fasta; saying so beats a bad layout."""
        p = tmp_path / "v.fa"
        p.write_text(">v\nACGTXXXXACGT\n")
        with pytest.raises(LayoutError, match="--vector-fasta"):
            load_vector(p)

    def test_several_records_are_refused(self, tmp_path):
        p = tmp_path / "v.fa"
        p.write_text(">a\nACGT\n>b\nACGT\n")
        with pytest.raises(LayoutError, match="one record"):
            load_vector(p)


class TestLocateInserts:

    def test_scan_members_locate_the_parent_span(self):
        p = Plasmid()
        strand, start, end, located = locate_inserts(p.sequence, p.scan(30))
        v0 = len(p.up) + len(p.template_flank_5)
        assert (strand, start, end) == ("+", v0, v0 + len(p.parent))
        assert located == 30

    def test_reverse_strand(self):
        p = Plasmid()
        strand, start, end, _ = locate_inserts(_rc(p.sequence), p.scan(20))
        assert strand == "-"
        assert end - start == len(p.parent)

    def test_span_across_the_origin(self):
        p = Plasmid()
        seq = p.sequence
        v0 = len(p.up) + len(p.template_flank_5)
        cut = v0 + 100                      # inside the parent
        rotated = seq[cut:] + seq[:cut]
        strand, start, end, _ = locate_inserts(rotated, p.scan(20))
        assert end - start == len(p.parent)
        assert end > len(rotated)           # reported as running past the origin

    def test_unrelated_sequences_are_not_located(self):
        p = Plasmid()
        others = [_random_seq(p.rng, 300) for _ in range(10)]
        assert locate_inserts(p.sequence, others) is None


class TestBarcodePlacement:

    def test_shared_barcode_is_not_its_own_competitor(self):
        """RB13-RB96 are the same sequences as FB13-FB96.  A read carrying one
        must still place, and must not claim to know which set it came from."""
        assert LEVSEQ_FBC[40] == LEVSEQ_RBC[40]
        cat = _catalog({"fbc": LEVSEQ_FBC, "rbc": LEVSEQ_RBC})
        hit = _place_barcode(OUTER_5P + LEVSEQ_FBC[40] + "ACGTACGTAC", cat)
        assert hit == (frozenset({"fbc", "rbc"}), len(OUTER_5P))

    def test_unshared_barcode_names_its_set(self):
        cat = _catalog({"fbc": LEVSEQ_FBC, "rbc": LEVSEQ_RBC})
        sets, _ = _place_barcode(OUTER_5P + LEVSEQ_FBC[3] + "ACGTACGTAC", cat)
        assert sets == frozenset({"fbc"})

    def test_noise_is_not_placed(self):
        cat = _catalog({"fbc": LEVSEQ_FBC, "rbc": LEVSEQ_RBC})
        rng = np.random.default_rng(0)
        assert _place_barcode(_random_seq(rng, 46), cat) is None


@requires_minimap2
class TestDetectLayout:

    def test_scan_library(self, tmp_path):
        p = Plasmid()
        inserts = p.scan(30)
        layout = _detect(tmp_path, p.sequence, inserts, p.reads(inserts))
        assert layout.variable_source == "expected sequences"
        assert not layout.reverse_complemented
        assert layout.orientation_votes[0] > 0 and layout.orientation_votes[1] == 0
        _assert_recovers(layout, p, tmp_path, len(p.parent))

    def test_reverse_strand_vector(self, tmp_path):
        p = Plasmid()
        inserts = p.scan(30)
        layout = _detect(tmp_path, _rc(p.sequence), inserts, p.reads(inserts))
        assert layout.reverse_complemented
        _assert_recovers(layout, p, tmp_path, len(p.parent))

    def test_amplicon_across_the_origin(self, tmp_path):
        p = Plasmid()
        inserts = p.scan(30)
        seq = p.sequence
        cut = len(p.up) + 60                # inside the 5' flank
        layout = _detect(tmp_path, seq[cut:] + seq[:cut], inserts, p.reads(inserts))
        _assert_recovers(layout, p, tmp_path, len(p.parent))

    def test_inserts_unrelated_to_the_vector(self, tmp_path):
        """A stuffer in the vector, replaced by inserts of another length: the
        variable region is the stretch of vector the reads never contain."""
        p = Plasmid()
        inserts = [_random_seq(p.rng, 400) for _ in range(20)]
        layout = _detect(tmp_path, p.sequence, inserts, p.reads(inserts, n=3000))
        assert layout.variable_source == "reads"
        _assert_recovers(layout, p, tmp_path, 400)

    def test_insert_sharing_the_stuffers_first_bases(self, tmp_path):
        """Bases the insert and stuffer happen to share at the junction read as
        flank; left there they would appear twice in the reference."""
        p = Plasmid()
        shared = p.parent[:2]
        inserts = [shared + _random_seq(p.rng, 398) for _ in range(20)]
        layout = _detect(tmp_path, p.sequence, inserts, p.reads(inserts, n=3000))
        _assert_recovers(layout, p, tmp_path, 400)

    def test_forward_barcode_3prime_is_refused(self, tmp_path):
        """Demultiplexed as is, every well would be transposed."""
        p = Plasmid()
        inserts = p.scan(30)
        with pytest.raises(LayoutError, match="wrong well"):
            _detect(tmp_path, p.sequence, inserts, p.reads(inserts, flipped=True))

    def test_orientation_unchecked_when_every_barcode_is_shared(self, tmp_path):
        """Plates 4-8 use RB13-RB32, shared with the forward set; with no
        first-row wells either, nothing can vote and the layout says so."""
        p = Plasmid()
        inserts = p.scan(30)
        reads = []
        r = p.rng
        for _ in range(1500):
            insert = inserts[int(r.integers(len(inserts)))]
            fbc = LEVSEQ_FBC[int(r.integers(12, 96))]
            rbc = LEVSEQ_RBC[int(r.integers(12, 32))]
            amp = p.template_flank_5 + insert + p.template_flank_3
            read = OUTER_5P + fbc + amp + _rc(rbc) + OUTER_3P
            reads.append(_apply_errors(r, read, 0.03))
        layout = _detect(tmp_path, p.sequence, inserts, reads)
        assert layout.orientation_votes == (0, 0)
        assert any("not checked" in line for line in layout.describe())
        _assert_recovers(layout, p, tmp_path, len(p.parent))

    def test_primer_between_barcode_and_given_amplicon(self, tmp_path):
        """The amplicon given can start inside the barcoding primer: on a real
        Fordyce-primer run, 32 bp lay between the forward barcode and it and
        17 bp after it.  Those are found at their full length and carried in
        the template, and the masks are the primer either side of each
        barcode."""
        p = Plasmid()
        inserts = p.scan(30)
        outer_5, outer_3 = "CACCCAAGACCACTCTCCGG", "GCACCTACTTCGCACACCG"
        spacer_5 = "CGCGCACATTTCCCCGAAAAGTGCTAGTGGTG"
        spacer_3 = "GCACTGACTCGCTGCGC"
        r = p.rng
        reads = []
        for _ in range(1500):
            insert = inserts[int(r.integers(len(inserts)))]
            fbc = LEVSEQ_FBC[int(r.integers(96))]
            rbc = LEVSEQ_RBC[int(r.integers(0, 4))]
            amp = p.template_flank_5 + insert + p.template_flank_3
            read = outer_5 + fbc + spacer_5 + amp + spacer_3 + _rc(rbc) + outer_3
            if r.random() < 0.5:
                read = _rc(read)
            reads.append(_apply_errors(r, read, 0.03))
        amplicon = p.template_flank_5 + p.parent + p.template_flank_3
        v0 = len(p.template_flank_5)
        vec, fq = _write(tmp_path, amplicon, reads)
        layout = detect_layout(None, inserts, fq, shutil.which("minimap2"),
                               tmp_path / "work", threads=2, amplicon=amplicon,
                               variable=(v0, v0 + len(p.parent)))
        assert layout.spacer_5p == spacer_5 and layout.spacer_3p == spacer_3
        t = parse_read_template(write_read_template(layout, tmp_path / "t.fa"))
        assert t.flank_5p == spacer_5 + p.template_flank_5
        assert t.flank_3p == p.template_flank_3 + spacer_3
        assert t.masks["mask1_front"].endswith(outer_5)
        assert t.masks["mask2_rear"].startswith(outer_3)
        assert t.masks["mask1_rear"] == spacer_5[:22]

    def test_reads_from_another_construct_are_refused(self, tmp_path):
        p, other = Plasmid(seed=1), Plasmid(seed=2)
        inserts = other.scan(20)
        with pytest.raises(LayoutError, match="align to the vector"):
            _detect(tmp_path, p.sequence, p.scan(20), other.reads(inserts, n=500))
