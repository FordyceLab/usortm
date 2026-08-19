"""Tests for calling a well's variant from its translated consensus.

The cases that matter are the ones where choosing a winner from a list of
near-identical references cannot help: a well carrying no mutation, and a well
carrying something the library never designed.
"""


import shutil

import numpy as np
import pytest
from Bio.Seq import Seq

from usortm.demux.protein_call import (
    WellCall,
    build_wt_reference,
    call_well,
    call_wells,
)


def _has(name):
    return shutil.which(name) is not None


requires_tools = pytest.mark.skipif(
    not (_has("minimap2") and _has("samtools")),
    reason="needs minimap2 and samtools",
)

# A 32-codon insert with generous flanks, so minimap2 has somewhere to anchor.
WT_INSERT = ("ATGGCTGAAGGCAACACCCTGATTAGCGTTGACTATGAAATCTTTGGCAAAGTGCAGGGC"
             "GTTTTCTTCCGCAAACACACCCAGGCGGAAGGCAAAAAACTGGGCTAA")[:96]
FLANK_5P = ("CTGACCGTTAGCCAGGATTTACGCAGTTCACGTGGAACCTTAGCGGTCAGATCCTGAAAC"
            "GTTAGCCAGTTACGGATCCAGTTACGCAGTTCAGGTGAACCTTAGCGGTCAGATCCTGAA")
FLANK_3P = ("GGATCCAGTTACGCAGTTCAGGTGAACCTTAGCGGTCAGATCCTGAAACGTTAGCCAGTT"
            "ACGGATCCAGTTACGCAGTTCAGGTGAACCTTAGCGGTCAGATCCTGAAACGTTAGCCAG")
WT_AA = str(Seq(WT_INSERT).translate())


def _mutate(insert, codon, new_codon):
    """Return *insert* with 1-based *codon* replaced."""
    i = (codon - 1) * 3
    return insert[:i] + new_codon + insert[i + 3:]


def _write_reads(path, insert, n=40, error_rate=0.0, seed=0, truncate=0):
    """Write *n* noisy copies of the construct carrying *insert*."""
    rng = np.random.default_rng(seed)
    amplicon = FLANK_5P + insert + FLANK_3P
    with open(path, "w") as fh:
        for k in range(n):
            seq = list(amplicon)
            if error_rate:
                for i in range(len(seq)):
                    if rng.random() < error_rate:
                        seq[i] = "ACGT"[int(rng.integers(4))]
            seq = "".join(seq)
            if truncate:
                seq = seq[:-truncate]
            fh.write(f"@read_{k}\n{seq}\n+\n{'I' * len(seq)}\n")
    return path


@pytest.fixture
def wt_ref(tmp_path):
    return str(build_wt_reference(WT_INSERT, FLANK_5P, FLANK_3P,
                                  tmp_path / "wt.fasta"))


def _call(tmp_path, wt_ref, insert, **kw):
    fq = _write_reads(str(tmp_path / "w.fastq"), insert, **kw)
    return call_well("A1", fq, wt_ref, len(FLANK_5P), len(WT_INSERT), WT_AA)


class TestUnmutatedWells:
    """The case a library-of-variants alignment cannot express."""

    @requires_tools
    def test_a_well_with_no_mutation_is_called_wt(self, tmp_path, wt_ref):
        assert _call(tmp_path, wt_ref, WT_INSERT).call == "WT"

    @requires_tools
    def test_still_wt_with_sequencing_error(self, tmp_path, wt_ref):
        """Errors are uncorrelated between reads, so the consensus holds."""
        call = _call(tmp_path, wt_ref, WT_INSERT, error_rate=0.03, seed=1)
        assert call.call == "WT"


class TestSubstitutions:

    @requires_tools
    def test_a_single_substitution_is_named(self, tmp_path, wt_ref):
        # Codon 5 is AAC (N); make it GCG (A).
        assert WT_INSERT[12:15] == "AAC"
        call = _call(tmp_path, wt_ref, _mutate(WT_INSERT, 5, "GCG"))
        assert call.call == "N5A"
        assert call.aa_changes == ["N5A"]

    @requires_tools
    def test_a_stop_is_named_with_a_star(self, tmp_path, wt_ref):
        call = _call(tmp_path, wt_ref, _mutate(WT_INSERT, 5, "TAA"))
        assert call.call == "N5*"

    @requires_tools
    def test_survives_sequencing_error(self, tmp_path, wt_ref):
        call = _call(tmp_path, wt_ref, _mutate(WT_INSERT, 5, "GCG"),
                     error_rate=0.03, seed=2)
        assert call.call == "N5A"

    @requires_tools
    def test_a_codon_differing_at_every_base_is_still_named(self, tmp_path,
                                                            wt_ref):
        """Three differences in one codon are cheaper for minimap2 to align as
        a deletion plus an insertion than as three mismatches.  Consensus taken
        a base at a time drops the inserted bases and loses one, so the well
        came back as an indel rather than the substitution it is.
        """
        assert WT_INSERT[12:15] == "AAC"
        call = _call(tmp_path, wt_ref, _mutate(WT_INSERT, 5, "TGA"))
        assert call.call == "N5*"
        assert call.insert_len == len(WT_INSERT)


class TestWellsTheLibraryCannotExpress:

    @requires_tools
    def test_an_undesigned_substitution_is_reported_as_itself(self, tmp_path,
                                                              wt_ref):
        """Nothing constrains the answer to a list, so an unexpected residue
        is reported rather than rounded to the nearest designed variant."""
        call = _call(tmp_path, wt_ref, _mutate(WT_INSERT, 7, "TGG"))
        assert call.call.startswith(WT_AA[6])
        assert call.call.endswith("W")

    @requires_tools
    def test_two_substitutions_are_flagged_not_forced(self, tmp_path, wt_ref):
        insert = _mutate(_mutate(WT_INSERT, 5, "GCG"), 9, "GCG")
        call = _call(tmp_path, wt_ref, insert)
        assert call.call == "multi(2)"
        assert not call.is_clean


class TestWhenItDeclinesToCall:
    """Failing loudly beats naming a variant the reads do not support."""

    @requires_tools
    def test_too_few_reads_is_reported_not_guessed(self, tmp_path, wt_ref):
        call = _call(tmp_path, wt_ref, WT_INSERT, n=1)
        assert call.call == "low-coverage"

    @requires_tools
    def test_an_empty_well_is_reported(self, tmp_path, wt_ref):
        fq = str(tmp_path / "empty.fastq")
        open(fq, "w").close()
        call = call_well("A1", fq, wt_ref, len(FLANK_5P), len(WT_INSERT), WT_AA)
        assert call.call == "no-reads"

    @requires_tools
    def test_a_deleted_codon_is_reported_as_an_indel(self, tmp_path, wt_ref):
        insert = WT_INSERT[:12] + WT_INSERT[15:]
        fq = _write_reads(str(tmp_path / "d.fastq"), insert)
        call = call_well("A1", fq, wt_ref, len(FLANK_5P), len(WT_INSERT), WT_AA)
        assert call.call.startswith("indel(")


class TestSupport:

    @requires_tools
    def test_a_clean_well_reports_full_support(self, tmp_path, wt_ref):
        call = _call(tmp_path, wt_ref, _mutate(WT_INSERT, 5, "GCG"))
        assert call.support == 1.0

    @requires_tools
    def test_a_mixed_well_reports_partial_support(self, tmp_path, wt_ref):
        """Half the reads mutant, half wild type: the call carries the split."""
        fq = str(tmp_path / "mixed.fastq")
        _write_reads(fq, _mutate(WT_INSERT, 5, "GCG"), n=20, seed=3)
        second = _write_reads(str(tmp_path / "b.fastq"), WT_INSERT, n=20, seed=4)
        with open(fq, "a") as fh:
            fh.write(open(second).read())
        call = call_well("A1", fq, wt_ref, len(FLANK_5P), len(WT_INSERT), WT_AA)
        assert 0.3 < call.support < 0.75


class TestCallWells:

    @requires_tools
    def test_calls_every_fastq_in_a_directory(self, tmp_path, wt_ref):
        wells = tmp_path / "fastqs"
        wells.mkdir()
        _write_reads(str(wells / "1A1.fastq"), WT_INSERT)
        _write_reads(str(wells / "1A2.fastq"), _mutate(WT_INSERT, 5, "GCG"))
        calls = call_wells(str(wells), WT_INSERT, FLANK_5P, FLANK_3P,
                           out_dir=str(tmp_path / "out"), workers=2)
        assert [c.well for c in calls] == ["1A1", "1A2"]
        assert [c.call for c in calls] == ["WT", "N5A"]


class TestReference:

    def test_the_reference_is_the_construct_around_wild_type(self, tmp_path):
        p = build_wt_reference(WT_INSERT, FLANK_5P, FLANK_3P,
                               tmp_path / "r.fasta")
        text = open(p).read()
        assert text.startswith(">wt_construct\n")
        assert FLANK_5P + WT_INSERT + FLANK_3P in text

    def test_one_reference_not_one_per_variant(self, tmp_path):
        """The point of the approach: a single target, so there is no list for
        an alignment score to pick a winner from."""
        p = build_wt_reference(WT_INSERT, FLANK_5P, FLANK_3P,
                               tmp_path / "r.fasta")
        assert open(p).read().count(">") == 1


class TestWellCall:

    def test_a_single_change_is_clean(self):
        assert WellCall(well="A1", call="N5A", aa_changes=["N5A"]).is_clean

    def test_wild_type_is_clean(self):
        assert WellCall(well="A1", call="WT").is_clean

    def test_several_changes_are_not(self):
        assert not WellCall(well="A1", call="multi(2)",
                            aa_changes=["N5A", "K9A"]).is_clean
