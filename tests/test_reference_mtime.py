"""A reference that has not changed keeps its mtime.

The resume check reads mtimes: a well's consensus is reused only when it is
newer than the reference it was built against.  Each segment of a run rewrote
all 377 per-variant references at startup with identical bytes, so every
segment invalidated the consensus the one before it had just built, and a
re-run that changed only a QC setting rebuilt 2,676 wells over three hours.
"""
import os

import pytest

from usortm.demux.pipeline import (_prepare_full_length_ref_fastas,
                                   _prepare_single_ref_fastas)

FLANK_5P = "AAAACCCC"
FLANK_3P = "GGGGTTTT"


def _multi(tmp_path):
    path = tmp_path / "library.fasta"
    path.write_text(">V1\nACGTACGTAC\n>V2\nTGCATGCATG\n")
    return path


def _stamp(directory):
    return {p.name: os.stat(p).st_mtime_ns
            for p in (directory / "single_ref_fastas").iterdir()
            if p.suffix == ".fasta"}


def test_an_unchanged_reference_is_not_rewritten(tmp_path):
    multi = _multi(tmp_path)
    out = tmp_path / "out"
    _prepare_single_ref_fastas(multi, out)
    before = _stamp(out)
    assert before, "no reference files were written"

    _prepare_single_ref_fastas(multi, out)
    assert _stamp(out) == before


def test_an_unchanged_full_length_reference_is_not_rewritten(tmp_path):
    """The path a run with a vector takes, which is the one that regressed."""
    multi = _multi(tmp_path)
    out = tmp_path / "out"
    _prepare_full_length_ref_fastas(multi, out, FLANK_5P, FLANK_3P)
    before = _stamp(out)
    assert before

    _prepare_full_length_ref_fastas(multi, out, FLANK_5P, FLANK_3P)
    assert _stamp(out) == before


def test_a_changed_reference_is_rewritten(tmp_path):
    """The guard must not stop a real change from landing."""
    multi = _multi(tmp_path)
    out = tmp_path / "out"
    _prepare_full_length_ref_fastas(multi, out, FLANK_5P, FLANK_3P)
    first = (out / "single_ref_fastas" / "V1.fasta").read_text()

    _prepare_full_length_ref_fastas(multi, out, "TTTTTTTT", FLANK_3P)
    second = (out / "single_ref_fastas" / "V1.fasta").read_text()
    assert second != first
    assert "TTTTTTTT" in second.replace("\n", "")


def test_the_written_sequence_is_still_flanked(tmp_path):
    multi = _multi(tmp_path)
    out = tmp_path / "out"
    _prepare_full_length_ref_fastas(multi, out, FLANK_5P, FLANK_3P)
    body = "".join(
        l for l in (out / "single_ref_fastas" / "V1.fasta").read_text()
        .splitlines() if not l.startswith(">"))
    assert body == FLANK_5P + "ACGTACGTAC" + FLANK_3P


def test_an_index_survives_a_rewrite_that_changes_nothing(tmp_path):
    """A .fai matching its FASTA is not stale, and rebuilding it costs time."""
    multi = _multi(tmp_path)
    out = tmp_path / "out"
    _prepare_full_length_ref_fastas(multi, out, FLANK_5P, FLANK_3P)
    fai = out / "single_ref_fastas" / "V1.fasta.fai"
    fai.write_text("V1\t26\t4\t60\t61\n")

    _prepare_full_length_ref_fastas(multi, out, FLANK_5P, FLANK_3P)
    assert fai.exists()


def test_unchanged_well_reads_keep_their_mtime(tmp_path):
    """The second half of the same problem.

    _consensus_is_reusable takes a consensus only when it is newer than both
    the reference and the reads.  Rewriting per-well FASTQs with identical
    bytes invalidated the consensus just as rewriting the references did.
    """
    pd = pytest.importorskip("pandas")
    from usortm.demux.utils import write_per_well_fastqs

    df = pd.DataFrame({
        "well_pos": ["1A1", "1A1", "1A2"],
        "read_name": ["r1", "r2", "r3"],
        "read_seq": ["ACGT", "TTTT", "GGGG"],
        "read_qual": ["IIII", "IIII", "IIII"],
    })
    write_per_well_fastqs(df, str(tmp_path))
    d = tmp_path / "wells" / "fastqs"
    before = {p.name: os.stat(p).st_mtime_ns for p in d.iterdir()}
    assert before

    write_per_well_fastqs(df, str(tmp_path))
    assert {p.name: os.stat(p).st_mtime_ns for p in d.iterdir()} == before


def test_changed_well_reads_are_rewritten(tmp_path):
    pd = pytest.importorskip("pandas")
    from usortm.demux.utils import write_per_well_fastqs

    df = pd.DataFrame({
        "well_pos": ["1A1"], "read_name": ["r1"],
        "read_seq": ["ACGT"], "read_qual": ["IIII"],
    })
    write_per_well_fastqs(df, str(tmp_path))
    path = tmp_path / "wells" / "fastqs" / "1A1.fastq"
    assert "ACGT" in path.read_text()

    df.loc[0, "read_seq"] = "TGCA"
    write_per_well_fastqs(df, str(tmp_path))
    assert "TGCA" in path.read_text()


def test_a_real_change_still_drops_the_index(tmp_path):
    multi = _multi(tmp_path)
    out = tmp_path / "out"
    _prepare_full_length_ref_fastas(multi, out, FLANK_5P, FLANK_3P)
    fai = out / "single_ref_fastas" / "V1.fasta.fai"
    fai.write_text("V1\t26\t4\t60\t61\n")

    _prepare_full_length_ref_fastas(multi, out, "TTTTTTTT", FLANK_3P)
    assert not fai.exists()
