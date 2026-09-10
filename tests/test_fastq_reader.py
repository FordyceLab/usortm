"""The htslib FASTQ reader returns what the Biopython loop returned.

create_read_df used to collect each read with SeqIO.parse, decode the
quality string into integers, and re-encode it one character at a time.
iter_fastq_reads reads through pysam instead, 10x faster on the project's
oriented FASTQ.  A faster parser is worth nothing if a single sequence,
quality string or mean quality differs, so this compares the two on files
built to have the awkward cases: gzip, header comments, empty-comment
headers, the full printable quality range, and a read one base long.
"""
import gzip
import random

import pytest
from Bio import SeqIO

from usortm.demux.utils import iter_fastq_reads

QUAL_CHARS = "".join(chr(33 + q) for q in range(0, 94))  # phred 0..93


def _records(n, rng):
    for i in range(n):
        length = rng.choice([1, 7, 50, 300, 2048])
        seq = "".join(rng.choice("ACGTN") for _ in range(length))
        qual = "".join(rng.choice(QUAL_CHARS) for _ in range(length))
        comment = rng.choice(["", " runid=abc ch=12", "|ref=V1|dir=fwd", "\tx=1"])
        yield f"read{i}", comment, seq, qual


def _write(path, recs, gz=False):
    opener = gzip.open if gz else open
    with opener(path, "wt") as fh:
        for name, comment, seq, qual in recs:
            fh.write(f"@{name}{comment}\n{seq}\n+\n{qual}\n")


def _biopython(path, gz=False):
    opener = gzip.open if gz else open
    out = []
    with opener(path, "rt") as fh:
        for rec in SeqIO.parse(fh, "fastq"):
            quals = rec.letter_annotations["phred_quality"]
            out.append((rec.id, str(rec.seq),
                        "".join(chr(q + 33) for q in quals),
                        sum(quals) / len(quals)))
    return out


@pytest.mark.parametrize("gz", [False, True])
def test_matches_the_biopython_loop(tmp_path, gz):
    rng = random.Random(11)
    recs = list(_records(400, rng))
    path = tmp_path / ("reads.fastq.gz" if gz else "reads.fastq")
    _write(path, recs, gz=gz)

    ours = list(iter_fastq_reads(path))
    theirs = _biopython(path, gz=gz)
    assert len(ours) == len(theirs) == 400

    for (n1, s1, q1, a1), (n2, s2, q2, a2) in zip(ours, theirs):
        assert n1 == n2
        assert s1 == s2
        assert q1 == q2
        assert a1 == pytest.approx(a2, abs=1e-12)


def test_the_name_stops_at_whitespace_like_rec_id(tmp_path):
    path = tmp_path / "r.fastq"
    path.write_text("@abc def ghi\nACGT\n+\nIIII\n")
    (name, seq, qual, avgq), = iter_fastq_reads(path)
    assert name == "abc"
    assert (seq, qual, avgq) == ("ACGT", "IIII", 40.0)


def test_the_full_quality_range_survives_untouched(tmp_path):
    path = tmp_path / "r.fastq"
    seq = "A" * len(QUAL_CHARS)
    path.write_text(f"@r\n{seq}\n+\n{QUAL_CHARS}\n")
    (_, _, qual, avgq), = iter_fastq_reads(path)
    assert qual == QUAL_CHARS
    assert avgq == pytest.approx(sum(range(94)) / 94)


def test_a_fasta_is_refused_rather_than_read_without_qualities(tmp_path):
    path = tmp_path / "r.fasta"
    path.write_text(">r\nACGT\n")
    with pytest.raises(ValueError, match="no quality string"):
        list(iter_fastq_reads(path))


def test_the_read_table_uses_it(tmp_path):
    """create_read_df must go through the new reader, not the old loop."""
    import inspect

    from usortm.demux import utils

    src = inspect.getsource(utils.create_read_df)
    assert "iter_fastq_reads(" in src
    assert 'SeqIO.parse(fh, "fastq")' not in src
