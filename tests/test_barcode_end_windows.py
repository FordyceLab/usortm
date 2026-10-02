"""Tests for cutting reads to their barcode ends before the barcode passes.

From RB13 up, the reverse barcodes are the same sequences as forward ones, so
each Dorado pass is given only its own end of every read.  The reads arrive
as sequenced, reverse-strand ones marked dir=rev, and have to be turned first
or half the "forward" ends would hold the reverse barcode.
"""

from usortm.demux.barcodes import LEVSEQ_FBC, LEVSEQ_RBC
from usortm.demux.pipeline import SHARED_RBC_FROM, _barcode_end_windows

_COMP = str.maketrans("ACGT", "TGCA")


def _rc(s):
    return s.translate(_COMP)[::-1]


def _read(fbc, rbc, insert="ACGT" * 150):
    return "AATATAAATT" + fbc + insert + _rc(rbc) + "ATAATTTATA"


def _records(path):
    lines = path.read_text().splitlines()
    return [(lines[i], lines[i + 1], lines[i + 3]) for i in range(0, len(lines), 4)]


def test_the_shared_barcodes_start_at_rb13():
    assert SHARED_RBC_FROM == 13
    assert LEVSEQ_RBC[SHARED_RBC_FROM - 1] == LEVSEQ_FBC[SHARED_RBC_FROM - 1]
    assert LEVSEQ_RBC[SHARED_RBC_FROM - 2] != LEVSEQ_FBC[SHARED_RBC_FROM - 2]


def test_each_end_gets_its_own_barcode(tmp_path):
    fwd = _read(LEVSEQ_FBC[21], LEVSEQ_RBC[28])
    rev = _rc(_read(LEVSEQ_FBC[4], LEVSEQ_RBC[30]))         # as sequenced
    fq = tmp_path / "oriented.fastq"
    fq.write_text(
        f"@a|ref=r|dir=fwd\n{fwd}\n+\n{'I' * len(fwd)}\n"
        f"@b|ref=r|dir=rev\n{rev}\n+\n{'#' * 5 + 'I' * (len(rev) - 5)}\n"
    )
    heads, tails = _barcode_end_windows(str(fq), tmp_path / "ends", 120)
    h, t = _records(tmp_path / "ends" / "fbc_ends.fastq"), _records(tmp_path / "ends" / "rbc_ends.fastq")

    assert [r[0] for r in h] == ["@a|ref=r|dir=fwd", "@b|ref=r|dir=rev"]
    assert all(len(r[1]) == len(r[2]) == 120 for r in h + t)
    # The reverse-strand read was turned: its forward barcode leads.
    assert LEVSEQ_FBC[21] in h[0][1] and LEVSEQ_FBC[4] in h[1][1]
    assert _rc(LEVSEQ_RBC[28]) in t[0][1] and _rc(LEVSEQ_RBC[30]) in t[1][1]
    # Its qualities were turned with it: the low ones were at its sequenced start.
    assert t[1][2].endswith("#####")
    # And neither end can see the other's barcode.
    assert _rc(LEVSEQ_RBC[28]) not in h[0][1] and LEVSEQ_FBC[21] not in t[0][1]


def test_short_reads_are_kept_whole(tmp_path):
    fq = tmp_path / "o.fastq"
    fq.write_text("@s|dir=fwd\nACGTACGT\n+\nIIIIIIII\n")
    _barcode_end_windows(str(fq), tmp_path / "ends", 300)
    assert _records(tmp_path / "ends" / "fbc_ends.fastq")[0][1] == "ACGTACGT"
