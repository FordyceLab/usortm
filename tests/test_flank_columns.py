"""Every column of the construct is judged by one rule, flanks included.

Scanned over the variable region alone, a well could hold a position in its
5' flank where the reads split and be reported clean: that test never looked
there, and the separate flank check compares the consensus rather than the
reads -- and a consensus that declines to call a split position writes an N,
which that check then forgives.
"""
import pytest

pysam = pytest.importorskip("pysam")

from usortm.demux.utils import (MIXED_TEMPLATE_THRESHOLD,
                                MIXED_TEMPLATE_WATCH,
                                _check_column_agreement,
                                column_agreement_class)

REF_LEN = 120
ORF_START = 40
ORF_LEN = 30
#: A flank position, well outside the ORF, which is where the miss was.
FLANK_POS = 12


def _build(tmp_path, disagreeing, total=20, at=FLANK_POS, base="A"):
    """A well whose reads split at *at*, with everything else matching."""
    demux = tmp_path / "demux_output"
    refs = demux / "reference_fasta" / "single_ref_fastas"
    cons = demux / "wells" / "consensus"
    refs.mkdir(parents=True)
    cons.mkdir(parents=True)

    ref = "C" * REF_LEN
    (refs / "V.fasta").write_text(f">V\n{ref}\n")

    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6", "SO": "coordinate"},
         "SQ": [{"SN": "V", "LN": REF_LEN}]})
    path = str(cons / "1A1.bam")
    with pysam.AlignmentFile(path, "wb", header=header) as out:
        for i in range(total):
            seq = list(ref)
            if i < disagreeing:
                seq[at] = base
            rec = pysam.AlignedSegment(header)
            rec.query_name = f"r{i}"
            rec.query_sequence = "".join(seq)
            rec.flag = 0
            rec.reference_id = 0
            rec.reference_start = 0
            rec.mapping_quality = 60
            rec.cigartuples = [(0, REF_LEN)]
            rec.query_qualities = pysam.qualitystring_to_array("I" * REF_LEN)
            out.write(rec)

    return str(cons), ref[ORF_START:ORF_START + ORF_LEN]


def test_a_split_flank_position_is_flagged(tmp_path):
    """8 of 20 reads differing at a flank base is the well's worst column."""
    cons, orf = _build(tmp_path, disagreeing=8)

    got = _check_column_agreement("1A1", cons, orf, ORF_START)

    assert got["n_flagged_positions"] == 1
    assert got["max_mismatch_frac"] == pytest.approx(0.40)
    # And it reaches the class the plate maps and the tiers act on.
    assert column_agreement_class(got["max_mismatch_frac"]) == "mixed"


def test_a_clean_flank_leaves_the_well_clean(tmp_path):
    """The widened scan must not flag a construct that simply matches."""
    cons, orf = _build(tmp_path, disagreeing=0)

    got = _check_column_agreement("1A1", cons, orf, ORF_START)
    assert got["n_flagged_positions"] == 0
    assert got["max_mismatch_frac"] == pytest.approx(0.0)
    assert column_agreement_class(got["max_mismatch_frac"]) == "clean"


def test_the_flank_uses_the_same_threshold_as_the_variable_region(tmp_path):
    """One rule for the whole construct: past the watch fraction, flagged."""
    below = int((MIXED_TEMPLATE_WATCH - 0.02) * 20)
    cons, orf = _build(tmp_path, disagreeing=below)
    assert _check_column_agreement(
        "1A1", cons, orf, ORF_START)["n_flagged_positions"] == 0

    above = int((MIXED_TEMPLATE_WATCH + 0.10) * 20)
    cons, orf = _build(tmp_path / "b", disagreeing=above)
    assert _check_column_agreement(
        "1A1", cons, orf, ORF_START)["n_flagged_positions"] == 1


def test_a_split_in_the_variable_region_still_counts(tmp_path):
    """Widening the scan must not lose what it already covered."""
    cons, orf = _build(tmp_path, disagreeing=8, at=ORF_START + 5)

    got = _check_column_agreement("1A1", cons, orf, ORF_START)
    assert got["n_flagged_positions"] == 1
    assert got["max_mismatch_frac"] == pytest.approx(0.40)


def test_a_missing_reference_leaves_the_flanks_unread(tmp_path):
    """Without a base to compare against, a column is skipped rather than
    counted as every read disagreeing."""
    cons, orf = _build(tmp_path, disagreeing=8)
    # Remove the construct reference; only the ORF the caller carries is left.
    (tmp_path / "demux_output" / "reference_fasta" / "single_ref_fastas"
     / "V.fasta").unlink()
    from usortm.demux.utils import _construct_reference
    _construct_reference.cache_clear()

    got = _check_column_agreement("1A1", cons, orf, ORF_START)
    assert got["n_flagged_positions"] == 0
    assert got["max_mismatch_frac"] == pytest.approx(0.0)


def test_the_thresholds_are_the_ones_the_report_acts_on():
    """The scan flags at the watch fraction; the plate calls mixed past 25%."""
    assert MIXED_TEMPLATE_WATCH == pytest.approx(0.10)
    assert MIXED_TEMPLATE_THRESHOLD == pytest.approx(0.25)
    assert column_agreement_class(0.40) == "mixed"
    assert column_agreement_class(0.15) == "watch"
    assert column_agreement_class(0.05) == "clean"
