"""A masked substitution is forgiven; nothing else at that position is.

Two positions of the AFMtag run disagree in every well at a fraction that does
not vary with what the well holds, which is the signature of the sequencing
rather than of a construct.  Masking them is a claim about the chemistry, so
it is written in a file the run reads and records -- and the tests here fix
what that file can say and how far the mask travels.
"""
import pytest

from usortm.demux.qc_mask import (MaskedChange, QCMaskError, as_lookup,
                                  describe, find_qc_mask, parse_qc_mask,
                                  read_qc_mask)

pysam = pytest.importorskip("pysam")

from usortm.demux.utils import _check_column_agreement, extract_matches

REF_LEN = 120
ORF_START = 40
ORF_LEN = 30
#: A flank position, standing in for the run's 799 and 802.
ARTEFACT_POS = 12


def _build(tmp_path, disagreeing, total=20, at=ARTEFACT_POS, base="A"):
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

    # The consensus realigned to the reference, which is what the flank check
    # reads; without it a well reports "No alignment" and the column scan --
    # the thing under test -- never runs.
    align = str(cons / "1A1_consensus_align.bam")
    with pysam.AlignmentFile(align, "wb", header=header) as out:
        rec = pysam.AlignedSegment(header)
        rec.query_name = "consensus"
        rec.query_sequence = ref
        rec.flag = 0
        rec.reference_id = 0
        rec.reference_start = 0
        rec.mapping_quality = 60
        rec.cigartuples = [(0, REF_LEN)]
        rec.query_qualities = pysam.qualitystring_to_array("I" * REF_LEN)
        rec.set_tag("MD", str(REF_LEN))
        out.write(rec)

    return str(cons), ref


# --- what the file may say ----------------------------------------------

def test_a_mask_names_a_position_a_base_and_why():
    got = parse_qc_mask({"mask": [
        {"position": 799, "base": "a", "note": "5-9% in every well sampled"},
    ]})
    assert got == [MaskedChange(position=799, base="A",
                                note="5-9% in every well sampled")]


def test_positions_come_back_in_order_whatever_the_file_says():
    got = parse_qc_mask({"mask": [{"position": 802, "base": "G"},
                                  {"position": 799, "base": "A"}]})
    assert [m.position for m in got] == [799, 802]


def test_no_mask_table_is_an_empty_mask():
    assert parse_qc_mask({}) == []


@pytest.mark.parametrize("entry, says", [
    ({"base": "A"}, "missing 'position'"),
    ({"position": 0, "base": "A"}, "not 1-based"),
    ({"position": 799}, "is not one of"),
    ({"position": 799, "base": "N"}, "is not one of"),
    ({"position": "seven", "base": "A"}, "whole number"),
])
def test_an_entry_that_cannot_be_applied_is_refused(entry, says):
    with pytest.raises(QCMaskError, match=says):
        parse_qc_mask({"mask": [entry]})


def test_the_same_substitution_twice_is_refused():
    """A file that states a claim twice may state it two ways later."""
    with pytest.raises(QCMaskError, match="already masked"):
        parse_qc_mask({"mask": [{"position": 799, "base": "A"},
                                {"position": 799, "base": "A"}]})


def test_two_bases_at_one_position_are_separate_claims():
    got = as_lookup(parse_qc_mask({"mask": [{"position": 799, "base": "A"},
                                            {"position": 799, "base": "T"}]}))
    assert got == {798: {"A", "T"}}


def test_the_lookup_is_zero_based():
    """The file is read as a sequence is; a BAM counts from zero."""
    assert as_lookup([MaskedChange(799, "A")]) == {798: {"A"}}


def test_describe_names_each_change():
    assert describe([MaskedChange(799, "A"), MaskedChange(802, "G")]) == (
        "A at 799, G at 802")
    assert describe([]) == ""


def test_a_file_is_read_from_disk(tmp_path):
    path = tmp_path / "qc_mask.toml"
    path.write_text('[[mask]]\nposition = 799\nbase = "A"\n'
                    'note = "artefact"\n')
    assert read_qc_mask(path) == [MaskedChange(799, "A", "artefact")]


def test_a_file_that_is_not_toml_is_refused(tmp_path):
    path = tmp_path / "qc_mask.toml"
    path.write_text("position: 799\n")
    with pytest.raises(QCMaskError, match="not valid TOML"):
        read_qc_mask(path)


# --- where the file lives ------------------------------------------------

def test_a_project_keeps_its_mask_in_config(tmp_path):
    (tmp_path / "config").mkdir()
    path = tmp_path / "config" / "qc_mask.toml"
    path.write_text("")
    assert find_qc_mask(tmp_path) == str(path)


def test_a_project_without_one_has_none(tmp_path):
    assert find_qc_mask(tmp_path) is None


def test_a_round_may_carry_its_own(tmp_path):
    """A re-order round is a different preparation, so it may differ."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "qc_mask.toml").write_text("")
    round_config = tmp_path / "rounds" / "2" / "config"
    round_config.mkdir(parents=True)
    (round_config / "qc_mask.toml").write_text("")

    assert find_qc_mask(tmp_path, 2) == str(round_config / "qc_mask.toml")


def test_a_round_without_one_falls_back_to_the_project(tmp_path):
    (tmp_path / "config").mkdir()
    shared = tmp_path / "config" / "qc_mask.toml"
    shared.write_text("")
    assert find_qc_mask(tmp_path, 2) == str(shared)


# --- what the mask does to a well ----------------------------------------

def test_without_a_mask_the_position_is_flagged(tmp_path):
    """The comparison the next test is against: 30% disagreement flags."""
    cons, _ = _build(tmp_path, disagreeing=6)
    got = _check_column_agreement("1A1", cons, "C" * ORF_LEN, ORF_START)
    assert got["n_flagged_positions"] == 1
    assert got["max_mismatch_frac"] == pytest.approx(0.30)


def test_a_masked_base_counts_as_agreement(tmp_path):
    cons, _ = _build(tmp_path, disagreeing=6, base="A")
    got = _check_column_agreement("1A1", cons, "C" * ORF_LEN, ORF_START,
                                  masked={ARTEFACT_POS: {"A"}})
    assert got["n_flagged_positions"] == 0
    assert got["max_mismatch_frac"] == pytest.approx(0.0)


def test_another_base_at_a_masked_position_still_counts(tmp_path):
    """An artefact explains one substitution, not the position."""
    cons, _ = _build(tmp_path, disagreeing=6, base="T")
    got = _check_column_agreement("1A1", cons, "C" * ORF_LEN, ORF_START,
                                  masked={ARTEFACT_POS: {"A"}})
    assert got["n_flagged_positions"] == 1
    assert got["max_mismatch_frac"] == pytest.approx(0.30)


def test_a_masked_base_carried_by_the_construct_still_counts(tmp_path):
    """One round 2 well reads G at 802 in every read, which is the construct.

    An artefact appears in a share of the reads whatever the well holds.  A
    masked base that is most of the column is the well's own sequence, and
    forgiving it would report a real substitution as a clean position.
    """
    cons, _ = _build(tmp_path, disagreeing=20, total=20, base="A")
    got = _check_column_agreement("1A1", cons, "C" * ORF_LEN, ORF_START,
                                  masked={ARTEFACT_POS: {"A"}})
    assert got["n_flagged_positions"] == 1
    assert got["max_mismatch_frac"] == pytest.approx(1.0)


def test_the_mask_holds_up_to_half_the_column(tmp_path):
    """Half is still forgiven; past half the construct carries it."""
    half, past = _build(tmp_path, disagreeing=10, total=20, base="A")[0], None
    got = _check_column_agreement("1A1", half, "C" * ORF_LEN, ORF_START,
                                  masked={ARTEFACT_POS: {"A"}})
    assert got["n_flagged_positions"] == 0

    past = _build(tmp_path / "b", disagreeing=11, total=20, base="A")[0]
    got = _check_column_agreement("1A1", past, "C" * ORF_LEN, ORF_START,
                                  masked={ARTEFACT_POS: {"A"}})
    assert got["n_flagged_positions"] == 1


def test_the_mask_applies_only_where_it_is_declared(tmp_path):
    cons, _ = _build(tmp_path, disagreeing=6, base="A")
    got = _check_column_agreement("1A1", cons, "C" * ORF_LEN, ORF_START,
                                  masked={ARTEFACT_POS + 1: {"A"}})
    assert got["n_flagged_positions"] == 1


def test_the_mask_reaches_the_well_checks(tmp_path):
    """Through extract_matches, which is where a run applies it.

    The checks run in worker processes, so the mask has to survive being
    handed across; a mask that only worked in-process would pass the tests
    above and do nothing in a run.
    """
    pd = pytest.importorskip("pandas")
    cons, ref = _build(tmp_path, disagreeing=6, base="A")
    orf = ref[ORF_START:ORF_START + ORF_LEN]
    # A row as the pipeline builds one: ref_len and ref_seq are the variable
    # region, and the flanks are given by their lengths.
    row = {
        "global_well": "1A1",
        "ref_len": ORF_LEN,
        "ref_seq": orf,
        "CIGAR": f"{ORF_LEN}M",
        "cons_seq": orf,
        "max_mismatch_frac": 0.0,
        "n_flagged_positions": 0,
    }

    without = extract_matches(pd.DataFrame([row]), flank_5p_len=ORF_START,
                              flank_3p_len=REF_LEN - ORF_START - ORF_LEN,
                              consensus_dir=cons, workers=2)
    assert without.loc[0, "n_flagged_positions"] == 1

    with_mask = extract_matches(pd.DataFrame([row]), flank_5p_len=ORF_START,
                                flank_3p_len=REF_LEN - ORF_START - ORF_LEN,
                                consensus_dir=cons, workers=2,
                                masked={ARTEFACT_POS: {"A"}})
    assert with_mask.loc[0, "n_flagged_positions"] == 0
