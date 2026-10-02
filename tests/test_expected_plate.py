"""Tests for reading an expected-plate CSV.

The mapping from a 96-well position in a quadrant to a 384-well position is
checked against barcode_to_well itself, so the CSV cannot address a well the
pipeline would decode differently.
"""

import pytest

from usortm.demux.expected_plate import (
    QUADRANTS,
    ExpectedPlateError,
    parse_columns,
    quadrant_to_384,
    read_expected_plate,
)
from usortm.demux.utils import barcode_to_well


def _csv(tmp_path, text, name="plate.csv"):
    p = tmp_path / name
    p.write_text(text.strip() + "\n")
    return p


class TestQuadrantMapping:

    @pytest.mark.parametrize("quadrant", range(4))
    def test_matches_barcode_decoding(self, quadrant):
        for plate in (1, 3, 8):
            rb = (plate - 1) * 4 + quadrant + 1
            for fb in range(1, 97):
                row96, col96 = (fb - 1) // 12 + 1, (fb - 1) % 12 + 1
                expected = barcode_to_well(f"FB{fb:02d}", f"RB{rb:02d}")
                assert f"{plate}{quadrant_to_384(quadrant, row96, col96)}" == expected

    def test_quadrants_interleave(self):
        assert [quadrant_to_384(q, 1, 1) for q in range(4)] == ["A1", "A2", "B1", "B2"]


class TestRead384:

    def test_positions_are_taken_as_given(self, tmp_path):
        p = read_expected_plate(_csv(tmp_path, """
plate,well,name,sequence
1,A1,wt,ACGTACGT
1,P24,m1,ACGTTCGT
2,C05,m2,ACGTACGA
"""))
        assert p.layout == "384"
        assert set(p.wells) == {(1, "A1"), (1, "P24"), (2, "C5")}
        assert p.n_plates == 2

    def test_headers_ignore_case_and_space(self, tmp_path):
        p = read_expected_plate(_csv(tmp_path, """
 Plate , Well ,Name, SEQUENCE
1,B2,x,ACGT
"""))
        assert (1, "B2") in p.wells

    def test_lowercase_flanks_are_dropped(self, tmp_path):
        p = read_expected_plate(_csv(tmp_path, """
plate,well,name,sequence
1,A1,x,ggccACGTACGTttaa
1,A2,y,acgtacgt
"""))
        assert p.wells[(1, "A1")].sequence == "ACGTACGT"
        # Nothing uppercase: the whole sequence is the insert, not nothing.
        assert p.wells[(1, "A2")].sequence == "ACGTACGT"

    def test_blank_sequence_is_an_expected_empty_well(self, tmp_path):
        p = read_expected_plate(_csv(tmp_path, """
plate,well,name,sequence
1,A1,x,ACGT
1,A2,,
"""))
        assert p.wells[(1, "A2")].empty
        assert p.reference_name(p.wells[(1, "A2")]) is None

    def test_top_left_only_is_noted(self, tmp_path):
        lines = ["plate,well,name,sequence"] + [
            f"1,{r}{c},v{r}{c},ACGT" for r in "ABCDEFGH" for c in range(1, 13)
        ]
        p = read_expected_plate(_csv(tmp_path, "\n".join(lines)))
        assert p.layout == "384"
        assert any("quadrant" in n for n in p.notes)


class TestRead96:

    def test_quadrant_column(self, tmp_path):
        p = read_expected_plate(_csv(tmp_path, """
plate,quadrant,well,name,sequence
1,TL,A1,a,ACGT
1,br,A1,b,ACGA
2,2,H12,c,ACGG
"""))
        assert p.layout == "96"
        assert set(p.wells) == {(1, "A1"), (1, "B2"), (2, "O24")}
        assert p.wells[(1, "B2")].label == "BR:A1"

    def test_rbc_column(self, tmp_path):
        p = read_expected_plate(_csv(tmp_path, """
rbc,well,name,sequence
RB06,A1,a,ACGT
7,A1,b,ACGA
"""))
        # RB06 is plate 2, TR; RB07 is plate 2, BL.
        assert set(p.wells) == {(2, "A2"), (2, "B1")}

    def test_rbc_disagreeing_with_plate_is_refused(self, tmp_path):
        with pytest.raises(ExpectedPlateError, match="RB06 is on plate 2"):
            read_expected_plate(_csv(tmp_path, """
plate,rbc,well,name,sequence
1,RB06,A1,a,ACGT
"""))

    def test_384_position_with_a_quadrant_is_refused(self, tmp_path):
        with pytest.raises(ExpectedPlateError, match="outside A1-H12"):
            read_expected_plate(_csv(tmp_path, """
plate,quadrant,well,name,sequence
1,TL,J3,a,ACGT
"""))

    def test_quadrant_needs_a_plate(self, tmp_path):
        with pytest.raises(ExpectedPlateError, match="plate column"):
            read_expected_plate(_csv(tmp_path, """
quadrant,well,name,sequence
TL,A1,a,ACGT
"""))


class TestRefusals:

    @pytest.mark.parametrize("body, match", [
        ("1,A1,a,ACGT\n1,A01,b,ACGA", "already given on line 2"),
        ("9,A1,a,ACGT", "outside 1-8"),
        ("1,Q1,a,ACGT", "not a well name"),
        ("1,A1,a,ACGTNNACGT", "contains N"),
        ("1,A1,,", "no well has an expected sequence"),
    ])
    def test_bad_rows(self, tmp_path, body, match):
        with pytest.raises(ExpectedPlateError, match=match):
            read_expected_plate(_csv(tmp_path, "plate,well,name,sequence\n" + body))

    def test_missing_column(self, tmp_path):
        with pytest.raises(ExpectedPlateError, match="sequence"):
            read_expected_plate(_csv(tmp_path, "plate,well,name\n1,A1,a"))


class TestReferences:

    def test_one_reference_per_distinct_sequence(self, tmp_path):
        p = read_expected_plate(_csv(tmp_path, """
plate,well,name,sequence
1,A1,wt rep1,ACGTACGT
1,A2,wt rep2,ACGTACGT
1,A3,m/1,ACGTTCGT
"""))
        fasta = p.write_reference_fasta(tmp_path / "ref.fa").read_text()
        assert fasta == ">wt_rep1\nACGTACGT\n>m_1\nACGTTCGT\n"
        assert p.reference_name(p.wells[(1, "A2")]) == "wt_rep1"

    def test_clashing_names_stay_distinct(self, tmp_path):
        p = read_expected_plate(_csv(tmp_path, """
plate,well,name,sequence
1,A1,v 1,ACGT
1,A2,v/1,ACGA
"""))
        assert sorted(p.constructs().values()) == ["v_1", "v_1_2"]


class TestOtherSheets:
    """A LevSeq mapping sheet names its columns its own way, and may give
    whole amplicons rather than marked-up inserts."""

    FLANK_5 = "ttaatacgactcactatagggagaccacaacggtttccctctagaaataattttgtttaactttaag"
    FLANK_3 = "ggatccgaattcgagctccgtcgacaagcttgcggccgcactcgagcaccaccaccaccaccactga"

    def _sheet(self, tmp_path,
               header="id,name,bc_plate,bc_well,clone_plate,clone_well,amplicon_seq"):
        inserts = ["atgaaacgtgcaattggc", "atgcccgggtttaaatag", "atgaaacgtgcaattggc"]
        lines = [header] + [
            f"lib.{i},g{i % 2},8,{w},1,{w},{self.FLANK_5}{ins}{self.FLANK_3}"
            for i, (w, ins) in enumerate(zip(("A1", "A2", "B13"), inserts))
        ]
        return _csv(tmp_path, "\n".join(lines))

    def test_mapping_sheet_columns_are_recognised(self, tmp_path):
        p = read_expected_plate(self._sheet(tmp_path))
        assert p.plates == [8] and set(p.wells) == {(8, "A1"), (8, "A2"), (8, "B13")}
        assert p.wells[(8, "A1")].source == "1:A1"

    def test_whole_amplicons_are_split_at_the_shared_ends(self, tmp_path):
        p = read_expected_plate(self._sheet(tmp_path))
        assert p.amplicons
        assert p.flank_5p.startswith(self.FLANK_5.upper())
        assert p.wells[(8, "A1")].sequence.endswith("TGCAATTGGC")
        assert len(p.constructs()) == 2
        assert any("whole amplicons" in n for n in p.notes)

    def test_columns_can_be_named(self, tmp_path):
        path = self._sheet(tmp_path, header="ident,construct name,lp,lw,cp,cw,amp")
        cols = parse_columns("plate=lp,well=lw,sequence=amp,name=construct name,"
                             "clone_plate=cp,clone_well=cw")
        p = read_expected_plate(path, columns=cols)
        assert p.wells[(8, "B13")].name == "g0"
        assert p.wells[(8, "B13")].source == "1:B13"

    def test_a_named_column_must_exist(self, tmp_path):
        with pytest.raises(ExpectedPlateError, match="no column 'wellz'"):
            read_expected_plate(self._sheet(tmp_path), columns={"well": "wellz"})

    @pytest.mark.parametrize("text, match", [
        ("well", "not field=column"),
        ("colour=red", "not a field"),
    ])
    def test_bad_columns_option(self, text, match):
        with pytest.raises(ExpectedPlateError, match=match):
            parse_columns(text)

    def test_marked_inserts_are_not_split(self, tmp_path):
        p = read_expected_plate(_csv(tmp_path, f"""
plate,well,name,sequence
1,A1,a,{self.FLANK_5}ATGAAACGT{self.FLANK_3}
1,A2,b,{self.FLANK_5}ATGCCCGGG{self.FLANK_3}
"""))
        assert not p.amplicons and p.wells[(1, "A1")].sequence == "ATGAAACGT"
