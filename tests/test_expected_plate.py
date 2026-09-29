"""Tests for reading an expected-plate CSV.

The mapping from a 96-well position in a quadrant to a 384-well position is
checked against barcode_to_well itself, so the CSV cannot address a well the
pipeline would decode differently.
"""

import pytest

from usortm.demux.expected_plate import (
    QUADRANTS,
    ExpectedPlateError,
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
