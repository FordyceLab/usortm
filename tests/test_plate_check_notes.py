"""Tests for how the plate check heads and references each well's summary page."""

from usortm.cli.demux_expected import _assigned_references, _verdict_note
from usortm.demux.verify import WellVerdict


def test_assigned_references_come_from_the_pipeline(tmp_path):
    (tmp_path / "well_df.csv").write_text(
        "plate,well,major_ref\n"
        "6,I4,CF-g051\n"
        "6,J4,something_not_on_the_plate\n"
        "x,K4,CF-g051\n"
    )
    out = _assigned_references(tmp_path, {"CF-g049", "CF-g051"})
    # A construct not on the plate, or an unreadable row, is left out.
    assert out == {(6, "I4"): "CF-g051"}


def test_assigned_references_without_a_well_table_is_empty(tmp_path):
    assert _assigned_references(tmp_path, {"a"}) == {}


def test_note_for_an_expected_well_states_the_verdict():
    v = WellVerdict(plate=6, well="A4", label="A4", expected="CF-g049",
                    verdict="match", reads=13, sample="1:A3")
    assert _verdict_note(v, "CF-g049") == (
        "Barcode plate 6, well A4 (sample 1:A3) · expected CF-g049: "
        "match · 13 reads")


def test_note_for_a_swapped_well_names_what_it_holds():
    v = WellVerdict(plate=1, well="B2", label="TL:B1", expected="a",
                    verdict="wrong construct", observed="b", reads=20)
    assert "expected a: wrong construct, holds b" in _verdict_note(v, "a")


def test_note_for_an_unlisted_well_names_the_reference_drawn():
    v = WellVerdict(plate=6, well="M4", label="M4", expected="",
                    verdict="unexpected reads", reads=13)
    assert "not in the expected plate; drawn against EV" in _verdict_note(v, "EV")
