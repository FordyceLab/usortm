"""Tests for picking into a destination plate that was designed in advance.

Ordinarily pick decides where a hit lands. A layout takes that decision, which
is what a designed plate needs: the wells were chosen so the scans fall in
known quadrants, and a hit has to arrive where the design says however many
other variants were recovered.
"""

import pytest

from usortm.cli.pick import LayoutError, _apply_layout, _load_layout


def _write(tmp_path, text, name="layout.csv"):
    path = tmp_path / name
    path.write_text(text)
    return path


def _hit(variant, plate="1", well="A1", reads=500):
    return {"variant": variant, "source_plate": plate, "source_well": well,
            "reads": reads, "consensus_fraction": 0.99}


class TestLoading:

    def test_reads_wells_and_variants(self, tmp_path):
        path = _write(tmp_path, "well,variant\nA1,G3A\nB1,G3F\n")
        assert _load_layout(path) == [
            {"plate": "0", "well": "A1", "variant": "G3A"},
            {"plate": "0", "well": "B1", "variant": "G3F"},
        ]

    def test_file_order_is_kept(self, tmp_path):
        path = _write(tmp_path, "well,variant\nH12,x\nA1,y\nC3,z\n")
        assert [r["well"] for r in _load_layout(path)] == ["H12", "A1", "C3"]

    def test_a_row_naming_no_variant_is_a_blank_well(self, tmp_path):
        path = _write(tmp_path, "well,variant\nA1,G3A\nA2,\n")
        assert _load_layout(path)[1]["variant"] is None

    def test_a_plate_column_spreads_it_over_plates(self, tmp_path):
        path = _write(tmp_path, "plate,well,variant\n1,A1,x\n2,A1,y\n")
        assert [r["plate"] for r in _load_layout(path)] == ["1", "2"]

    def test_the_same_well_twice_on_one_plate_is_refused(self, tmp_path):
        path = _write(tmp_path, "well,variant\nA1,x\nA1,y\n")
        with pytest.raises(LayoutError, match="appears twice"):
            _load_layout(path)

    def test_the_same_well_on_different_plates_is_fine(self, tmp_path):
        path = _write(tmp_path, "plate,well,variant\n1,A1,x\n2,A1,y\n")
        assert len(_load_layout(path)) == 2

    def test_alternative_column_names(self, tmp_path):
        path = _write(tmp_path, "target_well,name\nA1,G3A\n")
        assert _load_layout(path)[0] == {"plate": "0", "well": "A1",
                                         "variant": "G3A"}

    def test_a_missing_column_says_what_it_found(self, tmp_path):
        path = _write(tmp_path, "well,notes\nA1,something\n")
        with pytest.raises(LayoutError, match="notes"):
            _load_layout(path)

    def test_an_empty_file_is_refused(self, tmp_path):
        with pytest.raises(LayoutError, match="no rows"):
            _load_layout(_write(tmp_path, "well,variant\n"))

    def test_a_missing_file_is_refused(self, tmp_path):
        with pytest.raises(LayoutError, match="could not read"):
            _load_layout(tmp_path / "absent.csv")


class TestColumnsThatMustNotBeRead:
    """A designed layout usually carries its own source columns, meaning where
    the variant came from in the synthesis plates.  Those are not the sorted
    wells pick draws from, and reading them would put hits in the wrong
    place."""

    def test_the_layout_s_own_source_columns_are_ignored(self, tmp_path):
        path = _write(
            tmp_path,
            "well,source_plate,source_well,variant\nA1,2,H12,G3A\n",
        )
        assert _load_layout(path) == [
            {"plate": "0", "well": "A1", "variant": "G3A"}
        ]

    def test_a_hit_keeps_the_source_it_was_recovered_from(self, tmp_path):
        """The layout says where a variant goes, never where it came from."""
        layout = _load_layout(_write(
            tmp_path,
            "well,source_plate,source_well,variant\nA1,2,H12,G3A\n",
        ))
        picks = [_hit("G3A", plate="7", well="C4")]
        _apply_layout(picks, layout)
        assert picks[0]["source_plate"] == "7"
        assert picks[0]["source_well"] == "C4"
        assert picks[0]["target_well"] == "A1"


class TestPlacing:

    def test_a_hit_lands_where_the_design_says(self, tmp_path):
        layout = _load_layout(_write(tmp_path,
                                     "well,variant\nA1,G3A\nP24,G3F\n"))
        picks = [_hit("G3F"), _hit("G3A")]
        _apply_layout(picks, layout)
        assert [(h["variant"], h["target_well"]) for h in picks] == [
            ("G3A", "A1"), ("G3F", "P24"),
        ]

    def test_the_list_follows_the_layout_s_order_not_the_hits(self, tmp_path):
        layout = _load_layout(_write(tmp_path,
                                     "well,variant\nB2,b\nA1,a\nC3,c\n"))
        picks = [_hit("a"), _hit("b"), _hit("c")]
        _apply_layout(picks, layout)
        assert [h["variant"] for h in picks] == ["b", "a", "c"]

    def test_an_unrecovered_variant_leaves_its_well_empty(self, tmp_path):
        layout = _load_layout(_write(tmp_path,
                                     "well,variant\nA1,got\nA2,missing\n"))
        picks = [_hit("got")]
        stats = _apply_layout(picks, layout)
        assert len(picks) == 2
        assert picks[1]["empty"] is True
        assert picks[1]["target_well"] == "A2"
        assert picks[1]["reads"] == 0
        assert stats["not_recovered"] == 1

    def test_a_well_the_design_leaves_blank_gets_no_entry(self, tmp_path):
        """Nothing is picked into it, so it is not a line on a pick list."""
        layout = _load_layout(_write(tmp_path,
                                     "well,variant\nA1,got\nA2,\n"))
        picks = [_hit("got")]
        stats = _apply_layout(picks, layout)
        assert len(picks) == 1
        assert stats["designed_blank"] == 1

    def test_a_recovered_variant_with_no_well_is_reported(self, tmp_path):
        """It is dropped, so it has to be said rather than silently lost."""
        layout = _load_layout(_write(tmp_path, "well,variant\nA1,known\n"))
        picks = [_hit("known"), _hit("stranger")]
        stats = _apply_layout(picks, layout)
        assert stats["unplaced"] == ["stranger"]
        assert [h["variant"] for h in picks] == ["known"]

    def test_counts_add_up(self, tmp_path):
        layout = _load_layout(_write(
            tmp_path, "well,variant\nA1,a\nA2,b\nA3,\nA4,d\n"))
        picks = [_hit("a"), _hit("d"), _hit("elsewhere")]
        stats = _apply_layout(picks, layout)
        assert stats == {"filled": 2, "not_recovered": 1,
                         "designed_blank": 1, "unplaced": ["elsewhere"]}

    def test_existing_empty_placeholders_do_not_take_a_well(self, tmp_path):
        """Placeholders from an earlier stage are not recoveries."""
        layout = _load_layout(_write(tmp_path, "well,variant\nA1,a\n"))
        picks = [{"variant": "a", "empty": True}]
        stats = _apply_layout(picks, layout)
        assert stats["filled"] == 0
        assert stats["not_recovered"] == 1

    def test_the_first_well_offered_for_a_variant_wins(self, tmp_path):
        """pick sorts its candidates before this, so the best is already
        first."""
        layout = _load_layout(_write(tmp_path, "well,variant\nA1,a\n"))
        picks = [_hit("a", well="BEST", reads=900),
                 _hit("a", well="worse", reads=10)]
        _apply_layout(picks, layout)
        assert len(picks) == 1
        assert picks[0]["source_well"] == "BEST"


class TestRealLayout:
    """The 384-well plate designed for the AFMtag scan."""

    LAYOUT = ("/Users/micaholivas/Downloads/"
              "plate_384well_final_library.csv")

    @pytest.fixture
    def layout(self):
        import os
        if not os.path.exists(self.LAYOUT):
            pytest.skip("designed layout not present on this machine")
        return _load_layout(self.LAYOUT)

    def test_it_is_a_full_384_plate(self, layout):
        assert len(layout) == 384
        assert len({r["well"] for r in layout}) == 384

    def test_every_designed_variant_gets_one_well(self, layout):
        named = [r["variant"] for r in layout if r["variant"]]
        assert len(named) == 376
        assert len(set(named)) == 376

    def test_the_rest_are_blanks(self, layout):
        assert sum(1 for r in layout if r["variant"] is None) == 8
