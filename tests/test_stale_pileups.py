"""Pileup pages must not outlive the run that made them.

Each call writes only the wells it was asked for, so a well rendered by an
earlier run and not by this one kept its old page. On a project run several
times at different depths that shows as a well whose pileup holds a fraction
of its reads — reading as collapsed depth rather than as a stale file.
"""

import os

import pytest

from usortm.demux.streakout import _clear_stale_pileups


def _pages(tmp_path, names):
    d = tmp_path / "pileup"
    d.mkdir(exist_ok=True)
    for n in names:
        (d / n).write_text("old")
    return d


class TestClearing:

    def test_pages_not_regenerated_are_removed(self, tmp_path):
        d = _pages(tmp_path, ["well_1_A12.html", "well_1_A9.html",
                              "well_10_D15.html"])
        removed = _clear_stale_pileups(d, keep={"well_10_D15.html"})

        assert removed == 2
        assert sorted(os.listdir(d)) == ["well_10_D15.html"]

    def test_pages_being_regenerated_are_left(self, tmp_path):
        """No point deleting a file that is about to be rewritten."""
        d = _pages(tmp_path, ["well_1_A12.html"])
        _clear_stale_pileups(d, keep={"well_1_A12.html"})
        assert (d / "well_1_A12.html").exists()

    def test_no_keep_set_clears_everything(self, tmp_path):
        d = _pages(tmp_path, ["well_1_A1.html", "well_2_B2.html"])
        assert _clear_stale_pileups(d, keep=None) == 2
        assert os.listdir(d) == []

    def test_other_files_are_untouched(self, tmp_path):
        """The directory also holds an index and a cache; leave them."""
        d = _pages(tmp_path, ["well_1_A1.html"])
        (d / "index.html").write_text("index")
        (d / "notes.txt").write_text("x")

        _clear_stale_pileups(d, keep=set())
        assert sorted(os.listdir(d)) == ["index.html", "notes.txt"]

    def test_empty_directory(self, tmp_path):
        d = tmp_path / "pileup"
        d.mkdir()
        assert _clear_stale_pileups(d, keep=set()) == 0

    def test_an_undeletable_file_does_not_raise(self, tmp_path, monkeypatch):
        """This is housekeeping; it must not take the run down."""
        d = _pages(tmp_path, ["well_1_A1.html"])
        monkeypatch.setattr(
            os, "remove",
            lambda *a, **k: (_ for _ in ()).throw(OSError("busy")),
        )
        assert _clear_stale_pileups(d, keep=set()) == 0


class TestNamesMatchWhatIsWritten:

    def test_keep_names_use_the_written_form(self, tmp_path):
        """The keep set is built from tasks; it has to match the filenames
        _generate_one_pick_pileup writes, or every page is deleted each run."""
        plate, well = "10", "D15"
        written = f"well_{plate}_{well}.html"
        d = _pages(tmp_path, [written])

        keep = {f"well_{t['source_plate']}_{t['source_well']}.html"
                for t in [{"source_plate": plate, "source_well": well}]}
        _clear_stale_pileups(d, keep=keep)
        assert (d / written).exists()
