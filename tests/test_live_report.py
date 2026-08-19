"""Tests for the dashboard that fills in while a run is in progress.

It has to be written as figures are established rather than at the end, and
it must never be able to stop the run it is reporting on.
"""

import json

import pandas as pd
import pytest

from usortm.demux.live import DATA_FILE, PAGE_FILE, STAGES, LiveReport
from usortm.demux.pipeline import _wells_per_plate


def _read(out_dir):
    raw = (out_dir / DATA_FILE).read_text()
    return json.loads(raw[raw.index("=") + 1:].rstrip().rstrip(";"))


class TestWritesAsItGoes:

    def test_both_files_exist_before_any_figure(self, tmp_path):
        """The page is openable from the moment the run starts."""
        LiveReport(tmp_path)
        assert (tmp_path / PAGE_FILE).exists()
        assert (tmp_path / DATA_FILE).exists()

    def test_figures_accumulate(self, tmp_path):
        live = LiveReport(tmp_path)
        live.update(input_reads=4000)
        live.update(aligned=895)
        live.update(fbc=615, rbc=434)

        d = _read(tmp_path)
        assert d["input_reads"] == 4000
        assert d["aligned"] == 895
        assert d["fbc"] == 615 and d["rbc"] == 434

    def test_stage_advances(self, tmp_path):
        live = LiveReport(tmp_path)
        assert _read(tmp_path)["stage"] == "deps"
        live.set_stage("align")
        assert _read(tmp_path)["stage"] == "align"

    def test_every_stage_key_is_known_to_the_page(self, tmp_path):
        """The page renders the stage list from the data, so a key the list
        does not contain would show no active step."""
        live = LiveReport(tmp_path)
        known = {k for k, _ in STAGES}
        for key in ("config", "hist", "readdf", "wells", "variants",
                    "streakout", "done"):
            assert key in known
            live.set_stage(key)
            assert _read(tmp_path)["stage"] == key

    def test_none_values_do_not_overwrite(self, tmp_path):
        live = LiveReport(tmp_path)
        live.update(aligned=895)
        live.update(aligned=None)
        assert _read(tmp_path)["aligned"] == 895

    def test_label_is_carried(self, tmp_path):
        LiveReport(tmp_path, label="run2")
        assert _read(tmp_path)["label"] == "run2"

    def test_data_is_a_script_assignment(self, tmp_path):
        """A file:// page cannot fetch() a sibling, but can load a script."""
        LiveReport(tmp_path)
        assert (tmp_path / DATA_FILE).read_text().startswith("window.USORTM_LIVE =")

    def test_page_reloads_the_data_file(self, tmp_path):
        LiveReport(tmp_path)
        page = (tmp_path / PAGE_FILE).read_text()
        assert "live_data.js?t=" in page      # cache-busted reload
        assert "setInterval" in page


class TestNeverBreaksTheRun:

    def test_an_unwritable_directory_disables_it(self, tmp_path, monkeypatch):
        live = LiveReport(tmp_path)

        def _boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr("pathlib.Path.write_text", _boom)
        live.update(aligned=1)               # must not raise
        assert live.enabled is False

    def test_a_later_failure_stops_it_writing_but_not_the_run(self, tmp_path,
                                                              monkeypatch):
        live = LiveReport(tmp_path)
        live.update(aligned=5)
        monkeypatch.setattr("pathlib.Path.write_text",
                            lambda *a, **k: (_ for _ in ()).throw(OSError()))
        live.set_stage("done")
        assert live.enabled is False


class TestWellsPerPlate:

    def test_counts_by_plate(self):
        df = pd.DataFrame({"plate": [1, 1, 2, 3, 3, 3]})
        assert _wells_per_plate(df) == {"1": 2, "2": 1, "3": 3}

    def test_empty_frame(self):
        assert _wells_per_plate(pd.DataFrame({"plate": []})) == {}

    def test_missing_column(self):
        assert _wells_per_plate(pd.DataFrame({"well": ["A1"]})) == {}

    def test_none(self):
        assert _wells_per_plate(None) == {}

    def test_unparseable_plate_values_do_not_raise(self):
        """This feeds a display; a malformed table must not stop the run."""
        assert _wells_per_plate(pd.DataFrame({"plate": ["x", "y"]})) == {}
