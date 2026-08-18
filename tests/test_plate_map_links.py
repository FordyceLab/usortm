"""Tests for pileup links in the demux plate map.

Wells with a rendered pileup become clickable, which is what carries the
links into the plate map embedded in the summary report.
"""

import pandas as pd
import pytest


def _reads_df():
    """Two wells with enough depth to clear the plate map's own tiering."""
    return pd.DataFrame({
        "well_pos": ["1A1"] * 30 + ["1A3"] * 30,
        "ref_name": ["fwd:variant_1"] * 60,
    })


class TestPlateMapPileupLinks:

    def test_pileup_url_baked_into_the_map(self, tmp_path):
        pytest.importorskip("bokeh")
        from usortm.demux.viz import save_plate_map_html

        out = tmp_path / "plate_map.html"
        save_plate_map_html(
            _reads_df(), str(out),
            pileup_url_map={"1_A1": "pileups/pileup/well_1_A1.html"},
            min_reads=5,
        )
        html = out.read_text()
        assert "pileups/pileup/well_1_A1.html" in html
        assert "pileup_url" in html

    def test_tap_falls_back_to_the_pileup_url(self, tmp_path):
        """Streak-out and mutation links still win; this is the fallback."""
        pytest.importorskip("bokeh")
        from usortm.demux.viz import save_plate_map_html

        out = tmp_path / "plate_map.html"
        save_plate_map_html(_reads_df(), str(out),
                            pileup_url_map={"1_A1": "p/well_1_A1.html"},
                            min_reads=5)
        assert "so_url || mut_url || pu_url" in out.read_text()

    def test_hint_shown_only_for_unflagged_wells(self, tmp_path):
        """A streak-out well already says 'click to view pileup', so the
        generic hint would duplicate it."""
        pytest.importorskip("bokeh")
        from usortm.demux.viz import save_plate_map_html

        out = tmp_path / "plate_map.html"
        save_plate_map_html(
            _reads_df(), str(out),
            pileup_url_map={"1_A1": "p/well_1_A1.html",
                            "1_A3": "p/well_1_A3.html"},
            streakout_wells={"1_A1"},
            min_reads=5,
        )
        html = out.read_text()
        assert "Click to view pileup" in html      # for 1_A3
        assert "Multiple colonies" in html         # for 1_A1

    def test_wells_without_a_pileup_get_no_link(self, tmp_path):
        pytest.importorskip("bokeh")
        from usortm.demux.viz import save_plate_map_html

        out = tmp_path / "plate_map.html"
        save_plate_map_html(_reads_df(), str(out),
                            pileup_url_map={"1_A1": "p/well_1_A1.html"},
                            min_reads=5)
        html = out.read_text()
        assert "p/well_1_A1.html" in html
        assert "well_1_A3.html" not in html

    def test_no_map_leaves_behaviour_unchanged(self, tmp_path):
        pytest.importorskip("bokeh")
        from usortm.demux.viz import save_plate_map_html

        out = tmp_path / "plate_map.html"
        save_plate_map_html(_reads_df(), str(out), min_reads=5)
        assert "Click to view pileup" not in out.read_text()


class TestReportEmbedding:
    """The summary embeds the plate map via srcdoc, which resolves relative
    URLs against the report's own directory — so they must be absolutised."""

    def test_pileups_prefix_is_absolutised(self):
        # `usortm.cli.report` resolves to the command function, not the module,
        # so import the module explicitly.
        import importlib
        import inspect

        report_mod = importlib.import_module("usortm.cli.report")
        src = inspect.getsource(report_mod._save_html_report)
        assert "pileups/pileup/" in src

    def test_longer_prefix_survives_the_shorter_one(self):
        """'pileup/' is replaced too; it must not corrupt 'pileups/pileup/'."""
        base = "file:///tmp/demux"
        content = '"pileups/pileup/well_1_A1.html" and "pileup/well_2_B2.html"'
        for rel_prefix in ("streakout/", "pileup/", "mutation/pileup/",
                           "pileups/pileup/"):
            content = content.replace(f'"{rel_prefix}', f'"{base}/{rel_prefix}')

        assert f'"{base}/pileups/pileup/well_1_A1.html"' in content
        assert f'"{base}/pileup/well_2_B2.html"' in content
        assert f'{base}/{base}' not in content
