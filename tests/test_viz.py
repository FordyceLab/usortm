"""Tests for HTML theming in plate-map visualizations."""

import pandas as pd
import pytest


def test_save_plate_map_html_injects_theme_sync(tmp_path):
    """Demux plate map HTML should include summary-style theme sync."""
    pytest.importorskip("bokeh")
    from usortm.demux.viz import save_plate_map_html

    df = pd.DataFrame(
        {
            "well_pos": ["1A1"],
            "ref_name": ["fwd:variant_1"],
        }
    )

    out = tmp_path / "plate_map.html"
    save_plate_map_html(df, str(out), title="Plate Map")

    html = out.read_text()
    assert "--usortm-bg: #fafafa" in html
    assert '[data-theme="dark"] { --usortm-bg: #1a1a2e; }' in html
    assert "localStorage.getItem('usortm-theme')" in html


def test_save_pick_plate_map_html_injects_theme_sync(tmp_path):
    """Pick plate map HTML should include summary-style theme sync."""
    pytest.importorskip("bokeh")
    from usortm.demux.viz import save_pick_plate_map_html

    pick_list = [
        {
            "variant": "variant_1",
            "source_plate": "1",
            "source_well": "A1",
            "target_plate": "1",
            "target_well": "A1",
            "reads": 120,
            "consensus_fraction": 0.95,
        }
    ]

    out = tmp_path / "pick_plate_map.html"
    save_pick_plate_map_html(pick_list, str(out), title="Pick Plate Map")

    html = out.read_text()
    assert "--usortm-bg: #fafafa" in html
    assert '[data-theme="dark"] { --usortm-bg: #1a1a2e; }' in html
    assert "localStorage.getItem('usortm-theme')" in html
