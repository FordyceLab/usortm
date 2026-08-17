"""Tests for HTML theming in plate-map visualizations."""

import pandas as pd
import pytest


def _write_read_df_csv(path, n_reads=200, read_len=1000):
    """Write a read_df.csv shaped like a real one, sequences included.

    Sequences are made distinct per read so the frame's memory footprint
    behaves like a real run rather than sharing one interned string.
    """
    import random
    rng = random.Random(0)
    seqs = [
        "".join(rng.choice("ACGT") for _ in range(read_len))
        for _ in range(n_reads)
    ]
    quals = ["I" * read_len for _ in range(n_reads)]
    pd.DataFrame({
        "read_name": [f"read_{i}" for i in range(n_reads)],
        "fbc_name": ["FB01"] * n_reads,
        "rbc_name": ["RB01"] * n_reads,
        "well_pos": ["1A1"] * n_reads,
        "ref_name": ["fwd:variant_1"] * n_reads,
        "read_seq": seqs,
        "read_qual": quals,
        "avg_qual": [40.0] * n_reads,
    }).to_csv(path, index=False)
    return path


class TestLoadPlateMapReads:
    """The plate map needs two columns; read_df.csv carries eight."""

    def test_returns_only_plate_map_columns(self, tmp_path):
        from usortm.demux.viz import load_plate_map_reads

        csv = _write_read_df_csv(tmp_path / "read_df.csv")
        df = load_plate_map_reads(csv)

        assert list(df.columns) == ["well_pos", "ref_name"]
        assert "read_seq" not in df.columns
        assert "read_qual" not in df.columns

    def test_all_rows_preserved(self, tmp_path):
        from usortm.demux.viz import load_plate_map_reads

        csv = _write_read_df_csv(tmp_path / "read_df.csv", n_reads=57)
        assert len(load_plate_map_reads(csv)) == 57

    def test_values_match_full_read(self, tmp_path):
        """Narrowing the read must not change the values that survive."""
        from usortm.demux.viz import load_plate_map_reads

        csv = _write_read_df_csv(tmp_path / "read_df.csv", n_reads=10)
        narrow = load_plate_map_reads(csv)
        full = pd.read_csv(csv)

        pd.testing.assert_frame_equal(
            narrow.reset_index(drop=True),
            full[["well_pos", "ref_name"]].reset_index(drop=True),
        )

    def test_sequences_are_never_materialised(self, tmp_path):
        """The point of the narrowed read: the resident frame must not scale
        with the sequence columns, which dominate a real read_df.csv."""
        from usortm.demux.viz import load_plate_map_reads

        csv = _write_read_df_csv(tmp_path / "read_df.csv", n_reads=2000,
                                 read_len=1000)

        narrow = load_plate_map_reads(csv)
        full = pd.read_csv(csv)

        narrow_bytes = narrow.memory_usage(deep=True).sum()
        full_bytes = full.memory_usage(deep=True).sum()

        assert len(narrow) == len(full)
        # 2000 reads x 1000 bp of sequence + quality is ~4 MB of the ~4.3 MB
        # frame, so dropping them must cut the footprint by an order of
        # magnitude.
        assert narrow_bytes < full_bytes / 10, (
            f"narrow={narrow_bytes/1e6:.2f}MB full={full_bytes/1e6:.2f}MB"
        )

    def test_falls_back_when_columns_absent(self, tmp_path):
        """An older or degenerate read_df.csv still loads rather than raising,
        so callers keep showing their own 'no reads assigned' message."""
        from usortm.demux.viz import load_plate_map_reads

        csv = tmp_path / "read_df.csv"
        pd.DataFrame({"read_name": ["read_0"]}).to_csv(csv, index=False)

        df = load_plate_map_reads(csv)
        assert list(df.columns) == ["read_name"]

    def test_empty_table_with_headers_loads_empty(self, tmp_path):
        from usortm.demux.viz import load_plate_map_reads

        csv = _write_read_df_csv(tmp_path / "read_df.csv", n_reads=0)
        df = load_plate_map_reads(csv)

        assert df.empty
        assert list(df.columns) == ["well_pos", "ref_name"]

    def test_output_feeds_save_plate_map_html(self, tmp_path):
        """The narrowed frame must still be enough to draw the map."""
        pytest.importorskip("bokeh")
        from usortm.demux.viz import load_plate_map_reads, save_plate_map_html

        csv = _write_read_df_csv(tmp_path / "read_df.csv", n_reads=5)
        out = tmp_path / "plate_map.html"
        save_plate_map_html(load_plate_map_reads(csv), str(out))

        assert out.exists() and out.stat().st_size > 0


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
