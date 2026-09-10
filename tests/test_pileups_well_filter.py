"""Rendering named wells, and not losing the rest of the plate's links.

The plate map is rebuilt whole from the URL map it is given, so a map built
from only the wells just rendered drops the links for every other well.  That
is harmless while every well is rendered every time, and destructive the
moment a single well can be asked for.
"""
import csv

import pytest

from usortm.cli.pileups import _relink_plate_map


def _pileups(tmp_path, keys):
    d = tmp_path / "pileups" / "pileup"
    d.mkdir(parents=True)
    for k in keys:
        (d / f"well_{k}.html").write_text("<html></html>")
    return tmp_path / "pileups"


def test_relinking_keeps_every_pileup_on_disk(tmp_path, monkeypatch):
    """Rendering one well must not unlink the other ninety-five."""
    demux = tmp_path / "demux_output"
    demux.mkdir()
    (demux / "plate_map.html").write_text("<html></html>")
    (demux / "read_df.csv").write_text("well_pos\n1A1\n")
    out = _pileups(demux, ["1_A1", "1_A2", "1_A3", "4_D21"])

    captured = {}

    def fake_save(read_df, path, **kw):
        captured.update(kw)

    import usortm.demux.viz as viz
    monkeypatch.setattr(viz, "save_plate_map_html", fake_save)
    monkeypatch.setattr(viz, "load_plate_map_reads",
                        lambda p: _NonEmpty())

    # Only one well was rendered this time.
    _relink_plate_map(demux, out, [{"plate": "4", "well": "D21"}], 20)

    url_map = captured.get("pileup_url_map") or {}
    assert "4_D21" in url_map, "the rendered well must be linked"
    for other in ("1_A1", "1_A2", "1_A3"):
        assert other in url_map, (
            f"{other} has a pileup on disk and lost its link")


def test_a_well_with_no_pileup_is_not_linked(tmp_path, monkeypatch):
    demux = tmp_path / "demux_output"
    demux.mkdir()
    (demux / "plate_map.html").write_text("<html></html>")
    (demux / "read_df.csv").write_text("well_pos\n1A1\n")
    out = _pileups(demux, ["1_A1"])

    captured = {}
    import usortm.demux.viz as viz
    monkeypatch.setattr(viz, "save_plate_map_html",
                        lambda read_df, path, **kw: captured.update(kw))
    monkeypatch.setattr(viz, "load_plate_map_reads", lambda p: _NonEmpty())

    _relink_plate_map(demux, out, [{"plate": "9", "well": "Z9"}], 20)
    assert "9_Z9" not in (captured.get("pileup_url_map") or {})


class _NonEmpty:
    """A stand-in for the read table, which the relink only checks is there."""

    empty = False


def test_the_command_offers_a_well_option():
    """The option exists and says what it takes."""
    import re

    from typer.testing import CliRunner

    from usortm.cli import app

    out = re.sub(r"\x1b\[[0-9;:]*m", "",
                 CliRunner().invoke(app, ["pileups", "--help"]).stdout)
    assert "--well" in out
    joined = " ".join(out.split())
    assert "comma-separated" in joined
