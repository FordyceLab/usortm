"""Tests for the plate check page."""

import html as hesc
import json
import re

from usortm.demux.expected_plate import quadrant_to_384, read_expected_plate
from usortm.demux.verify import WellVerdict
from usortm.report.plate_check import (
    plate_maps,
    quadrant_of_384,
    write_plate_check_page,
)


def _plate(tmp_path, text):
    p = tmp_path / "plate.csv"
    p.write_text(text.strip() + "\n")
    return read_expected_plate(p)


def _cells(html):
    """``(classes, links)`` of every well cell, in grid order."""
    out = []
    for c, rest in re.findall(r'<i class="(w[^"]*)"([^>]*)>', html):
        m = re.search(r'data-links="([^"]*)"', rest)
        out.append((c, json.loads(hesc.unescape(m.group(1))) if m else None))
    return out


def test_quadrant_inverse():
    for q in range(4):
        for r in range(1, 9):
            for c in range(1, 13):
                assert quadrant_of_384(quadrant_to_384(q, r, c)) == (q, r, c)


def test_96_well_plate_is_drawn_in_its_own_coordinates(tmp_path):
    plate = _plate(tmp_path, """
plate,quadrant,well,name,sequence
1,BR,A1,a,ACGT
1,BR,H12,b,ACGA
""")
    # A1 in BR is 384-well B2; H12 in BR is P24.
    verdicts = [
        WellVerdict(1, "B2", "BR:A1", "a", "match", observed="a", reads=40),
        WellVerdict(1, "P24", "BR:H12", "b", "wrong construct", observed="a", reads=40),
    ]
    html = plate_maps(plate, verdicts)
    assert "BR quadrant (RB04)" in html
    cells = _cells(html)
    assert len(cells) == 96
    assert cells[0][0] == "w"                  # A1 holds what it should
    assert cells[-1][0] == "w mut"             # H12 does not


def test_384_well_plate_is_drawn_whole(tmp_path):
    plate = _plate(tmp_path, """
plate,well,name,sequence
2,P24,a,ACGT
""")
    cells = _cells(plate_maps(plate, [WellVerdict(2, "P24", "P24", "a", "changed", reads=30)]))
    assert len(cells) == 384 and cells[-1][0] == "w mut"


def test_expected_empty_wells_are_hatched(tmp_path):
    plate = _plate(tmp_path, """
plate,well,name,sequence
1,A1,a,ACGT
1,A2,,
""")
    cells = _cells(plate_maps(plate, [WellVerdict(1, "A1", "A1", "a", "match", reads=40)]))
    assert [c[0] for c in cells[:3]] == ["w", "w blank", "w"]


def test_wells_open_their_pileup_and_summary(tmp_path):
    plate = _plate(tmp_path, """
plate,well,name,sequence
1,A1,a,ACGT
1,A2,b,ACGA
""")
    verdicts = [WellVerdict(1, "A1", "A1", "a", "match", reads=40),
                WellVerdict(1, "A2", "A2", "b", "no reads")]
    links = {"1_A1": {"pileup": "pileup/well_1_A1.html",
                      "summary": "pileup/well_1_A1_summary.html"}}
    cells = _cells(plate_maps(plate, verdicts, links))
    assert cells[0] == ("w", links["1_A1"])
    assert cells[1][1] is None


def test_names_are_escaped(tmp_path):
    plate = _plate(tmp_path, """
plate,well,name,sequence
1,A1,<b>x</b>,ACGT
""")
    v = [WellVerdict(1, "A1", "A1", "_b_x_b_", "unrecognised", reads=40,
                     note='<script>alert("x")</script>')]
    page = write_plate_check_page(plate, v, tmp_path / "p.html").read_text()
    assert "<script>alert" not in page
    assert page.count("<script>") == 1     # the page's own hover script


def test_page_is_the_maps_and_legend(tmp_path):
    """The page carries the plate maps, a depth ramp beside each, and the
    legend; the per-well table is verification.csv, not the page."""
    plate = _plate(tmp_path, """
plate,well,name,sequence
1,A1,a,ACGT
""")
    v = [WellVerdict(1, "A1", "A1", "a", "no reads")]
    page = write_plate_check_page(plate, v, tmp_path / "p.html").read_text()
    body = page.split("<body>")[1]
    assert "<table" not in body
    assert body.count('class="cbar"') == 1
    assert 'class="legend"' in body
    assert 'class="themetoggle"' in body


def test_rows_and_columns_are_labelled(tmp_path):
    plate = _plate(tmp_path, """
plate,quadrant,well,name,sequence
1,TL,A1,a,ACGT
""")
    html = plate_maps(plate, [])
    top = re.findall(r'<span class="ax top">(\d*)</span>', html)
    side = re.findall(r'<span class="ax">([A-P])</span>', html)
    assert top == [""] + [str(c) for c in range(1, 13)]
    assert side == list("ABCDEFGH")
