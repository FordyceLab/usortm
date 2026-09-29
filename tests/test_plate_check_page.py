"""Tests for the plate map component and the plate check page built on it."""

import re

from usortm.demux.expected_plate import quadrant_to_384, read_expected_plate
from usortm.demux.verify import WellVerdict
from usortm.report.plate_check import plate_maps, quadrant_of_384, write_plate_check_page
from usortm.report.platemap import (
    Well,
    legend,
    plate_map_assets,
    plate_map_page,
    render_plate,
)


def _cells(html):
    """``(tag, classes, href)`` of every well, in grid order."""
    return [(t, c, h or None) for t, c, h in
            re.findall(r'<(a|i) class="(upm-w[^"]*)"(?: href="([^"]*)")?', html)]


def _plate(tmp_path, text):
    p = tmp_path / "plate.csv"
    p.write_text(text.strip() + "\n")
    return read_expected_plate(p)


# --- the component ----------------------------------------------------------

class TestRenderPlate:

    def test_grid_and_axes(self):
        html = render_plate([], rows=8, cols=12)
        assert len(_cells(html)) == 96
        top = re.findall(r'<span class="upm-ax top">(\d*)</span>', html)
        side = re.findall(r'<span class="upm-ax">([A-P])</span>', html)
        assert top == [""] + [str(c) for c in range(1, 13)]
        assert side == list("ABCDEFGH")
        assert len(_cells(render_plate([], rows=16, cols=24))) == 384

    def test_wells_land_where_they_say(self):
        html = render_plate([Well(8, 12, depth=40, flag="mut", href="x.html")])
        cells = _cells(html)
        assert cells[-1] == ("a", "upm-w upm-mut", "x.html")
        assert all(c[0] == "i" for c in cells[:-1])

    def test_blank_well_has_no_depth_fill(self):
        html = render_plate([Well(1, 1, flag="blank")])
        assert re.search(r'<i class="upm-w upm-blank" style="">', html)

    def test_ramp_and_caption(self):
        html = render_plate([], caption="Plate 1")
        assert html.count('class="cbar"') == 1 and 'class="upm-cap">Plate 1<' in html
        assert 'class="cbar"' not in render_plate([], scale=False)

    def test_tip_and_caption_are_escaped(self):
        html = render_plate([Well(1, 1, tip='<b>x</b><script>alert(1)</script>')],
                            caption="<script>")
        assert "<script>" not in html

    def test_scoped_to_the_component(self):
        """Every selector the fragment brings is under .upm, so embedding a map
        restyles nothing else on the host page."""
        css = re.search(r"<style>(.*?)</style>", plate_map_assets(), re.S).group(1)
        css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
        css = re.sub(r"@media[^{]*\{", "", css)
        for rule in re.findall(r"([^{}]+)\{[^{}]*\}", css):
            for sel in rule.split(","):
                sel = sel.strip()
                if sel:
                    assert "upm" in sel, sel

    def test_legend(self):
        html = legend([("mut", "flagged"), ("blank", "empty")], note="click a well")
        assert '<i class="upm-mut">' in html and '<i class="upm-blank">' in html
        assert "click a well" in html

    def test_page_has_toggle_and_assets(self):
        page = plate_map_page("Maps", render_plate([]), meta="m")
        assert 'class="upm-toggle"' in page
        assert "__upmTip" in page and page.count("<h1>Maps</h1>") == 1


# --- the plate check --------------------------------------------------------

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
    html = plate_maps(plate, verdicts, {"1_P24": "pileup/well_1_P24_summary.html"})
    assert "BR quadrant (RB04)" in html
    cells = _cells(html)
    assert len(cells) == 96
    assert cells[0][1] == "upm-w"
    assert cells[-1] == ("a", "upm-w upm-mut", "pileup/well_1_P24_summary.html")


def test_384_well_plate_is_drawn_whole(tmp_path):
    plate = _plate(tmp_path, """
plate,well,name,sequence
2,P24,a,ACGT
""")
    cells = _cells(plate_maps(plate, [WellVerdict(2, "P24", "P24", "a", "changed", reads=30)]))
    assert len(cells) == 384 and cells[-1][1] == "upm-w upm-mut"


def test_expected_empty_wells_are_hatched(tmp_path):
    plate = _plate(tmp_path, """
plate,well,name,sequence
1,A1,a,ACGT
1,A2,,
""")
    cells = _cells(plate_maps(plate, [WellVerdict(1, "A1", "A1", "a", "match", reads=40)]))
    assert [c[1] for c in cells[:3]] == ["upm-w", "upm-w upm-blank", "upm-w"]


def test_page_is_the_maps_and_legend(tmp_path):
    plate = _plate(tmp_path, """
plate,well,name,sequence
1,A1,<b>x</b>,ACGT
""")
    v = [WellVerdict(1, "A1", "A1", "_b_x_b_", "unrecognised", reads=40,
                     note='<script>alert("x")</script>')]
    page = write_plate_check_page(plate, v, tmp_path / "p.html").read_text()
    body = page.split("<body>")[1]
    assert "<table" not in body and 'class="upm upm-legend"' in body
    assert "<script>alert" not in page
