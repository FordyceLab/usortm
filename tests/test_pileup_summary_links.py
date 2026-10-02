"""A well's pileup and its summary link to each other, on any seqviewer.

A seqviewer that takes the counterpart's href draws the link itself; the
version usortm pins does not, and the link is placed on the page instead.
"""

from usortm.demux.streakout import _render_linked, summary_path_for


def test_summary_sits_beside_its_pileup():
    assert summary_path_for("out/pileup/well_8_A1.html") == "out/pileup/well_8_A1_summary.html"


def test_seqviewer_draws_the_link_when_it_can():
    seen = {}

    def render(view, summary_href=None):
        seen["href"] = summary_href
        return "<html><body>page</body></html>"

    page = _render_linked(render, object(), "summary_href", "well_1_A1_summary.html", "← Summary")
    assert seen["href"] == "well_1_A1_summary.html"
    assert page == "<html><body>page</body></html>"


def test_link_is_placed_when_seqviewer_cannot():
    def render_summary(view, max_lanes=2):
        return '<html><body class="x"><main>summary</main></body></html>'

    page = _render_linked(render_summary, object(), "pileup_href",
                          'well_1_A1.html"><script>', "All reads →")
    assert page.startswith('<html><body class="x"><a href="well_1_A1.html&quot;&gt;&lt;script&gt;"')
    assert "All reads →</a><main>" in page
    assert "<script>" not in page
