"""A plate map well opens the summary, and the summary leads to the reads.

The summary names what changed in 34 KB; the pileup draws every read in 828.
Opening the small page first is the point, so the maps link to it where it
exists -- without losing the wells that have only a pileup.
"""
import os

import pytest

from usortm.report.plates import PILEUP_SOURCES, pileup_links


def _write(root, rel, names):
    d = os.path.join(str(root), rel)
    os.makedirs(d, exist_ok=True)
    for n in names:
        with open(os.path.join(d, n), "w") as fh:
            fh.write("<html></html>")
    return d


def test_a_well_with_a_summary_links_to_it(tmp_path):
    _write(tmp_path, PILEUP_SOURCES[0],
           ["well_1_A1.html", "well_1_A1_summary.html"])
    links = pileup_links(tmp_path)
    assert links["1_A1"].endswith("well_1_A1_summary.html")


def test_a_well_without_one_keeps_its_pileup_link(tmp_path):
    _write(tmp_path, PILEUP_SOURCES[0], ["well_2_B2.html"])
    links = pileup_links(tmp_path)
    assert links["2_B2"].endswith("well_2_B2.html")


def test_a_summary_does_not_invent_a_well(tmp_path):
    """well_1_A1_summary.html must not register a well '1_A1_summary'."""
    _write(tmp_path, PILEUP_SOURCES[0],
           ["well_1_A1.html", "well_1_A1_summary.html"])
    links = pileup_links(tmp_path)
    assert "1_A1_summary" not in links
    assert set(links) == {"1_A1"}


def test_both_kinds_together(tmp_path):
    _write(tmp_path, PILEUP_SOURCES[0],
           ["well_1_A1.html", "well_1_A1_summary.html",
            "well_1_A2.html",
            "well_3_C7.html", "well_3_C7_summary.html"])
    links = pileup_links(tmp_path)
    assert set(links) == {"1_A1", "1_A2", "3_C7"}
    assert links["1_A1"].endswith("_summary.html")
    assert links["3_C7"].endswith("_summary.html")
    assert links["1_A2"].endswith("well_1_A2.html")


def test_the_newest_summary_wins_across_directories(tmp_path):
    """The same rule the pileups already follow: when, not which command."""
    if len(PILEUP_SOURCES) < 2:
        pytest.skip("only one pileup source")
    old = _write(tmp_path, PILEUP_SOURCES[0], ["well_1_A1_summary.html"])
    new = _write(tmp_path, PILEUP_SOURCES[1], ["well_1_A1_summary.html"])
    os.utime(os.path.join(old, "well_1_A1_summary.html"), (1, 1))
    links = pileup_links(tmp_path)
    assert links["1_A1"].startswith(PILEUP_SOURCES[1])
