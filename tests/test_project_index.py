"""Tests for the project front page.

Output is organised by the stage that produced it, so the pages a person
opens are spread across several directories among machine artefacts. The
index gathers them.
"""

import json
import re

import pytest

from usortm.cli.project_index import INDEX_FILE, write_index


def _sections(html):
    return re.findall(r"<h2>(.*?)</h2>", html)


def _links(html):
    return re.findall(r'<a href="([^"]+)"', html)


@pytest.fixture
def project(tmp_path):
    p = tmp_path / "proj"
    (p / "demux_output").mkdir(parents=True)
    (p / "report").mkdir()
    (p / "usortm_project.json").write_text("{}")
    return p


class TestFindsWhatExists:

    def test_empty_project_says_so(self, project):
        html = write_index(project).read_text()
        assert "Nothing produced yet" in html
        assert "usortm demux" in html

    def test_summary_and_plate_map_are_listed(self, project):
        (project / "report" / "summary.html").write_text("x")
        (project / "demux_output" / "plate_map.html").write_text("x")

        html = write_index(project).read_text()
        assert "report/summary.html" in _links(html)
        assert "demux_output/plate_map.html" in _links(html)
        assert "Results" in _sections(html)

    def test_pileup_directories_are_counted(self, project):
        so = project / "demux_output" / "streakout"
        so.mkdir()
        for i in range(3):
            (so / f"well_1_A{i}.html").write_text("x")

        html = write_index(project).read_text()
        assert "Pileups" in _sections(html)
        assert "3 wells" in html

    def test_data_files_are_listed(self, project):
        (project / "demux_output" / "well_assignments.csv").write_text("x")
        html = write_index(project).read_text()
        assert "demux_output/well_assignments.csv" in _links(html)
        assert "Data" in _sections(html)

    def test_absent_things_are_not_linked(self, project):
        """A link to a file that was never produced is worse than no link."""
        html = write_index(project).read_text()
        assert "summary.html" not in html
        assert "plate_map.html" not in html

    def test_live_dashboard_appears_while_running(self, project):
        (project / "demux_output" / "live.html").write_text("x")
        html = write_index(project).read_text()
        assert "While running" in _sections(html)
        assert "demux_output/live.html" in _links(html)

    def test_sections_are_ordered_by_what_is_wanted_first(self, project):
        (project / "demux_output" / "live.html").write_text("x")
        (project / "report" / "summary.html").write_text("x")
        (project / "demux_output" / "well_assignments.csv").write_text("x")

        assert _sections(write_index(project).read_text()) == [
            "While running", "Results", "Data",
        ]


class TestRounds:

    def test_a_later_round_reads_its_own_directories(self, tmp_path):
        p = tmp_path / "proj"
        rd = p / "rounds" / "2"
        (rd / "demux_output").mkdir(parents=True)
        (rd / "report").mkdir()
        (rd / "report" / "summary.html").write_text("x")

        html = write_index(p, round_num=2).read_text()
        assert "rounds/2/report/summary.html" in _links(html)
        assert "round 2" in html


class TestWriting:

    def test_written_at_the_project_root(self, project):
        out = write_index(project)
        assert out == project / INDEX_FILE
        assert out.exists()

    def test_rewriting_replaces_rather_than_appends(self, project):
        (project / "report" / "summary.html").write_text("x")
        first = write_index(project).read_text()
        second = write_index(project).read_text()
        assert first.count("summary.html") == second.count("summary.html")

    def test_theme_aware(self, project):
        html = write_index(project).read_text()
        assert "prefers-color-scheme" in html
        assert 'data-theme="dark"' in html
