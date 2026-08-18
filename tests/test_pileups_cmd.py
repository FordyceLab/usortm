"""Tests for `usortm pileups`, which renders a pileup for every well.

The rendering itself is covered by the streak-out and pick paths; what
matters here is which wells get selected, and that the index makes thousands
of loose HTML files navigable.
"""

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from usortm.cli import app

runner = CliRunner()


@pytest.fixture
def project(tmp_path):
    """A project with wells across three plates and a spread of depths."""
    project_dir = tmp_path / "proj"
    demux = project_dir / "demux_output"
    demux.mkdir(parents=True)
    (project_dir / "usortm_project.json").write_text(json.dumps(
        {"library_size": 10, "barcode_kit": "levseq", "n_plates": 3,
         "workflow_steps": {}}
    ))

    rows = []
    for plate in (1, 2, 3):
        for i, depth in enumerate((5, 25, 150), start=1):
            rows.append({
                "plate": str(plate), "well": f"A{i}", "reads": depth,
                "variant": f"var_{i}", "consensus_fraction": 0.95,
                "cons_check": "",
            })
    with open(demux / "well_assignments.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return project_dir


def _fake_render(**kwargs):
    """Stand in for generate_pick_pileups by writing the files it would."""
    out = Path(kwargs["output_dir"]) / "pileup"
    out.mkdir(parents=True, exist_ok=True)
    for hit in kwargs["pick_list"]:
        (out / f"well_{hit['source_plate']}_{hit['source_well']}.html").write_text(
            "<html>pileup</html>"
        )
    return {}


class TestWellSelection:

    def test_shallow_wells_are_skipped(self, project):
        with patch("usortm.demux.streakout.generate_pick_pileups",
                   side_effect=_fake_render) as gen:
            result = runner.invoke(app, ["pileups", str(project), "--min-reads", "20"])

        assert result.exit_code == 0, result.stdout
        picked = gen.call_args.kwargs["pick_list"]
        # 3 plates x (25, 150) reads; the 5-read wells are below the cutoff.
        assert len(picked) == 6
        assert all(h["reads"] >= 20 for h in picked)

    def test_min_reads_zero_takes_everything(self, project):
        with patch("usortm.demux.streakout.generate_pick_pileups",
                   side_effect=_fake_render) as gen:
            runner.invoke(app, ["pileups", str(project), "--min-reads", "0"])
        assert len(gen.call_args.kwargs["pick_list"]) == 9

    def test_plate_filter(self, project):
        with patch("usortm.demux.streakout.generate_pick_pileups",
                   side_effect=_fake_render) as gen:
            runner.invoke(app, ["pileups", str(project), "--min-reads", "0",
                                "--plate", "2"])
        picked = gen.call_args.kwargs["pick_list"]
        assert {h["source_plate"] for h in picked} == {"2"}

    def test_plate_filter_accepts_a_list(self, project):
        with patch("usortm.demux.streakout.generate_pick_pileups",
                   side_effect=_fake_render) as gen:
            runner.invoke(app, ["pileups", str(project), "--min-reads", "0",
                                "--plate", "1,3"])
        picked = gen.call_args.kwargs["pick_list"]
        assert {h["source_plate"] for h in picked} == {"1", "3"}

    def test_source_and_target_match_so_one_file_per_well(self, project):
        """generate_pick_pileups names output after the source well; pointing
        target at the same well keeps it one file per well."""
        with patch("usortm.demux.streakout.generate_pick_pileups",
                   side_effect=_fake_render) as gen:
            runner.invoke(app, ["pileups", str(project), "--min-reads", "0"])
        for hit in gen.call_args.kwargs["pick_list"]:
            assert hit["source_plate"] == hit["target_plate"]
            assert hit["source_well"] == hit["target_well"]

    def test_nothing_to_render_exits_cleanly(self, project):
        with patch("usortm.demux.streakout.generate_pick_pileups",
                   side_effect=_fake_render) as gen:
            result = runner.invoke(app, ["pileups", str(project),
                                         "--min-reads", "10000"])
        assert result.exit_code == 0
        assert "No wells to render" in result.stdout
        gen.assert_not_called()

    def test_missing_demux_output_is_reported(self, tmp_path):
        empty = tmp_path / "empty"
        (empty / "x").mkdir(parents=True)
        (empty / "usortm_project.json").write_text("{}")
        result = runner.invoke(app, ["pileups", str(empty)])
        assert result.exit_code == 1
        assert "usortm demux" in result.stdout


class TestIndex:

    def _run(self, project, *args):
        with patch("usortm.demux.streakout.generate_pick_pileups",
                   side_effect=_fake_render):
            result = runner.invoke(app, ["pileups", str(project), *args])
        assert result.exit_code == 0, result.stdout
        return project / "demux_output" / "pileups" / "index.html"

    def test_index_links_every_rendered_pileup(self, project):
        import re

        index = self._run(project, "--min-reads", "0")
        html = index.read_text()
        links = re.findall(r"href='(pileup/[^']+)'", html)

        assert len(links) == 9
        for link in links:
            assert (index.parent / link).exists(), f"broken link: {link}"

    def test_index_groups_by_plate(self, project):
        import re

        html = self._run(project, "--min-reads", "0").read_text()
        assert re.findall(r"<h2>Plate (\d+)", html) == ["1", "2", "3"]

    def test_index_reports_depth_and_variant(self, project):
        html = self._run(project, "--min-reads", "0").read_text()
        assert "var_3" in html
        assert "150" in html
        assert "95%" in html

    def test_index_omits_wells_that_were_not_rendered(self, project):
        """A well filtered out must not appear as a dead link."""
        html = self._run(project, "--min-reads", "100").read_text()
        assert "A3" in html      # the 150-read well
        assert "A1</a>" not in html   # the 5-read well

    def test_index_numerals_are_tabular_and_theme_aware(self, project):
        html = self._run(project, "--min-reads", "0").read_text()
        assert "tabular-nums" in html
        assert "prefers-color-scheme" in html
