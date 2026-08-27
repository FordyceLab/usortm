"""Tests for the report CLI command."""

import pytest
from pathlib import Path
import csv
import json
from typer.testing import CliRunner
from usortm.cli import app

runner = CliRunner()


@pytest.fixture
def mock_project_with_library(tmp_path):
    """Create a mock project directory with demux results and library file."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # Create library file
    library_file = project_dir / "library.csv"
    with open(library_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "sequence"])
        writer.writerow(["var1", "ATGC" * 75])
        writer.writerow(["var2", "GCTA" * 75])
        writer.writerow(["var3", "TACG" * 75])
        writer.writerow(["var4", "CGAT" * 75])
        writer.writerow(["var5", "ATAT" * 75])  # Missing variant

    # Create project state file
    state = {
        "library_size": 5,
        "seq_length": 300,
        "fold_sampling": 4,
        "library_file": str(library_file),
        "workflow_steps": {
            "demux": {
                "completed": True,
                "timestamp": "2024-01-01T00:00:00",
                "total_reads": 1000,
                "assigned_reads": 950,
                "wells_with_data": 4,
            }
        }
    }
    with open(project_dir / "usortm_project.json", "w") as f:
        json.dump(state, f)

    # Create demux output directory
    demux_dir = project_dir / "demux_output"
    demux_dir.mkdir()

    # Create well assignments file (missing var5)
    well_data = [
        {"plate": "1", "well": "A1", "variant": "var1", "reads": 100, "consensus_fraction": 0.95},
        {"plate": "1", "well": "B1", "variant": "var2", "reads": 200, "consensus_fraction": 0.98},
        {"plate": "1", "well": "C1", "variant": "var1", "reads": 150, "consensus_fraction": 0.93},
        {"plate": "1", "well": "D1", "variant": "var3", "reads": 80, "consensus_fraction": 0.91},
        {"plate": "2", "well": "A1", "variant": "var4", "reads": 120, "consensus_fraction": 0.96},
    ]

    with open(demux_dir / "well_assignments.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["plate", "well", "variant", "reads", "consensus_fraction"])
        writer.writeheader()
        writer.writerows(well_data)

    # Create demux summary
    summary = {
        "total_reads": 1000,
        "assigned_reads": 950,
        "wells_with_data": 5,
        "wells_passing": 4,
    }
    with open(demux_dir / "demux_summary.json", "w") as f:
        json.dump(summary, f)

    return project_dir


def test_report_all_formats(mock_project_with_library):
    """Test generating all report formats."""
    result = runner.invoke(app, ["report", str(mock_project_with_library), "--format", "all"])

    assert result.exit_code == 0
    assert "Reporting" in result.stdout

    report_dir = mock_project_with_library / "report"
    assert report_dir.exists()

    # Check all expected files exist
    assert (report_dir.parent / "summary.html").exists()
    assert (report_dir / "plate_maps.csv").exists()
    assert (report_dir / "final_mapping.csv").exists()
    assert (report_dir / "missing_variants.csv").exists()
    assert (report_dir / "report.json").exists()


def test_report_csv_only(mock_project_with_library):
    """Test generating only CSV reports."""
    result = runner.invoke(app, ["report", str(mock_project_with_library), "--format", "csv"])

    assert result.exit_code == 0

    report_dir = mock_project_with_library / "report"

    # CSV files should exist
    assert (report_dir / "plate_maps.csv").exists()
    assert (report_dir / "final_mapping.csv").exists()
    assert (report_dir / "missing_variants.csv").exists()

    # HTML and JSON should not exist
    assert not (report_dir.parent / "summary.html").exists()
    assert not (report_dir / "report.json").exists()


def test_report_html_only(mock_project_with_library):
    """Test generating only HTML report."""
    result = runner.invoke(app, ["report", str(mock_project_with_library), "--format", "html"])

    assert result.exit_code == 0

    report_dir = mock_project_with_library / "report"

    # HTML should exist
    assert (report_dir.parent / "summary.html").exists()

    # CSV and JSON should not exist
    assert not (report_dir / "plate_maps.csv").exists()
    assert not (report_dir / "report.json").exists()


def test_report_json_only(mock_project_with_library):
    """Test generating only JSON report."""
    result = runner.invoke(app, ["report", str(mock_project_with_library), "--format", "json"])

    assert result.exit_code == 0

    report_dir = mock_project_with_library / "report"

    # JSON should exist
    assert (report_dir / "report.json").exists()

    # CSV and HTML should not exist
    assert not (report_dir / "plate_maps.csv").exists()
    assert not (report_dir.parent / "summary.html").exists()


def test_report_plate_maps_content(mock_project_with_library):
    """Test plate maps CSV content."""
    runner.invoke(app, ["report", str(mock_project_with_library), "--format", "csv"])

    plate_maps = mock_project_with_library / "report" / "plate_maps.csv"
    with open(plate_maps, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        assert len(rows) == 5  # 5 wells with data
        assert rows[0]["plate"] == "1"
        assert rows[0]["well"] == "A1"
        assert rows[0]["variant"] == "var1"
        assert rows[0]["reads"] == "100"


def test_report_final_mapping_content(mock_project_with_library):
    """Test final mapping CSV content."""
    runner.invoke(app, ["report", str(mock_project_with_library), "--format", "csv"])

    final_mapping = mock_project_with_library / "report" / "final_mapping.csv"
    with open(final_mapping, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        # Should have 4 unique variants (var1-4, var5 is missing)
        assert len(rows) == 4

        # Find var1 row - should have 2 wells, best is C1 with 150 reads
        var1_row = next(row for row in rows if row["variant"] == "var1")
        assert var1_row["num_wells"] == "2"
        assert var1_row["best_well"] == "C1"
        assert var1_row["best_reads"] == "150"


def test_report_missing_variants_content(mock_project_with_library):
    """Test missing variants CSV content."""
    runner.invoke(app, ["report", str(mock_project_with_library), "--format", "csv"])

    missing_variants = mock_project_with_library / "report" / "missing_variants.csv"
    with open(missing_variants, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        # Should have 1 missing variant (var5)
        assert len(rows) == 1
        assert rows[0]["variant"] == "var5"
        assert rows[0]["status"] == "missing"


def test_report_json_content(mock_project_with_library):
    """Test JSON report content."""
    runner.invoke(app, ["report", str(mock_project_with_library), "--format", "json"])

    report_json = mock_project_with_library / "report" / "report.json"
    with open(report_json) as f:
        report = json.load(f)

        assert "generated" in report
        assert report["project"]["library_size"] == 5
        assert report["project"]["seq_length"] == 300
        assert report["demux"]["total_reads"] == 1000
        assert report["variants"]["unique"] == 4
        assert report["coverage"]["recovered"] == 4
        assert report["coverage"]["percent"] == 80.0  # 4/5 * 100


def test_report_html_content(mock_project_with_library):
    """Test HTML report contains expected content."""
    runner.invoke(app, ["report", str(mock_project_with_library), "--format", "html"])

    html_file = mock_project_with_library / "summary.html"
    with open(html_file) as f:
        html_content = f.read()

        # Check for key elements
        assert "uSort-M Summary" in html_content
        assert "Library size" in html_content
        assert "Input reads" in html_content
        # The well count is stated by fold sampling, as the figure it
        # divides, rather than as a metric of its own.
        assert "Fold sampling" in html_content
        assert "wells &ge;20 reads" in html_content
        assert "Library recovery" in html_content
        assert "Demux plate maps" in html_content


def test_report_invalid_format(mock_project_with_library):
    """Test error with invalid format."""
    result = runner.invoke(app, ["report", str(mock_project_with_library), "--format", "invalid"])

    assert result.exit_code == 1
    assert "Invalid format" in result.stdout


def test_report_no_demux_results(tmp_path):
    """Test error when no demux results exist."""
    project_dir = tmp_path / "no_demux_project"
    project_dir.mkdir()

    # Create project state without demux
    state = {
        "library_size": 100,
        "workflow_steps": {}
    }
    with open(project_dir / "usortm_project.json", "w") as f:
        json.dump(state, f)

    result = runner.invoke(app, ["report", str(project_dir)])

    assert result.exit_code == 1
    assert "No demultiplexing results" in result.stdout


def test_report_invalid_project(tmp_path):
    """Test error with invalid project directory."""
    result = runner.invoke(app, ["report", str(tmp_path)])

    assert result.exit_code == 1
    assert "Not a valid uSort-M project" in result.stdout


def test_report_no_library_file(tmp_path):
    """Test report works without library file (no missing variants)."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # Create project state without library_file
    state = {
        "library_size": 100,
        "seq_length": 300,
        "fold_sampling": 4,
        "workflow_steps": {
            "demux": {
                "completed": True,
                "timestamp": "2024-01-01T00:00:00",
            }
        }
    }
    with open(project_dir / "usortm_project.json", "w") as f:
        json.dump(state, f)

    # Create minimal demux output
    demux_dir = project_dir / "demux_output"
    demux_dir.mkdir()

    with open(demux_dir / "well_assignments.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["plate", "well", "variant", "reads", "consensus_fraction"])
        writer.writerow(["1", "A1", "var1", 100, 0.95])

    with open(demux_dir / "demux_summary.json", "w") as f:
        json.dump({"total_reads": 100, "assigned_reads": 95, "wells_with_data": 1, "wells_passing": 1}, f)

    result = runner.invoke(app, ["report", str(project_dir), "--format", "csv"])

    assert result.exit_code == 0

    # Missing variants file should exist but indicate no library file
    missing_file = project_dir / "report" / "missing_variants.csv"
    with open(missing_file, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert "No library file" in rows[0]["status"]


def test_report_coverage_strips_cons_check(tmp_path):
    """Coverage should not exceed 100% due to |cons_check suffixes."""
    project_dir = tmp_path / "cons_check_project"
    project_dir.mkdir()

    state = {
        "library_size": 2,
        "seq_length": 300,
        "fold_sampling": 4,
        "workflow_steps": {"demux": {"completed": True}},
    }
    with open(project_dir / "usortm_project.json", "w") as f:
        json.dump(state, f)

    demux_dir = project_dir / "demux_output"
    demux_dir.mkdir()

    # var1 appears with two different cons_check values — same underlying variant
    with open(demux_dir / "well_assignments.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["plate", "well", "variant", "reads", "consensus_fraction"])
        writer.writerow(["1", "A1", "var1|match", 200, 0.95])
        writer.writerow(["1", "B1", "var1|mismatch", 150, 0.88])
        writer.writerow(["1", "C1", "var2|match", 180, 0.97])

    with open(demux_dir / "demux_summary.json", "w") as f:
        json.dump({"input_reads": 600, "assigned_reads": 530, "wells_with_data": 3, "wells_passing": 3}, f)

    result = runner.invoke(app, ["report", str(project_dir), "--format", "json"])
    assert result.exit_code == 0

    with open(project_dir / "report" / "report.json") as f:
        report = json.load(f)

    # 2 unique variants after stripping |cons_check, library_size = 2 → 100%
    assert report["variants"]["unique"] == 2
    assert report["coverage"]["percent"] == 100.0


def test_report_html_has_bar_chart(mock_project_with_library):
    """HTML report should contain per-plate bar chart SVG."""
    result = runner.invoke(app, ["report", str(mock_project_with_library), "--format", "html"])
    assert result.exit_code == 0

    html_file = mock_project_with_library / "summary.html"
    with open(html_file) as f:
        html_content = f.read()

    # Figures are inline SVG, and every plate gets a map of its own.
    assert "<svg" in html_content
    assert 'class="cols24"' in html_content
    assert 'data-p="1"' in html_content


def test_report_html_no_minimum_reads_row(mock_project_with_library):
    """HTML report should not contain 'Minimum reads' row."""
    result = runner.invoke(app, ["report", str(mock_project_with_library), "--format", "html"])
    assert result.exit_code == 0

    html_file = mock_project_with_library / "summary.html"
    with open(html_file) as f:
        html_content = f.read()

    assert "Minimum reads" not in html_content


def test_report_statistics_accuracy(mock_project_with_library):
    """Test that statistics are calculated correctly."""
    result = runner.invoke(app, ["report", str(mock_project_with_library), "--format", "all"])

    assert result.exit_code == 0

    # Check console output for statistics
    assert "Library Size" in result.stdout or "5" in result.stdout
    assert "Wells with data" in result.stdout or "5" in result.stdout

    # Verify JSON has correct calculations
    report_json = mock_project_with_library / "report" / "report.json"
    with open(report_json) as f:
        report = json.load(f)

        # Average reads per well = (100 + 200 + 150 + 80 + 120) / 5 = 130
        assert report["variants"]["avg_reads_per_well"] == 130.0

        # 2 variants with multiple wells (var1 appears twice)
        assert report["variants"]["variants_with_multiple_wells"] == 1


def test_report_quality_bins(mock_project_with_library):
    """Verify bin classification logic with known well data."""
    from usortm.cli.report import _compute_quality_bins

    well_data = [
        # Bin 1: >90% consensus, >=100 reads
        {"variant": "v1", "reads": 200, "consensus_fraction": 0.95},
        {"variant": "v2", "reads": 100, "consensus_fraction": 0.91},
        # Bin 2: >90% consensus, 50-99 reads
        {"variant": "v3", "reads": 75, "consensus_fraction": 0.92},
        # Bin 3: >90% consensus, 20-49 reads
        {"variant": "v4", "reads": 30, "consensus_fraction": 0.95},
        # Unbinned: <=90% consensus
        {"variant": "v5", "reads": 500, "consensus_fraction": 0.90},
        # Unbinned: <20 reads
        {"variant": "v6", "reads": 10, "consensus_fraction": 0.99},
    ]

    result = _compute_quality_bins(well_data, library_size=10)
    qb = result["quality_bins"]

    assert qb["bin1"] == 2   # v1, v2
    assert qb["bin2"] == 1   # v3
    assert qb["bin3"] == 1   # v4
    assert qb["unbinned"] == 2  # v5 (<=90%), v6 (<20 reads)


def test_report_quality_bins_picks_best_well():
    """When multiple wells have the same base variant, pick the best."""
    from usortm.cli.report import _compute_quality_bins

    well_data = [
        {"variant": "v1|match", "reads": 50, "consensus_fraction": 0.95},
        {"variant": "v1|mismatch", "reads": 200, "consensus_fraction": 0.98},
        {"variant": "v2", "reads": 30, "consensus_fraction": 0.92},
    ]

    result = _compute_quality_bins(well_data, library_size=5)
    qb = result["quality_bins"]

    # v1 best well is 200 reads -> bin1; v2 is 30 reads -> bin3
    assert qb["bin1"] == 1
    assert qb["bin3"] == 1


def test_report_recovery_tiers(mock_project_with_library):
    """Verify tier accumulation: B includes A, C includes B."""
    from usortm.cli.report import _compute_quality_bins

    well_data = [
        {"variant": "v1", "reads": 200, "consensus_fraction": 0.95},  # bin1
        {"variant": "v2", "reads": 60, "consensus_fraction": 0.92},   # bin2
        {"variant": "v3", "reads": 25, "consensus_fraction": 0.91},   # bin3
    ]

    result = _compute_quality_bins(well_data, library_size=10)
    tiers = result["recovery_tiers"]

    assert tiers["A"]["count"] == 1       # bin1 only
    assert tiers["B"]["count"] == 2       # bin1 + bin2
    assert tiers["C"]["count"] == 3       # bin1 + bin2 + bin3
    assert tiers["A"]["pct"] == 10.0      # 1/10 * 100
    assert tiers["B"]["pct"] == 20.0
    assert tiers["C"]["pct"] == 30.0


def test_report_json_has_tiers(mock_project_with_library):
    """JSON report should include quality_bins and recovery_tiers."""
    runner.invoke(app, ["report", str(mock_project_with_library), "--format", "json"])

    with open(mock_project_with_library / "report" / "report.json") as f:
        report = json.load(f)

    assert "quality_bins" in report
    assert "recovery_tiers" in report
    assert "A" in report["recovery_tiers"]
    assert "count" in report["recovery_tiers"]["A"]
    assert "pct" in report["recovery_tiers"]["A"]


def test_report_dark_mode_toggle(mock_project_with_library):
    """HTML report should contain dark mode CSS and JS toggle."""
    runner.invoke(app, ["report", str(mock_project_with_library), "--format", "html"])

    html_file = mock_project_with_library / "summary.html"
    with open(html_file) as f:
        html_content = f.read()

    # CSS custom properties
    assert "--surface-1:" in html_content
    assert "--text-primary:" in html_content
    assert '[data-theme="dark"]' in html_content
    assert "prefers-color-scheme" in html_content

    # JS toggle
    assert "themeToggle" in html_content
    assert "usortm-theme" in html_content
    assert "localStorage" in html_content


def test_report_html_library_recovery(mock_project_with_library):
    """HTML report should contain Library Recovery section with tiers."""
    runner.invoke(app, ["report", str(mock_project_with_library), "--format", "html"])

    html_file = mock_project_with_library / "summary.html"
    with open(html_file) as f:
        html_content = f.read()

    assert "Library recovery" in html_content
    assert "Tier A" in html_content
    assert "Tier B" in html_content
    assert "Tier C" in html_content


def test_report_html_selected_tier_indicator(mock_project_with_library):
    """HTML report should highlight selected recovery tier when pick tier is present."""
    state_file = mock_project_with_library / "usortm_project.json"
    with open(state_file) as f:
        state = json.load(f)

    state["workflow_steps"]["pick"] = {
        "completed": True,
        "tier": "B",
        "total_hits": 10,
        "unique_variants": 4,
        "target_format": 384,
    }
    with open(state_file, "w") as f:
        json.dump(state, f)

    runner.invoke(app, ["report", str(mock_project_with_library), "--format", "html"])

    html_file = mock_project_with_library / "summary.html"
    with open(html_file) as f:
        html_content = f.read()

    assert "Selected tier" in html_content
    # Exactly one tier is marked, and it is the one the pick used.
    assert html_content.count("Selected tier") == 1
    assert html_content.count('<tr class="sel">') == 1
    row = html_content.split('<tr class="sel">')[1]
    assert "Tier B" in row.split("</tr>")[0]


def test_final_mapping_groups_by_base_variant(tmp_path):
    """_save_final_mapping should group by base name, stripping |cons_check suffixes."""
    from usortm.cli.report import _save_final_mapping

    well_data = [
        {"plate": "1", "well": "A1", "variant": "AIRE;254;400", "reads": 50, "consensus_fraction": 0.80},
        {"plate": "1", "well": "B1", "variant": "AIRE;254;400|Perfect Match", "reads": 200, "consensus_fraction": 0.98},
        {"plate": "1", "well": "C1", "variant": "BRCA1;10;20", "reads": 120, "consensus_fraction": 0.95},
    ]

    output_file = tmp_path / "final_mapping.csv"
    _save_final_mapping(well_data, output_file)

    with open(output_file, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Should have 2 unique base variants, not 3
    assert len(rows) == 2
    variants = {row["variant"] for row in rows}
    assert variants == {"AIRE;254;400", "BRCA1;10;20"}

    # AIRE row should show 2 wells and best reads = 200
    aire_row = next(r for r in rows if r["variant"] == "AIRE;254;400")
    assert aire_row["num_wells"] == "2"
    assert aire_row["best_reads"] == "200"


def test_report_seq_len_range_from_demux_summary(tmp_path):
    """HTML report should show measured seq length range when available in demux_summary."""
    project_dir = tmp_path / "seq_len_project"
    project_dir.mkdir()

    state = {
        "library_size": 2,
        "seq_length": 300,
        "fold_sampling": 4,
        "workflow_steps": {"demux": {"completed": True}},
    }
    with open(project_dir / "usortm_project.json", "w") as f:
        json.dump(state, f)

    demux_dir = project_dir / "demux_output"
    demux_dir.mkdir()

    with open(demux_dir / "well_assignments.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["plate", "well", "variant", "reads", "consensus_fraction"])
        writer.writerow(["1", "A1", "var1", 200, 0.95])
        writer.writerow(["1", "B1", "var2", 150, 0.92])

    # Include measured seq_len range in summary
    with open(demux_dir / "demux_summary.json", "w") as f:
        json.dump({
            "input_reads": 400,
            "assigned_reads": 350,
            "wells_with_data": 2,
            "wells_passing": 2,
            "seq_len_min": 285,
            "seq_len_max": 510,
            "seq_len_median": 400,
        }, f)

    result = runner.invoke(app, ["report", str(project_dir), "--format", "html"])
    assert result.exit_code == 0

    html_file = project_dir / "summary.html"
    with open(html_file) as f:
        html_content = f.read()

    # Should show range "285–510 bp", NOT the plan value "300 bp"
    assert "285" in html_content
    assert "510" in html_content
    # The plan-step-only value should not appear as the seq length
    assert "300 bp" not in html_content


def test_report_seq_len_single_value_from_demux_summary(tmp_path):
    """HTML report shows single value when min == max in demux_summary."""
    project_dir = tmp_path / "seq_len_single_project"
    project_dir.mkdir()

    state = {
        "library_size": 1,
        "seq_length": 300,
        "fold_sampling": 4,
        "workflow_steps": {"demux": {"completed": True}},
    }
    with open(project_dir / "usortm_project.json", "w") as f:
        json.dump(state, f)

    demux_dir = project_dir / "demux_output"
    demux_dir.mkdir()

    with open(demux_dir / "well_assignments.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["plate", "well", "variant", "reads", "consensus_fraction"])
        writer.writerow(["1", "A1", "var1", 200, 0.95])

    with open(demux_dir / "demux_summary.json", "w") as f:
        json.dump({
            "input_reads": 200,
            "assigned_reads": 200,
            "wells_with_data": 1,
            "wells_passing": 1,
            "seq_len_min": 450,
            "seq_len_max": 450,
        }, f)

    result = runner.invoke(app, ["report", str(project_dir), "--format", "html"])
    assert result.exit_code == 0

    html_file = project_dir / "summary.html"
    with open(html_file) as f:
        html_content = f.read()

    assert "450 bp" in html_content


def test_summary_sits_above_the_directories_it_links_to(mock_project_with_library):
    """The page reaches the pileups under pick/ and demux_output/.

    Safari grants a file:// page read access to its own directory and below,
    so a page written into report/ cannot open a sibling directory and every
    well link fails.
    """
    runner.invoke(app, ["report", str(mock_project_with_library),
                        "--format", "html"])

    assert (mock_project_with_library / "summary.html").exists()
    assert not (mock_project_with_library / "report" / "summary.html").exists()


def test_a_summary_left_in_the_report_directory_is_dropped(mock_project_with_library):
    """The old copy is the one reached for out of habit, and it is broken."""
    report_dir = mock_project_with_library / "report"
    report_dir.mkdir(exist_ok=True)
    stale = report_dir / "summary.html"
    stale.write_text("<html>an earlier run</html>")

    runner.invoke(app, ["report", str(mock_project_with_library),
                        "--format", "html"])

    assert not stale.exists()
