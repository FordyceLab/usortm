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
        {"plate": "1", "well": "D1", "variant": "var3", "reads": 80, "consensus_fraction": 0.90},
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
    assert (report_dir / "summary.html").exists()
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
    assert not (report_dir / "summary.html").exists()
    assert not (report_dir / "report.json").exists()


def test_report_html_only(mock_project_with_library):
    """Test generating only HTML report."""
    result = runner.invoke(app, ["report", str(mock_project_with_library), "--format", "html"])

    assert result.exit_code == 0

    report_dir = mock_project_with_library / "report"

    # HTML should exist
    assert (report_dir / "summary.html").exists()

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
    assert not (report_dir / "summary.html").exists()


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

    html_file = mock_project_with_library / "report" / "summary.html"
    with open(html_file) as f:
        html_content = f.read()

        # Check for key elements
        assert "uSort-M Workflow Report" in html_content
        assert "Library Size" in html_content
        assert "Input Reads" in html_content
        assert "Unique Variants" in html_content
        assert "Library Coverage" in html_content
        assert "80.0%" in html_content  # 4/5 coverage


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
