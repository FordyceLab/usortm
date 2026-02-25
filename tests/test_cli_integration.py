"""Integration tests for the full CLI workflow.

Tests that require external tools (dorado, minimap2, samtools) are skipped
when those tools are not available. Mock-based tests verify CLI wiring
without external dependencies.
"""

import shutil
from unittest.mock import patch
import csv
import json

import pytest
from typer.testing import CliRunner
from usortm.cli import app
from usortm.demux.deps import DependencyError

runner = CliRunner()


def _tool_available(name: str) -> bool:
    """Check if an external tool is available using the project's finders."""
    from usortm.demux import deps
    try:
        finder = getattr(deps, f"find_{name}", None)
        if finder:
            finder()
            return True
    except DependencyError:
        return False
    return shutil.which(name) is not None


# Skip marker for tests needing all demux tools
requires_demux_tools = pytest.mark.skipif(
    not _tool_available("dorado")
    or not _tool_available("minimap2")
    or not _tool_available("samtools"),
    reason="dorado, minimap2, or samtools not installed",
)


@pytest.fixture
def library_csv(tmp_path):
    """Create a test library CSV file."""
    library_file = tmp_path / "test_library.csv"
    with open(library_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "sequence"])
        for i in range(10):
            writer.writerow([f"variant_{i + 1}", "ATGC" * 75])
    return library_file


@pytest.fixture
def mock_fastq(tmp_path):
    """Create a mock FASTQ file."""
    fastq_file = tmp_path / "reads.fastq"
    with open(fastq_file, "w") as f:
        for i in range(100):
            f.write(f"@read_{i}\n")
            f.write("ATGCATGCATGCATGC\n")
            f.write("+\n")
            f.write("IIIIIIIIIIIIIIII\n")
    return fastq_file


@pytest.fixture
def project_with_demux_results(tmp_path):
    """Create a project with pre-made demux results for pick/report tests.

    This avoids needing external tools for testing downstream commands.
    """
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    state = {
        "library_size": 10,
        "seq_length": 300,
        "fold_sampling": 4,
        "barcode_kit": "levseq",
        "n_plates": 1,
        "workflow_steps": {
            "plan": {"completed": True},
            "demux": {
                "completed": True,
                "total_reads": 100,
                "assigned_reads": 65,
                "wells_with_data": 8,
            },
        },
    }
    with open(project_dir / "usortm_project.json", "w") as f:
        json.dump(state, f)

    # Create mock demux output
    demux_output = project_dir / "demux_output"
    demux_output.mkdir()

    with open(demux_output / "well_assignments.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "plate", "well", "reads", "variant", "consensus_fraction",
        ])
        for i in range(8):
            writer.writerow([
                "1", f"A{i + 1}", 50 + i * 10, f"variant_{i + 1}", 0.95,
            ])

    with open(demux_output / "demux_summary.json", "w") as f:
        json.dump({
            "total_reads": 100,
            "assigned_reads": 65,
            "wells_with_data": 8,
            "wells_passing": 6,
        }, f)

    return project_dir


# ===================================================================
# Mock-based demux CLI test
# ===================================================================

def test_demux_with_mock_pipeline(tmp_path, mock_fastq):
    """Test demux CLI with a mocked pipeline (no external tools needed)."""
    project_dir = tmp_path / "mock_project"
    project_dir.mkdir()

    state = {
        "library_size": 10,
        "barcode_kit": "levseq",
        "n_plates": 1,
        "workflow_steps": {},
    }
    with open(project_dir / "usortm_project.json", "w") as f:
        json.dump(state, f)

    mock_results = {
        "input_reads": 500,
        "aligned_reads": 200,
        "demuxed_reads": 100,
        "total_reads": 500,
        "assigned_reads": 65,
        "wells_with_data": 8,
        "wells_passing": 6,
        "well_assignments": {
            "1_A1": {
                "plate": "1",
                "well": "A1",
                "reads": 80,
                "variant": "GFP_test",
                "consensus_fraction": 0.95,
            },
        },
    }

    with patch(
        "usortm.cli.demux_cmd._run_demux",
        return_value=mock_results,
    ), patch(
        "usortm.cli.demux_cmd.check_all_dependencies",
        return_value={
            "dorado": "/usr/bin/dorado",
            "minimap2": "/usr/bin/minimap2",
            "samtools": "/usr/bin/samtools",
        },
    ):
        result = runner.invoke(app, [
            "demux",
            str(project_dir),
            "--fastq", str(mock_fastq),
        ])

    assert result.exit_code == 0
    assert "Demultiplexing" in result.stdout
    assert (project_dir / "demux_output").exists()
    assert (project_dir / "demux_output" / "well_assignments.csv").exists()


# ===================================================================
# Pick and report tests (using pre-made demux results)
# ===================================================================

def test_pick_command(project_with_demux_results):
    """Test pick command with pre-existing demux results."""
    result = runner.invoke(app, [
        "pick",
        str(project_with_demux_results),
        "--unique-only",
    ])

    assert result.exit_code == 0
    assert "Hit Picking" in result.stdout
    assert (project_with_demux_results / "hitlist.csv").exists()

    # Verify Integra ASSIST format
    with open(project_with_demux_results / "hitlist.csv", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        assert header == [
            "SampleID", "SourcePlateID", "SourceWell",
            "TargetPlateID", "TargetWell", "TransferVolume",
        ]


def test_pick_with_volume_option(project_with_demux_results):
    """Test pick command with custom volume."""
    result = runner.invoke(app, [
        "pick",
        str(project_with_demux_results),
        "--volume", "10.0",
        "--fill-order", "row",
    ])
    assert result.exit_code == 0

    hitlist = project_with_demux_results / "hitlist.csv"
    with open(hitlist, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        first_hit = next(reader)
        assert first_hit[5] == "10.0"


def test_report_formats(project_with_demux_results):
    """Test report command with each output format."""
    for fmt in ["csv", "html", "json"]:
        result = runner.invoke(app, [
            "report",
            str(project_with_demux_results),
            "--format", fmt,
        ])
        assert result.exit_code == 0


# ===================================================================
# Estimate command
# ===================================================================

def test_estimate_command():
    """Test estimate command works."""
    result = runner.invoke(app, [
        "estimate",
        "--library-size", "500",
        "--seq-length", "300",
        "--fold-sampling", "4",
    ])
    assert result.exit_code == 0
    assert "$" in result.stdout or "cost" in result.stdout.lower()


def test_plan_command_with_round_option(tmp_path, library_csv):
    """Plan should accept --round without shadowing Python's round()."""
    project_dir = tmp_path / "round_test"
    result = runner.invoke(app, [
        "plan",
        str(library_csv),
        "--output", str(project_dir),
        "--seq-length", "300",
        "--round", "2",
    ])
    assert result.exit_code == 0

    with open(project_dir / "usortm_project.json") as f:
        state = json.load(f)
    assert state["round"] == 2


# ===================================================================
# Error handling
# ===================================================================

def test_pick_without_demux(tmp_path):
    """Pick should fail when demux hasn't been run."""
    project_dir = tmp_path / "error_test"
    project_dir.mkdir()

    state = {"library_size": 10, "workflow_steps": {}}
    with open(project_dir / "usortm_project.json", "w") as f:
        json.dump(state, f)

    result = runner.invoke(app, ["pick", str(project_dir)])
    assert result.exit_code == 1
    assert "No demultiplexing results" in result.stdout


def test_report_without_demux(tmp_path):
    """Report should fail when demux hasn't been run."""
    project_dir = tmp_path / "error_test"
    project_dir.mkdir()

    state = {"library_size": 10, "workflow_steps": {}}
    with open(project_dir / "usortm_project.json", "w") as f:
        json.dump(state, f)

    result = runner.invoke(app, ["report", str(project_dir)])
    assert result.exit_code == 1
    assert "No demultiplexing results" in result.stdout


# ===================================================================
# Help and registration
# ===================================================================

def test_help_commands():
    """Test that all commands have help text."""
    commands = ["estimate", "plan", "demux", "pick", "report"]
    for cmd in commands:
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0
        assert len(result.stdout) > 0


def test_command_registration():
    """Test that all commands are registered."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "estimate" in result.stdout
    assert "plan" in result.stdout
    assert "demux" in result.stdout
    assert "pick" in result.stdout
    assert "report" in result.stdout


def test_cli_output_formatting():
    """Test that CLI output is properly formatted with Rich."""
    result = runner.invoke(app, [
        "estimate",
        "--library-size", "100",
        "--seq-length", "300",
    ])
    assert result.exit_code == 0
    lines = result.stdout.split('\n')
    assert len(lines) > 5


# ===================================================================
# Full workflow (requires external tools)
# ===================================================================

@requires_demux_tools
def test_full_workflow_with_tools(tmp_path, library_csv, mock_fastq):
    """Test complete workflow with real demux tools installed."""
    project_dir = tmp_path / "full_test"

    # Plan
    result = runner.invoke(app, [
        "plan",
        str(library_csv),
        "--output", str(project_dir),
        "--seq-length", "300",
        "--fold-sampling", "4",
    ])
    assert result.exit_code == 0

    # Demux
    result = runner.invoke(app, [
        "demux",
        str(project_dir),
        "--fastq", str(mock_fastq),
    ])
    assert result.exit_code == 0
    assert (project_dir / "demux_output" / "well_assignments.csv").exists()

    # Pick
    result = runner.invoke(app, ["pick", str(project_dir)])
    assert result.exit_code == 0

    # Report
    result = runner.invoke(app, [
        "report", str(project_dir), "--format", "all",
    ])
    assert result.exit_code == 0
