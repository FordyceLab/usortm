"""Tests for the pick CLI command."""

import pytest
from pathlib import Path
import csv
import json
from typer.testing import CliRunner
from usortm.cli import app

runner = CliRunner()


def _hitlist_path(project_dir):
    """Return the first per-plate hitlist file."""
    d = project_dir / "pick" / "integra_assist_input"
    files = sorted(d.glob("hitlist_plate_*.csv"))
    return files[0] if files else d / "hitlist_plate_0.csv"


@pytest.fixture
def mock_project_dir(tmp_path):
    """Create a mock project directory with demux results."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # Create project state file
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

    # Create demux output directory
    demux_dir = project_dir / "demux_output"
    demux_dir.mkdir()

    # Create well assignments file
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
        "wells_passing": 5,
    }
    with open(demux_dir / "demux_summary.json", "w") as f:
        json.dump(summary, f)

    return project_dir


def test_pick_basic(mock_project_dir):
    """Test basic pick list generation (defaults to Tier A, row-wise fill)."""
    result = runner.invoke(app, ["pick", str(mock_project_dir)])

    assert result.exit_code == 0
    assert "Hit Picking" in result.stdout

    # Check output file was created
    hitlist = _hitlist_path(mock_project_dir)
    assert hitlist.exists()

    # Read and validate output
    with open(hitlist, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        assert header == ["SampleID", "SourcePlateID", "SourceWell", "TargetPlateID", "TargetWell", "TransferVolume"]

        rows = list(reader)
        # Default Tier A: >=100 reads and >90% consensus
        # var2 (200, 0.98), var1/C1 (150, 0.93), var4 (120, 0.96) pass
        # var3 (80, 0.90) fails reads + consensus
        assert len(rows) == 3
        variants = {row[0] for row in rows}
        assert variants == {"var1", "var2", "var4"}


def test_pick_unique_only(mock_project_dir):
    """Test unique-only flag picks one well per variant."""
    result = runner.invoke(app, ["pick", str(mock_project_dir), "--unique-only", "--tier", ""])

    assert result.exit_code == 0

    hitlist = _hitlist_path(mock_project_dir)
    with open(hitlist, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # Skip header
        rows = list(reader)

        # Should have 4 unique variants (var1, var2, var3, var4)
        variants = [row[0] for row in rows]
        assert len(variants) == len(set(variants))  # All unique

        # var1 appears twice in input - should pick the one with higher reads
        var1_rows = [row for row in rows if row[0] == "var1"]
        assert len(var1_rows) == 1
        # Should pick C1 (150 reads) over A1 (100 reads)
        assert var1_rows[0][2] == "C1"


def test_pick_all_hits(mock_project_dir):
    """Test all-hits flag picks all wells."""
    result = runner.invoke(app, ["pick", str(mock_project_dir), "--all-hits", "--tier", ""])

    assert result.exit_code == 0

    hitlist = _hitlist_path(mock_project_dir)
    with open(hitlist, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # Skip header
        rows = list(reader)

        # Should have all 5 wells
        assert len(rows) == 5


def test_pick_custom_output(mock_project_dir):
    """Test custom output path."""
    output_path = mock_project_dir / "custom_output" / "custom_hits.csv"
    result = runner.invoke(app, [
        "pick",
        str(mock_project_dir),
        "--output", str(output_path),
        "--tier", "",
    ])

    assert result.exit_code == 0
    # Per-plate files are written to the parent directory of --output
    plate_files = sorted(output_path.parent.glob("hitlist_plate_*.csv"))
    assert len(plate_files) >= 1


def test_pick_custom_volume(mock_project_dir):
    """Test custom transfer volume."""
    result = runner.invoke(app, [
        "pick",
        str(mock_project_dir),
        "--volume", "10.0",
        "--tier", "",
    ])

    assert result.exit_code == 0

    hitlist = _hitlist_path(mock_project_dir)
    with open(hitlist, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # Skip header
        row = next(reader)
        # Volume should be 10.0
        assert row[5] == "10.0"


def test_pick_target_filter(mock_project_dir, tmp_path):
    """Test filtering by target variants."""
    # Create targets file
    targets_file = tmp_path / "targets.csv"
    with open(targets_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant"])
        writer.writerow(["var2"])
        writer.writerow(["var4"])

    result = runner.invoke(app, [
        "pick",
        str(mock_project_dir),
        "--targets", str(targets_file),
        "--tier", "",
    ])

    assert result.exit_code == 0

    hitlist = _hitlist_path(mock_project_dir)
    with open(hitlist, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # Skip header
        rows = list(reader)

        # Should only have var2 and var4
        variants = [row[0] for row in rows]
        assert set(variants) == {"var2", "var4"}


def test_pick_fill_order_column(mock_project_dir):
    """Test column-wise fill order."""
    result = runner.invoke(app, [
        "pick",
        str(mock_project_dir),
        "--fill-order", "column",
        "--target-format", "96",
        "--tier", "",
    ])

    assert result.exit_code == 0

    hitlist = _hitlist_path(mock_project_dir)
    with open(hitlist, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # Skip header
        rows = list(reader)

        # First well should be A1, second should be B1 (column-wise)
        assert rows[0][4] == "A1"
        assert rows[1][4] == "B1"


def test_pick_fill_order_row(mock_project_dir):
    """Test row-wise fill order."""
    result = runner.invoke(app, [
        "pick",
        str(mock_project_dir),
        "--fill-order", "row",
        "--target-format", "96",
        "--tier", "",
    ])

    assert result.exit_code == 0

    hitlist = _hitlist_path(mock_project_dir)
    with open(hitlist, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # Skip header
        rows = list(reader)

        # First well should be A1, second should be A2 (row-wise)
        assert rows[0][4] == "A1"
        assert rows[1][4] == "A2"


def test_pick_no_demux_results(tmp_path):
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

    result = runner.invoke(app, ["pick", str(project_dir)])

    assert result.exit_code == 1
    assert "No demultiplexing results" in result.stdout


def test_pick_invalid_project(tmp_path):
    """Test error with invalid project directory."""
    result = runner.invoke(app, ["pick", str(tmp_path)])

    assert result.exit_code == 1
    assert "Not a valid uSort-M project" in result.stdout


@pytest.fixture
def tier_project(tmp_path):
    """Project with wells spanning different quality tiers."""
    project_dir = tmp_path / "tier_project"
    project_dir.mkdir()

    state = {
        "library_size": 100,
        "seq_length": 300,
        "fold_sampling": 4,
        "workflow_steps": {"demux": {"completed": True}},
    }
    with open(project_dir / "usortm_project.json", "w") as f:
        json.dump(state, f)

    demux_dir = project_dir / "demux_output"
    demux_dir.mkdir()

    # Wells: var1 (Tier A), var2 (Tier B), var3 (Tier C), var4 (below C)
    well_data = [
        {"plate": "1", "well": "A1", "variant": "var1", "reads": 200, "consensus_fraction": 0.95},
        {"plate": "1", "well": "B1", "variant": "var2", "reads": 75, "consensus_fraction": 0.92},
        {"plate": "1", "well": "C1", "variant": "var3", "reads": 30, "consensus_fraction": 0.91},
        {"plate": "1", "well": "D1", "variant": "var4", "reads": 10, "consensus_fraction": 0.99},
        {"plate": "1", "well": "E1", "variant": "var5", "reads": 500, "consensus_fraction": 0.85},
    ]

    with open(demux_dir / "well_assignments.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["plate", "well", "variant", "reads", "consensus_fraction"])
        writer.writeheader()
        writer.writerows(well_data)

    with open(demux_dir / "demux_summary.json", "w") as f:
        json.dump({"total_reads": 1000, "assigned_reads": 815, "wells_with_data": 5, "wells_passing": 4}, f)

    return project_dir


def test_pick_with_tier_A(tier_project):
    """Tier A: only wells with >=100 reads and >90% consensus."""
    result = runner.invoke(app, ["pick", str(tier_project), "--tier", "A"])
    assert result.exit_code == 0

    hitlist = _hitlist_path(tier_project)
    with open(hitlist, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        rows = list(reader)

    variants = {row[0] for row in rows}
    assert variants == {"var1"}


def test_pick_with_tier_B(tier_project):
    """Tier B: >=50 reads and >90% consensus."""
    result = runner.invoke(app, ["pick", str(tier_project), "--tier", "B"])
    assert result.exit_code == 0

    hitlist = _hitlist_path(tier_project)
    with open(hitlist, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        rows = list(reader)

    variants = {row[0] for row in rows}
    assert variants == {"var1", "var2"}


def test_pick_with_tier_C(tier_project):
    """Tier C: >=20 reads and >90% consensus."""
    result = runner.invoke(app, ["pick", str(tier_project), "--tier", "C"])
    assert result.exit_code == 0

    hitlist = _hitlist_path(tier_project)
    with open(hitlist, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        rows = list(reader)

    variants = {row[0] for row in rows}
    # var4 has 10 reads (<20) so excluded; var5 has 0.85 consensus (<=0.9) so excluded
    assert variants == {"var1", "var2", "var3"}


def test_pick_empty_wells(tmp_path):
    """Empty placeholder rows appear for unrecovered library variants."""
    project_dir = tmp_path / "empty_wells_project"
    project_dir.mkdir()

    # Library has 5 variants but only 2 are recovered
    library_file = tmp_path / "library.csv"
    with open(library_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Sequence"])
        writer.writerow(["libA", "ATCG"])
        writer.writerow(["libB", "GCTA"])
        writer.writerow(["libC", "TTTT"])
        writer.writerow(["libD", "AAAA"])
        writer.writerow(["libE", "CCCC"])

    state = {
        "library_size": 5,
        "library_file": str(library_file),
        "workflow_steps": {"demux": {"completed": True}},
    }
    with open(project_dir / "usortm_project.json", "w") as f:
        json.dump(state, f)

    demux_dir = project_dir / "demux_output"
    demux_dir.mkdir()

    # Only libA and libC are recovered
    well_data = [
        {"plate": "1", "well": "A1", "variant": "libA", "reads": 200, "consensus_fraction": 0.95},
        {"plate": "1", "well": "B1", "variant": "libC", "reads": 150, "consensus_fraction": 0.92},
    ]
    with open(demux_dir / "well_assignments.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["plate", "well", "variant", "reads", "consensus_fraction"])
        writer.writeheader()
        writer.writerows(well_data)
    with open(demux_dir / "demux_summary.json", "w") as f:
        json.dump({"total_reads": 500, "assigned_reads": 350, "wells_with_data": 2, "wells_passing": 2}, f)

    result = runner.invoke(app, ["pick", str(project_dir), "--tier", ""])
    assert result.exit_code == 0

    hitlist = _hitlist_path(project_dir)
    with open(hitlist, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # skip header
        rows = list(reader)

    # Only recovered variants are written (empty wells are excluded)
    assert len(rows) == 2
    assert [row[0] for row in rows] == ["libA", "libC"]

    # Recovered wells have real data
    assert rows[0][1] == "1"
    assert rows[0][2] == "A1"
    assert rows[0][5] == "5.0"


def test_pick_empty_wells_with_legacy_suffix(tmp_path):
    """Variants with |cons_check suffix in well_assignments still match library names."""
    project_dir = tmp_path / "legacy_suffix_project"
    project_dir.mkdir()

    library_file = tmp_path / "library.csv"
    with open(library_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Sequence"])
        writer.writerow(["libA", "ATCG"])
        writer.writerow(["libB", "GCTA"])
        writer.writerow(["libC", "TTTT"])

    state = {
        "library_size": 3,
        "library_file": str(library_file),
        "workflow_steps": {"demux": {"completed": True}},
    }
    with open(project_dir / "usortm_project.json", "w") as f:
        json.dump(state, f)

    demux_dir = project_dir / "demux_output"
    demux_dir.mkdir()

    # libA recovered with legacy |Perfect Match suffix, libC without suffix
    well_data = [
        {"plate": "1", "well": "A1", "variant": "libA|Perfect Match", "reads": 200, "consensus_fraction": 0.98},
        {"plate": "1", "well": "B1", "variant": "libC", "reads": 150, "consensus_fraction": 0.92},
    ]
    with open(demux_dir / "well_assignments.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["plate", "well", "variant", "reads", "consensus_fraction"])
        writer.writeheader()
        writer.writerows(well_data)
    with open(demux_dir / "demux_summary.json", "w") as f:
        json.dump({"total_reads": 500, "assigned_reads": 350, "wells_with_data": 2, "wells_passing": 2}, f)

    result = runner.invoke(app, ["pick", str(project_dir), "--tier", ""])
    assert result.exit_code == 0

    hitlist = _hitlist_path(project_dir)
    with open(hitlist, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        rows = list(reader)

    # Only recovered variants are written (empty wells excluded)
    assert len(rows) == 2
    assert [row[0] for row in rows] == ["libA", "libC"]

    # libA matched despite |Perfect Match suffix — has real source data
    lib_a_row = rows[0]
    assert lib_a_row[1] == "1"   # source plate
    assert lib_a_row[2] == "A1"  # source well
