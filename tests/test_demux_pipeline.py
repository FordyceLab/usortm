"""Tests for the LevSeq demultiplexing pipeline.

Unit tests use synthetic FASTQ/FASTA fixtures and mock external tools.
Integration tests (marked with requires_*) need dorado, minimap2, samtools.
"""

import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from usortm.demux.barcodes import (
    LEVSEQ_FBC,
    LEVSEQ_RBC,
    get_rbc_count_for_plates,
    write_levseq_fbc_fasta,
    write_levseq_fbc_toml,
    write_levseq_rbc_fasta,
    write_levseq_rbc_toml,
)
from usortm.demux.deps import DependencyError, find_tool
from usortm.demux.pipeline import (
    _build_barcode_name_dfs,
    _prepare_single_ref_fastas,
    _translate_to_cli_format,
)


# ---------------------------------------------------------------------------
# Skip markers for tests requiring external tools
# Uses the project's own find_*() functions which check env vars and
# common installation locations, not just PATH.
# ---------------------------------------------------------------------------
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


requires_dorado = pytest.mark.skipif(
    not _tool_available("dorado"),
    reason="dorado not installed",
)
requires_minimap2 = pytest.mark.skipif(
    not _tool_available("minimap2"),
    reason="minimap2 not installed",
)
requires_samtools = pytest.mark.skipif(
    not _tool_available("samtools"),
    reason="samtools not installed",
)

# Reference sequence used across fixtures (~300 bp GFP fragment)
FAKE_REFERENCE_SEQ = (
    "ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGAC"
    "GGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTAC"
    "GGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACC"
    "CTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAG"
    "CAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTC"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_reference_fasta(tmp_path):
    """Create a single-entry reference FASTA (~300 bp GFP fragment)."""
    fasta_path = tmp_path / "reference.fasta"
    fasta_path.write_text(f">GFP_test\n{FAKE_REFERENCE_SEQ}\n")
    return fasta_path


@pytest.fixture
def fake_multi_ref_fasta(tmp_path):
    """Create a multi-entry reference FASTA with three sequences."""
    fasta_path = tmp_path / "multi_reference.fasta"
    # Use substrings of the reference as different "genes"
    seq1 = FAKE_REFERENCE_SEQ[:100]
    seq2 = FAKE_REFERENCE_SEQ[100:200]
    seq3 = FAKE_REFERENCE_SEQ[200:]
    fasta_path.write_text(
        f">gene_A\n{seq1}\n"
        f">gene_B\n{seq2}\n"
        f">gene_C\n{seq3}\n"
    )
    return fasta_path


@pytest.fixture
def fake_fastq(tmp_path):
    """Generate a synthetic FASTQ with reads containing LevSeq barcodes.

    Each read is structured as: FBC + reference_fragment + RBC_revcomp
    This mimics what a real nanopore read with LevSeq barcodes looks like.
    """
    fastq_path = tmp_path / "reads.fastq"
    lines = []

    # Generate 100 reads, each with a barcode pair
    for i in range(100):
        fbc_idx = i % 96       # cycle through forward barcodes
        rbc_idx = i % 4        # use 4 reverse barcodes (1 plate)
        fbc_seq = LEVSEQ_FBC[fbc_idx]
        rbc_seq = LEVSEQ_RBC[rbc_idx]

        # Reverse complement the RBC for the 3' end
        rbc_rc = _reverse_complement(rbc_seq)

        # Take a fragment of the reference as the insert
        start = (i * 3) % (len(FAKE_REFERENCE_SEQ) - 50)
        insert = FAKE_REFERENCE_SEQ[start:start + 50]

        read_seq = fbc_seq + insert + rbc_rc
        qual = "I" * len(read_seq)

        lines.append(f"@read_{i:04d}")
        lines.append(read_seq)
        lines.append("+")
        lines.append(qual)

    fastq_path.write_text("\n".join(lines) + "\n")
    return fastq_path


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    comp = {"A": "T", "T": "A", "G": "C", "C": "G",
            "a": "t", "t": "a", "g": "c", "c": "g"}
    return "".join(comp.get(b, b) for b in reversed(seq))


# ===================================================================
# Barcode data tests
# ===================================================================

class TestLevSeqBarcodes:
    """Tests for LevSeq barcode data integrity."""

    def test_fbc_count(self):
        """There should be exactly 96 forward barcodes."""
        assert len(LEVSEQ_FBC) == 96

    def test_rbc_count(self):
        """There should be exactly 96 reverse barcodes."""
        assert len(LEVSEQ_RBC) == 96

    def test_fbc_length(self):
        """All forward barcodes should be 24 nucleotides."""
        for i, seq in enumerate(LEVSEQ_FBC):
            assert len(seq) == 24, f"NB{i + 1:02d} has length {len(seq)}"

    def test_rbc_length(self):
        """All reverse barcodes should be 20-25 nucleotides."""
        for i, seq in enumerate(LEVSEQ_RBC):
            assert 20 <= len(seq) <= 25, (
                f"RB{i + 1:02d} has length {len(seq)}, "
                "expected 20-25 nt"
            )

    def test_fbc_valid_dna(self):
        """All forward barcodes should contain only valid DNA bases."""
        valid = set("ATGC")
        for i, seq in enumerate(LEVSEQ_FBC):
            assert set(seq.upper()) <= valid, (
                f"NB{i + 1:02d} contains invalid bases: "
                f"{set(seq.upper()) - valid}"
            )

    def test_rbc_valid_dna(self):
        """All reverse barcodes should contain only valid DNA bases."""
        valid = set("ATGC")
        for i, seq in enumerate(LEVSEQ_RBC):
            assert set(seq.upper()) <= valid, (
                f"RB{i + 1:02d} contains invalid bases: "
                f"{set(seq.upper()) - valid}"
            )

    def test_fbc_unique(self):
        """All forward barcodes should be unique."""
        assert len(set(LEVSEQ_FBC)) == 96

    def test_rbc_first_12_unique(self):
        """The first 12 reverse barcodes (unique RBs) should be distinct."""
        assert len(set(LEVSEQ_RBC[:12])) == 12


# ===================================================================
# Barcode config generation tests
# ===================================================================

class TestBarcodeConfigGeneration:
    """Tests for TOML and FASTA config file generation."""

    def test_rbc_count_for_plates(self):
        """Verify RBC count calculation for different plate numbers."""
        assert get_rbc_count_for_plates(1) == 4
        assert get_rbc_count_for_plates(2) == 8
        assert get_rbc_count_for_plates(8) == 32
        assert get_rbc_count_for_plates(24) == 96
        # Cap at 96
        assert get_rbc_count_for_plates(30) == 96

    def test_fbc_toml_generation(self, tmp_path):
        """FBC TOML should be a valid file with correct structure."""
        toml_path = write_levseq_fbc_toml(tmp_path)
        assert toml_path.exists()
        content = toml_path.read_text()
        assert "[arrangement]" in content
        assert 'name = "levSeq_bcs_map"' in content
        assert 'kit = "levSeq"' in content
        assert 'LevSeq-fbc-%02i' in content
        assert "last_index = 96" in content
        # Masks must be non-empty (dorado requires at least one per end)
        assert 'mask1_front = "AATATAAATT"' in content
        assert 'mask1_rear' in content
        assert 'mask2_front' in content
        assert 'mask2_rear' in content

    def test_rbc_toml_generation(self, tmp_path):
        """RBC TOML should respect the n_barcodes parameter."""
        toml_path = write_levseq_rbc_toml(tmp_path, n_barcodes=4)
        assert toml_path.exists()
        content = toml_path.read_text()
        assert "last_index = 4" in content
        assert 'LevSeq-rbc-%02i' in content
        # Masks must be non-empty
        assert 'mask1_front = "TATAAATTAT"' in content
        assert 'mask2_front = "GCTCACGCTGTAGGTATCTCAG"' in content

    def test_rbc_toml_cap_at_96(self, tmp_path):
        """RBC TOML should cap at 96 even if more are requested."""
        toml_path = write_levseq_rbc_toml(tmp_path, n_barcodes=200)
        content = toml_path.read_text()
        assert "last_index = 96" in content

    def test_fbc_fasta_generation(self, tmp_path):
        """FBC FASTA should contain all 96 barcodes with correct names."""
        fasta_path = write_levseq_fbc_fasta(tmp_path)
        assert fasta_path.exists()
        content = fasta_path.read_text()

        # Check first and last entries (lowercase naming)
        assert ">LevSeq-fbc-01" in content
        assert ">LevSeq-fbc-96" in content
        assert LEVSEQ_FBC[0] in content
        assert LEVSEQ_FBC[95] in content

        # Count entries
        headers = [l for l in content.strip().split("\n") if l.startswith(">")]
        assert len(headers) == 96

    def test_rbc_fasta_generation(self, tmp_path):
        """RBC FASTA should contain the requested number of barcodes."""
        fasta_path = write_levseq_rbc_fasta(tmp_path, n_barcodes=4)
        assert fasta_path.exists()
        content = fasta_path.read_text()

        headers = [l for l in content.strip().split("\n") if l.startswith(">")]
        assert len(headers) == 4
        assert ">LevSeq-rbc-01" in content
        assert ">LevSeq-rbc-04" in content

    def test_rbc_fasta_all(self, tmp_path):
        """RBC FASTA with no limit should write all 96 barcodes."""
        fasta_path = write_levseq_rbc_fasta(tmp_path)
        content = fasta_path.read_text()
        headers = [l for l in content.strip().split("\n") if l.startswith(">")]
        assert len(headers) == 96


# ===================================================================
# Dependency checker tests
# ===================================================================

class TestDependencyChecker:
    """Tests for external tool detection."""

    def test_find_tool_missing(self):
        """Should raise DependencyError for a nonexistent tool."""
        with pytest.raises(DependencyError, match="not found on PATH"):
            find_tool("nonexistent_tool_xyz_12345")

    def test_find_tool_found(self):
        """Should find a tool that exists (python is always available)."""
        path = find_tool("python3")
        assert path is not None
        assert Path(path).exists()

    def test_find_tool_env_var_override(self, tmp_path):
        """Should use the env var path when set."""
        # Create a fake executable
        fake_tool = tmp_path / "fake_dorado"
        fake_tool.write_text("#!/bin/sh\necho fake")
        fake_tool.chmod(0o755)

        with patch.dict("os.environ", {"TEST_TOOL_PATH": str(fake_tool)}):
            path = find_tool("nonexistent_tool", env_var="TEST_TOOL_PATH")
            assert path == str(fake_tool.resolve())

    def test_find_tool_env_var_invalid(self):
        """Should fall back to PATH if env var points to nonexistent file."""
        with patch.dict("os.environ", {"BAD_PATH": "/no/such/file"}):
            with pytest.raises(DependencyError):
                find_tool("nonexistent_tool", env_var="BAD_PATH")


# ===================================================================
# Pipeline helper tests
# ===================================================================

class TestPipelineHelpers:
    """Tests for pipeline helper functions."""

    def test_build_barcode_name_dfs(self):
        """Should create DataFrames with correct shape and naming."""
        fbc_df, rbc_df = _build_barcode_name_dfs(n_fbc=96, n_rbc=4)

        assert len(fbc_df) == 96
        assert len(rbc_df) == 4
        assert fbc_df["name"].iloc[0] == "FB01"
        assert fbc_df["name"].iloc[95] == "FB96"
        assert rbc_df["name"].iloc[0] == "RB01"
        assert rbc_df["name"].iloc[3] == "RB04"

    def test_prepare_single_ref_fastas(self, fake_multi_ref_fasta, tmp_path):
        """Should split a multi-entry FASTA into individual files."""
        output_dir = tmp_path / "refs"
        _prepare_single_ref_fastas(fake_multi_ref_fasta, output_dir)

        single_dir = output_dir / "single_ref_fastas"
        assert single_dir.exists()
        assert (single_dir / "gene_A.fasta").exists()
        assert (single_dir / "gene_B.fasta").exists()
        assert (single_dir / "gene_C.fasta").exists()

        # Verify content of one file (BioPython wraps at 60 chars per line)
        content = (single_dir / "gene_A.fasta").read_text()
        assert ">gene_A" in content
        # Remove line breaks to check the full sequence
        seq_only = content.replace("\n", "").replace(">gene_A", "")
        assert FAKE_REFERENCE_SEQ[:100] in seq_only

    def test_translate_to_cli_format(self):
        """Should correctly translate DataFrames to CLI output dict."""
        # Create sample read DataFrame
        read_df = pd.DataFrame({
            "read_name": [f"read_{i}" for i in range(50)],
            "well_pos": ["1A1"] * 30 + ["1A3"] * 15 + [None] * 5,
        })

        # Create sample well DataFrame
        well_df = pd.DataFrame({
            "plate": [1, 1],
            "well": ["A1", "A3"],
            "global_well": ["1A1", "1A3"],
            "depth": [30, 15],
            "major_ref": ["fwd:GFP_test", "fwd:GFP_test"],
            "major_freq": [0.95, 0.87],
            "cons_check": ["Perfect Match", "Silent Mutation"],
        })

        pipeline_stats = {
            "input_reads": 1000,
            "align": {"mapped": 800, "unmapped": 200, "fwd": 500, "rev": 300},
            "demux": {"complete_assignments": 50},
        }
        results = _translate_to_cli_format(
            read_df=read_df,
            well_df=well_df,
            min_reads=20,
            pipeline_stats=pipeline_stats,
        )

        assert results["input_reads"] == 1000
        assert results["aligned_reads"] == 800
        assert results["demuxed_reads"] == 50
        assert results["total_reads"] == 1000  # backward compat alias
        assert results["assigned_reads"] == 45  # 50 - 5 with None
        assert results["wells_with_data"] == 2
        assert results["wells_passing"] == 1   # only depth=30 >= 20
        assert "1_A1" in results["well_assignments"]
        assert "1_A3" in results["well_assignments"]

        # Check well assignment content
        a1 = results["well_assignments"]["1_A1"]
        assert a1["reads"] == 30
        # cons_check is stored separately — variant name is clean
        assert a1["variant"] == "GFP_test"
        assert a1["cons_check"] == "Perfect Match"
        assert a1["consensus_fraction"] == 0.95

    def test_translate_empty_well_df(self):
        """Should handle empty DataFrames gracefully."""
        read_df = pd.DataFrame(columns=["read_name", "well_pos"])
        well_df = pd.DataFrame(columns=[
            "plate", "well", "global_well", "depth",
            "major_ref", "major_freq",
        ])

        results = _translate_to_cli_format(read_df, well_df, min_reads=100, pipeline_stats={})

        assert results["input_reads"] == 0
        assert results["total_reads"] == 0
        assert results["assigned_reads"] == 0
        assert results["wells_with_data"] == 0
        assert results["wells_passing"] == 0
        assert results["well_assignments"] == {}

    def test_generate_well_df_empty_assignments(self):
        """generate_well_df should return an empty, well-formed frame."""
        from usortm.demux.utils import generate_well_df

        read_df = pd.DataFrame({
            "read_name": ["read_1", "read_2"],
            "ref_name": ["fwd:GFP_test", "rev:GFP_test"],
            "well_pos": [None, None],
        })

        well_df = generate_well_df(read_df)

        assert well_df.empty
        assert set([
            "plate", "well", "global_well", "depth",
            "major_ref", "major_freq", "ref_len", "ref_seq",
        ]).issubset(set(well_df.columns))

    def test_format_df_handles_missing_assignment_columns(self):
        """format_df should not crash when demux columns are missing."""
        from usortm.demux.utils import format_df

        read_df = pd.DataFrame({
            "read_name": ["read_1"],
            "read_seq": ["ATGC"],
            "read_qual": ["IIII"],
            "avg_qual": [40.0],
        })

        formatted = format_df(read_df, fbc_df=None, rbc_df=None, ref_fasta=None)

        assert formatted.empty
        assert set(["fbc_name", "rbc_name", "ref_name", "well_pos"]).issubset(
            set(formatted.columns)
        )


# ===================================================================
# Fake FASTQ fixture validation tests
# ===================================================================

class TestFakeFixtures:
    """Tests that validate the fake FASTQ/FASTA fixtures themselves."""

    def test_fake_fastq_is_valid(self, fake_fastq):
        """The fake FASTQ should have the correct format (4 lines per read)."""
        lines = fake_fastq.read_text().strip().split("\n")
        # 100 reads * 4 lines each = 400 lines
        assert len(lines) == 400

        # Check first read structure
        assert lines[0].startswith("@read_")
        assert lines[2] == "+"
        # Seq and qual should have the same length
        assert len(lines[1]) == len(lines[3])

    def test_fake_fastq_contains_barcodes(self, fake_fastq):
        """Reads should contain actual LevSeq barcode sequences."""
        content = fake_fastq.read_text()
        # First read should start with NB01 barcode
        assert LEVSEQ_FBC[0] in content
        # Should contain reverse complement of RB01
        rb01_rc = _reverse_complement(LEVSEQ_RBC[0])
        assert rb01_rc in content

    def test_fake_reference_fasta(self, fake_reference_fasta):
        """Reference FASTA should be properly formatted."""
        content = fake_reference_fasta.read_text()
        assert content.startswith(">GFP_test\n")
        # Should contain the full reference sequence
        assert FAKE_REFERENCE_SEQ in content


# ===================================================================
# CSV to reference FASTA conversion tests
# ===================================================================

class TestCsvToFasta:
    """Tests for csv_to_reference_fasta() utility."""

    def test_basic_conversion(self, tmp_path):
        """Convert a simple Name,Sequence CSV to FASTA."""
        from usortm.demux.utils import csv_to_reference_fasta

        csv_path = tmp_path / "library.csv"
        csv_path.write_text("Name,Sequence\ngene_A,ATGCGATCG\ngene_B,TTGGCCAA\n")

        fasta_path = tmp_path / "ref.fasta"
        result = csv_to_reference_fasta(str(csv_path), str(fasta_path), strip_flanking=False)

        assert result == str(fasta_path)
        content = fasta_path.read_text()
        assert ">gene_A\nATGCGATCG\n" in content
        assert ">gene_B\nTTGGCCAA\n" in content

    def test_strip_flanking(self, tmp_path):
        """Lowercase flanking regions should be stripped when enabled."""
        from usortm.demux.utils import csv_to_reference_fasta

        csv_path = tmp_path / "library.csv"
        csv_path.write_text(
            "Name,Sequence\n"
            "gene_X,acgtACGTTTGGCCacgt\n"
        )

        fasta_path = tmp_path / "ref.fasta"
        csv_to_reference_fasta(str(csv_path), str(fasta_path), strip_flanking=True)

        content = fasta_path.read_text()
        assert ">gene_X\nACGTTTGGCC\n" in content
        # Lowercase flanking should NOT appear
        assert "acgt" not in content

    def test_no_strip(self, tmp_path):
        """Full sequences should be preserved when strip_flanking=False."""
        from usortm.demux.utils import csv_to_reference_fasta

        csv_path = tmp_path / "library.csv"
        csv_path.write_text(
            "Name,Sequence\n"
            "gene_Y,acgtACGTTTGGCCacgt\n"
        )

        fasta_path = tmp_path / "ref.fasta"
        csv_to_reference_fasta(str(csv_path), str(fasta_path), strip_flanking=False)

        content = fasta_path.read_text()
        assert ">gene_Y\nacgtACGTTTGGCCacgt\n" in content

    def test_entry_count(self, tmp_path):
        """Number of FASTA entries should match CSV rows."""
        from usortm.demux.utils import csv_to_reference_fasta
        from Bio import SeqIO

        csv_path = tmp_path / "library.csv"
        lines = ["Name,Sequence"] + [f"seq_{i},ATGC" for i in range(50)]
        csv_path.write_text("\n".join(lines) + "\n")

        fasta_path = tmp_path / "ref.fasta"
        csv_to_reference_fasta(str(csv_path), str(fasta_path))

        records = list(SeqIO.parse(str(fasta_path), "fasta"))
        assert len(records) == 50


# ===================================================================
# Alignment and strand-split tests (require minimap2 + samtools)
# ===================================================================

@requires_minimap2
@requires_samtools
class TestAlignAndSplit:
    """Tests for align_and_split_by_strand()."""

    def test_output_structure(self, tmp_path, fake_fastq, fake_reference_fasta):
        """Verify the output FASTQ and ref_map are produced."""
        from usortm.demux.utils import align_and_split_by_strand

        align_dir = tmp_path / "alignment"
        oriented_fq, ref_map, align_stats = align_and_split_by_strand(
            multi_ref_fasta=str(fake_reference_fasta),
            fastq=str(fake_fastq),
            output_dir=str(align_dir),
            threads=1,
        )

        # Output FASTQ should exist and be non-empty
        assert Path(oriented_fq).exists()
        assert Path(oriented_fq).stat().st_size > 0

        # ref_map should be a dict with read_name -> {ref, direction}
        assert isinstance(ref_map, dict)
        if len(ref_map) > 0:
            first_entry = next(iter(ref_map.values()))
            assert "ref" in first_entry
            assert "direction" in first_entry
            assert first_entry["direction"] in ("fwd", "rev")

        # align_stats should have the expected keys
        assert isinstance(align_stats, dict)
        assert "fwd" in align_stats
        assert "rev" in align_stats
        assert "mapped" in align_stats
        assert "unmapped" in align_stats
        assert align_stats["mapped"] == align_stats["fwd"] + align_stats["rev"]

    def test_bam_created(self, tmp_path, fake_fastq, fake_reference_fasta):
        """BAM alignment file should be created."""
        from usortm.demux.utils import align_and_split_by_strand

        align_dir = tmp_path / "alignment"
        align_and_split_by_strand(
            multi_ref_fasta=str(fake_reference_fasta),
            fastq=str(fake_fastq),
            output_dir=str(align_dir),
            threads=1,
        )

        bam_path = align_dir / "ref_alignment.bam"
        assert bam_path.exists()
        assert bam_path.stat().st_size > 0


# ===================================================================
# Integration tests (require external tools)
# ===================================================================

@requires_dorado
@requires_minimap2
@requires_samtools
class TestFullPipeline:
    """End-to-end pipeline tests that require external tools."""

    def test_full_levseq_pipeline(self, tmp_path, fake_fastq, fake_reference_fasta):
        """Run the full pipeline with fake data and verify outputs."""
        from usortm.demux.pipeline import run_levseq_pipeline

        output_dir = tmp_path / "demux_output"
        stages_seen = []

        def track_progress(msg):
            stages_seen.append(msg)

        results = run_levseq_pipeline(
            fastq=fake_fastq,
            output_dir=output_dir,
            reference=fake_reference_fasta,
            n_plates=1,
            min_reads=5,
            min_fraction=0.5,
            threads=1,
            progress_callback=track_progress,
        )

        # Verify result structure
        assert "input_reads" in results
        assert "aligned_reads" in results
        assert "demuxed_reads" in results
        assert "assigned_reads" in results
        assert "wells_with_data" in results
        assert "wells_passing" in results
        assert "well_assignments" in results
        assert results["input_reads"] > 0

        # Verify output files were created
        assert (output_dir / "dorado_config").exists()
        assert (output_dir / "alignment").exists()
        assert (output_dir / "fbc").exists()
        assert (output_dir / "rbc").exists()
        assert (output_dir / "read_df.csv").exists()
        assert (output_dir / "well_df.csv").exists()

        # Verify progress was tracked
        assert len(stages_seen) > 0


# ---------------------------------------------------------------------------
# Tests for _process_single_well and generate_per_well_consensus workers param
# ---------------------------------------------------------------------------

class TestProcessSingleWell:
    """Unit tests for the per-well consensus helper."""

    def test_returns_tuple_of_three(self, tmp_path):
        """_process_single_well always returns (well, cigar, cons) even on failure."""
        from usortm.demux.utils import _process_single_well

        paths = {
            "ref_fa": str(tmp_path / "nonexistent_ref.fasta"),
            "fq": str(tmp_path / "nonexistent.fastq"),
            "bam": str(tmp_path / "out.bam"),
            "cons_fa": str(tmp_path / "cons.fasta"),
            "cons_bam": str(tmp_path / "cons_align.bam"),
        }
        result = _process_single_well("1A1", paths, "minimap2_fake", "samtools_fake")
        assert len(result) == 3
        well, cigar, cons = result
        assert well == "1A1"
        # Should return None values on failure (subprocess will fail with fake paths)
        assert cigar is None
        assert cons is None

    def test_workers_param_accepted(self):
        """generate_per_well_consensus accepts a workers keyword argument."""
        import inspect
        from usortm.demux.utils import generate_per_well_consensus

        sig = inspect.signature(generate_per_well_consensus)
        assert "workers" in sig.parameters
        assert sig.parameters["workers"].default == 4


class TestConcatFastqDir:
    """Unit tests for _concat_fastq_dir."""

    def test_concatenates_plain_fastqs(self, tmp_path):
        """Combines multiple plain FASTQ files into one."""
        from usortm.cli.demux_cmd import _concat_fastq_dir

        fq_dir = tmp_path / "fastqs"
        fq_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        reads_a = "@read1\nACGT\n+\nIIII\n@read2\nTTTT\n+\nIIII\n"
        reads_b = "@read3\nGGGG\n+\nIIII\n"
        (fq_dir / "lane1.fastq").write_text(reads_a)
        (fq_dir / "lane2.fastq").write_text(reads_b)

        result = _concat_fastq_dir(fq_dir, out_dir)

        assert result == out_dir / "combined.fastq"
        combined = result.read_text()
        assert "@read1" in combined
        assert "@read2" in combined
        assert "@read3" in combined

    def test_concatenates_gzipped_fastqs(self, tmp_path):
        """Decompresses and combines gzip FASTQ files."""
        import gzip
        from usortm.cli.demux_cmd import _concat_fastq_dir

        fq_dir = tmp_path / "fastqs"
        fq_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        reads = "@read1\nACGT\n+\nIIII\n"
        with gzip.open(fq_dir / "lane1.fastq.gz", "wt") as f:
            f.write(reads)
        (fq_dir / "lane2.fastq").write_text("@read2\nTTTT\n+\nIIII\n")

        result = _concat_fastq_dir(fq_dir, out_dir)

        combined = result.read_text()
        assert "@read1" in combined
        assert "@read2" in combined

    def test_raises_on_empty_dir(self, tmp_path):
        """Exits with typer.Exit when no FASTQ files are found."""
        import click
        from usortm.cli.demux_cmd import _concat_fastq_dir

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with pytest.raises(click.exceptions.Exit):
            _concat_fastq_dir(empty_dir, out_dir)
