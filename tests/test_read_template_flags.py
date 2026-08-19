"""A read template handed to the wrong flag must be caught.

As a --reference its masked spans align to nothing; as a --vector-fasta its
three masked spans are not one variable region. Either way the barcode masks
go underived, which reads as an empty library rather than a wrong flag.
"""

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from usortm.cli import app
from usortm.cli.demux_cmd import _looks_like_read_template

runner = CliRunner()


def _flat(text):
    """Collapse wrapping so assertions do not depend on terminal width."""
    return " ".join(text.split())

FRONT_PAD = "GGATCCTTAAGCACTCAATG"
FLANK_5P = "ACGT" * 40
FLANK_3P = "TTGA" * 50
REAR_PAD = "CCTAGGATTCGTGAGTTACC"


def _template(tmp_path):
    seq = (FRONT_PAD + "N" * 24 + FLANK_5P + "N" * 300 + FLANK_3P
           + "N" * 24 + REAR_PAD)
    path = tmp_path / "Reference_read.fa"
    path.write_text(f">Reference_read\n{seq}\n")
    return path


@pytest.fixture
def project(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "usortm_project.json").write_text(json.dumps(
        {"library_size": 4, "barcode_kit": "levseq", "n_plates": 1,
         "workflow_steps": {}}
    ))
    lib = tmp_path / "variants.csv"
    lib.write_text("name,sequence\nvar_1,ACGTACGTACGT\n")
    fq = tmp_path / "reads.fastq"
    fq.write_text("@r\nACGT\n+\nIIII\n")
    return proj, lib, fq


def _run(project, template, flag):
    proj, lib, fq = project
    with patch("usortm.cli.demux_cmd._run_demux") as run_demux, patch(
        "usortm.cli.demux_cmd.check_all_dependencies",
        return_value={"dorado": "d", "minimap2": "m", "samtools": "s"},
    ):
        result = runner.invoke(app, [
            "demux", str(proj), "--library-csv", str(lib),
            flag, str(template), "--fastq", str(fq),
        ])
    return result, run_demux


class TestWrongFlagIsCaught:

    def test_reference_flag_is_caught(self, project, tmp_path):
        result, run_demux = _run(project, _template(tmp_path), "--reference")

        assert result.exit_code == 1
        assert "looks like a read template" in _flat(result.stdout)
        assert "--read-template" in _flat(result.stdout)
        run_demux.assert_not_called()

    def test_vector_fasta_flag_is_caught(self, project, tmp_path):
        result, run_demux = _run(project, _template(tmp_path), "--vector-fasta")

        assert result.exit_code == 1
        assert "--read-template" in _flat(result.stdout)
        run_demux.assert_not_called()

    def test_caught_even_though_library_csv_overrides_reference(
        self, project, tmp_path
    ):
        """--library-csv replaces `reference` further down, so the check has to
        run against what was actually typed."""
        result, _ = _run(project, _template(tmp_path), "--reference")
        assert "looks like a read template" in _flat(result.stdout)


class TestOrdinaryInputsAreNotFlagged:

    def test_a_vector_has_one_masked_span(self, tmp_path):
        vector = tmp_path / "vector.fa"
        vector.write_text(f">v\n{FLANK_5P}{'N' * 300}{FLANK_3P}\n")
        assert not _looks_like_read_template(vector)

    def test_a_library_reference_has_none(self, tmp_path):
        ref = tmp_path / "ref.fa"
        ref.write_text(">var_1\nACGTACGTACGT\n>var_2\nTTGACCTTGACC\n")
        assert not _looks_like_read_template(ref)

    def test_a_missing_file_is_not_flagged(self, tmp_path):
        assert not _looks_like_read_template(tmp_path / "nope.fa")
