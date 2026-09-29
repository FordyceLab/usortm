"""Tests for the argument handling of ``usortm demux --expected``.

These stop before the pipeline, so they need none of the demux toolchain.
"""

import json
import re

from typer.testing import CliRunner

from usortm.cli import app

runner = CliRunner()


def _plain(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _inputs(tmp_path):
    csv = tmp_path / "plate.csv"
    csv.write_text("plate,well,name,sequence\n1,A1,wt,ACGTACGTACGT\n")
    fq = tmp_path / "reads.fastq"
    fq.write_text("@r\nACGT\n+\nIIII\n")
    return csv, fq


def test_needs_a_construct(tmp_path):
    csv, fq = _inputs(tmp_path)
    result = runner.invoke(app, ["demux", "--expected", str(csv), "--fastq", str(fq),
                                 "-o", str(tmp_path / "out")])
    assert result.exit_code == 1
    assert "--vector" in result.output and "--read-template" in result.output


def test_library_flags_are_refused(tmp_path):
    csv, fq = _inputs(tmp_path)
    result = runner.invoke(app, ["demux", "--expected", str(csv), "--fastq", str(fq),
                                 "--library-csv", str(csv)])
    assert result.exit_code == 1
    assert "--library-csv does not apply" in result.output


def test_a_project_is_not_an_output_directory(tmp_path):
    csv, fq = _inputs(tmp_path)
    project = tmp_path / "proj"
    project.mkdir()
    (project / "usortm_project.json").write_text(json.dumps({}))
    result = runner.invoke(app, ["demux", str(project), "--expected", str(csv),
                                 "--fastq", str(fq)])
    assert result.exit_code == 1
    assert "outside a project" in result.output


def test_bad_expected_plate_is_reported(tmp_path):
    csv, fq = _inputs(tmp_path)
    csv.write_text("plate,well,name,sequence\n9,A1,wt,ACGT\n")
    result = runner.invoke(app, ["demux", "--expected", str(csv), "--fastq", str(fq)])
    assert result.exit_code == 1
    assert "outside 1-8" in _plain(result.output)


def test_project_mode_still_needs_a_project(tmp_path):
    result = runner.invoke(app, ["demux", "--fastq", str(tmp_path)])
    assert result.exit_code == 1
    assert "project directory is required" in result.output


def test_vector_without_expected_is_refused(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "usortm_project.json").write_text(json.dumps({}))
    vec = tmp_path / "v.fa"
    vec.write_text(">v\nACGT\n")
    result = runner.invoke(app, ["demux", str(project), "--vector", str(vec)])
    assert result.exit_code == 1
    assert "only used with --expected" in result.output
