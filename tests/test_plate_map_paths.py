"""Tests for resolving the FASTQ paths a plate map lists.

A relative path in the file is ambiguous. It may have been written relative
to the directory the run was launched from, which is where the user typed it,
rather than relative to the file. Resolving it only against the file's own
directory doubles the prefix when the file sits inside that directory:
"usortm_project/fastqs/x.gz" in usortm_project/plate_map.toml became
"usortm_project/usortm_project/fastqs/x.gz".
"""

import os

import pytest

from usortm.demux.plate_map import (
    Segment, check_segment_paths, load_plate_map, write_plate_map,
)


def _write_map(path, fastq_path):
    path.write_text(
        '[[fastq]]\nname = "run1"\n'
        f'path = "{fastq_path}"\n'
        "plates = { 1 = 1 }\n"
    )
    return path


class TestRelativeResolution:

    def test_relative_to_the_config_directory(self, tmp_path):
        (tmp_path / "reads.fastq").write_text("@r\nACGT\n+\nIIII\n")
        cfg = _write_map(tmp_path / "plate_map.toml", "reads.fastq")

        seg = load_plate_map(cfg)[0]
        assert seg.path == tmp_path / "reads.fastq"
        assert seg.path.exists()

    def test_relative_to_the_working_directory(self, tmp_path, monkeypatch):
        """The regression: the path was typed from the parent, and the config
        lives one level down, so the file-relative reading doubles it."""
        proj = tmp_path / "usortm_project"
        (proj / "fastqs").mkdir(parents=True)
        (proj / "fastqs" / "reads.fastq.gz").write_text("x")
        cfg = _write_map(proj / "plate_map.toml",
                         "usortm_project/fastqs/reads.fastq.gz")

        monkeypatch.chdir(tmp_path)
        seg = load_plate_map(cfg)[0]
        assert seg.path.exists(), f"did not resolve: {seg.path}"
        assert "usortm_project/usortm_project" not in str(seg.path)

    def test_config_relative_wins_when_both_exist(self, tmp_path, monkeypatch):
        """Ambiguous only when one is missing; prefer the file's own directory."""
        sub = tmp_path / "proj"
        sub.mkdir()
        (sub / "reads.fastq").write_text("config-relative")
        (tmp_path / "reads.fastq").write_text("cwd-relative")
        cfg = _write_map(sub / "plate_map.toml", "reads.fastq")

        monkeypatch.chdir(tmp_path)
        assert load_plate_map(cfg)[0].path.read_text() == "config-relative"

    def test_absolute_paths_are_untouched(self, tmp_path):
        fq = tmp_path / "reads.fastq"
        fq.write_text("x")
        cfg = _write_map(tmp_path / "sub_map.toml", str(fq))

        assert load_plate_map(cfg)[0].path == fq

    def test_a_path_that_exists_nowhere_still_names_something(self, tmp_path):
        cfg = _write_map(tmp_path / "plate_map.toml", "absent/reads.fastq")
        seg = load_plate_map(cfg)[0]

        assert not seg.path.exists()
        assert "absent/reads.fastq" in str(seg.path)


class TestMissingPathsAreReported:

    def test_missing_segments_are_listed(self, tmp_path):
        segs = [Segment(name="a", path=tmp_path / "gone.fastq", plates={1: 1}),
                Segment(name="b", path=tmp_path / "here.fastq", plates={2: 2})]
        (tmp_path / "here.fastq").write_text("x")

        missing = check_segment_paths(segs)
        assert [s.name for s in missing] == ["a"]

    def test_all_present_is_empty(self, tmp_path):
        fq = tmp_path / "r.fastq"
        fq.write_text("x")
        assert check_segment_paths([Segment(name="a", path=fq, plates={1: 1})]) == []

    def test_a_directory_counts_as_present(self, tmp_path):
        d = tmp_path / "fastq_pass"
        d.mkdir()
        assert check_segment_paths([Segment(name="a", path=d, plates={1: 1})]) == []


class TestWritingIsUnambiguous:

    def test_written_paths_are_absolute(self, tmp_path, monkeypatch):
        """New files avoid the ambiguity entirely."""
        proj = tmp_path / "proj"
        proj.mkdir()
        fq = proj / "reads.fastq"
        fq.write_text("x")

        monkeypatch.chdir(tmp_path)
        out = write_plate_map(
            [Segment(name="run1", path="proj/reads.fastq", plates={1: 1})],
            proj / "plate_map.toml",
        )
        assert os.path.isabs(
            [l for l in out.read_text().splitlines() if l.startswith("path")][0]
            .split('"')[1]
        )
