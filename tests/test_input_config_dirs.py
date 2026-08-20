"""Tests for keeping a project's inputs and configuration in their own places.

A run fills the top of a project with its own output, and the things a person
supplied get lost among it. They now go in ``inputs/`` and ``config/``.

Projects made before that split kept everything loose, and are still read
without a migration step -- one that would have to be got right on data nobody
wants to re-derive.
"""

import pytest

from usortm.paths import config_file, input_file, paths_for


class TestFindingInputs:

    def test_found_in_the_inputs_directory(self, tmp_path):
        (tmp_path / "inputs").mkdir()
        (tmp_path / "inputs" / "variants.csv").write_text("Name,Sequence\n")
        assert input_file(tmp_path, "variants.csv") == (
            tmp_path / "inputs" / "variants.csv")

    def test_an_older_project_keeps_working(self, tmp_path):
        """Everything at the top level, as projects were made before."""
        (tmp_path / "variants.csv").write_text("Name,Sequence\n")
        assert input_file(tmp_path, "variants.csv") == (
            tmp_path / "variants.csv")

    def test_the_organised_place_wins(self, tmp_path):
        """A project part-way through the change reads the new one."""
        (tmp_path / "inputs").mkdir()
        (tmp_path / "inputs" / "variants.csv").write_text("new\n")
        (tmp_path / "variants.csv").write_text("old\n")
        assert input_file(tmp_path, "variants.csv").read_text() == "new\n"

    def test_a_missing_file_names_where_it_should_go(self, tmp_path):
        """So a caller reporting it absent points at the right place."""
        assert input_file(tmp_path, "variants.csv") == (
            tmp_path / "inputs" / "variants.csv")

    def test_a_directory_counts(self, tmp_path):
        (tmp_path / "inputs" / "fastqs").mkdir(parents=True)
        assert input_file(tmp_path, "fastqs") == tmp_path / "inputs" / "fastqs"


class TestFindingConfig:

    def test_found_in_the_config_directory(self, tmp_path):
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "plate_map.toml").write_text("")
        assert config_file(tmp_path, "plate_map.toml") == (
            tmp_path / "config" / "plate_map.toml")

    def test_an_older_project_keeps_working(self, tmp_path):
        (tmp_path / "plate_map.toml").write_text("")
        assert config_file(tmp_path, "plate_map.toml") == (
            tmp_path / "plate_map.toml")

    def test_barcodes_are_configuration(self, tmp_path):
        (tmp_path / "config" / "barcodes").mkdir(parents=True)
        assert config_file(tmp_path, "barcodes") == (
            tmp_path / "config" / "barcodes")

    def test_a_missing_file_names_where_it_should_go(self, tmp_path):
        assert config_file(tmp_path, "mask_config.toml") == (
            tmp_path / "config" / "mask_config.toml")


class TestTheTwoAreSeparate:
    """What a person supplied, and how the run is set up, are different
    questions and are not looked for in the same place."""

    def test_an_input_is_not_found_in_config(self, tmp_path):
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "variants.csv").write_text("")
        assert not input_file(tmp_path, "variants.csv").exists()

    def test_config_is_not_found_in_inputs(self, tmp_path):
        (tmp_path / "inputs").mkdir()
        (tmp_path / "inputs" / "plate_map.toml").write_text("")
        assert not config_file(tmp_path, "plate_map.toml").exists()

    def test_they_agree_with_the_layout_module(self, tmp_path):
        p = paths_for(tmp_path)
        assert input_file(tmp_path, "variants.csv").parent == p.inputs
        assert config_file(tmp_path, "plate_map.toml").parent == p.config


class TestWhatPlanWrites:

    @pytest.fixture
    def planned(self, tmp_path):
        from usortm.cli.plan import _save_variants

        variants = [{"name": f"v{i}", "sequence": "ACGT" * 10}
                    for i in range(8)]
        out = tmp_path / "proj"
        (out / "inputs").mkdir(parents=True)
        _save_variants(variants, out / "inputs" / "variants.csv")
        return out

    def test_the_library_goes_under_inputs(self, planned):
        assert (planned / "inputs" / "variants.csv").exists()
        assert not (planned / "variants.csv").exists()

    def test_and_is_found_there(self, planned):
        assert input_file(planned, "variants.csv").exists()

    def test_barcodes_go_under_config(self, tmp_path):
        from usortm.cli.plan import _generate_barcode_assignments

        out = tmp_path / "proj"
        out.mkdir()
        _generate_barcode_assignments(2, "levseq", out)
        assert (out / "config" / "barcodes").is_dir()
        assert not (out / "barcodes").exists()

    def test_and_are_found_there(self, tmp_path):
        from usortm.cli.plan import _generate_barcode_assignments

        out = tmp_path / "proj"
        out.mkdir()
        _generate_barcode_assignments(2, "levseq", out)
        assert config_file(out, "barcodes").is_dir()
