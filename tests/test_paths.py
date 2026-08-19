"""Tests for the project layout.

Round 1 writes to the top of the project and later rounds nest under
rounds/<n>/. Fourteen places used to work that out for themselves; these pin
the rule so the single implementation can be trusted.
"""

import pytest

from usortm.paths import ProjectPaths, paths_for


class TestRoundRoot:

    def test_round_one_is_the_project_root(self, tmp_path):
        p = paths_for(tmp_path)
        assert p.round_root == tmp_path
        assert p.results == tmp_path / "results"
        assert p.demux == tmp_path / "demux"

    def test_later_rounds_nest(self, tmp_path):
        p = paths_for(tmp_path, round_num=2)
        assert p.round_root == tmp_path / "rounds" / "2"
        assert p.results == tmp_path / "rounds" / "2" / "results"
        assert p.demux == tmp_path / "rounds" / "2" / "demux"

    def test_round_zero_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="1 or greater"):
            paths_for(tmp_path, round_num=0)


class TestSharedVersusPerRound:

    def test_config_is_shared_across_rounds(self, tmp_path):
        """One barcode layout and mask set describe the whole project."""
        assert paths_for(tmp_path).plate_map == paths_for(
            tmp_path, round_num=3).plate_map
        assert paths_for(tmp_path).barcodes == paths_for(
            tmp_path, round_num=3).barcodes

    def test_state_file_is_shared(self, tmp_path):
        assert paths_for(tmp_path).state == paths_for(tmp_path, 2).state

    def test_round_state_is_per_round(self, tmp_path):
        assert paths_for(tmp_path, 2).round_state == (
            tmp_path / "rounds" / "2" / "usortm_round.json")

    def test_variants_are_per_round_after_the_first(self, tmp_path):
        """A re-order round orders its own subset of the library."""
        assert paths_for(tmp_path).variants == tmp_path / "inputs" / "variants.csv"
        assert paths_for(tmp_path, 2).variants == (
            tmp_path / "rounds" / "2" / "variants.csv")

    def test_index_is_always_at_the_top(self, tmp_path):
        assert paths_for(tmp_path, 4).index == tmp_path / "index.html"


class TestSplitByLifetime:

    def test_results_hold_what_is_kept(self, tmp_path):
        p = paths_for(tmp_path)
        for path in (p.summary, p.plate_map_html, p.wells_csv,
                     p.well_details_csv, p.run_stats, p.pileups, p.picks):
            assert p.results in path.parents or path == p.results

    def test_demux_holds_what_is_rebuildable(self, tmp_path):
        p = paths_for(tmp_path)
        for path in (p.reads_csv, p.references, p.segments, p.live):
            assert p.demux in path.parents or path == p.demux

    def test_clean_targets_only_the_rebuildable_half(self, tmp_path):
        p = paths_for(tmp_path)
        assert p.rebuildable() == (p.demux,)
        assert p.results not in p.rebuildable()

    def test_the_big_read_table_is_not_a_result(self, tmp_path):
        """reads.csv runs to gigabytes and nothing needs it after the run."""
        p = paths_for(tmp_path)
        assert p.demux in p.reads_csv.parents


class TestSegments:

    def test_named_under_demux(self, tmp_path):
        p = paths_for(tmp_path)
        assert p.segment("run1") == tmp_path / "demux" / "segments" / "run1"

    def test_segments_follow_the_round(self, tmp_path):
        assert paths_for(tmp_path, 2).segment("run1") == (
            tmp_path / "rounds" / "2" / "demux" / "segments" / "run1")


class TestEnsure:

    def test_creates_what_a_run_writes_into(self, tmp_path):
        p = paths_for(tmp_path).ensure()
        for path in (p.inputs, p.config, p.results, p.demux):
            assert path.is_dir()

    def test_is_idempotent(self, tmp_path):
        paths_for(tmp_path).ensure()
        paths_for(tmp_path).ensure()          # must not raise

    def test_returns_itself_for_chaining(self, tmp_path):
        p = paths_for(tmp_path)
        assert p.ensure() is p

    def test_a_later_round_creates_its_own_tree(self, tmp_path):
        p = paths_for(tmp_path, 2).ensure()
        assert (tmp_path / "rounds" / "2" / "results").is_dir()
        assert (tmp_path / "rounds" / "2" / "demux").is_dir()
