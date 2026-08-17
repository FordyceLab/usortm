"""Tests for barcode-plate to sort-plate mapping.

The motivating case throughout is a ten-plate run split across two FASTQs,
where the kit's eight barcode plates are not enough and plates 1 and 2 are
reused::

    fastq 1: barcode 1-6  -> sort 1-6
    fastq 2: barcode 7,8  -> sort 7,8
             barcode 1,2  -> sort 9,10
"""

import pandas as pd
import pytest

from usortm.demux.plate_map import (
    MAX_BARCODE_PLATES,
    PlateMapError,
    Segment,
    format_plate_map_toml,
    identity_segment,
    load_plate_map,
    parse_plate_map,
    total_sort_plates,
    write_plate_map,
)
from usortm.demux.utils import barcode_to_well, format_df


TEN_PLATE_TOML = """
[[fastq]]
name = "run1"
path = "run1.fastq"
plates = { 1 = 1, 2 = 2, 3 = 3, 4 = 4, 5 = 5, 6 = 6 }

[[fastq]]
name = "run2"
path = "run2.fastq"
plates = { 7 = 7, 8 = 8, 1 = 9, 2 = 10 }
"""


@pytest.fixture
def ten_plate_config(tmp_path):
    (tmp_path / "run1.fastq").write_text("@r\nACGT\n+\nIIII\n")
    (tmp_path / "run2.fastq").write_text("@r\nACGT\n+\nIIII\n")
    cfg = tmp_path / "plate_map.toml"
    cfg.write_text(TEN_PLATE_TOML)
    return cfg


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestLoad:

    def test_loads_both_segments(self, ten_plate_config):
        segments = load_plate_map(ten_plate_config)
        assert [s.name for s in segments] == ["run1", "run2"]

    def test_barcode_and_sort_plates(self, ten_plate_config):
        run1, run2 = load_plate_map(ten_plate_config)
        assert run1.barcode_plates == [1, 2, 3, 4, 5, 6]
        assert run1.sort_plates == [1, 2, 3, 4, 5, 6]
        # Barcode plates 1 and 2 reappear, now standing for sort plates 9, 10.
        assert run2.barcode_plates == [1, 2, 7, 8]
        assert run2.sort_plates == [7, 8, 9, 10]

    def test_ten_sort_plates_covered(self, ten_plate_config):
        segments = load_plate_map(ten_plate_config)
        covered = sorted(p for s in segments for p in s.sort_plates)
        assert covered == list(range(1, 11))
        assert total_sort_plates(segments) == 10

    def test_relative_paths_resolve_against_config(self, ten_plate_config):
        segments = load_plate_map(ten_plate_config)
        assert segments[0].path == ten_plate_config.parent / "run1.fastq"
        assert segments[0].path.exists()

    def test_name_defaults_to_file_stem(self, tmp_path):
        cfg = tmp_path / "pm.toml"
        cfg.write_text('[[fastq]]\npath = "readsA.fastq"\nplates = { 1 = 1 }\n')
        assert load_plate_map(cfg)[0].name == "readsA"

    def test_rbc_count_covers_highest_barcode_plate(self, ten_plate_config):
        run1, run2 = load_plate_map(ten_plate_config)
        assert run1.n_rbc == 24          # plates 1-6
        # run2 needs RB01-RB32 because it uses plate 8, even though plates
        # 3-6 are not in its pool.
        assert run2.n_rbc == 32

    def test_describe_reads_in_sort_plate_order(self, ten_plate_config):
        _, run2 = load_plate_map(ten_plate_config)
        assert run2.describe() == "barcode 7,8,1,2 -> sort 7,8,9,10"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def _cfg(self, tmp_path, body):
        cfg = tmp_path / "pm.toml"
        cfg.write_text(body)
        return cfg

    def test_barcode_plate_above_kit_limit_rejected(self, tmp_path):
        """The kit has 8 barcode plates; 9 would silently produce no wells."""
        cfg = self._cfg(tmp_path, '[[fastq]]\npath = "a.fastq"\nplates = { 9 = 9 }\n')
        with pytest.raises(PlateMapError, match="out of range"):
            load_plate_map(cfg)

    def test_error_names_the_kit_limit(self, tmp_path):
        cfg = self._cfg(tmp_path, '[[fastq]]\npath = "a.fastq"\nplates = { 12 = 1 }\n')
        with pytest.raises(PlateMapError, match=str(MAX_BARCODE_PLATES)):
            load_plate_map(cfg)

    def test_same_sort_plate_from_two_fastqs_rejected(self, tmp_path):
        """Two FASTQs claiming one sort plate would merge into the same wells."""
        cfg = self._cfg(tmp_path, (
            '[[fastq]]\nname = "a"\npath = "a.fastq"\nplates = { 1 = 5 }\n'
            '[[fastq]]\nname = "b"\npath = "b.fastq"\nplates = { 2 = 5 }\n'
        ))
        with pytest.raises(PlateMapError, match="Sort plate 5"):
            load_plate_map(cfg)

    def test_duplicate_sort_plate_within_one_fastq_rejected(self, tmp_path):
        cfg = self._cfg(tmp_path, (
            '[[fastq]]\npath = "a.fastq"\nplates = { 1 = 3, 2 = 3 }\n'
        ))
        with pytest.raises(PlateMapError, match="more than one"):
            load_plate_map(cfg)

    def test_duplicate_segment_name_rejected(self, tmp_path):
        cfg = self._cfg(tmp_path, (
            '[[fastq]]\nname = "x"\npath = "a.fastq"\nplates = { 1 = 1 }\n'
            '[[fastq]]\nname = "x"\npath = "b.fastq"\nplates = { 2 = 2 }\n'
        ))
        with pytest.raises(PlateMapError, match="duplicate segment name"):
            load_plate_map(cfg)

    def test_missing_path_rejected(self, tmp_path):
        cfg = self._cfg(tmp_path, "[[fastq]]\nplates = { 1 = 1 }\n")
        with pytest.raises(PlateMapError, match="missing 'path'"):
            load_plate_map(cfg)

    def test_missing_plates_rejected(self, tmp_path):
        cfg = self._cfg(tmp_path, '[[fastq]]\npath = "a.fastq"\n')
        with pytest.raises(PlateMapError, match="plates"):
            load_plate_map(cfg)

    def test_no_entries_rejected(self, tmp_path):
        cfg = self._cfg(tmp_path, "# nothing here\n")
        with pytest.raises(PlateMapError, match="No \\[\\[fastq\\]\\] entries"):
            load_plate_map(cfg)

    def test_invalid_toml_names_the_file(self, tmp_path):
        cfg = self._cfg(tmp_path, "[[fastq]\npath = broken\n")
        with pytest.raises(PlateMapError, match="invalid TOML"):
            load_plate_map(cfg)

    def test_zero_sort_plate_rejected(self, tmp_path):
        cfg = self._cfg(tmp_path, '[[fastq]]\npath = "a.fastq"\nplates = { 1 = 0 }\n')
        with pytest.raises(PlateMapError, match="must be 1 or greater"):
            load_plate_map(cfg)

    def test_non_numeric_sort_plate_rejected(self, tmp_path):
        cfg = self._cfg(tmp_path, '[[fastq]]\npath = "a.fastq"\nplates = { 1 = "x" }\n')
        with pytest.raises(PlateMapError, match="must be a number"):
            load_plate_map(cfg)


# ---------------------------------------------------------------------------
# Identity mapping (the ordinary single-FASTQ run)
# ---------------------------------------------------------------------------

class TestIdentity:

    def test_maps_each_plate_to_itself(self, tmp_path):
        seg = identity_segment(tmp_path / "reads.fastq", n_plates=6)
        assert seg.plates == {i: i for i in range(1, 7)}

    def test_clamped_to_kit_limit(self, tmp_path):
        seg = identity_segment(tmp_path / "reads.fastq", n_plates=20)
        assert seg.barcode_plates == list(range(1, MAX_BARCODE_PLATES + 1))

    def test_at_least_one_plate(self, tmp_path):
        assert identity_segment(tmp_path / "r.fastq", n_plates=0).plates == {1: 1}


# ---------------------------------------------------------------------------
# Round-tripping the config
# ---------------------------------------------------------------------------

class TestRoundTrip:

    def test_written_config_reloads_identically(self, ten_plate_config, tmp_path):
        original = load_plate_map(ten_plate_config)
        out = write_plate_map(original, tmp_path / "saved.toml")
        reloaded = load_plate_map(out)

        assert [s.name for s in reloaded] == [s.name for s in original]
        assert [s.plates for s in reloaded] == [s.plates for s in original]

    def test_rendered_toml_is_commented(self, ten_plate_config):
        text = format_plate_map_toml(load_plate_map(ten_plate_config))
        assert text.startswith("#")
        assert "--plate-map" in text


# ---------------------------------------------------------------------------
# Applying the mapping to barcodes
# ---------------------------------------------------------------------------

class TestBarcodeToWell:

    FASTQ1 = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
    FASTQ2 = {7: 7, 8: 8, 1: 9, 2: 10}

    def test_without_map_barcode_plate_is_sort_plate(self):
        assert barcode_to_well("FB01", "RB05") == "2A1"

    def test_identity_map_matches_no_map(self):
        assert (barcode_to_well("FB01", "RB05", self.FASTQ1)
                == barcode_to_well("FB01", "RB05"))

    def test_reused_barcode_plate_becomes_later_sort_plate(self):
        """The whole point: RB01 is sort plate 1 in one FASTQ, 9 in the other."""
        assert barcode_to_well("FB01", "RB01", self.FASTQ1) == "1A1"
        assert barcode_to_well("FB01", "RB01", self.FASTQ2) == "9A1"
        assert barcode_to_well("FB01", "RB05", self.FASTQ2) == "10A1"

    def test_high_barcode_plates_unchanged(self):
        assert barcode_to_well("FB01", "RB25", self.FASTQ2) == "7A1"
        assert barcode_to_well("FB01", "RB29", self.FASTQ2) == "8A1"

    def test_barcode_plate_outside_pool_is_dropped(self):
        """Plates 3-6 are not in FASTQ 2's pool, so a hit there is carry-over."""
        assert barcode_to_well("FB01", "RB09", self.FASTQ2) is None
        assert barcode_to_well("FB01", "RB25", self.FASTQ1) is None

    def test_well_coordinates_are_untouched_by_remap(self):
        """Only the plate number changes; row/column come from the barcodes."""
        plain = barcode_to_well("FB40", "RB02")
        remapped = barcode_to_well("FB40", "RB02", self.FASTQ2)
        assert plain[1:] == remapped[1:]
        assert plain.startswith("1") and remapped.startswith("9")


class TestBarcodeGeneration:
    """Reverse barcodes are generated contiguously from RB01, so a segment
    must generate enough of them to reach its highest barcode plate — even
    when the plates in between are not part of its pool."""

    def test_segment_generates_barcodes_up_to_its_highest_plate(self, tmp_path):
        from usortm.demux.barcodes import write_levseq_rbc_fasta
        from Bio import SeqIO

        run2 = Segment(name="run2", path=tmp_path / "r.fastq",
                       plates={7: 7, 8: 8, 1: 9, 2: 10})
        fasta = write_levseq_rbc_fasta(tmp_path / "cfg", n_barcodes=run2.n_rbc)
        names = [rec.id for rec in SeqIO.parse(str(fasta), "fasta")]

        assert len(names) == 32
        # Barcode plate 8 lives on reverse barcodes 29-32; without those the
        # reads for sort plate 8 would never be classified.
        assert "LevSeq-rbc-29" in names
        assert "LevSeq-rbc-32" in names

    def test_six_plate_segment_stops_at_24(self, tmp_path):
        from usortm.demux.barcodes import write_levseq_rbc_fasta
        from Bio import SeqIO

        run1 = Segment(name="run1", path=tmp_path / "r.fastq",
                       plates={i: i for i in range(1, 7)})
        fasta = write_levseq_rbc_fasta(tmp_path / "cfg", n_barcodes=run1.n_rbc)
        assert len(list(SeqIO.parse(str(fasta), "fasta"))) == 24

    def test_every_pool_plate_is_reachable(self, tmp_path):
        """Each barcode plate in the pool maps to a well within the generated
        barcode range."""
        run2 = Segment(name="run2", path=tmp_path / "r.fastq",
                       plates={7: 7, 8: 8, 1: 9, 2: 10})
        for bc_plate in run2.barcode_plates:
            first_rb = (bc_plate - 1) * 4 + 1
            assert first_rb <= run2.n_rbc
            well = barcode_to_well("FB01", f"RB{first_rb:02d}", run2.plates)
            assert well is not None
            assert well.startswith(str(run2.plates[bc_plate]))


class TestInteractivePairParsing:
    """The prompt collects pairs as free text, so parsing carries the risk."""

    def _parse(self, text):
        from usortm.cli.demux_cmd import _parse_plate_pairs
        return _parse_plate_pairs(text)

    def test_parses_the_ten_plate_second_fastq(self):
        assert self._parse("7:7, 8:8, 1:9, 2:10") == {7: 7, 8: 8, 1: 9, 2: 10}

    def test_tolerates_spacing_and_separators(self):
        assert self._parse(" 1 : 9 ;2:10 ") == {1: 9, 2: 10}

    def test_accepts_equals_as_separator(self):
        assert self._parse("1=9, 2=10") == {1: 9, 2: 10}

    def test_single_pair(self):
        assert self._parse("3:4") == {3: 4}

    def test_rejects_text_without_a_separator(self):
        with pytest.raises(PlateMapError, match="not a barcode:sort pair"):
            self._parse("1 9")

    def test_rejects_barcode_plate_beyond_the_kit(self):
        with pytest.raises(PlateMapError, match="out of range"):
            self._parse("9:9")

    def test_rejects_duplicate_sort_plate(self):
        with pytest.raises(PlateMapError, match="more than one"):
            self._parse("1:5, 2:5")

    def test_shares_validation_with_the_config_file(self):
        """Typed pairs and file entries must be judged by the same rules."""
        typed = self._parse("7:7, 8:8, 1:9, 2:10")
        from_file = parse_plate_map({"fastq": [{
            "path": "x", "plates": {"7": 7, "8": 8, "1": 9, "2": 10},
        }]})[0].plates
        assert typed == from_file


class TestFormatDf:

    def _reads(self, rbc_names):
        return pd.DataFrame({
            "read_name": [f"r{i}" for i in range(len(rbc_names))],
            "fbc_name": ["FB01"] * len(rbc_names),
            "rbc_name": rbc_names,
            "ref_name": ["fwd:var_001"] * len(rbc_names),
        })

    def test_plate_map_applied_to_well_positions(self):
        df = format_df(self._reads(["RB01", "RB05"]), plate_map={1: 9, 2: 10})
        assert sorted(df["well_pos"]) == ["10A1", "9A1"]

    def test_off_pool_reads_get_no_well(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            df = format_df(self._reads(["RB01", "RB09"]), plate_map={1: 9})
        assert df["well_pos"].isna().sum() == 1
        assert "carry-over" in caplog.text

    def test_no_plate_map_leaves_behaviour_unchanged(self):
        df = format_df(self._reads(["RB01", "RB05"]))
        assert sorted(df["well_pos"]) == ["1A1", "2A1"]
