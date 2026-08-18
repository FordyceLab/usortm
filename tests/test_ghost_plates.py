"""Tests for the ghost-plate filter on the demux plate map.

The filter exists to drop plates that were never loaded but picked up a few
switched barcodes. It previously split well_pos on "_", a separator that is
not there, which made it a per-well depth filter instead — so a real plate
whose wells were individually shallow disappeared from the plate map even
though its reads were in the table.
"""

import pandas as pd

from usortm.cli.demux_cmd import GHOST_PLATE_MIN_READS, _drop_ghost_plates


def _reads(spec):
    """Build a read table from {well_pos: n_reads}."""
    wells = [w for w, n in spec.items() for _ in range(n)]
    return pd.DataFrame({
        "well_pos": wells,
        "ref_name": ["fwd:var_1"] * len(wells),
    })


def _plates(df):
    return sorted({w[:-1].rstrip("ABCDEFGHIJKLMNOP") or w[0]
                   for w in df["well_pos"]}, key=int) if len(df) else []


class TestGhostPlateFilter:

    def test_shallow_wells_do_not_hide_a_real_plate(self):
        """The regression: 200 wells of 7 reads is a real plate, not a ghost."""
        spec = {f"1{chr(65 + i // 24)}{i % 24 + 1}": 7 for i in range(200)}
        df = _reads(spec)

        kept = _drop_ghost_plates(df)
        assert len(kept) == len(df)
        assert kept["well_pos"].str.startswith("1").all()

    def test_genuinely_empty_plate_is_dropped(self):
        df = _reads({**{f"1A{i}": 30 for i in range(1, 6)},   # plate 1: 150
                     "2A1": 3, "2B2": 4})                      # plate 2: 7
        kept = _drop_ghost_plates(df)

        surviving = {w[0] for w in kept["well_pos"]}
        assert surviving == {"1"}

    def test_threshold_is_per_plate_total(self):
        below = _reads({f"3A{i}": 1 for i in range(1, GHOST_PLATE_MIN_READS)})
        assert _drop_ghost_plates(below).empty

        at = _reads({f"3A{i}": 1 for i in range(1, GHOST_PLATE_MIN_READS + 1)})
        assert len(_drop_ghost_plates(at)) == GHOST_PLATE_MIN_READS

    def test_multi_digit_plate_numbers_parse(self):
        """A ten-plate run has wells like '10P24'; the plate is '10', not '1'."""
        df = _reads({**{f"1A{i}": 30 for i in range(1, 4)},
                     **{f"10P{i}": 30 for i in range(1, 4)}})
        kept = _drop_ghost_plates(df)

        assert len(kept) == len(df)
        plates = {
            w[:2] if w.startswith("10") else w[0] for w in kept["well_pos"]
        }
        assert plates == {"1", "10"}

    def test_plate_10_is_not_confused_with_plate_1(self):
        """Plate 1 real, plate 10 a ghost — they must be counted separately."""
        df = _reads({**{f"1A{i}": 30 for i in range(1, 6)},   # plate 1: 150
                     "10A1": 2})                               # plate 10: 2
        kept = _drop_ghost_plates(df)

        assert "10A1" not in set(kept["well_pos"])
        assert len(kept) == 150

    def test_empty_frame_passes_through(self):
        empty = pd.DataFrame({"well_pos": [], "ref_name": []})
        assert _drop_ghost_plates(empty).empty

    def test_frame_without_well_pos_passes_through(self):
        df = pd.DataFrame({"read_name": ["r1", "r2"]})
        assert len(_drop_ghost_plates(df)) == 2
