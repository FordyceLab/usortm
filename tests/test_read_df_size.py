"""Tests for what the per-read table carries, and who reads it.

The table is written once per run and read several times afterwards, so a
column nothing needs is paid for repeatedly. On a real run these two facts were
worth about two gigabytes of disk and a few minutes of wall clock.
"""

import pandas as pd
import pytest

from usortm.demux.utils import READ_DF_HEAVY_COLUMNS, write_read_df_csv


def _read_df(n=50):
    """A frame shaped like the pipeline's, with the heavy columns present."""
    ref_seq = "ACGT" * 400          # one reference, repeated on every row
    return pd.DataFrame({
        "read_name": [f"read_{i}" for i in range(n)],
        "fbc_name": ["bc01"] * n,
        "rbc_name": ["rb01"] * n,
        "well_pos": [f"1A{i % 12 + 1}" for i in range(n)],
        "ref_name": ["fwd:var_1"] * n,
        "avg_qual": [20.0] * n,
        "ref_id": ["var_1"] * n,
        "ref_seq": [ref_seq] * n,
        "ref_len": [len(ref_seq)] * n,
        "read_seq": ["ACGT" * 250] * n,
        "read_qual": ["I" * 1000] * n,
        "segment": ["run1"] * n,
    })


class TestWhatIsWritten:

    def test_the_sequence_columns_are_dropped(self, tmp_path):
        path = tmp_path / "read_df.csv"
        write_read_df_csv(_read_df(), path)
        cols = pd.read_csv(path, nrows=0).columns
        assert "read_seq" not in cols
        assert "read_qual" not in cols

    def test_the_reference_sequence_is_dropped(self, tmp_path):
        """ref_seq repeats one of a few hundred sequences on every row, and
        ref_id already names which one."""
        path = tmp_path / "read_df.csv"
        write_read_df_csv(_read_df(), path)
        assert "ref_seq" not in pd.read_csv(path, nrows=0).columns

    def test_what_identifies_a_read_survives(self, tmp_path):
        path = tmp_path / "read_df.csv"
        write_read_df_csv(_read_df(), path)
        cols = set(pd.read_csv(path, nrows=0).columns)
        for kept in ("read_name", "well_pos", "ref_name", "ref_id", "ref_len",
                     "avg_qual", "segment"):
            assert kept in cols, kept

    def test_every_row_survives(self, tmp_path):
        path = tmp_path / "read_df.csv"
        write_read_df_csv(_read_df(n=37), path)
        assert len(pd.read_csv(path)) == 37

    def test_the_caller_s_frame_is_left_alone(self, tmp_path):
        """Dropping happens on the way out, not in place."""
        df = _read_df()
        write_read_df_csv(df, tmp_path / "read_df.csv")
        assert "ref_seq" in df.columns
        assert "read_seq" in df.columns

    def test_a_frame_without_the_heavy_columns_still_writes(self, tmp_path):
        df = _read_df().drop(columns=list(READ_DF_HEAVY_COLUMNS))
        path = tmp_path / "read_df.csv"
        write_read_df_csv(df, path)
        assert len(pd.read_csv(path)) == 50

    def test_the_file_is_far_smaller(self, tmp_path):
        full = tmp_path / "full.csv"
        slim = tmp_path / "slim.csv"
        df = _read_df(n=200)
        df.to_csv(full, index=False)
        write_read_df_csv(df, slim)
        assert slim.stat().st_size < full.stat().st_size / 10


class TestPickDoesNotReadWhatItCannotUse:
    """When sequences are not in the table, the pileups come from the per-well
    FASTQs and the table's body is never needed."""

    def test_the_body_is_not_read_when_sequences_are_absent(self, tmp_path,
                                                            monkeypatch):
        from usortm.demux import streakout

        demux = tmp_path / "demux_output"
        (demux / "reference_fasta" / "single_ref_fastas").mkdir(parents=True)
        (demux / "wells" / "fastqs").mkdir(parents=True)

        rows = "\n".join(f"read_{i},1A1,fwd:var_1,var_1" for i in range(500))
        (demux / "read_df.csv").write_text(
            "read_name,well_pos,ref_name,ref_id\n" + rows + "\n"
        )

        calls = []
        real = pd.read_csv

        def spy(*args, **kwargs):
            calls.append(kwargs.get("nrows"))
            return real(*args, **kwargs)

        monkeypatch.setattr(streakout.pd, "read_csv", spy)

        streakout.generate_pick_pileups(
            pick_list=[{"source_plate": "1", "source_well": "A1",
                        "variant": "var_1", "reads": 10,
                        "consensus_fraction": 1.0, "target_plate": "1",
                        "target_well": "A1"}],
            demux_output_dir=str(demux),
            output_dir=str(tmp_path / "pick"),
            workers=1,
            minimap2_path="minimap2",
            samtools_path="samtools",
        )

        assert calls, "the table was never opened at all"
        assert all(n == 0 for n in calls), (
            f"the table's body was read: read_csv nrows={calls}"
        )

    def test_older_outputs_with_sequences_still_work(self, tmp_path):
        """A table that does carry sequences is still used as before."""
        from usortm.demux import streakout

        demux = tmp_path / "demux_output"
        (demux / "reference_fasta" / "single_ref_fastas").mkdir(parents=True)
        df = _read_df(n=4)
        df["well_pos"] = "1A1"
        df.to_csv(demux / "read_df.csv", index=False)

        # No reference FASTA for the variant, so nothing renders; the point is
        # that the sequence-carrying path is taken without error.
        out = streakout.generate_pick_pileups(
            pick_list=[{"source_plate": "1", "source_well": "A1",
                        "variant": "var_1", "reads": 4,
                        "consensus_fraction": 1.0, "target_plate": "1",
                        "target_well": "A1"}],
            demux_output_dir=str(demux),
            output_dir=str(tmp_path / "pick"),
            workers=1,
        )
        assert isinstance(out, dict)
