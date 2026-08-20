"""Tests for --orient-ref used without --vector-fasta.

Both options route the pipeline down the same branch, but only
--vector-fasta supplies the flanking sequences.  The concatemer filter is
derived from those flanks, so it has to be skipped when they are absent.
"""

import pytest

from usortm.demux.pipeline import run_levseq_pipeline


def _fake_reads():
    import pandas as pd

    return pd.DataFrame({
        "read_name": [f"r{i}" for i in range(4)],
        "fbc_name": ["FB01"] * 4,
        "rbc_name": ["RB01"] * 4,
        "well_pos": ["1A1"] * 4,
        "ref_name": ["fwd:var_1"] * 4,
        # A short read that a flank-derived cutoff would have dropped.
        "read_seq": ["ACGT" * 60, "ACGT" * 5, "ACGT" * 60, "ACGT" * 60],
        "read_qual": ["I" * 240, "I" * 20, "I" * 240, "I" * 240],
    })


class TestOrientRefWithoutVector:
    """A bare --orient-ref must not reach for flanks that were never parsed."""

    def test_concatemer_filter_is_skipped(self, tmp_path, monkeypatch):
        """Previously this raised TypeError: len(None)."""
        import pandas as pd
        from usortm.demux import pipeline as pl

        reference = tmp_path / "ref.fasta"
        reference.write_text(">var_1\n" + "ACGT" * 50 + "\n")
        orient_ref = tmp_path / "orient.fasta"
        orient_ref.write_text(">orient\n" + "ACGT" * 50 + "\n")

        read_df = _fake_reads()
        well_df = pd.DataFrame({
            "plate": [1], "well": ["A1"], "global_well": ["1A1"],
            "depth": [4], "major_ref": ["var_1"], "major_freq": [1.0],
            "ref_len": [200],
        })

        monkeypatch.setattr(pl, "check_all_dependencies",
                            lambda: {"dorado": "d", "minimap2": "m",
                                     "samtools": "s"})
        monkeypatch.setattr(pl.utils, "demux", lambda **kw: None)
        monkeypatch.setattr(pl.utils, "align_and_split_by_strand",
                            lambda **kw: ("oriented.fastq", {},
                                          {"mapped": 4, "unmapped": 0,
                                           "fwd": 4, "rev": 0}))
        monkeypatch.setattr(pl.utils, "create_read_df", lambda **kw: read_df)
        monkeypatch.setattr(pl.utils, "format_df", lambda *a, **kw: read_df)
        monkeypatch.setattr(pl.utils, "generate_well_df", lambda *a, **kw: well_df)
        monkeypatch.setattr(pl, "_compute_read_length_hist", lambda *a: {})

        captured = {}

        def _fake_write_per_well(df, out_root):
            captured["n_reads"] = len(df)

        monkeypatch.setattr(pl.utils, "write_per_well_fastqs", _fake_write_per_well)
        monkeypatch.setattr(pl.utils, "assign_variants_from_reads",
                            lambda wdf, *a, **kw: wdf)
        monkeypatch.setattr(pl.utils, "generate_per_well_consensus",
                            lambda wdf, *a, **kw: wdf)
        monkeypatch.setattr(pl.utils, "extract_matches", lambda wdf, **kw: wdf)
        monkeypatch.setattr(pl.utils, "detect_consensus_hotspots",
                            lambda *a, **kw: {})
        monkeypatch.setattr(
            "usortm.demux.streakout.detect_streakout_candidates",
            lambda *a, **kw: [],
        )

        results = run_levseq_pipeline(
            fastq=tmp_path / "reads.fastq",
            output_dir=tmp_path / "out",
            reference=reference,
            orient_ref=orient_ref,      # no vector_fasta
            n_plates=1,
            min_reads=1,
        )

        # Every read survives: without flanks there is no amplicon length to
        # filter against.
        assert captured["n_reads"] == 4
        # The well is reported even though four reads is below the depth at
        # which it counts as having data -- what is being tested is the
        # filter, not the depth threshold.
        assert results["well_assignments"]

    def test_no_flank_lengths_reported(self, tmp_path, monkeypatch):
        """flank_5p_len/3p_len are only meaningful with a vector."""
        import pandas as pd
        from usortm.demux import pipeline as pl

        reference = tmp_path / "ref.fasta"
        reference.write_text(">var_1\n" + "ACGT" * 50 + "\n")
        orient_ref = tmp_path / "orient.fasta"
        orient_ref.write_text(">orient\n" + "ACGT" * 50 + "\n")

        read_df = _fake_reads()
        well_df = pd.DataFrame({
            "plate": [1], "well": ["A1"], "global_well": ["1A1"],
            "depth": [4], "major_ref": ["var_1"], "major_freq": [1.0],
            "ref_len": [200],
        })
        monkeypatch.setattr(pl, "check_all_dependencies",
                            lambda: {"dorado": "d", "minimap2": "m",
                                     "samtools": "s"})
        monkeypatch.setattr(pl.utils, "demux", lambda **kw: None)
        monkeypatch.setattr(pl.utils, "align_and_split_by_strand",
                            lambda **kw: ("oriented.fastq", {},
                                          {"mapped": 4, "unmapped": 0,
                                           "fwd": 4, "rev": 0}))
        monkeypatch.setattr(pl.utils, "create_read_df", lambda **kw: read_df)
        monkeypatch.setattr(pl.utils, "format_df", lambda *a, **kw: read_df)
        monkeypatch.setattr(pl.utils, "generate_well_df", lambda *a, **kw: well_df)
        monkeypatch.setattr(pl, "_compute_read_length_hist", lambda *a: {})
        monkeypatch.setattr(pl.utils, "write_per_well_fastqs", lambda *a, **kw: None)
        monkeypatch.setattr(pl.utils, "assign_variants_from_reads",
                            lambda wdf, *a, **kw: wdf)
        monkeypatch.setattr(pl.utils, "generate_per_well_consensus",
                            lambda wdf, *a, **kw: wdf)
        monkeypatch.setattr(pl.utils, "extract_matches", lambda wdf, **kw: wdf)
        monkeypatch.setattr(pl.utils, "detect_consensus_hotspots",
                            lambda *a, **kw: {})
        monkeypatch.setattr(
            "usortm.demux.streakout.detect_streakout_candidates",
            lambda *a, **kw: [],
        )

        results = run_levseq_pipeline(
            fastq=tmp_path / "reads.fastq",
            output_dir=tmp_path / "out",
            reference=reference,
            orient_ref=orient_ref,
            n_plates=1,
            min_reads=1,
        )
        assert "flank_5p_len" not in results
