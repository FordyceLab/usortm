"""Tests for read-table row order and the two long stages' progress reports.

``create_read_df`` and ``write_per_well_fastqs`` are the two single-pass stages
over every read in a run.  Both take minutes on a real run, so both report how
far they have got; these tests pin that reporting down, along with the row
order the read table is built in.
"""

import os

import pandas as pd

from usortm.demux.utils import create_read_df, write_per_well_fastqs


def _write_fastq(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("".join(f"@{rid}\n{seq}\n+\n{qual}\n"
                         for rid, seq, qual in records))


def _run(tmp_path, n_reads):
    """A run of *n_reads* reads, all classified, in a known FASTQ order."""
    base = str(tmp_path)
    reads = [(f"read_{i:04d}", "ACGT", "IIII") for i in range(n_reads)]
    _write_fastq(os.path.join(base, "fbc", "barcode01.fastq"), reads)
    _write_fastq(os.path.join(base, "rbc", "barcode01.fastq"), reads)
    ref_map = {rid: {"ref": "var_001", "direction": "fwd"} for rid, _, _ in reads}
    _write_fastq(os.path.join(base, "alignment", "oriented_reads.fastq"),
                 [(f"{rid}|ref=var_001|dir=fwd", s, q) for rid, s, q in reads])
    return base, ref_map, os.path.join(base, "alignment", "oriented_reads.fastq")


class TestReadOrder:

    def test_rows_follow_the_oriented_fastq(self, tmp_path):
        """Row order is the FASTQ's, not the hash order of a set union.

        Per-well FASTQs are written in this order and the variant-assignment
        step reads the first reads of each well, so an order that varies
        between runs changes which reads that step sees.
        """
        base, ref_map, oriented = _run(tmp_path, 50)
        df = create_read_df(base, ref_map=ref_map, oriented_fastq=oriented)

        assert list(df["read_name"]) == [f"read_{i:04d}" for i in range(50)]

    def test_order_is_stable_across_calls(self, tmp_path):
        base, ref_map, oriented = _run(tmp_path, 50)
        first = create_read_df(base, ref_map=ref_map, oriented_fastq=oriented)
        second = create_read_df(base, ref_map=ref_map, oriented_fastq=oriented)

        assert list(first["read_name"]) == list(second["read_name"])

    def test_a_read_missing_from_the_fastq_still_gets_a_row(self, tmp_path):
        """Reads the barcode calls know about but the FASTQ lacks come last."""
        base, ref_map, oriented = _run(tmp_path, 3)
        ref_map["read_late"] = {"ref": "var_001", "direction": "fwd"}

        df = create_read_df(base, ref_map=ref_map, oriented_fastq=oriented)

        assert list(df["read_name"])[-1] == "read_late"
        assert pd.isna(df.set_index("read_name").loc["read_late", "read_seq"])


class TestQualityDecoding:

    def test_qual_string_is_the_fastq_line(self, tmp_path):
        base, ref_map, oriented = _run(tmp_path, 1)
        df = create_read_df(base, ref_map=ref_map, oriented_fastq=oriented)

        assert df.loc[0, "read_qual"] == "IIII"

    def test_avg_qual_is_mean_phred(self, tmp_path):
        """'!' is Q0 and 'I' is Q40, so a half-and-half read averages 20."""
        base = str(tmp_path)
        _write_fastq(os.path.join(base, "fbc", "barcode01.fastq"),
                     [("read_a", "ACGT", "!!II")])
        _write_fastq(os.path.join(base, "alignment", "oriented_reads.fastq"),
                     [("read_a|ref=var_001|dir=fwd", "ACGT", "!!II")])
        ref_map = {"read_a": {"ref": "var_001", "direction": "fwd"}}

        df = create_read_df(base, ref_map=ref_map,
                            oriented_fastq=os.path.join(
                                base, "alignment", "oriented_reads.fastq"))

        assert df.loc[0, "avg_qual"] == 20.0


class TestProgressReporting:

    def test_read_df_reports_progress_every_50k_reads(self, tmp_path):
        base, ref_map, oriented = _run(tmp_path, 100_001)
        seen = []

        create_read_df(base, ref_map=ref_map, oriented_fastq=oriented,
                       progress_callback=seen.append)

        assert seen == [50_000, 100_000]

    def test_read_df_progress_is_optional(self, tmp_path):
        base, ref_map, oriented = _run(tmp_path, 10)
        df = create_read_df(base, ref_map=ref_map, oriented_fastq=oriented)
        assert len(df) == 10

    def test_per_well_write_reports_wells_done_and_total(self, tmp_path):
        read_df = pd.DataFrame({
            "read_name": [f"read_{i}" for i in range(60)],
            "read_seq": ["ACGT"] * 60,
            "read_qual": ["IIII"] * 60,
            "well_pos": [f"1A{i + 1}" for i in range(60)],
        })
        seen = []

        write_per_well_fastqs(read_df, str(tmp_path),
                              progress_callback=lambda n, t: seen.append((n, t)))

        # Every 25 wells, plus a final call on the last well.
        assert seen == [(25, 60), (50, 60), (60, 60)]

    def test_per_well_write_still_writes_every_well(self, tmp_path):
        read_df = pd.DataFrame({
            "read_name": ["read_a", "read_b", "read_c"],
            "read_seq": ["ACGT", "TTTT", "GGGG"],
            "read_qual": ["IIII", "IIII", "IIII"],
            "well_pos": ["1A1", "1A1", "1B2"],
        })

        write_per_well_fastqs(read_df, str(tmp_path))

        fastqs = os.path.join(str(tmp_path), "wells", "fastqs")
        assert sorted(os.listdir(fastqs)) == ["1A1.fastq", "1B2.fastq"]
        assert open(os.path.join(fastqs, "1A1.fastq")).read() == (
            "@read_a\nACGT\n+\nIIII\n@read_b\nTTTT\n+\nIIII\n"
        )
