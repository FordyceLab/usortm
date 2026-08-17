"""Characterization tests for ``utils.create_read_df``.

``create_read_df`` merges three independent sources into one per-read table:
Dorado forward-barcode output (``fbc/``), Dorado reverse-barcode output
(``rbc/``), and reference/direction assignments from the alignment step.

These tests pin down its current, observable behaviour — column layout,
read-name normalisation, barcode index conventions, and which reads make it
into the table at all.  They exist so the read table can later be rebuilt to
stream sequences to disk instead of holding them in memory, without silently
changing what the pipeline sees.
"""

import gzip
import os

import pandas as pd
import pytest

from usortm.demux.utils import create_read_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_fastq(path, records, gzipped=False):
    """Write ``records`` as a FASTQ file.

    Args:
        path: Destination path.
        records: Iterable of ``(read_id, seq, qual)`` tuples.
        gzipped: Write gzip-compressed output.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = "".join(f"@{rid}\n{seq}\n+\n{qual}\n" for rid, seq, qual in records)
    open_fn = gzip.open if gzipped else open
    with open_fn(path, "wt") as fh:
        fh.write(text)


def _oriented(base_dir, records, gzipped=False):
    """Write an oriented FASTQ alongside *base_dir* and return its path."""
    name = "oriented_reads.fastq.gz" if gzipped else "oriented_reads.fastq"
    path = os.path.join(base_dir, "alignment", name)
    _write_fastq(path, records, gzipped=gzipped)
    return path


def _by_name(df):
    """Index a read DataFrame by read_name for order-independent assertions."""
    return df.set_index("read_name")


@pytest.fixture
def simple_run(tmp_path):
    """A minimal three-read run with full FBC + RBC + reference agreement.

    read_a and read_b land in barcode01/barcode01; read_c in barcode02/barcode03.
    """
    base = str(tmp_path)
    _write_fastq(
        os.path.join(base, "fbc", "barcode01.fastq"),
        [("read_a", "ACGT", "IIII"), ("read_b", "ACGTA", "IIIII")],
    )
    _write_fastq(
        os.path.join(base, "fbc", "barcode02.fastq"),
        [("read_c", "TTTT", "IIII")],
    )
    _write_fastq(
        os.path.join(base, "rbc", "barcode01.fastq"),
        [("read_a", "ACGT", "IIII"), ("read_b", "ACGTA", "IIIII")],
    )
    _write_fastq(
        os.path.join(base, "rbc", "barcode03.fastq"),
        [("read_c", "TTTT", "IIII")],
    )

    ref_map = {
        "read_a": {"ref": "var_001", "direction": "fwd"},
        "read_b": {"ref": "var_001", "direction": "rev"},
        "read_c": {"ref": "var_002", "direction": "fwd"},
    }
    oriented = _oriented(base, [
        ("read_a|ref=var_001|dir=fwd", "ACGT", "IIII"),
        ("read_b|ref=var_001|dir=rev", "ACGTA", "IIIII"),
        ("read_c|ref=var_002|dir=fwd", "TTTT", "IIII"),
    ])
    return base, ref_map, oriented


# ---------------------------------------------------------------------------
# Column contract
# ---------------------------------------------------------------------------

class TestColumnContract:

    def test_expected_columns_present(self, simple_run):
        base, ref_map, oriented = simple_run
        df = create_read_df(base, ref_map=ref_map, oriented_fastq=oriented)

        assert set(df.columns) == {
            "read_name", "fbc", "rbc", "ref_name",
            "read_seq", "read_qual", "avg_qual",
        }

    def test_one_row_per_read(self, simple_run):
        base, ref_map, oriented = simple_run
        df = create_read_df(base, ref_map=ref_map, oriented_fastq=oriented)

        assert len(df) == 3
        assert set(df["read_name"]) == {"read_a", "read_b", "read_c"}

    def test_sequences_and_quals_recovered(self, simple_run):
        base, ref_map, oriented = simple_run
        df = _by_name(create_read_df(base, ref_map=ref_map, oriented_fastq=oriented))

        assert df.loc["read_a", "read_seq"] == "ACGT"
        assert df.loc["read_a", "read_qual"] == "IIII"
        assert df.loc["read_b", "read_seq"] == "ACGTA"

    def test_avg_qual_is_mean_phred(self, tmp_path):
        """avg_qual averages Phred scores, not the raw ASCII characters."""
        base = str(tmp_path)
        # '!' is Q0, 'I' is Q40 -> mean 20.0
        _write_fastq(
            os.path.join(base, "fbc", "barcode01.fastq"),
            [("read_a", "ACGT", "!!II")],
        )
        ref_map = {"read_a": {"ref": "var_001", "direction": "fwd"}}
        oriented = _oriented(base, [("read_a|ref=var_001|dir=fwd", "ACGT", "!!II")])

        df = _by_name(create_read_df(base, ref_map=ref_map, oriented_fastq=oriented))
        assert df.loc["read_a", "avg_qual"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Barcode parsing
# ---------------------------------------------------------------------------

class TestBarcodeParsing:

    def test_barcode_index_is_zero_based(self, simple_run):
        """Dorado emits barcodeNN 1-based; the table stores NN-1."""
        base, ref_map, oriented = simple_run
        df = _by_name(create_read_df(base, ref_map=ref_map, oriented_fastq=oriented))

        assert df.loc["read_a", "fbc"] == 0    # barcode01
        assert df.loc["read_a", "rbc"] == 0    # barcode01
        assert df.loc["read_c", "fbc"] == 1    # barcode02
        assert df.loc["read_c", "rbc"] == 2    # barcode03

    def test_high_barcode_numbers(self, tmp_path):
        base = str(tmp_path)
        _write_fastq(
            os.path.join(base, "fbc", "barcode96.fastq"),
            [("read_a", "ACGT", "IIII")],
        )
        ref_map = {"read_a": {"ref": "var_001", "direction": "fwd"}}
        oriented = _oriented(base, [("read_a|ref=var_001|dir=fwd", "ACGT", "IIII")])

        df = _by_name(create_read_df(base, ref_map=ref_map, oriented_fastq=oriented))
        assert df.loc["read_a", "fbc"] == 95

    # NOTE: these two tests deliberately avoid the words "unclassified" and
    # "barcode<digits>" in their names.  pytest derives tmp_path from the test
    # name, and create_read_df matches both patterns against the *whole* path,
    # so a test named after them would poison its own fixture directory and
    # pass vacuously.  See TestPathSensitivity below.

    def test_decoy_bin_is_skipped(self, tmp_path):
        """Dorado's unclassified bin must not be read as a barcode."""
        base = str(tmp_path)
        _write_fastq(
            os.path.join(base, "fbc", "barcode01.fastq"),
            [("read_a", "ACGT", "IIII")],
        )
        _write_fastq(
            os.path.join(base, "fbc", "unclassified.fastq"),
            [("read_junk", "GGGG", "IIII")],
        )
        ref_map = {"read_a": {"ref": "var_001", "direction": "fwd"}}
        oriented = _oriented(base, [("read_a|ref=var_001|dir=fwd", "ACGT", "IIII")])

        df = create_read_df(base, ref_map=ref_map, oriented_fastq=oriented)
        assert "read_junk" not in set(df["read_name"])
        # ...and the real barcode file was still read.
        assert _by_name(df).loc["read_a", "fbc"] == 0

    def test_decoy_bin_in_numbered_dir_skipped(self, tmp_path):
        """The barcode number is matched against the whole path, not just the
        file name, so a decoy bin nested under a numbered barcode directory
        would otherwise be mis-assigned to that barcode."""
        base = str(tmp_path)
        _write_fastq(
            os.path.join(base, "fbc", "barcode07", "unclassified.fastq"),
            [("read_junk", "GGGG", "IIII")],
        )
        _write_fastq(
            os.path.join(base, "fbc", "barcode07", "reads.fastq"),
            [("read_a", "ACGT", "IIII")],
        )
        ref_map = {"read_a": {"ref": "var_001", "direction": "fwd"}}
        oriented = _oriented(base, [("read_a|ref=var_001|dir=fwd", "ACGT", "IIII")])

        df = create_read_df(base, ref_map=ref_map, oriented_fastq=oriented)
        names = set(df["read_name"])
        assert "read_junk" not in names
        assert _by_name(df).loc["read_a", "fbc"] == 6

    def test_nested_barcode_directories_are_found(self, tmp_path):
        """Dorado may nest output one level deep; the glob is recursive."""
        base = str(tmp_path)
        _write_fastq(
            os.path.join(base, "fbc", "run1", "barcode04.fastq"),
            [("read_a", "ACGT", "IIII")],
        )
        ref_map = {"read_a": {"ref": "var_001", "direction": "fwd"}}
        oriented = _oriented(base, [("read_a|ref=var_001|dir=fwd", "ACGT", "IIII")])

        df = _by_name(create_read_df(base, ref_map=ref_map, oriented_fastq=oriented))
        assert df.loc["read_a", "fbc"] == 3


# ---------------------------------------------------------------------------
# Path sensitivity (characterisation — current behaviour is surprising)
# ---------------------------------------------------------------------------

class TestPathSensitivity:
    """Barcode parsing inspects the full absolute path of each FASTQ.

    Both the ``unclassified`` guard and the ``barcode(\\d+)`` match are applied
    to the whole path rather than to the file's location relative to
    *base_dir*.  That makes the result depend on what the enclosing project
    directory happens to be called.  These tests document the current
    behaviour so a refactor cannot change it unnoticed.
    """

    def test_enclosing_dir_named_unclassified_drops_all_barcodes(self, tmp_path):
        """A project path containing 'unclassified' silently skips every
        barcode file, yielding no FBC assignments at all."""
        base = os.path.join(str(tmp_path), "unclassified_run", "demux")
        _write_fastq(
            os.path.join(base, "fbc", "barcode01.fastq"),
            [("read_a", "ACGT", "IIII")],
        )
        ref_map = {"read_a": {"ref": "var_001", "direction": "fwd"}}
        oriented = _oriented(base, [("read_a|ref=var_001|dir=fwd", "ACGT", "IIII")])

        df = create_read_df(base, ref_map=ref_map, oriented_fastq=oriented)
        assert df.attrs["fbc_classified"] == 0
        assert pd.isna(_by_name(df).loc["read_a", "fbc"])

    def test_enclosing_dir_with_barcode_number_wins_over_filename(self, tmp_path):
        """A project path containing 'barcode<digits>' captures that number
        for every file, because re.search takes the leftmost match."""
        base = os.path.join(str(tmp_path), "barcode99_experiment", "demux")
        _write_fastq(
            os.path.join(base, "fbc", "barcode01.fastq"),
            [("read_a", "ACGT", "IIII")],
        )
        ref_map = {"read_a": {"ref": "var_001", "direction": "fwd"}}
        oriented = _oriented(base, [("read_a|ref=var_001|dir=fwd", "ACGT", "IIII")])

        df = create_read_df(base, ref_map=ref_map, oriented_fastq=oriented)
        # Reads land in barcode 99 (from the directory), not 01 (from the file).
        assert _by_name(df).loc["read_a", "fbc"] == 98


# ---------------------------------------------------------------------------
# Read-name normalisation
# ---------------------------------------------------------------------------

class TestReadNameNormalisation:

    def test_ref_and_dir_tags_stripped(self, simple_run):
        """Oriented-FASTQ headers carry |ref=/|dir= tags that must be removed
        so they join against the untagged Dorado read names."""
        base, ref_map, oriented = simple_run
        df = create_read_df(base, ref_map=ref_map, oriented_fastq=oriented)

        for name in df["read_name"]:
            assert "|ref=" not in name
            assert "|dir=" not in name

    def test_mate_suffix_stripped(self, tmp_path):
        base = str(tmp_path)
        _write_fastq(
            os.path.join(base, "fbc", "barcode01.fastq"),
            [("read_a/1", "ACGT", "IIII")],
        )
        ref_map = {"read_a": {"ref": "var_001", "direction": "fwd"}}
        oriented = _oriented(base, [("read_a|ref=var_001|dir=fwd", "ACGT", "IIII")])

        df = create_read_df(base, ref_map=ref_map, oriented_fastq=oriented)
        # The /1 mate suffix is stripped, so this joins to the same read.
        assert set(df["read_name"]) == {"read_a"}
        assert len(df) == 1

    def test_tagged_and_untagged_names_join_to_one_row(self, simple_run):
        """A read tagged in the oriented FASTQ and untagged in Dorado output
        must collapse to a single row, not two."""
        base, ref_map, oriented = simple_run
        df = _by_name(create_read_df(base, ref_map=ref_map, oriented_fastq=oriented))

        # read_a has fbc, rbc, ref and sequence all populated on one row.
        row = df.loc["read_a"]
        assert row["fbc"] == 0
        assert row["rbc"] == 0
        assert row["ref_name"] == "fwd:var_001"
        assert row["read_seq"] == "ACGT"


# ---------------------------------------------------------------------------
# Reference assignment
# ---------------------------------------------------------------------------

class TestReferenceAssignment:

    def test_ref_name_carries_direction_prefix(self, simple_run):
        base, ref_map, oriented = simple_run
        df = _by_name(create_read_df(base, ref_map=ref_map, oriented_fastq=oriented))

        assert df.loc["read_a", "ref_name"] == "fwd:var_001"
        assert df.loc["read_b", "ref_name"] == "rev:var_001"
        assert df.loc["read_c", "ref_name"] == "fwd:var_002"

    def test_without_alignment_results_nothing_is_assigned(self, tmp_path, caplog):
        """Barcodes alone are not enough: with no ref_map the table has no
        reference and no sequence, and format_df drops every row.

        The pipeline must therefore always run the alignment stage first;
        the CLI enforces that by requiring a reference.
        """
        import logging
        from usortm.demux.utils import format_df

        base = str(tmp_path)
        _write_fastq(
            os.path.join(base, "fbc", "barcode01.fastq"),
            [("read_a", "ACGT", "IIII")],
        )
        _write_fastq(
            os.path.join(base, "rbc", "barcode01.fastq"),
            [("read_a", "ACGT", "IIII")],
        )

        with caplog.at_level(logging.WARNING):
            df = create_read_df(base)

        assert len(df) == 1
        assert df["ref_name"].isna().all()
        assert df["read_seq"].isna().all()
        assert "align_and_split_by_strand" in caplog.text

        # ...and the row does not survive well assignment.
        assert len(format_df(df, fbc_df=None, rbc_df=None, ref_fasta=None)) == 0


# ---------------------------------------------------------------------------
# Partial assignments and membership
# ---------------------------------------------------------------------------

class TestPartialAssignments:

    def test_read_missing_rbc_keeps_row_with_null(self, tmp_path):
        """A read classified by FBC only still gets a row; format_df is what
        later drops incomplete assignments, not create_read_df."""
        base = str(tmp_path)
        _write_fastq(
            os.path.join(base, "fbc", "barcode01.fastq"),
            [("read_a", "ACGT", "IIII")],
        )
        ref_map = {"read_a": {"ref": "var_001", "direction": "fwd"}}
        oriented = _oriented(base, [("read_a|ref=var_001|dir=fwd", "ACGT", "IIII")])

        df = _by_name(create_read_df(base, ref_map=ref_map, oriented_fastq=oriented))
        assert df.loc["read_a", "fbc"] == 0
        assert pd.isna(df.loc["read_a", "rbc"])

    def test_read_with_barcodes_but_no_alignment_has_null_ref(self, tmp_path):
        base = str(tmp_path)
        _write_fastq(
            os.path.join(base, "fbc", "barcode01.fastq"),
            [("read_a", "ACGT", "IIII")],
        )
        _write_fastq(
            os.path.join(base, "rbc", "barcode01.fastq"),
            [("read_a", "ACGT", "IIII")],
        )
        oriented = _oriented(base, [])

        df = _by_name(create_read_df(base, ref_map={}, oriented_fastq=oriented))
        assert pd.isna(df.loc["read_a", "ref_name"])
        # No oriented sequence for it either.
        assert pd.isna(df.loc["read_a", "read_seq"])

    def test_read_only_in_oriented_fastq_is_excluded(self, tmp_path):
        """Table membership is the union of FBC, RBC and reference
        assignments — a sequence alone does not create a row."""
        base = str(tmp_path)
        _write_fastq(
            os.path.join(base, "fbc", "barcode01.fastq"),
            [("read_a", "ACGT", "IIII")],
        )
        ref_map = {"read_a": {"ref": "var_001", "direction": "fwd"}}
        oriented = _oriented(base, [
            ("read_a|ref=var_001|dir=fwd", "ACGT", "IIII"),
            ("read_ghost|ref=var_001|dir=fwd", "GGGG", "IIII"),
        ])

        df = create_read_df(base, ref_map=ref_map, oriented_fastq=oriented)
        assert set(df["read_name"]) == {"read_a"}


# ---------------------------------------------------------------------------
# Reported statistics
# ---------------------------------------------------------------------------

class TestAttrs:

    def test_classification_counts_reported(self, simple_run):
        base, ref_map, oriented = simple_run
        df = create_read_df(base, ref_map=ref_map, oriented_fastq=oriented)

        assert df.attrs["fbc_classified"] == 3
        assert df.attrs["rbc_classified"] == 3
        assert df.attrs["ref_assigned"] == 3

    def test_counts_track_partial_classification(self, tmp_path):
        base = str(tmp_path)
        _write_fastq(
            os.path.join(base, "fbc", "barcode01.fastq"),
            [("read_a", "ACGT", "IIII"), ("read_b", "ACGT", "IIII")],
        )
        _write_fastq(
            os.path.join(base, "rbc", "barcode01.fastq"),
            [("read_a", "ACGT", "IIII")],
        )
        ref_map = {"read_a": {"ref": "var_001", "direction": "fwd"}}
        oriented = _oriented(base, [("read_a|ref=var_001|dir=fwd", "ACGT", "IIII")])

        df = create_read_df(base, ref_map=ref_map, oriented_fastq=oriented)
        assert df.attrs["fbc_classified"] == 2
        assert df.attrs["rbc_classified"] == 1
        assert df.attrs["ref_assigned"] == 1


# ---------------------------------------------------------------------------
# Input format handling
# ---------------------------------------------------------------------------

class TestInputFormats:

    def test_gzipped_oriented_fastq(self, tmp_path):
        """The oriented FASTQ is detected by magic bytes, not extension."""
        base = str(tmp_path)
        _write_fastq(
            os.path.join(base, "fbc", "barcode01.fastq"),
            [("read_a", "ACGT", "IIII")],
        )
        ref_map = {"read_a": {"ref": "var_001", "direction": "fwd"}}
        oriented = _oriented(
            base, [("read_a|ref=var_001|dir=fwd", "ACGT", "IIII")], gzipped=True
        )

        df = _by_name(create_read_df(base, ref_map=ref_map, oriented_fastq=oriented))
        assert df.loc["read_a", "read_seq"] == "ACGT"

    def test_gzipped_dorado_output(self, tmp_path):
        base = str(tmp_path)
        _write_fastq(
            os.path.join(base, "fbc", "barcode01.fastq.gz"),
            [("read_a", "ACGT", "IIII")],
            gzipped=True,
        )
        ref_map = {"read_a": {"ref": "var_001", "direction": "fwd"}}
        oriented = _oriented(base, [("read_a|ref=var_001|dir=fwd", "ACGT", "IIII")])

        df = create_read_df(base, ref_map=ref_map, oriented_fastq=oriented)
        assert "read_a" in set(df["read_name"])


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------

class TestEmptyInputs:

    def test_no_inputs_returns_empty_frame(self, tmp_path):
        df = create_read_df(str(tmp_path), ref_map={}, oriented_fastq=None)
        assert len(df) == 0

    def test_empty_frame_reports_zero_counts(self, tmp_path):
        df = create_read_df(str(tmp_path), ref_map={}, oriented_fastq=None)
        assert df.attrs["fbc_classified"] == 0
        assert df.attrs["rbc_classified"] == 0
        assert df.attrs["ref_assigned"] == 0
