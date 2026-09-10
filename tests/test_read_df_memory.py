"""The per-read sequences are freed once the per-well FASTQs hold them.

They are 94% of the table -- 16 GB against 1 GB on a run of 3.6M reads at a
median 2,028 bp -- and nothing after the FASTQs are written reads them from
here.  Carried to the end of a run they put the machine into swap, which is
what made consensus generation take an hour rather than a quarter of one.
"""
import pytest

pd = pytest.importorskip("pandas")

from usortm.demux.utils import drop_read_sequences, write_per_well_fastqs


def _df(n=8):
    return pd.DataFrame({
        "read_name": [f"r{i}" for i in range(n)],
        "well_pos": ["1A1"] * n,
        "read_seq": ["ACGT" * 400] * n,
        "read_qual": ["I" * 1600] * n,
        "ref_name": ["fwd:v1"] * n,
    })


def test_the_sequences_go_once_the_fastqs_are_written(tmp_path):
    df = _df()
    write_per_well_fastqs(df, str(tmp_path))

    slim = drop_read_sequences(df, str(tmp_path / "wells" / "fastqs"))
    assert "read_seq" not in slim.columns
    assert "read_qual" not in slim.columns
    # What later stages do need is still there.
    assert list(slim["read_name"]) == list(df["read_name"])
    assert "well_pos" in slim.columns


def test_the_table_gets_smaller(tmp_path):
    df = _df(200)
    write_per_well_fastqs(df, str(tmp_path))
    before = df.memory_usage(deep=True).sum()
    after = drop_read_sequences(
        df, str(tmp_path / "wells" / "fastqs")).memory_usage(deep=True).sum()
    assert after < before / 2


def test_without_fastqs_the_sequences_stay(tmp_path):
    """The fallback in assign_variants_from_reads still reads these columns.

    Dropping them with nothing to read back from would turn a slow path into
    a broken one.
    """
    df = _df()
    kept = drop_read_sequences(df, str(tmp_path / "does_not_exist"))
    assert "read_seq" in kept.columns
    assert "read_qual" in kept.columns


def test_an_empty_fastq_directory_counts_as_no_fastqs(tmp_path):
    empty = tmp_path / "fastqs"
    empty.mkdir()
    kept = drop_read_sequences(_df(), str(empty))
    assert "read_seq" in kept.columns


def test_dropping_twice_is_harmless(tmp_path):
    df = _df()
    write_per_well_fastqs(df, str(tmp_path))
    d = str(tmp_path / "wells" / "fastqs")
    once = drop_read_sequences(df, d)
    assert list(drop_read_sequences(once, d).columns) == list(once.columns)
