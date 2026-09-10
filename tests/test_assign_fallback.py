"""A well the translation cannot name is named by aligning its reads.

Translation reads a well's variant off its consensus protein, and names it
only when that differs from the parent at exactly one residue.  A well
carrying an indel differs at every residue after it, so it was left
unassigned -- and an unassigned well gets no reference, so no consensus, so no
QC value at all.  Round 2 had two such wells, 1D11 and 1E16, each holding the
construct its layout expected and each named confidently by the alignment.

Only those wells are re-aligned: against a whole library that path costs about
45 ms a read, so paying it for a plate would undo the reason translation is
used first.
"""
import pytest

pd = pytest.importorskip("pandas")

from usortm.demux.pipeline import (_RETRY_MIN_DEPTH,
                                   _name_the_rest_by_alignment)


class _Spy:
    """Stands in for the alignment assignment and records its arguments."""

    def __init__(self, names=None):
        self.calls = []
        self.names = names or {}

    def __call__(self, well_df, read_df, ref, **kw):
        self.calls.append(kw)
        out = well_df.copy()
        for well, name in self.names.items():
            out.loc[out["global_well"] == well, "major_ref"] = name
        return out


def _frame():
    return pd.DataFrame({
        "global_well": ["1A1", "1D11", "1E16", "1B8"],
        "major_ref": ["G3F", "unassigned", "unassigned", "unassigned"],
        "depth": [98, 263, 238, 0],
    })


def _run(monkeypatch, spy, df=None):
    from usortm.demux import utils
    monkeypatch.setattr(utils, "assign_variants_from_reads", spy)
    return _name_the_rest_by_alignment(
        df if df is not None else _frame(), pd.DataFrame(), "lib.fasta",
        "/fastqs", __import__("pathlib").Path("/refs"),
        {"minimap2": "minimap2"}, 2, 20, lambda *_: None,
    )


def test_only_the_unnamed_wells_are_re_aligned(monkeypatch):
    spy = _Spy()
    _run(monkeypatch, spy)
    assert len(spy.calls) == 1
    assert sorted(spy.calls[0]["wells"]) == ["1D11", "1E16"]


def test_a_well_the_translation_named_is_left_alone(monkeypatch):
    spy = _Spy()
    _run(monkeypatch, spy)
    assert "1A1" not in spy.calls[0]["wells"]


def test_a_shallow_well_is_not_retried(monkeypatch):
    """1B8 holds no reads; aligning them answers nothing."""
    spy = _Spy()
    _run(monkeypatch, spy)
    assert "1B8" not in spy.calls[0]["wells"]
    assert _RETRY_MIN_DEPTH >= 1


def test_the_retry_names_the_well(monkeypatch):
    spy = _Spy({"1D11": "V47*", "1E16": "Y91M"})
    out = _run(monkeypatch, spy)
    got = dict(zip(out["global_well"], out["major_ref"]))
    assert got["1D11"] == "V47*"
    assert got["1E16"] == "Y91M"
    assert got["1A1"] == "G3F"


def test_nothing_to_retry_does_not_align(monkeypatch):
    """The common case: translation named every well, so no slow path runs."""
    spy = _Spy()
    df = pd.DataFrame({
        "global_well": ["1A1", "1A2"],
        "major_ref": ["G3F", "Parent"],
        "depth": [98, 50],
    })
    out = _run(monkeypatch, spy, df)
    assert spy.calls == []
    assert list(out["major_ref"]) == ["G3F", "Parent"]


def test_a_parent_well_is_not_treated_as_unnamed(monkeypatch):
    """Parent is an answer, not the absence of one."""
    spy = _Spy()
    df = pd.DataFrame({
        "global_well": ["1A1"], "major_ref": ["Parent"], "depth": [80],
    })
    _run(monkeypatch, spy, df)
    assert spy.calls == []
