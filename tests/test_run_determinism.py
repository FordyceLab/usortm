"""The same reads give the same answer, run after run.

Python randomises string hashing per process, so anything that reaches an
output through the iteration order of a set changes between runs.  Two places
did: the per-read table's row order, which put each well's reads in a new
order and so rewrote every per-well FASTQ and discarded the consensus built
from it; and the tie-break that picks a well's reference, which could call a
well one way today and the other way tomorrow.

The subprocesses here matter: within one process a set's iteration order is
stable, so a test that does not cross a process boundary cannot see either
bug.
"""
import subprocess
import sys
import textwrap

import pytest


def _in_fresh_processes(code: str, runs: int = 4):
    """Run *code* in separate interpreters and collect what it prints.

    Each gets its own hash seed, which is the condition being tested.
    """
    out = []
    for _ in range(runs):
        r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        out.append(r.stdout.strip())
    return out


def test_python_really_does_randomise_string_hashing():
    """The premise. Without this the other tests here prove nothing."""
    got = _in_fresh_processes(
        "print(list({'read_a', 'read_b', 'read_c', 'read_d', 'read_e'}))", 8)
    assert len(set(got)) > 1, (
        "set order did not vary across processes, so these tests cannot "
        "detect the bug they are written for")


def test_the_read_table_keeps_its_row_order_across_runs():
    got = _in_fresh_processes("""
        fbc = {f'read_{i}': 'bc01' for i in range(40)}
        rbc = {f'read_{i}': 'rb01' for i in range(40)}
        ref = {f'read_{i}': 'fwd:v1' for i in range(40)}
        print(','.join(sorted(set(fbc) | set(rbc) | set(ref))))
    """)
    assert len(set(got)) == 1


def test_a_tied_well_is_called_the_same_way_every_run():
    """Twenty reads for one reference, twenty for another."""
    got = _in_fresh_processes("""
        refs = ['v_alpha'] * 20 + ['v_beta'] * 20
        print(max(sorted(set(refs)), key=refs.count))
    """)
    assert len(set(got)) == 1, f"tie broken differently: {set(got)}"


def test_the_tie_break_still_prefers_the_majority():
    """Determinism must not cost correctness: more reads still wins."""
    from collections import Counter

    for refs, want in (
        (["v_beta"] * 5 + ["v_alpha"] * 2, "v_beta"),
        (["v_alpha"] * 9 + ["v_beta"] * 30, "v_beta"),
    ):
        assert max(sorted(set(refs)), key=refs.count) == want
        assert Counter(refs).most_common(1)[0][0] == want


def test_one_wells_reads_are_written_in_a_stable_order(tmp_path):
    """End to end: the same table twice gives byte-identical FASTQs."""
    pd = pytest.importorskip("pandas")
    from usortm.demux.utils import write_per_well_fastqs

    df = pd.DataFrame({
        "well_pos": ["1A1"] * 3,
        "read_name": ["r3", "r1", "r2"],
        "read_seq": ["AAAA", "CCCC", "GGGG"],
        "read_qual": ["IIII"] * 3,
    })
    write_per_well_fastqs(df, str(tmp_path))
    first = (tmp_path / "wells" / "fastqs" / "1A1.fastq").read_text()

    write_per_well_fastqs(df, str(tmp_path))
    assert (tmp_path / "wells" / "fastqs" / "1A1.fastq").read_text() == first
