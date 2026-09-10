"""The consensus stage runs across processes and gives the same answer.

Each well launches six subprocesses.  From the pipeline's 4-5 GB parent a
fork costs 100-170 ms and holds the GIL, so eight worker threads queued
behind each other's forks and the stage ran on one core: 20.7 minutes for
segment 1 of the AFMtag round 1 project.  Worker processes are small, their
forks are cheap, and they do not share the parent's GIL.

What must hold: every well gets a result, the result is the one the serial
path gives, the process pool is actually used rather than quietly replaced
by threads, and the one pool failure that means "a worker is running the
whole pipeline" is not swallowed.
"""
import concurrent.futures
import logging
import random
import shutil

import pytest

from usortm.demux import utils
from usortm.demux.utils import _run_consensus_tasks

BOOTSTRAP_MSG = (
    "An attempt has been made to start a new process before the current "
    "process has finished its bootstrapping phase."
)


def _fake_wells(tmp_path, n):
    """Wells whose tools do not exist, so each returns (well, None, None).

    Enough to test the pool's plumbing without minimap2 or samtools.
    """
    paths = {}
    for i in range(n):
        well = f"1A{i + 1}"
        paths[well] = {
            "ref_fa": str(tmp_path / "ref.fasta"),
            "ref_mmi": str(tmp_path / "ref.fasta"),
            "fq": str(tmp_path / f"{well}.fastq"),
            "bam": str(tmp_path / f"{well}.bam"),
            "cons_fa": str(tmp_path / f"{well}_consensus.fasta"),
            "cons_bam": str(tmp_path / f"{well}_consensus_align.bam"),
        }
    return paths


def test_every_well_gets_a_result_from_the_process_pool(tmp_path, caplog,
                                                        monkeypatch):
    """Processes are used, and threads are not so much as constructed.

    Without the second half this test would pass against a thread pool,
    which is the regression it exists to catch.
    """
    class ThreadsWereUsed:
        def __init__(self, *a, **k):
            raise AssertionError("the consensus stage fell back to threads")

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor",
                        ThreadsWereUsed)
    paths = _fake_wells(tmp_path, 5)
    with caplog.at_level(logging.WARNING, logger="usortm"):
        results = _run_consensus_tasks(paths, "minimap2_fake", "samtools_fake",
                                       False, workers=2)
    assert set(results) == set(paths)
    assert all(results[w] == (None, None) for w in paths)
    assert "falling back to threads" not in caplog.text


def test_the_process_pool_matches_the_serial_path(tmp_path):
    paths = _fake_wells(tmp_path, 4)
    serial = _run_consensus_tasks(paths, "minimap2_fake", "samtools_fake",
                                  False, workers=1)
    pooled = _run_consensus_tasks(paths, "minimap2_fake", "samtools_fake",
                                  False, workers=3)
    assert serial == pooled


def test_a_single_well_does_not_open_a_pool(tmp_path, monkeypatch):
    """One task is not worth a pool; the serial path must be taken."""
    class Boom:
        def __init__(self, *a, **k):
            raise AssertionError("a pool was opened for one well")

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", Boom)
    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", Boom)
    paths = _fake_wells(tmp_path, 1)
    results = _run_consensus_tasks(paths, "minimap2_fake", "samtools_fake",
                                   False, workers=8)
    assert results == {"1A1": (None, None)}


def test_an_ordinary_pool_failure_falls_back_to_threads(tmp_path, monkeypatch,
                                                        caplog):
    """A process pool that cannot start is an environment limit, not an
    error in the run: the stage completes on threads and says so."""
    class NoProcesses:
        def __init__(self, *a, **k):
            raise OSError("no fork for you")

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", NoProcesses)
    paths = _fake_wells(tmp_path, 3)
    with caplog.at_level(logging.WARNING, logger="usortm"):
        results = _run_consensus_tasks(paths, "minimap2_fake", "samtools_fake",
                                       False, workers=2)
    assert set(results) == set(paths)
    assert "falling back to threads" in caplog.text


def test_the_bootstrapping_error_is_not_swallowed(tmp_path, monkeypatch):
    """This one means a worker is running the whole pipeline.  Falling back
    to threads here is how eight demuxes once shared one output directory."""
    class InsideAWorker:
        def __init__(self, *a, **k):
            raise RuntimeError(BOOTSTRAP_MSG)

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", InsideAWorker)
    paths = _fake_wells(tmp_path, 3)
    with pytest.raises(RuntimeError, match="bootstrapping"):
        _run_consensus_tasks(paths, "minimap2_fake", "samtools_fake",
                             False, workers=2)


# --- with the real tools ---------------------------------------------------

def _tool(name):
    finder = getattr(utils, f"find_{name}", None)
    try:
        return finder() if finder else shutil.which(name)
    except Exception:                                  # noqa: BLE001
        return shutil.which(name)


def _write_fasta(path, name, seq):
    path.write_text(f">{name}\n{seq}\n")


def _write_fastq(path, reads):
    with open(path, "w") as fh:
        for i, seq in enumerate(reads):
            fh.write(f"@r{i}\n{seq}\n+\n{'I' * len(seq)}\n")


@pytest.mark.skipif(not (_tool("minimap2") and _tool("samtools")),
                    reason="minimap2 and samtools are needed")
def test_processes_and_serial_build_the_same_consensus(tmp_path):
    """The claim that matters: same CIGAR, same consensus, per well.

    Three wells, each holding 12 reads of a 400 bp reference; one well
    carries a substitution in every read so that the consensus has
    something to differ on.
    """
    rng = random.Random(7)
    ref = "".join(rng.choice("ACGT") for _ in range(400))
    ref_fa = tmp_path / "ref.fasta"
    _write_fasta(ref_fa, "ref", ref)

    mutant = ref[:200] + ("A" if ref[200] != "A" else "C") + ref[201:]
    wells = {"1A1": ref, "1A2": mutant, "1A3": ref}
    paths = {}
    for well, seq in wells.items():
        fq = tmp_path / f"{well}.fastq"
        _write_fastq(fq, [seq] * 12)
        paths[well] = {
            "ref_fa": str(ref_fa), "ref_mmi": str(ref_fa), "fq": str(fq),
            "bam": str(tmp_path / f"{well}.bam"),
            "cons_fa": str(tmp_path / f"{well}_consensus.fasta"),
            "cons_bam": str(tmp_path / f"{well}_consensus_align.bam"),
        }

    mm2, sam = _tool("minimap2"), _tool("samtools")
    serial = _run_consensus_tasks(paths, mm2, sam, False, workers=1)
    pooled = _run_consensus_tasks(paths, mm2, sam, False, workers=3)

    assert set(serial) == set(pooled) == set(wells)
    assert serial == pooled
    # and the test is not vacuous: the tools ran and produced sequences
    assert all(cons for _, cons in pooled.values()), pooled
    assert pooled["1A2"][1] != pooled["1A1"][1]
    assert pooled["1A1"][1] == pooled["1A3"][1]
