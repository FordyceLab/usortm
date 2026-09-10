"""A worker process must not start a demux of its own.

The consensus stage runs its per-well checks across a ProcessPoolExecutor.
Where the start method is spawn -- the default on macOS -- each worker is a
fresh interpreter that re-imports the module the run was launched from.  A
launcher that calls the CLI at module level is re-entered by every worker,
and each one begins a whole demux: eight copies sharing one output directory
and one live_data.js, overwriting each other's tables.

The run looks like it cycles back to the read table and then produces
nothing.  It gave no other signal, because the RuntimeError Python raises in
each child was caught by the pool's fallback to threads and reported through
a logger no child had configured.  A hundred minutes went into finding it.
"""
import multiprocessing
import os
import sys
from pathlib import Path

import pytest

from usortm.demux.pipeline import (SpawnedChildError,
                                   _refuse_in_a_spawned_child)
from usortm.demux.utils import _reraise_if_bootstrapping

BOOTSTRAP_MSG = (
    "An attempt has been made to start a new process before the current "
    "process has finished its bootstrapping phase."
)


def test_the_main_process_is_allowed_through():
    assert multiprocessing.current_process().name == "MainProcess"
    _refuse_in_a_spawned_child()


def test_a_worker_is_refused(monkeypatch):
    class Fake:
        name = "SpawnProcess-3"

    monkeypatch.setattr(multiprocessing, "current_process", lambda: Fake())
    with pytest.raises(SpawnedChildError) as err:
        _refuse_in_a_spawned_child()
    assert "SpawnProcess-3" in str(err.value)
    assert "__main__" in str(err.value)


def test_the_refusal_happens_before_any_output_is_written(monkeypatch,
                                                          tmp_path):
    """The guard is first in the body, so a worker writes nothing at all.

    A worker that got as far as creating the output directory would already
    be racing the real run over the files inside it.
    """
    from usortm.demux import pipeline

    class Fake:
        name = "SpawnProcess-1"

    monkeypatch.setattr(multiprocessing, "current_process", lambda: Fake())
    out = tmp_path / "demux_output"
    with pytest.raises(SpawnedChildError):
        pipeline.run_levseq_pipeline(fastq=tmp_path / "reads.fastq",
                                     output_dir=out)
    assert not out.exists()


# --- the pool fallback -------------------------------------------------

def test_the_bootstrapping_error_is_not_swallowed(capsys):
    """Falling back to threads here would run a duplicate demux."""
    exc = RuntimeError(BOOTSTRAP_MSG)
    with pytest.raises(RuntimeError):
        _reraise_if_bootstrapping(exc)
    assert "__main__" in capsys.readouterr().err


def test_an_ordinary_pool_failure_still_falls_back():
    """The fallback exists for real environment limits; keep it working."""
    _reraise_if_bootstrapping(OSError("too many open files"))
    _reraise_if_bootstrapping(RuntimeError("something else entirely"))


# --- the real thing ----------------------------------------------------

def _pipeline_in_child(queue, out_dir):
    try:
        from usortm.demux.pipeline import run_levseq_pipeline
        run_levseq_pipeline(fastq=Path(out_dir) / "reads.fastq",
                            output_dir=Path(out_dir) / "demux_output")
        queue.put("NO ERROR")
    except Exception as exc:                       # noqa: BLE001
        queue.put(type(exc).__name__)


@pytest.mark.skipif("spawn" not in multiprocessing.get_all_start_methods(),
                    reason="spawn is not available here")
def test_a_genuinely_spawned_worker_is_refused(tmp_path):
    """Not a monkeypatched name: a real spawned process, as in the run.

    This is the test that would have caught it.  Without the guard the
    child returns "NO ERROR" and proceeds to demultiplex.
    """
    ctx = multiprocessing.get_context("spawn")
    env_src = str(Path(__file__).resolve().parents[1] / "src")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [env_src, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)

    queue = ctx.Queue()
    proc = ctx.Process(target=_pipeline_in_child, args=(queue, str(tmp_path)))
    proc.start()
    proc.join(timeout=120)
    assert proc.exitcode is not None, "the child never finished"
    assert not queue.empty(), "the child reported nothing"
    assert queue.get() == "SpawnedChildError"
    assert not (tmp_path / "demux_output").exists()
