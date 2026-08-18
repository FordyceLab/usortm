"""Tests for the interactive plate-map prompt.

The prompt runs whenever no plate map was supplied.  It confirms the sort
plate count first, then the mapping, because the count is what decides
whether barcode plates have to be reused at all.

questionary needs a real terminal, so its answers are scripted here — the
branching is the part that carries risk, not the widget drawing.
"""

import pytest

from usortm.demux.plate_map import load_plate_map


class _Scripted:
    """Feed questionary scripted answers, in order, per prompt type."""

    def __init__(self, text=None, confirm=None, select=None):
        self.text = list(text or [])
        self.confirm = list(confirm or [])
        self.select = list(select or [])
        self.asked = {"text": 0, "confirm": 0, "select": 0}

    def _pop(self, kind):
        queue = getattr(self, kind)
        self.asked[kind] += 1
        if not queue:
            raise AssertionError(f"prompt asked for an unscripted {kind}")
        return queue.pop(0)

    def install(self, monkeypatch):
        import questionary

        def make(kind):
            def factory(*args, **kwargs):
                value = self._pop(kind)

                class _Q:
                    def ask(self_inner):
                        return value

                return _Q()

            return factory

        monkeypatch.setattr(questionary, "text", make("text"))
        monkeypatch.setattr(questionary, "confirm", make("confirm"))
        monkeypatch.setattr(questionary, "select", make("select"))
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)


def _prompt(monkeypatch, script, fastq, n_plates, project_dir):
    from usortm.cli.demux_cmd import _prompt_plate_map

    script.install(monkeypatch)
    return _prompt_plate_map(fastq, n_plates, project_dir)


@pytest.fixture
def single_fastq(tmp_path):
    fq = tmp_path / "reads.fastq"
    fq.write_text("@r\nACGT\n+\nIIII\n")
    return fq


@pytest.fixture
def run_dir(tmp_path):
    """A directory holding two FASTQ files, as a sequencing run would."""
    d = tmp_path / "fastq_pass"
    d.mkdir()
    for name in ("run1.fastq", "run2.fastq"):
        (d / name).write_text("@r\nACGT\n+\nIIII\n")
    return d


class TestPromptFlow:

    def test_non_interactive_shell_does_not_prompt(self, monkeypatch, single_fastq,
                                                   tmp_path):
        from usortm.cli.demux_cmd import _prompt_plate_map

        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        assert _prompt_plate_map(single_fastq, 6, tmp_path) is None

    def test_plate_count_is_asked_first(self, monkeypatch, single_fastq, tmp_path):
        """One FASTQ within the kit's limit needs only the count confirmed."""
        script = _Scripted(text=["6"])
        segments = _prompt(monkeypatch, script, single_fastq, 6, tmp_path)

        assert script.asked["text"] == 1
        assert len(segments) == 1
        assert segments[0].plates == {i: i for i in range(1, 7)}

    def test_plate_count_can_differ_from_the_plan(self, monkeypatch, single_fastq,
                                                  tmp_path):
        """The plan's figure comes from library size, not what was sorted."""
        segments = _prompt(monkeypatch, _Scripted(text=["3"]), single_fastq, 6,
                           tmp_path)
        assert segments[0].sort_plates == [1, 2, 3]

    def test_directory_from_one_run_stays_a_single_segment(
        self, monkeypatch, run_dir, tmp_path
    ):
        """A nanopore fastq_pass holds many files from one run sharing one
        barcode layout — it must not become a segment per file."""
        script = _Scripted(text=["6"], confirm=[True])
        segments = _prompt(monkeypatch, script, run_dir, 6, tmp_path)

        assert len(segments) == 1
        assert segments[0].path == run_dir
        assert segments[0].plates == {i: i for i in range(1, 7)}

    def test_directory_from_separate_runs_maps_each_file(
        self, monkeypatch, run_dir, tmp_path
    ):
        """Answering 'no' to the one-run question maps the files separately,
        which is what stops two layouts being concatenated together."""
        files = sorted(run_dir.glob("*.fastq"))
        script = _Scripted(
            text=["6", "1:1, 2:2, 3:3", "4:4, 5:5, 6:6"],
            confirm=[False, False],  # not one run; then decline saving
            select=[files[0], files[1]],
        )
        segments = _prompt(monkeypatch, script, run_dir, 6, tmp_path)

        assert [s.plates for s in segments] == [
            {1: 1, 2: 2, 3: 3},
            {4: 4, 5: 5, 6: 6},
        ]

    def test_more_sort_plates_than_barcode_plates_maps_per_fastq(
        self, monkeypatch, run_dir, tmp_path
    ):
        """The motivating case: ten sort plates, so reuse is unavoidable and
        the single-segment shortcut is never offered."""
        files = sorted(run_dir.glob("*.fastq"))
        script = _Scripted(
            text=["10", "1:1, 2:2, 3:3, 4:4, 5:5, 6:6", "7:7, 8:8, 1:9, 2:10"],
            confirm=[False],  # only the save question is asked
            select=[files[0], files[1]],
        )
        segments = _prompt(monkeypatch, script, run_dir, 10, tmp_path)

        assert len(segments) == 2
        assert script.asked["confirm"] == 1
        covered = sorted(p for s in segments for p in s.sort_plates)
        assert covered == list(range(1, 11))
        assert segments[1].plates == {7: 7, 8: 8, 1: 9, 2: 10}

    def test_bad_pairs_are_re_asked(self, monkeypatch, run_dir, tmp_path):
        files = sorted(run_dir.glob("*.fastq"))
        script = _Scripted(
            text=["10", "not-pairs", "1:1, 2:2, 3:3, 4:4, 5:5, 6:6",
                  "7:7, 8:8, 1:9, 2:10"],
            confirm=[False],
            select=[files[0], files[1]],
        )
        segments = _prompt(monkeypatch, script, run_dir, 10, tmp_path)

        assert len(segments) == 2
        assert not script.text, "every scripted answer should have been consumed"

    def test_saved_config_is_reusable(self, monkeypatch, run_dir, tmp_path):
        files = sorted(run_dir.glob("*.fastq"))
        script = _Scripted(
            text=["10", "1:1, 2:2, 3:3, 4:4, 5:5, 6:6", "7:7, 8:8, 1:9, 2:10"],
            confirm=[True],  # save it
            select=[files[0], files[1]],
        )
        segments = _prompt(monkeypatch, script, run_dir, 10, tmp_path)

        saved = tmp_path / "plate_map.toml"
        assert saved.exists()
        assert [s.plates for s in load_plate_map(saved)] == [
            s.plates for s in segments
        ]

    def test_cancelling_the_plate_count_aborts(self, monkeypatch, single_fastq,
                                               tmp_path):
        assert _prompt(monkeypatch, _Scripted(text=[None]), single_fastq, 6,
                       tmp_path) is None
