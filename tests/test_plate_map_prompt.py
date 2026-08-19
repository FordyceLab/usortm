"""Tests for the interactive plate-map prompt.

The prompt runs whenever no plate map was supplied.  It establishes the sort
plate count, then for each FASTQ asks two things in the order the experiment
happened: which sort plates that file holds, and which barcode plates carried
them.

questionary needs a real terminal, so its answers are scripted here — the
branching and parsing carry the risk, not the widget drawing.
"""

import pytest

from usortm.demux.plate_map import PlateMapError, load_plate_map


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


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

class TestPlateListParsing:

    def _parse(self, text):
        from usortm.cli.demux_cmd import _parse_plate_list
        return _parse_plate_list(text)

    def test_range(self):
        assert self._parse("1-6") == [1, 2, 3, 4, 5, 6]

    def test_comma_list(self):
        assert self._parse("7,8,9,10") == [7, 8, 9, 10]

    def test_mixed_range_and_list(self):
        assert self._parse("1-4, 9") == [1, 2, 3, 4, 9]

    def test_order_is_preserved(self):
        """The list is matched positionally against barcode plates."""
        assert self._parse("9,7,8") == [9, 7, 8]

    def test_whitespace_and_semicolons(self):
        assert self._parse(" 1 ; 2 , 3 ") == [1, 2, 3]

    def test_duplicates_rejected(self):
        with pytest.raises(PlateMapError, match="more than once"):
            self._parse("1,2,2")

    def test_backwards_range_rejected(self):
        with pytest.raises(PlateMapError, match="backwards"):
            self._parse("6-1")

    def test_garbage_rejected(self):
        with pytest.raises(PlateMapError, match="not a plate number"):
            self._parse("one,two")


class TestBarcodeAssignment:

    def _assign(self, text, sort_plates):
        from usortm.cli.demux_cmd import _parse_barcode_assignment
        return _parse_barcode_assignment(text, sort_plates)

    def test_positional_list(self):
        """'7,8,1,2' against sort plates 7,8,9,10 reuses barcode 1 and 2."""
        assert self._assign("7,8,1,2", [7, 8, 9, 10]) == {7: 7, 8: 8, 1: 9, 2: 10}

    def test_identity(self):
        assert self._assign("1,2,3", [1, 2, 3]) == {1: 1, 2: 2, 3: 3}

    def test_explicit_pairs_still_accepted(self):
        assert self._assign("7:7, 8:8, 1:9, 2:10", [7, 8, 9, 10]) == {
            7: 7, 8: 8, 1: 9, 2: 10
        }

    def test_length_mismatch_names_the_counts(self):
        with pytest.raises(PlateMapError, match="3 barcode plate"):
            self._assign("1,2,3", [1, 2, 3, 4])

    def test_pairs_missing_a_sort_plate_rejected(self):
        with pytest.raises(PlateMapError, match="No barcode plate"):
            self._assign("7:7, 8:8", [7, 8, 9, 10])

    def test_barcode_plate_beyond_the_kit_rejected(self):
        with pytest.raises(PlateMapError, match="out of range"):
            self._assign("9,10", [1, 2])


# ---------------------------------------------------------------------------
# Prompt flow
# ---------------------------------------------------------------------------

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

    def test_each_fastq_is_asked_sort_plates_then_barcode_plates(
        self, monkeypatch, run_dir, tmp_path
    ):
        files = sorted(run_dir.glob("*.fastq"))
        script = _Scripted(
            text=["6", "1-3", "1,2,3", "4-6", "4,5,6"],
            confirm=[False],          # these are separate runs
            select=[files[0], files[1]],
        )
        segments = _prompt(monkeypatch, script, run_dir, 6, tmp_path)

        assert [s.plates for s in segments] == [
            {1: 1, 2: 2, 3: 3},
            {4: 4, 5: 5, 6: 6},
        ]

    def test_reused_barcode_plates_across_fastqs(
        self, monkeypatch, run_dir, tmp_path
    ):
        """The motivating case: ten sort plates, so barcode plates 1 and 2 are
        used twice and the one-to-one shortcut is never offered."""
        files = sorted(run_dir.glob("*.fastq"))
        script = _Scripted(
            text=["10", "1-6", "1,2,3,4,5,6", "7-10", "7,8,1,2"],
            select=[files[0], files[1]],
        )
        segments = _prompt(monkeypatch, script, run_dir, 10, tmp_path)

        assert len(segments) == 2
        # Over the kit's limit, so the one-run shortcut is never offered and
        # nothing else is asked either — the mapping is simply recorded.
        assert script.asked["confirm"] == 0
        assert segments[0].plates == {i: i for i in range(1, 7)}
        assert segments[1].plates == {7: 7, 8: 8, 1: 9, 2: 10}
        covered = sorted(p for s in segments for p in s.sort_plates)
        assert covered == list(range(1, 11))

    def test_bad_answers_are_re_asked(self, monkeypatch, run_dir, tmp_path):
        files = sorted(run_dir.glob("*.fastq"))
        script = _Scripted(
            text=["10",
                  "not-plates", "1-6", "1,2,3,4,5,6",   # bad sort list, retried
                  "7-10", "1,2", "7,8,1,2"],            # wrong length, retried
            select=[files[0], files[1]],
        )
        segments = _prompt(monkeypatch, script, run_dir, 10, tmp_path)

        assert len(segments) == 2
        assert segments[1].plates == {7: 7, 8: 8, 1: 9, 2: 10}
        assert not script.text, "every scripted answer should have been consumed"

    def test_mapping_is_recorded_without_being_asked(self, monkeypatch, run_dir,
                                                     tmp_path):
        """The mapping is part of the run's configuration, so it is written to
        the project rather than offered as an optional export."""
        files = sorted(run_dir.glob("*.fastq"))
        script = _Scripted(
            text=["10", "1-6", "1,2,3,4,5,6", "7-10", "7,8,1,2"],
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


# ---------------------------------------------------------------------------
# Re-confirming a saved mapping
# ---------------------------------------------------------------------------

class TestSavedMapConfirmation:
    """A saved plate map is last run's answer; this run may load different
    plates, so it is offered for review rather than applied silently."""

    def _resolve(self, monkeypatch, script, project_dir, fastq, n_plates=10):
        from usortm.cli.demux_cmd import _resolve_plate_map

        script.install(monkeypatch)
        return _resolve_plate_map(None, project_dir, fastq, n_plates)

    @pytest.fixture
    def saved(self, tmp_path, run_dir):
        from usortm.demux.plate_map import Segment, write_plate_map

        files = sorted(run_dir.glob("*.fastq"))
        write_plate_map(
            [Segment(name="run1", path=files[0], plates={i: i for i in range(1, 7)}),
             Segment(name="run2", path=files[1], plates={7: 7, 8: 8, 1: 9, 2: 10})],
            tmp_path / "plate_map.toml",
        )
        return tmp_path

    def test_confirming_keeps_the_saved_map(self, monkeypatch, saved, run_dir):
        script = _Scripted(confirm=[True])
        segments = self._resolve(monkeypatch, script, saved, run_dir)

        assert [s.name for s in segments] == ["run1", "run2"]
        assert segments[1].plates == {7: 7, 8: 8, 1: 9, 2: 10}

    def test_declining_re_prompts(self, monkeypatch, saved, run_dir):
        files = sorted(run_dir.glob("*.fastq"))
        script = _Scripted(
            text=["4", "1-2", "1,2", "3-4", "3,4"],
            confirm=[False, False, False],   # not still correct; not one run; no save
            select=[files[0], files[1]],
        )
        segments = self._resolve(monkeypatch, script, saved, run_dir, n_plates=4)

        covered = sorted(p for s in segments for p in s.sort_plates)
        assert covered == [1, 2, 3, 4]

    def test_non_interactive_uses_the_saved_map_without_asking(
        self, monkeypatch, saved, run_dir
    ):
        """Scripted and remote runs must not block on a question."""
        from usortm.cli.demux_cmd import _resolve_plate_map

        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        segments = _resolve_plate_map(None, saved, run_dir, 10)
        assert [s.name for s in segments] == ["run1", "run2"]
