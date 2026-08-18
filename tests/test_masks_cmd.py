"""Tests for `usortm masks derive` and the low-barcode-yield guard.

Masks are specific to a plasmid backbone. Masks from another construct
classify almost nothing while alignment still succeeds, which produces a
finished-looking run with empty wells — the failure these cover.
"""

import gzip
import json

import pytest
from typer.testing import CliRunner

from usortm.cli import app
from usortm.demux.barcodes import LEVSEQ_FBC, LEVSEQ_RBC
from usortm.demux.pipeline import _check_barcode_yield

runner = CliRunner()

FRONT = "GGTCTAACGCTTAAGCACTCAA"     # what sits before the forward barcode
REAR = "TTCAGGCACCTTAAGGGCTATA"      # and after it
_COMP = str.maketrans("ACGT", "TGCA")


def _rc(s):
    return s.translate(_COMP)[::-1]


def _write_reads(path, n=400, gzipped=False):
    """Reads laid out as the arrangement expects, around known masks."""
    lines = []
    for i in range(n):
        fbc = LEVSEQ_FBC[i % 96]
        rbc = LEVSEQ_RBC[i % 4]
        seq = (FRONT + fbc + REAR + "ACGT" * 40
               + _rc(REAR) + _rc(rbc) + _rc(FRONT))
        lines.append(f"@read_{i}\n{seq}\n+\n{'I' * len(seq)}\n")
    text = "".join(lines)
    open_fn = gzip.open if gzipped else open
    with open_fn(path, "wt") as fh:
        fh.write(text)
    return path


@pytest.fixture
def project(tmp_path):
    proj = tmp_path / "proj"
    (proj / "demux_output" / "alignment").mkdir(parents=True)
    (proj / "usortm_project.json").write_text(json.dumps({"n_plates": 1}))
    _write_reads(proj / "demux_output" / "alignment" / "oriented_reads.fastq")
    return proj


class TestDerive:

    def test_recovers_the_masks_the_reads_were_built_with(self, project):
        result = runner.invoke(app, ["masks", "derive", str(project)])

        assert result.exit_code == 0, result.stdout
        out = (project / "mask_config.derived.toml").read_text()
        assert f'mask1_front = "{FRONT}"' in out
        assert f'mask1_rear  = "{REAR}"' in out

    def test_reverse_masks_are_the_complements(self, project):
        runner.invoke(app, ["masks", "derive", str(project)])
        out = (project / "mask_config.derived.toml").read_text()
        assert f'mask2_front = "{_rc(REAR)}"' in out
        assert f'mask2_rear  = "{_rc(FRONT)}"' in out

    def test_mask_length_is_respected(self, project):
        runner.invoke(app, ["masks", "derive", str(project),
                            "--mask-length", "10"])
        out = (project / "mask_config.derived.toml").read_text()
        assert f'mask1_front = "{FRONT[-10:]}"' in out
        assert f'mask1_rear  = "{REAR[:10]}"' in out

    def test_reads_a_gzipped_fastq_directly(self, tmp_path, project):
        """Deriving must work before any demux has run."""
        raw = _write_reads(tmp_path / "raw.fastq.gz", gzipped=True)
        out = tmp_path / "derived.toml"
        result = runner.invoke(app, ["masks", "derive", str(project),
                                     "--reads", str(raw), "--output", str(out)])

        assert result.exit_code == 0, result.stdout
        assert f'mask1_front = "{FRONT}"' in out.read_text()

    def test_missing_reads_is_reported(self, tmp_path):
        proj = tmp_path / "bare"
        proj.mkdir()
        (proj / "usortm_project.json").write_text("{}")
        result = runner.invoke(app, ["masks", "derive", str(proj)])

        assert result.exit_code == 1
        assert "usortm demux" in result.stdout

    def test_reads_without_barcodes_are_reported(self, tmp_path, project):
        junk = tmp_path / "junk.fastq"
        junk.write_text("".join(
            f"@r{i}\n{'ACGT' * 50}\n+\n{'I' * 200}\n" for i in range(50)
        ))
        result = runner.invoke(app, ["masks", "derive", str(project),
                                     "--reads", str(junk)])
        assert result.exit_code == 1
        assert "no exact barcode matches" in result.stdout


class TestBarcodeYieldGuard:
    """The check that would have caught masks from the wrong construct."""

    def test_near_zero_yield_is_critical(self):
        w = _check_barcode_yield({"ref_assigned": 35015, "fbc_classified": 19,
                                  "rbc_classified": 230})
        assert w is not None
        assert w["severity"] == "critical"
        assert "masks" in w["detail"]
        assert "usortm masks derive" in w["detail"]

    def test_low_but_nonzero_yield_is_a_warning(self):
        w = _check_barcode_yield({"ref_assigned": 10000, "fbc_classified": 800,
                                  "rbc_classified": 900})
        assert w["severity"] == "low"

    def test_healthy_yield_is_silent(self):
        assert _check_barcode_yield({
            "ref_assigned": 10000, "fbc_classified": 5200,
            "rbc_classified": 6100,
        }) is None

    def test_judged_on_the_worse_of_the_two(self):
        """One good barcode does not excuse the other being absent."""
        w = _check_barcode_yield({"ref_assigned": 10000, "fbc_classified": 9000,
                                  "rbc_classified": 30})
        assert w is not None and w["severity"] == "critical"

    def test_no_reads_is_silent(self):
        assert _check_barcode_yield({"ref_assigned": 0, "union_reads": 0}) is None
