"""``usortm demux --expected`` end to end, against a plate with a known answer.

A 96-well plate barcoded as quadrant TL of LevSeq plate 1, holding single-codon
variants of one parent in a random backbone.  Most wells hold what they
should; the rest are each one way a plate goes wrong.  The run is given only
the parent vector, the expected CSV and the reads, so the read layout is
detected as well.  Needs dorado, minimap2 and samtools.
"""

import csv
import re

import numpy as np
import pytest
from typer.testing import CliRunner

from usortm.cli import app
from usortm.demux.barcodes import LEVSEQ_FBC, LEVSEQ_RBC
from usortm.qc.synthetic import _apply_errors, _random_seq


def _tools_available():
    try:
        from usortm.demux.deps import check_all_dependencies
        check_all_dependencies()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _tools_available(),
                                reason="demux toolchain not installed")

_COMP = str.maketrans("ACGT", "TGCA")
OUTER_5P, OUTER_3P = "AATATAAATT", "ATAATTTATA"


def _rc(s):
    return s.translate(_COMP)[::-1]


def _build(tmp_path):
    rng = np.random.default_rng(11)
    up, down = _random_seq(rng, 1200), _random_seq(rng, 1200)
    p5, p3 = _random_seq(rng, 22), _random_seq(rng, 26)
    f5, f3 = _random_seq(rng, 150), _random_seq(rng, 180)
    parent = _random_seq(rng, 240)
    (tmp_path / "vector.fa").write_text(f">parent\n{up + p5 + f5 + parent + f3 + p3 + down}\n")

    def mutate(seq, pos):
        codon = seq[pos:pos + 3]
        while codon == seq[pos:pos + 3]:
            codon = _random_seq(rng, 3)
        return seq[:pos] + codon + seq[pos + 3:]

    constructs = {f"v{i}": mutate(parent, 3 * (4 + 6 * i)) for i in range(8)}
    wells = [f"{r}{c}" for r in "ABC" for c in range(1, 9)]          # 24 wells
    expected = {w: f"v{i % 8}" for i, w in enumerate(wells)}
    expected["C8"] = None                                              # expected empty

    # What each well actually holds, and the verdict that should follow.
    holds = {w: [(n, 1.0)] if n else [] for w, n in expected.items()}
    truth = {w: "match" for w in wells}
    truth["C8"] = "unexpected reads"
    holds["C8"] = [("v3", 1.0)]
    holds["A1"], holds["A2"] = [(expected["A2"], 1.0)], [(expected["A1"], 1.0)]
    truth["A1"] = truth["A2"] = "wrong construct"
    extra = {"B3_mut": mutate(constructs[expected["B3"]], 3 * 70)}
    holds["B3"], truth["B3"] = [("B3_mut", 1.0)], "changed"
    holds["B6"], truth["B6"] = [(expected["B6"], 0.5), (expected["B7"], 0.5)], "mixed"
    holds["C2"], truth["C2"] = [], "no reads"
    extra["foreign"] = _random_seq(rng, 240)
    holds["C5"], truth["C5"] = [("foreign", 1.0)], "unrecognised"
    seqs = {**constructs, **extra}

    with open(tmp_path / "expected.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["plate", "quadrant", "well", "name", "sequence"])
        for well in wells:
            n = expected[well]
            w.writerow([1, "TL", well, n or "", constructs[n] if n else ""])

    rbc = LEVSEQ_RBC[0]                                                # RB01: plate 1, TL
    with open(tmp_path / "reads.fastq", "w") as fh:
        for well in wells:
            if not holds[well]:
                continue
            fbc = LEVSEQ_FBC[(ord(well[0]) - 65) * 12 + int(well[1:]) - 1]
            names, probs = zip(*holds[well])
            for k in range(45):
                n = names[int(rng.choice(len(names), p=np.array(probs) / sum(probs)))]
                read = OUTER_5P + fbc + p5 + f5 + seqs[n] + f3 + p3 + _rc(rbc) + OUTER_3P
                if rng.random() < 0.5:
                    read = _rc(read)
                read = _apply_errors(rng, read, 0.02)
                fh.write(f"@{well}_{k}\n{read}\n+\n{'I' * len(read)}\n")
    return truth


def test_every_well_gets_its_true_verdict(tmp_path):
    truth = _build(tmp_path)
    out = tmp_path / "check"
    result = CliRunner().invoke(app, [
        "demux", "--expected", str(tmp_path / "expected.csv"),
        "--fastq", str(tmp_path / "reads.fastq"),
        "--vector", str(tmp_path / "vector.fa"),
        "-o", str(out), "--workers", "2",
    ])
    assert result.exit_code == 0, re.sub(r"\x1b\[[0-9;]*m", "", result.output)

    rows = {r["label"].split(":")[-1]: r
            for r in csv.DictReader(open(out / "verification.csv"))}
    # Every expected well has a row, and so does the one expected empty that
    # holds reads.
    assert set(rows) == set(truth)
    assert {w: rows[w]["verdict"] for w in truth} == truth

    assert rows["A1"]["note"] == "swapped with TL:A2"
    assert rows["C8"]["observed"] == "v3"
    assert rows["B3"]["differences"]
    for name in ("plate_check.html", "verification_summary.json",
                 "derived_read_template.fasta"):
        assert (out / name).exists()
