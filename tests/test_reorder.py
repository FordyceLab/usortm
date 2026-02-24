"""Tests for usortm reorder command."""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest

from usortm.cli.reorder import (
    _load_recovered,
    _load_variants,
    _normalize,
    _write_idt_eblocks,
    _write_idt_opools,
    _write_twist_gene_fragments,
    _write_twist_oligo_pools,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_project(tmp_path: Path, round: int = 1, pick_completed: bool = True) -> Path:
    state = {
        "round": round,
        "library_size": 5,
        "variants_file": str(tmp_path / "variants.csv"),
        "workflow_steps": {
            "pick": {"completed": pick_completed, "total_hits": 3},
        },
    }
    (tmp_path / "usortm_project.json").write_text(json.dumps(state))
    return tmp_path


def _make_variants_csv(tmp_path: Path, variants: list[dict]) -> Path:
    path = tmp_path / "variants.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "sequence"])
        writer.writeheader()
        writer.writerows(variants)
    return path


def _make_hitlist(tmp_path: Path, recovered: list[str], empties: list[str] | None = None) -> Path:
    """Create a hitlist CSV. recovered get volume=5.0; empties get volume=0.0."""
    path = tmp_path / "hitlist.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["SampleID", "SourcePlateID", "SourceWell", "TargetPlateID", "TargetWell", "TransferVolume"])
        for name in recovered:
            writer.writerow([name, "1", "A1", "0", "A1", "5.0"])
        for name in (empties or []):
            writer.writerow([name, "", "", "0", "", "0.0"])
    return path


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_normalize_strips_cons_check():
    assert _normalize("var1|cons_check") == "var1"


def test_normalize_strips_perfect_match():
    assert _normalize("var1|Perfect Match") == "var1"


def test_normalize_plain():
    assert _normalize("var1") == "var1"


def test_load_variants_standard_columns(tmp_path):
    _make_variants_csv(tmp_path, [
        {"name": "var1", "sequence": "ATGC"},
        {"name": "var2", "sequence": "GCTA"},
    ])
    variants = _load_variants(tmp_path / "variants.csv")
    assert len(variants) == 2
    assert variants[0] == {"name": "var1", "sequence": "ATGC"}


def test_load_variants_alternate_columns(tmp_path):
    path = tmp_path / "v.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "Sequence"])
        writer.writeheader()
        writer.writerow({"variant": "var1", "Sequence": "ATGC"})
    variants = _load_variants(path)
    assert variants[0]["name"] == "var1"
    assert variants[0]["sequence"] == "ATGC"


def test_load_recovered_strips_suffix(tmp_path):
    _make_hitlist(tmp_path, ["var1|cons_check", "var2"])
    recovered = _load_recovered(tmp_path / "hitlist.csv")
    assert "var1" in recovered
    assert "var2" in recovered


def test_load_recovered_excludes_empty_wells(tmp_path):
    """Unrecovered variants (TransferVolume=0.0) must not appear in recovered set."""
    _make_hitlist(tmp_path, recovered=["var1"], empties=["var2", "var3"])
    recovered = _load_recovered(tmp_path / "hitlist.csv")
    assert "var1" in recovered
    assert "var2" not in recovered
    assert "var3" not in recovered


def test_write_idt_eblocks_single_plate(tmp_path):
    dropouts = [{"name": f"var{i}", "sequence": "ATGC"} for i in range(5)]
    out = tmp_path / "out.csv"
    n_plates = _write_idt_eblocks(dropouts, out)
    assert n_plates == 1
    with open(out) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["Well Position", "Name", "Sequence"]
    assert rows[1] == ["A1", "var0", "ATGC"]
    assert rows[5] == ["A5", "var4", "ATGC"]


def test_write_idt_eblocks_multi_plate(tmp_path):
    dropouts = [{"name": f"var{i}", "sequence": "ATGC"} for i in range(100)]
    out = tmp_path / "out.csv"
    n_plates = _write_idt_eblocks(dropouts, out)
    assert n_plates == 2


def test_write_idt_eblocks_96_boundary(tmp_path):
    dropouts = [{"name": f"var{i}", "sequence": "ATGC"} for i in range(96)]
    out = tmp_path / "out.csv"
    n_plates = _write_idt_eblocks(dropouts, out)
    assert n_plates == 1


def test_write_twist_gene_fragments(tmp_path):
    dropouts = [{"name": "var1", "sequence": "ATGC"}, {"name": "var2", "sequence": "GCTA"}]
    out = tmp_path / "out.csv"
    _write_twist_gene_fragments(dropouts, out)
    with open(out) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["Sequence name", "Sequence"]
    assert rows[1] == ["var1", "ATGC"]
    assert rows[2] == ["var2", "GCTA"]


def test_write_twist_oligo_pools(tmp_path):
    dropouts = [{"name": "var1", "sequence": "ATGC"}, {"name": "var2", "sequence": "GCTA"}]
    out = tmp_path / "out.csv"
    _write_twist_oligo_pools(dropouts, out)
    with open(out) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["name", "sequence"]
    assert rows[1] == ["var1", "ATGC"]
    assert rows[2] == ["var2", "GCTA"]


def test_write_idt_opools(tmp_path):
    dropouts = [{"name": "var1", "sequence": "ATGC"}, {"name": "var2", "sequence": "GCTA"}]
    out = tmp_path / "out.csv"
    _write_idt_opools(dropouts, out, pool_name="round2_dropouts")
    with open(out) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["Pool name", "Sequence"]
    assert rows[1] == ["round2_dropouts", "ATGC"]
    assert rows[2] == ["round2_dropouts", "GCTA"]


def test_write_idt_opools_default_pool_name(tmp_path):
    dropouts = [{"name": "var1", "sequence": "ATGC"}]
    out = tmp_path / "out.csv"
    _write_idt_opools(dropouts, out, pool_name="dropout_pool")
    with open(out) as f:
        rows = list(csv.reader(f))
    assert rows[1][0] == "dropout_pool"


# ---------------------------------------------------------------------------
# Integration: dropout identification
# ---------------------------------------------------------------------------

def test_dropout_identification(tmp_path):
    variants = [
        {"name": "var1", "sequence": "AAAA"},
        {"name": "var2", "sequence": "CCCC"},
        {"name": "var3", "sequence": "GGGG"},
    ]
    _make_variants_csv(tmp_path, variants)
    _make_hitlist(tmp_path, ["var1", "var2"])

    from usortm.cli.reorder import _load_variants, _load_recovered, _normalize
    all_variants = _load_variants(tmp_path / "variants.csv")
    recovered = _load_recovered(tmp_path / "hitlist.csv")
    dropouts = [v for v in all_variants if _normalize(v["name"]) not in recovered]

    assert len(dropouts) == 1
    assert dropouts[0]["name"] == "var3"
