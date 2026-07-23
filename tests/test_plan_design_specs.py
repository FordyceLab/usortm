"""Tests for library-designer metadata carry-over in `usortm plan`.

library-designer (the upstream design tool) writes a `<name>_design_specs.json`
next to the `variants.csv` it emits. `plan` detects that file, pre-fills the
synthesis method / skew from `spec.platform`, and records the design provenance
in `usortm_project.json` so the upstream design and downstream sort share one
trail.
"""
import csv
import json

from typer.testing import CliRunner

from usortm.cli import app
from usortm.cli.plan import (
    _design_record,
    _load_design_specs,
    _skew_from_platform,
)

runner = CliRunner()


# --- fixtures ----------------------------------------------------------------

def _write_variants(path, n=10):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "sequence"])
        for i in range(n):
            # lowercase flanking (stripped) + uppercase variable region (kept)
            w.writerow([f"variant_{i}", "ggcgc" + "ATGC" * 10 + "cggcg"])


def _specs(platform="twist_oligo_pools", n_variants=10, name="hAcyP1"):
    return {
        "spec": {
            "name": name,
            "protein_sequence": "AEGNTLISVDYE",
            "substitutions": ["F", "Y", "M", "A", "TAG"],
            "adaptor_5": "ggcgcGGTCTCC",
            "adaptor_3": "CCTCTGGcggcg",
            "avoid_enzymes": ["BsaI"],
            "optimization": {"species": "e_coli", "method": "use_best_codon"},
            "platform": platform,
            "seed": 0,
        },
        "seed": 0,
        "versions": {"library_designer": "0.1.0", "dnachisel": "3.2.11"},
        "n_variants": n_variants,
    }


def _write_specs(path, **kwargs):
    path.write_text(json.dumps(_specs(**kwargs), indent=2))


# --- _skew_from_platform -----------------------------------------------------

def test_skew_from_pooled_method_slug():
    skew, slug, label = _skew_from_platform("twist_oligo_pools")
    assert skew == 4.0
    assert slug == "twist_oligo_pools"
    assert label  # human-readable name present


def test_skew_from_arrayed_method_slug_is_uniform():
    skew, slug, label = _skew_from_platform("idt_eblocks")
    assert skew == 1.0
    assert slug == "idt_eblocks"


def test_skew_from_bare_arrayed():
    assert _skew_from_platform("arrayed") == (1.0, None, "arrayed synthesis")


def test_skew_from_bare_pooled_defers_to_prompt():
    # A generic "pooled" platform names no specific method, so it can't fix a skew.
    assert _skew_from_platform("pooled") == (None, None, None)


def test_skew_from_unknown_or_missing_platform():
    assert _skew_from_platform("not_a_real_method") == (None, None, None)
    assert _skew_from_platform(None) == (None, None, None)
    assert _skew_from_platform("") == (None, None, None)


# --- _load_design_specs ------------------------------------------------------

def test_autodetect_design_specs_sibling(tmp_path):
    variants = tmp_path / "variants.csv"
    _write_variants(variants)
    _write_specs(tmp_path / "hAcyP1_design_specs.json")
    design = _load_design_specs(None, variants, library_size=10)
    assert design is not None
    assert design["spec"]["platform"] == "twist_oligo_pools"


def test_autodetect_falls_back_to_provenance_name(tmp_path):
    variants = tmp_path / "variants.csv"
    _write_variants(variants)
    _write_specs(tmp_path / "hAcyP1_provenance.json")  # legacy name
    design = _load_design_specs(None, variants, library_size=10)
    assert design is not None
    assert design["path"].name == "hAcyP1_provenance.json"


def test_autodetect_fixed_name(tmp_path):
    variants = tmp_path / "variants.csv"
    _write_variants(variants)
    _write_specs(tmp_path / "design_specs.json")
    design = _load_design_specs(None, variants, library_size=10)
    assert design is not None


def test_explicit_path(tmp_path):
    variants = tmp_path / "variants.csv"
    _write_variants(variants)
    specs = tmp_path / "elsewhere.json"
    _write_specs(specs)
    design = _load_design_specs(specs, variants, library_size=10)
    assert design is not None
    assert design["path"] == specs


def test_missing_explicit_path_is_soft_none(tmp_path):
    variants = tmp_path / "variants.csv"
    _write_variants(variants)
    assert _load_design_specs(tmp_path / "nope.json", variants, 10) is None


def test_no_specs_returns_none(tmp_path):
    variants = tmp_path / "variants.csv"
    _write_variants(variants)
    assert _load_design_specs(None, variants, 10) is None


def test_multiple_matches_returns_none(tmp_path):
    variants = tmp_path / "variants.csv"
    _write_variants(variants)
    _write_specs(tmp_path / "a_design_specs.json")
    _write_specs(tmp_path / "b_design_specs.json")
    assert _load_design_specs(None, variants, 10) is None


def test_malformed_json_is_soft_none(tmp_path):
    variants = tmp_path / "variants.csv"
    _write_variants(variants)
    (tmp_path / "hAcyP1_design_specs.json").write_text("{not valid json")
    assert _load_design_specs(None, variants, 10) is None


def test_json_without_spec_block_is_ignored(tmp_path):
    variants = tmp_path / "variants.csv"
    _write_variants(variants)
    (tmp_path / "hAcyP1_design_specs.json").write_text('{"something_else": 1}')
    assert _load_design_specs(None, variants, 10) is None


# --- _design_record ----------------------------------------------------------

def test_design_record_captures_provenance(tmp_path):
    specs = tmp_path / "hAcyP1_design_specs.json"
    _write_specs(specs)
    design = {"path": specs, "data": _specs(), "spec": _specs()["spec"]}
    rec = _design_record(design)
    assert rec["source"] == "library-designer"
    assert rec["library_name"] == "hAcyP1"
    assert rec["platform"] == "twist_oligo_pools"
    assert rec["adaptor_5"] == "ggcgcGGTCTCC"
    assert rec["optimization"]["species"] == "e_coli"
    assert rec["tool_versions"]["library_designer"] == "0.1.0"


# --- end-to-end through the CLI ----------------------------------------------

def test_plan_uses_design_specs_and_records_them(tmp_path):
    project = tmp_path / "proj"
    variants = tmp_path / "variants.csv"
    _write_variants(variants)
    _write_specs(tmp_path / "hAcyP1_design_specs.json", platform="twist_oligo_pools")

    result = runner.invoke(app, ["plan", str(variants), "--output", str(project)])
    assert result.exit_code == 0, result.output

    state = json.loads((project / "usortm_project.json").read_text())
    assert state["synthesis_method"] == "twist_oligo_pools"
    assert state["skew"] == 4.0
    assert state["library_design"]["platform"] == "twist_oligo_pools"
    assert state["library_design"]["adaptor_5"] == "ggcgcGGTCTCC"
    assert state["library_design"]["source"] == "library-designer"


def test_plan_explicit_skew_overrides_design_specs(tmp_path):
    project = tmp_path / "proj"
    variants = tmp_path / "variants.csv"
    _write_variants(variants)
    _write_specs(tmp_path / "hAcyP1_design_specs.json", platform="twist_oligo_pools")

    result = runner.invoke(
        app, ["plan", str(variants), "--output", str(project), "--skew", "2.5"]
    )
    assert result.exit_code == 0, result.output
    state = json.loads((project / "usortm_project.json").read_text())
    assert state["skew"] == 2.5
    # The design record is still captured even when skew is user-supplied.
    assert state["library_design"]["platform"] == "twist_oligo_pools"


def test_plan_without_design_specs_has_no_record(tmp_path):
    project = tmp_path / "proj"
    variants = tmp_path / "variants.csv"
    _write_variants(variants)

    result = runner.invoke(app, ["plan", str(variants), "--output", str(project)])
    assert result.exit_code == 0, result.output
    state = json.loads((project / "usortm_project.json").read_text())
    assert state["library_design"] is None
    assert state["synthesis_method"] is None
