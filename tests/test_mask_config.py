"""Tests for barcode mask RC derivation, TOML loading, and preset system."""

import pytest
from pathlib import Path

from usortm.demux.barcodes import (
    reverse_complement,
    fbc_to_rbc_masks,
    DEFAULT_MASKS,
)


def test_reverse_complement_basic():
    assert reverse_complement("ATGC") == "GCAT"


def test_reverse_complement_lowercase():
    assert reverse_complement("atgc") == "gcat"


def test_reverse_complement_mixed():
    assert reverse_complement("AtGc") == "gCaT"


def test_reverse_complement_roundtrip():
    seq = "AATATAAATT"
    assert reverse_complement(reverse_complement(seq)) == seq


def test_fbc_to_rbc_masks_matches_defaults():
    """Derived RBC masks should match the hand-written DEFAULT_MASKS."""
    derived = fbc_to_rbc_masks(DEFAULT_MASKS["fbc"])
    expected = DEFAULT_MASKS["rbc"]

    assert derived["mask1_front"] == expected["mask1_front"]
    assert derived["mask1_rear"] == expected["mask1_rear"]
    assert derived["mask2_front"] == expected["mask2_front"]
    assert derived["mask2_rear"] == expected["mask2_rear"]


def test_fbc_to_rbc_masks_swap_pattern():
    """Verify the swap+RC pattern: rbc.mask1_front = RC(fbc.mask2_rear), etc."""
    fbc = {
        "mask1_front": "AAAA",
        "mask1_rear": "CCCC",
        "mask2_front": "GGGG",
        "mask2_rear": "TTTT",
    }
    rbc = fbc_to_rbc_masks(fbc)

    assert rbc["mask1_front"] == reverse_complement("TTTT")   # RC(fbc.mask2_rear)
    assert rbc["mask1_rear"] == reverse_complement("GGGG")    # RC(fbc.mask2_front)
    assert rbc["mask2_front"] == reverse_complement("CCCC")   # RC(fbc.mask1_rear)
    assert rbc["mask2_rear"] == reverse_complement("AAAA")    # RC(fbc.mask1_front)


def test_load_mask_config_fbc_only(tmp_path):
    """Loading a TOML with only [fbc] should auto-derive [rbc]."""
    from usortm.cli.demux_cmd import _load_mask_config

    toml_content = """
[meta]
description = "test"

[fbc]
mask1_front = "AATATAAATT"
mask1_rear  = "CTGAGATACCTACAGCGTGAGC"
mask2_front = "CAAGTGAGAAATCACCATGAGTGACG"
mask2_rear  = "ATAATTTATA"
"""
    toml_file = tmp_path / "test_mask.toml"
    toml_file.write_text(toml_content)

    config = _load_mask_config(toml_file)

    assert "fbc" in config
    assert "rbc" in config
    assert config["rbc"]["mask1_front"] == reverse_complement("ATAATTTATA")


def test_load_mask_config_full_format(tmp_path):
    """Loading a TOML with both [fbc] and [rbc] should use both as-is."""
    from usortm.cli.demux_cmd import _load_mask_config

    toml_content = """
[fbc]
mask1_front = "AAA"
mask1_rear  = "CCC"
mask2_front = "GGG"
mask2_rear  = "TTT"

[rbc]
mask1_front = "CUSTOM1"
mask1_rear  = "CUSTOM2"
mask2_front = "CUSTOM3"
mask2_rear  = "CUSTOM4"
"""
    toml_file = tmp_path / "full_mask.toml"
    toml_file.write_text(toml_content)

    config = _load_mask_config(toml_file)

    assert config["rbc"]["mask1_front"] == "CUSTOM1"
    assert config["rbc"]["mask2_rear"] == "CUSTOM4"


def test_preset_list():
    """Built-in presets should include T7_default."""
    from usortm.demux.presets import list_presets

    presets = list_presets()
    names = {p["name"] for p in presets}
    assert "T7_default" in names

    t7 = next(p for p in presets if p["name"] == "T7_default")
    assert t7["source"] == "built-in"
    assert t7["description"]  # should have a description


def test_preset_get():
    """get_preset should resolve built-in names."""
    from usortm.demux.presets import get_preset

    path = get_preset("T7_default")
    assert path.is_file()
    assert path.name == "T7_default.toml"


def test_preset_get_not_found():
    """get_preset should raise for unknown names."""
    from usortm.demux.presets import get_preset

    with pytest.raises(FileNotFoundError, match="No preset named"):
        get_preset("nonexistent_preset_xyz")


def test_preset_add(tmp_path, monkeypatch):
    """add_preset should copy a TOML into a writable user dir."""
    from usortm.demux import presets

    source = tmp_path / "my_custom.toml"
    source.write_text('[meta]\ndescription = "custom"\n\n[fbc]\nmask1_front = "AAA"\n')

    # Redirect USER_DIR so the test never writes to ~/.usortm.
    monkeypatch.setattr(presets, "USER_DIR", tmp_path / "presets")

    # Use a custom name to avoid collisions with any real user preset names.
    dest = presets.add_preset(source, name="test_custom_preset_xyz")
    assert dest.is_file()
    assert "test_custom_preset_xyz" in dest.name
