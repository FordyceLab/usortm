"""Manage barcode mask presets for LevSeq demultiplexing.

Built-in presets ship with the package; users can install additional
presets into ``~/.usortm/presets/``.
"""
from __future__ import annotations

import shutil
from pathlib import Path

BUILTIN_DIR = Path(__file__).parent / "preset_data"
USER_DIR = Path.home() / ".usortm" / "presets"


def _read_description(toml_path: Path) -> str:
    """Extract the ``[meta] description`` field from a TOML file."""
    try:
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]

        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        return data.get("meta", {}).get("description", "")
    except Exception:
        return ""


def list_presets() -> list[dict]:
    """Return a list of available presets from built-in and user dirs.

    Each entry is ``{name, description, source, path}``.  User presets
    take priority over built-in ones with the same name.
    """
    presets: dict[str, dict] = {}

    # Built-in presets (lower priority)
    if BUILTIN_DIR.is_dir():
        for p in sorted(BUILTIN_DIR.glob("*.toml")):
            name = p.stem
            presets[name] = {
                "name": name,
                "description": _read_description(p),
                "source": "built-in",
                "path": p,
            }

    # User presets (higher priority — override built-in)
    if USER_DIR.is_dir():
        for p in sorted(USER_DIR.glob("*.toml")):
            name = p.stem
            presets[name] = {
                "name": name,
                "description": _read_description(p),
                "source": "user",
                "path": p,
            }

    return list(presets.values())


def get_preset(name: str) -> Path:
    """Resolve a preset name to its TOML file path.

    User directory is checked first so user presets can shadow built-ins.

    Raises:
        FileNotFoundError: If no preset with *name* exists.
    """
    user_path = USER_DIR / f"{name}.toml"
    if user_path.is_file():
        return user_path

    builtin_path = BUILTIN_DIR / f"{name}.toml"
    if builtin_path.is_file():
        return builtin_path

    raise FileNotFoundError(
        f"No preset named '{name}'. Run 'usortm config list' to see available presets."
    )


def add_preset(toml_path: Path, name: str | None = None) -> Path:
    """Install a TOML file as a user preset.

    Args:
        toml_path: Path to the source TOML file.
        name: Optional preset name (defaults to file stem).

    Returns:
        Path to the installed preset file.
    """
    if not toml_path.is_file():
        raise FileNotFoundError(f"TOML file not found: {toml_path}")

    preset_name = name or toml_path.stem
    USER_DIR.mkdir(parents=True, exist_ok=True)
    dest = USER_DIR / f"{preset_name}.toml"
    shutil.copy2(toml_path, dest)
    return dest
