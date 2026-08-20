"""Dependency checker for external tools required by the demux pipeline.

Locates dorado, minimap2, and samtools on the system PATH or via
environment variable overrides. Raises clear errors when tools are missing.
"""

from __future__ import annotations

import glob as glob_mod
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DependencyError(RuntimeError):
    """Raised when a required external tool is not found on PATH."""
    pass


# Common installation locations searched when a tool is not on PATH.
_COMMON_PATHS = {
    "dorado": [
        str(Path.home() / "Downloads" / "dorado-*" / "bin" / "dorado"),
        str(Path.home() / ".dorado" / "bin" / "dorado"),
        "/opt/homebrew/bin/dorado",
        "/usr/local/bin/dorado",
    ],
}


def _search_common_paths(name: str) -> Optional[str]:
    """Search common installation locations for a tool.

    Returns the path to the newest matching executable, or None.
    """
    patterns = _COMMON_PATHS.get(name)
    if not patterns:
        return None

    for pattern in patterns:
        # glob to expand wildcards (e.g. dorado-*/bin/dorado)
        matches = sorted(glob_mod.glob(pattern), reverse=True)
        for match in matches:
            p = Path(match)
            if p.is_file() and os.access(str(p), os.X_OK):
                return str(p.resolve())
    return None


def tool_versions(tools: Optional[dict] = None) -> dict:
    """Record what produced a run: the package, and each external tool.

    Worth storing rather than deriving later, because the answer changes under
    you.  Which dorado is found depends on PATH, and the layout of its output
    changed between versions -- a run that demultiplexed nothing looks the same
    afterwards whichever version did it, unless the version was written down at
    the time.

    Args:
        tools: ``{name: path}`` already resolved, so the versions recorded are
            the binaries actually used rather than whatever is found now.
            Resolved here when omitted.

    Returns:
        ``{name: {"path": ..., "version": ...}}`` plus ``usortm`` and
        ``seqviewer`` entries.  A tool that cannot be queried records its path
        with a null version rather than being left out, since knowing it was
        used and unidentifiable beats a silent gap.
    """
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    out: dict = {}
    for name in ("usortm", "seqviewer"):
        try:
            out[name] = {"version": _pkg_version(name)}
        except PackageNotFoundError:
            out[name] = {"version": None}

    if tools is None:
        tools = {}
        for name, finder in (("dorado", find_dorado),
                             ("minimap2", find_minimap2),
                             ("samtools", find_samtools)):
            try:
                tools[name] = finder()
            except Exception:
                continue

    for name, path in tools.items():
        entry = {"path": str(path), "version": None}
        try:
            result = subprocess.run([str(path), "--version"],
                                    capture_output=True, text=True, timeout=10)
            text = (result.stdout.strip() or result.stderr.strip())
            if text:
                # samtools puts its version on the first line after the name;
                # dorado and minimap2 print it alone.
                first = text.splitlines()[0].strip()
                entry["version"] = first.split()[-1] if " " in first else first
        except Exception:
            pass
        out[name] = entry
    return out


def _version_tuple(version_string: str) -> tuple:
    """Parse a version string like '1.3.1+abc' into (1, 3, 1)."""
    clean = version_string.strip().split("+")[0]
    parts = []
    for part in clean.split(".")[:3]:
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)


def find_tool(name: str, env_var: Optional[str] = None) -> str:
    """Locate an executable on PATH or via an environment variable.

    Search order:
        1. Environment variable override (e.g. DORADO_PATH)
        2. System PATH lookup
        3. Common installation locations (~/Downloads, /opt/homebrew, etc.)

    Args:
        name: Name of the executable (e.g. "dorado").
        env_var: Optional environment variable that overrides PATH lookup.

    Returns:
        Absolute path to the executable.

    Raises:
        DependencyError: If the tool cannot be found.
    """
    # 1. Check environment variable override first
    if env_var:
        custom_path = os.environ.get(env_var)
        if custom_path and Path(custom_path).is_file():
            return str(Path(custom_path).resolve())

    # 2. Fall back to PATH lookup
    path = shutil.which(name)
    if path is not None:
        return path

    # 3. Search common installation locations
    common = _search_common_paths(name)
    if common is not None:
        logger.info(
            "%s not on PATH, found at common location: %s", name, common
        )
        return common

    hint = f" or set the {env_var} environment variable" if env_var else ""
    raise DependencyError(
        f"'{name}' not found on PATH. "
        f"Please install {name}{hint}."
    )


def find_dorado(min_version: str = "1.0.0") -> str:
    """Locate the Dorado basecaller/demuxer.

    Args:
        min_version: Minimum recommended version. A warning is logged
            if the found binary is older than this.

    Returns:
        Absolute path to the dorado executable.
    """
    path = find_tool("dorado", env_var="DORADO_PATH")

    # Version check (advisory, does not raise)
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        version_str = result.stdout.strip() or result.stderr.strip()
        found = _version_tuple(version_str)
        required = _version_tuple(min_version)
        if found and required and found < required:
            logger.warning(
                "dorado %s found at %s, but >= %s is recommended. "
                "Set DORADO_PATH to override.",
                version_str, path, min_version,
            )
    except Exception:
        pass  # Don't fail on version check

    return path


def find_minimap2() -> str:
    """Locate minimap2 aligner."""
    return find_tool("minimap2", env_var="MINIMAP2_PATH")


def find_samtools() -> str:
    """Locate samtools."""
    return find_tool("samtools", env_var="SAMTOOLS_PATH")


#: What this package calls on seqviewer.  Checked by name at startup rather
#: than by version, because the install that matters is usually editable: a
#: checkout on disk whose version string says nothing about which commit is
#: there, and which shadows the pin in pyproject.toml entirely.
SEQVIEWER_REQUIRED = {
    "seqviewer": ("PileupGroup", "PileupView", "Read", "grid_from_reads",
                  "reads_from_alignment", "render"),
}
SEQVIEWER_REQUIRED_FIELDS = {"PileupGroup": ("parent",)}


def check_seqviewer() -> None:
    """Fail early when the installed seqviewer is not the one expected.

    The pileups are drawn by seqviewer, and the two packages have to agree on
    the shape of what is passed between them.  An editable checkout can drift
    from the pinned commit without anything saying so, and the disagreement
    then surfaces as a TypeError from the first pileup -- which on a real run
    is more than an hour in, after the expensive stages have already run.

    Raises:
        DependencyError: If seqviewer is absent, or is missing something this
            package uses.
    """
    import importlib

    try:
        module = importlib.import_module("seqviewer")
    except ImportError as exc:
        raise DependencyError(
            "seqviewer is required for pileups but is not installed.\n"
            "  Install it with: pip install -e '.[demux]'"
        ) from exc

    missing = [name for name in SEQVIEWER_REQUIRED["seqviewer"]
               if not hasattr(module, name)]
    for cls_name, fields in SEQVIEWER_REQUIRED_FIELDS.items():
        cls = getattr(module, cls_name, None)
        if cls is None:
            continue
        import dataclasses

        try:
            present = {f.name for f in dataclasses.fields(cls)}
        except TypeError:
            continue
        missing += [f"{cls_name}.{f}" for f in fields if f not in present]

    if missing:
        raise DependencyError(
            "The installed seqviewer is not the version uSort-M expects.\n"
            f"  Missing: {', '.join(missing)}\n"
            f"  Loaded from: {getattr(module, '__file__', 'unknown')}\n"
            "  An editable checkout overrides the pin in pyproject.toml, so "
            "updating one package without the other leaves them disagreeing.\n"
            "  Fix with: pip install -e '.[demux]' --force-reinstall\n"
            "  or bring the checkout to the pinned commit."
        )


def check_all_dependencies() -> dict[str, str]:
    """Validate that all required tools are available.

    Returns:
        Dict mapping tool name to its absolute path.

    Raises:
        DependencyError: On the first tool that cannot be found, or when the
            installed seqviewer does not match what this package calls.
    """
    check_seqviewer()
    return {
        "dorado": find_dorado(),
        "minimap2": find_minimap2(),
        "samtools": find_samtools(),
    }
