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


def check_all_dependencies() -> dict[str, str]:
    """Validate that all required tools are available.

    Returns:
        Dict mapping tool name to its absolute path.

    Raises:
        DependencyError: On the first tool that cannot be found.
    """
    return {
        "dorado": find_dorado(),
        "minimap2": find_minimap2(),
        "samtools": find_samtools(),
    }
