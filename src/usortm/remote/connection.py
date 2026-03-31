"""SSH connection management for remote uSort-M jobs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


CONFIG_DIR = Path.home() / ".usortm"
CONFIG_FILE = CONFIG_DIR / "remote.toml"


def _find_ssh_key() -> Optional[str]:
    """Auto-detect SSH key from common locations."""
    ssh_dir = Path.home() / ".ssh"
    for name in ("id_ed25519", "id_rsa", "id_ecdsa"):
        key = ssh_dir / name
        if key.exists():
            return str(key)
    return None


def load_config() -> dict:
    """Load remote configuration from ~/.usortm/remote.toml."""
    if not CONFIG_FILE.exists():
        return {}
    with open(CONFIG_FILE, "rb") as f:
        return tomllib.load(f)


def save_config(config: dict) -> None:
    """Save remote configuration to ~/.usortm/remote.toml."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for section, values in config.items():
        lines.append(f"[{section}]")
        for key, val in values.items():
            if isinstance(val, str):
                lines.append(f'{key} = "{val}"')
            elif isinstance(val, (int, float)):
                lines.append(f"{key} = {val}")
            elif isinstance(val, bool):
                lines.append(f"{key} = {'true' if val else 'false'}")
        lines.append("")
    CONFIG_FILE.write_text("\n".join(lines))


def get_connection(
    host: Optional[str] = None,
    user: Optional[str] = None,
    key_path: Optional[str] = None,
):
    """Create a Fabric SSH connection, falling back to config defaults.

    Returns a ``fabric.Connection`` instance.  Raises ``ImportError`` if
    fabric is not installed (it is an optional dependency).
    """
    try:
        from fabric import Connection
    except ImportError:
        print(
            "The 'remote' extra is required for remote execution.\n"
            "Install it with:  pip install usortm[remote]",
            file=sys.stderr,
        )
        raise

    cfg = load_config().get("connection", {})
    host = host or cfg.get("host")
    user = user or cfg.get("user")
    key_path = key_path or cfg.get("key_path") or _find_ssh_key()

    if not host or not user:
        raise ValueError(
            "host and user are required.  Provide them as arguments or "
            "configure defaults with: usortm remote config init"
        )

    connect_kwargs = {}
    if key_path:
        connect_kwargs["key_filename"] = str(Path(key_path).expanduser())

    return Connection(
        host=host,
        user=user,
        connect_timeout=10,
        connect_kwargs=connect_kwargs,
    )


def resolve_remote_home(conn) -> str:
    """Expand ``$HOME`` on the remote server."""
    result = conn.run("echo $HOME", hide=True)
    return result.stdout.strip()


def expand_remote_tilde(conn, path: str) -> str:
    """Replace leading ``~`` in *path* with the remote home directory."""
    if path.startswith("~"):
        return resolve_remote_home(conn) + path[1:]
    return path
