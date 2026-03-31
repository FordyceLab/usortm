"""CLI subcommands for remote job execution."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn, TextColumn
from rich import box

from usortm.cli.theme import get_console, BORDER_STYLE

console = get_console()

remote_app = typer.Typer(
    help="Run uSort-M jobs on a remote server via SSH.",
    add_completion=False,
    rich_markup_mode="rich",
)


# ── config ───────────────────────────────────────────────────────────


@remote_app.command(name="config")
def remote_config(
    init: bool = typer.Option(False, "--init", help="Interactive setup of SSH connection."),
    show: bool = typer.Option(False, "--show", help="Display current configuration."),
):
    """Configure or display SSH connection settings."""
    from usortm.remote.connection import load_config, save_config, CONFIG_FILE

    if show or (not init and not show):
        cfg = load_config()
        if not cfg:
            console.print(f"[yellow]No config found.[/yellow] Run: [cyan]usortm remote config --init[/cyan]")
            return
        console.print(Panel.fit(
            f"[bold]Config:[/bold] {CONFIG_FILE}",
            border_style=BORDER_STYLE,
        ))
        conn = cfg.get("connection", {})
        table = Table(box=box.SIMPLE)
        table.add_column("Setting", style="bold")
        table.add_column("Value")
        for key, val in conn.items():
            table.add_row(key, str(val))
        console.print(table)
        if not init:
            return

    if init:
        import questionary

        cfg = load_config()
        conn = cfg.get("connection", {})

        host = questionary.text(
            "Remote host (hostname or IP):",
            default=conn.get("host", ""),
        ).ask()
        if host is None:
            raise typer.Abort()

        user = questionary.text(
            "SSH username:",
            default=conn.get("user", ""),
        ).ask()
        if user is None:
            raise typer.Abort()

        from usortm.remote.connection import _find_ssh_key
        default_key = conn.get("key_path", _find_ssh_key() or "")
        key_path = questionary.text(
            "SSH key path:",
            default=default_key,
        ).ask()
        if key_path is None:
            raise typer.Abort()

        remote_job_dir = questionary.text(
            "Remote job directory:",
            default=conn.get("remote_job_dir", "~/usortm_jobs"),
        ).ask()
        if remote_job_dir is None:
            raise typer.Abort()

        usortm_path = questionary.text(
            "Path to usortm on remote (leave blank to auto-detect):",
            default=conn.get("usortm_path", ""),
        ).ask()
        if usortm_path is None:
            raise typer.Abort()

        connection_cfg = {
            "host": host,
            "user": user,
            "key_path": key_path,
            "remote_job_dir": remote_job_dir,
        }
        if usortm_path.strip():
            connection_cfg["usortm_path"] = usortm_path.strip()

        save_config({"connection": connection_cfg})
        console.print(f"[green]\u2713[/green] Config saved to {CONFIG_FILE}")


# ── verify ───────────────────────────────────────────────────────────


@remote_app.command(name="verify")
def remote_verify():
    """Check that required tools are installed on the remote server."""
    from usortm.remote import RemoteDemux

    console.print(Panel.fit(
        "[brand]uSort-M[/brand] Remote Environment Check",
        border_style=BORDER_STYLE,
    ))

    try:
        mgr = RemoteDemux()
    except Exception as e:
        console.print(f"[red]Connection failed:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"Connected to [bold]{mgr.conn.host}[/bold]")
    console.print()

    results = mgr.verify_remote()

    for tool, info in results["tools"].items():
        if info["found"]:
            ver = f" ({info['version']})" if info["version"] else ""
            console.print(f"  [green]\u2713[/green] {tool}{ver}")
        else:
            console.print(f"  [red]\u2717[/red] {tool} — not found")
            console.print(f"    Install: {info['install']}")

    if "disk" in results:
        console.print(f"\n  Disk: {results['disk']['available']} available")

    console.print()
    if results["ok"]:
        console.print("[green]All required tools found.[/green]")
    else:
        missing = [t for t, s in results["tools"].items() if not s["found"]]
        console.print(f"[red]Missing:[/red] {', '.join(missing)}")
        raise typer.Exit(1)


# ── demux ────────────────────────────────────────────────────────────


@remote_app.command(name="demux")
def remote_demux(
    project_dir: Path = typer.Argument(
        ...,
        help="Path to uSort-M project directory.",
        exists=True,
    ),
    fastq: Optional[Path] = typer.Option(
        None,
        "--fastq", "-f",
        help="Local FASTQ file to upload to the remote server.",
    ),
    remote_fastq: Optional[str] = typer.Option(
        None,
        "--remote-fastq",
        help="Path to FASTQ file already on the remote server.",
    ),
    reference: Optional[Path] = typer.Option(
        None,
        "--reference", "-r",
        help="Reference FASTA for alignment.",
    ),
    library_csv: Optional[Path] = typer.Option(
        None,
        "--library-csv", "-l",
        help="Library CSV with Name,Sequence columns (auto-converted to reference FASTA).",
    ),
    vector_fasta: Optional[Path] = typer.Option(
        None,
        "--vector-fasta",
        help="Full vector FASTA for flanking region detection.",
        exists=True,
    ),
    mask_config: Optional[Path] = typer.Option(
        None,
        "--mask-config",
        help="Barcode mask configuration TOML file.",
        exists=True,
    ),
    threads: int = typer.Option(8, "--threads", "-t", help="Threads for minimap2."),
    workers: int = typer.Option(4, "--workers", "-w", help="Parallel workers for consensus."),
    subsample: Optional[int] = typer.Option(None, "--subsample", help="Subsample N reads."),
):
    """Submit a demux job to the remote server.

    Either --fastq (local file, will be uploaded) or --remote-fastq
    (path on the server) must be provided.
    """
    from usortm.remote import RemoteDemux

    console.print(Panel.fit(
        "[brand]uSort-M[/brand] Remote Demux",
        border_style=BORDER_STYLE,
    ))

    if not fastq and not remote_fastq:
        console.print("[red]Error:[/red] Provide --fastq or --remote-fastq")
        raise typer.Exit(1)
    if not reference and not library_csv:
        console.print("[red]Error:[/red] Provide --reference or --library-csv")
        raise typer.Exit(1)

    try:
        mgr = RemoteDemux()
    except Exception as e:
        console.print(f"[red]Connection failed:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[green]\u2713[/green] Connected to [bold]{mgr.conn.host}[/bold]")

    # Show resolved usortm path so user can verify before uploading
    from usortm.remote.connection import load_config as _load_cfg
    _cfg = _load_cfg().get("connection", {})
    _usortm_path = _cfg.get("usortm_path") or mgr._find_remote_usortm()
    if _usortm_path == "usortm":
        console.print(
            "[yellow]Warning:[/yellow] Could not find usortm on remote. "
            "Run [cyan]usortm remote config --init[/cyan] to set the path."
        )
    else:
        console.print(f"[green]\u2713[/green] usortm: [dim]{_usortm_path}[/dim]")

    upload_callback = None
    progress_ctx = None

    if fastq:
        file_size = fastq.stat().st_size
        progress_ctx = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        progress_ctx.start()
        task_id = progress_ctx.add_task(f"Uploading {fastq.name}", total=file_size)

        def upload_callback(transferred: int, total: int):
            progress_ctx.update(task_id, completed=transferred)

    try:
        job_key, fastq_uploaded = mgr.submit(
            project_dir=project_dir,
            fastq=fastq,
            remote_fastq=remote_fastq,
            reference=reference,
            library_csv=library_csv,
            vector_fasta=vector_fasta,
            mask_config=mask_config,
            threads=threads,
            workers=workers,
            subsample=subsample,
            upload_callback=upload_callback,
        )
    finally:
        if progress_ctx:
            progress_ctx.stop()

    if fastq and not fastq_uploaded:
        console.print("[green]\u2713[/green] FASTQ already on remote — skipped upload")

    console.print(f"[green]\u2713[/green] Job submitted: [bold]{job_key}[/bold]")
    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print(f"  [cyan]usortm remote status {project_dir}/[/cyan]   \u2192 Check progress")
    console.print(f"  [cyan]usortm remote log {project_dir}/[/cyan]      \u2192 View remote log")
    console.print(f"  [cyan]usortm remote fetch {project_dir}/[/cyan]    \u2192 Download results")


# ── status ───────────────────────────────────────────────────────────


@remote_app.command(name="status")
def remote_status(
    project_dir: Path = typer.Argument(
        ...,
        help="Path to uSort-M project directory.",
        exists=True,
    ),
):
    """Check the status of a remote demux job."""
    from usortm.remote.demux import RemoteDemux

    try:
        mgr, job_id = RemoteDemux.from_project(project_dir)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Connection failed:[/red] {e}")
        raise typer.Exit(1)

    info = mgr.get_detailed_status(job_id)
    status = info["status"]

    status_color = {"RUNNING": "yellow", "COMPLETED": "green", "FAILED": "red"}.get(status, "white")

    console.print()
    job_key = info.get("job_key", "?")
    console.print(Panel.fit(
        f"[brand]uSort-M[/brand] Remote Demux  ·  [bold]{job_key}[/bold]  ·  [{status_color}]{status}[/{status_color}]",
        border_style=BORDER_STYLE,
    ))
    console.print()

    stages = info.get("stages", [])
    n_done = sum(1 for s in stages if s["done"])

    # Find the current (in-progress) stage index
    current_idx = None
    if status == "RUNNING":
        # Last done stage is current; next is pending
        for i in range(len(stages) - 1, -1, -1):
            if stages[i]["done"]:
                current_idx = i
                break
        if current_idx is None:
            current_idx = 0  # nothing done yet, first stage is current

    for i, stage in enumerate(stages):
        label = stage["label"]
        done = stage["done"]

        if done and (current_idx is None or i < current_idx or status != "RUNNING"):
            icon = "[green]\u2713[/green]"
            style = ""
        elif done and i == current_idx and status == "RUNNING":
            icon = "[yellow]\u25b6[/yellow]"
            style = "[bold]"
        elif i == current_idx and status == "RUNNING" and not done:
            icon = "[yellow]\u25b6[/yellow]"
            style = "[bold]"
        else:
            icon = "[dim]\u25cb[/dim]"
            style = "[dim]"

        console.print(f"  {icon}  {style}{label}{'[/bold]' if style == '[bold]' else '[/dim]' if style == '[dim]' else ''}")

    # Last log line for context while running
    last_line = info.get("last_log_line", "").strip()
    if last_line and status == "RUNNING":
        console.print()
        console.print(f"  [dim]{last_line}[/dim]")

    console.print()
    if status == "COMPLETED":
        console.print(f"[bold]Next:[/bold] [cyan]usortm remote fetch {project_dir}/[/cyan]")
    elif status == "FAILED":
        console.print(f"[bold]Check log:[/bold] [cyan]usortm remote log {project_dir}/[/cyan]")
    console.print()


# ── log ──────────────────────────────────────────────────────────────


@remote_app.command(name="log")
def remote_log(
    project_dir: Path = typer.Argument(
        ...,
        help="Path to uSort-M project directory.",
        exists=True,
    ),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of log lines to show."),
):
    """View the remote job log."""
    from usortm.remote.demux import RemoteDemux

    try:
        mgr, job_id = RemoteDemux.from_project(project_dir)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Connection failed:[/red] {e}")
        raise typer.Exit(1)

    log_text = mgr.get_log(job_id, lines=lines)
    console.print(log_text)


# ── fetch ────────────────────────────────────────────────────────────


@remote_app.command(name="fetch")
def remote_fetch(
    project_dir: Path = typer.Argument(
        ...,
        help="Path to uSort-M project directory.",
        exists=True,
    ),
    read_data: bool = typer.Option(
        False,
        "--read-data/--no-read-data",
        help="Also download read_df.csv and per-variant FASTAs (needed for pileups).",
    ),
):
    """Download demux results from the remote server.

    By default downloads only metadata (well_df.csv, well_assignments.csv,
    demux_summary.json).  Add --read-data to also fetch the files needed
    for pileup generation during pick.
    """
    from usortm.remote.demux import RemoteDemux

    try:
        mgr, job_key = RemoteDemux.from_project(project_dir)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Connection failed:[/red] {e}")
        raise typer.Exit(1)

    # Check status first
    info = mgr.status(job_key)
    if info["status"] not in ("COMPLETED",):
        console.print(
            f"[yellow]Warning:[/yellow] Job status is {info['status']}. "
            f"Results may be incomplete."
        )

    console.print(f"Fetching metadata for [bold]{job_key}[/bold]...")
    local_demux = mgr.fetch_metadata(job_key, project_dir)
    console.print(f"[green]\u2713[/green] Metadata saved to {local_demux}")

    if read_data:
        console.print("Fetching read data (this may take a while)...")
        mgr.fetch_read_data(job_key, project_dir)
        console.print(f"[green]\u2713[/green] Read data saved to {local_demux}")

    console.print()
    console.print(f"[bold]Next:[/bold] [cyan]usortm pick {project_dir}/[/cyan]")
    if not read_data:
        console.print(
            "  [dim](pileups will be skipped — re-run with --read-data to enable)[/dim]"
        )


# ── cancel ───────────────────────────────────────────────────────────


@remote_app.command(name="cancel")
def remote_cancel(
    project_dir: Path = typer.Argument(
        ...,
        help="Path to uSort-M project directory.",
        exists=True,
    ),
):
    """Cancel a running remote demux job."""
    from usortm.remote.demux import RemoteDemux

    try:
        mgr, job_key = RemoteDemux.from_project(project_dir)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Connection failed:[/red] {e}")
        raise typer.Exit(1)

    if mgr.cancel(job_key):
        console.print(f"[green]\u2713[/green] Cancelled job [bold]{job_key}[/bold]")
    else:
        console.print(f"[yellow]Could not cancel job {job_key} (may have already finished)[/yellow]")


# ── list ─────────────────────────────────────────────────────────────


@remote_app.command(name="list")
def remote_list():
    """List all remote job directories with their status and size."""
    from usortm.remote.demux import RemoteDemux

    try:
        mgr = RemoteDemux()
    except Exception as e:
        console.print(f"[red]Connection failed:[/red] {e}")
        raise typer.Exit(1)

    jobs = mgr.list_jobs()
    if not jobs:
        console.print("[dim]No remote jobs found.[/dim]")
        return

    from datetime import datetime

    table = Table(box=box.SIMPLE)
    table.add_column("Job Key", style="bold")
    table.add_column("Status")
    table.add_column("Size")
    table.add_column("Last Modified")

    status_colors = {"COMPLETED": "green", "FAILED": "red", "RUNNING": "yellow"}

    for job in sorted(jobs, key=lambda j: j["mtime"], reverse=True):
        status = job["status"]
        color = status_colors.get(status, "white")
        age = ""
        if job["mtime"]:
            dt = datetime.fromtimestamp(job["mtime"])
            age = dt.strftime("%Y-%m-%d %H:%M")
        table.add_row(job["job_key"], f"[{color}]{status}[/{color}]", job["size"], age)

    console.print(table)


# ── clean ────────────────────────────────────────────────────────────


@remote_app.command(name="clean")
def remote_clean(
    project_dir: Optional[Path] = typer.Argument(
        None,
        help="Project directory to keep (its job key is preserved). Omit to review all.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without deleting."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
):
    """Delete orphaned remote job directories.

    If a project directory is given, its job key is kept; all other job
    directories on the remote are candidates for deletion.  Without a
    project directory, all jobs are listed and you choose interactively.
    """
    from usortm.remote.demux import RemoteDemux

    try:
        mgr = RemoteDemux()
    except Exception as e:
        console.print(f"[red]Connection failed:[/red] {e}")
        raise typer.Exit(1)

    # Determine which keys to keep
    keep_keys: list[str] = []
    if project_dir:
        import json
        state_file = project_dir / "usortm_project.json"
        if state_file.exists():
            with open(state_file) as f:
                project = json.load(f)
            key = project.get("workflow_steps", {}).get("demux", {}).get("remote", {}).get("job_key")
            if key:
                keep_keys.append(key)
                console.print(f"Keeping current project job: [bold]{key}[/bold]")

    jobs = mgr.list_jobs()
    to_delete = [j for j in jobs if j["job_key"] not in keep_keys]

    if not to_delete:
        console.print("[green]Nothing to clean.[/green]")
        return

    from datetime import datetime
    from rich.status import Status as RichStatus

    table = Table(box=box.SIMPLE, title="Jobs to delete" if not dry_run else "Would delete")
    table.add_column("Job Key", style="bold")
    table.add_column("Status")
    table.add_column("Size")
    table.add_column("Last Modified")

    status_colors = {"COMPLETED": "green", "FAILED": "red", "RUNNING": "yellow"}
    for job in sorted(to_delete, key=lambda j: j["mtime"], reverse=True):
        status = job["status"]
        color = status_colors.get(status, "white")
        age = datetime.fromtimestamp(job["mtime"]).strftime("%Y-%m-%d %H:%M") if job["mtime"] else ""
        table.add_row(job["job_key"], f"[{color}]{status}[/{color}]", job["size"], age)

    console.print(table)

    if dry_run:
        console.print("[dim]Dry run — nothing deleted.[/dim]")
        return

    if not yes:
        import questionary
        confirmed = questionary.confirm(
            f"Delete {len(to_delete)} job director{'y' if len(to_delete) == 1 else 'ies'}?",
            default=False,
        ).ask()
        if not confirmed:
            console.print("[dim]Aborted.[/dim]")
            return

    deleted = mgr.clean(keep_keys=keep_keys)
    console.print(f"[green]\u2713[/green] Deleted {len(deleted)} job director{'y' if len(deleted) == 1 else 'ies'}.")
