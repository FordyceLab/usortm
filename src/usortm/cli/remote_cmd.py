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
    fastq_url: Optional[str] = typer.Option(
        None,
        "--fastq-url", "-u",
        help="URL for the remote server to wget directly (faster when remote has better connectivity).",
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

    One of --fastq (local upload), --remote-fastq (already on server),
    or --fastq-url (remote wget) must be provided.  Use --fastq-url when
    the sequencing provider gives a download link and the remote has faster
    network access than your local machine.  Gzipped files (.fastq.gz) are
    supported and passed through as-is.
    """
    from usortm.remote import RemoteDemux

    console.print(Panel.fit(
        "[brand]uSort-M[/brand] Remote Demux",
        border_style=BORDER_STYLE,
    ))

    if not fastq and not remote_fastq and not fastq_url:
        console.print("[red]Error:[/red] Provide --fastq, --remote-fastq, or --fastq-url")
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

    if fastq_url:
        url_filename = fastq_url.split("?")[0].rstrip("/").split("/")[-1] or "reads.fastq"
        console.print(f"[dim]FASTQ will be downloaded on remote: [bold]{url_filename}[/bold][/dim]")

    if fastq:
        # Compute total size: sum of files if directory, single file otherwise
        if fastq.is_dir():
            file_size = sum(
                p.stat().st_size for p in fastq.iterdir()
                if p.name.endswith((".fastq", ".fastq.gz", ".fq", ".fq.gz", ".zip"))
            )
        else:
            file_size = fastq.stat().st_size
        _progress_state: dict = {}
        _progress_transferred: dict = {"total": 0, "prev_file_done": 0}

        def upload_callback(transferred: int, total: int):
            if "ctx" not in _progress_state:
                ctx = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                )
                ctx.start()
                _progress_state["ctx"] = ctx
                _progress_state["task"] = ctx.add_task(f"Uploading {fastq.name}", total=file_size)
            # For directory uploads (multiple files), accumulate across files
            if fastq.is_dir():
                current = _progress_transferred["prev_file_done"] + transferred
                _progress_state["ctx"].update(_progress_state["task"], completed=current)
                if transferred == total:
                    _progress_transferred["prev_file_done"] = current
            else:
                _progress_state["ctx"].update(_progress_state["task"], completed=transferred)

        def _get_progress_ctx():
            return _progress_state.get("ctx")
    else:
        def _get_progress_ctx():
            return None

    def _log_status(msg: str):
        console.print(f"  [dim]{msg}[/dim]")

    try:
        job_key, fastq_uploaded = mgr.submit(
            project_dir=project_dir,
            fastq=fastq,
            remote_fastq=remote_fastq,
            fastq_url=fastq_url,
            reference=reference,
            library_csv=library_csv,
            vector_fasta=vector_fasta,
            mask_config=mask_config,
            threads=threads,
            workers=workers,
            subsample=subsample,
            upload_callback=upload_callback,
            log_status=_log_status,
        )
    finally:
        ctx = _get_progress_ctx()
        if ctx:
            ctx.stop()
        mgr.conn.close()

    if fastq_url and not fastq_uploaded:
        console.print("[green]\u2713[/green] FASTQ already on remote — download skipped")
    elif fastq and not fastq_uploaded:
        console.print("[green]\u2713[/green] FASTQ already on remote — skipped upload")

    console.print(f"[green]\u2713[/green] Job submitted: [bold]{job_key}[/bold]")
    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print(f"  [cyan]usortm remote status {project_dir}/[/cyan]   \u2192 Check progress")
    console.print(f"  [cyan]usortm remote log {project_dir}/[/cyan]      \u2192 View remote log")
    console.print(f"  [cyan]usortm remote fetch {project_dir}/[/cyan]    \u2192 Download results")


# ── status ───────────────────────────────────────────────────────────


def _render_status(info: dict, project_dir):
    """Build the status display as a Rich renderable Group."""
    from rich.console import Group
    from rich.text import Text

    parts = []
    status = info["status"]
    status_color = {"RUNNING": "yellow", "COMPLETED": "green", "FAILED": "red"}.get(status, "white")
    job_key = info.get("job_key", "?")

    parts.append(Text(""))
    parts.append(Panel.fit(
        f"[brand]uSort-M[/brand] Remote Demux  ·  [bold]{job_key}[/bold]  ·  [{status_color}]{status}[/{status_color}]",
        border_style=BORDER_STYLE,
    ))
    parts.append(Text(""))

    stages = info.get("stages", [])
    current_idx = None
    if status == "RUNNING":
        for i in range(len(stages) - 1, -1, -1):
            if stages[i]["done"]:
                current_idx = i
                break
        if current_idx is None:
            current_idx = 0

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

        close = "[/bold]" if style == "[bold]" else "[/dim]" if style == "[dim]" else ""
        parts.append(Text.from_markup(f"  {icon}  {style}{label}{close}"))

    last_line = info.get("last_log_line", "").strip()
    if last_line and status == "RUNNING":
        parts.append(Text(""))
        parts.append(Text.from_markup(f"  [dim]{last_line}[/dim]"))

    parts.append(Text(""))
    if status == "COMPLETED":
        parts.append(Text.from_markup(f"[bold]Next:[/bold] [cyan]usortm remote fetch {project_dir}/[/cyan]"))
    elif status == "FAILED":
        parts.append(Text.from_markup(f"[bold]Check log:[/bold] [cyan]usortm remote log {project_dir}/[/cyan]"))
    parts.append(Text(""))

    return Group(*parts)


@remote_app.command(name="status")
def remote_status(
    project_dir: Path = typer.Argument(
        ...,
        help="Path to uSort-M project directory.",
        exists=True,
    ),
    watch: bool = typer.Option(False, "--watch", "-w", help="Auto-refresh until job completes."),
    interval: int = typer.Option(15, "--interval", "-i", help="Refresh interval in seconds (with --watch)."),
):
    """Check the status of a remote demux job."""
    import time
    from usortm.remote.demux import RemoteDemux
    from rich.live import Live

    try:
        mgr, job_id = RemoteDemux.from_project(project_dir)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Connection failed:[/red] {e}")
        raise typer.Exit(1)

    if not watch:
        info = mgr.get_detailed_status(job_id)
        console.print(_render_status(info, project_dir))
        return

    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                info = mgr.get_detailed_status(job_id)
                output = _render_status(info, project_dir)
                live.update(output)
                if info["status"] in ("COMPLETED", "FAILED"):
                    break
                time.sleep(interval)
    except KeyboardInterrupt:
        pass


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
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        _csv_task = None
        _fasta_task = None

        _active_task = None

        def _on_file(fname: str, size_bytes: int):
            nonlocal _csv_task, _fasta_task, _active_task
            task = progress.add_task(f"  {fname}", total=size_bytes or None)
            if fname.startswith("variant FASTAs"):
                _fasta_task = task
            else:
                _csv_task = task
            _active_task = task

        def _transfer_cb(transferred: int, total: int):
            if _active_task is not None:
                progress.update(_active_task, completed=transferred, total=total)

        progress.start()
        try:
            mgr.fetch_read_data(
                job_key, project_dir,
                on_file=_on_file,
                transfer_callback=_transfer_cb,
            )
        finally:
            progress.stop()

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
    """Delete remote job directories.

    Lists all remote jobs and lets you select which to delete.
    If a project directory is given, its job key is marked as
    current and excluded from the selection.
    """
    from usortm.remote.demux import RemoteDemux
    import questionary

    try:
        mgr = RemoteDemux()
    except Exception as e:
        console.print(f"[red]Connection failed:[/red] {e}")
        raise typer.Exit(1)

    # Determine which keys to protect (current project)
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
    candidates = [j for j in jobs if j["job_key"] not in keep_keys]

    if not candidates:
        console.print("[green]Nothing to clean.[/green]")
        return

    from datetime import datetime

    status_colors = {"COMPLETED": "green", "FAILED": "red", "RUNNING": "yellow"}
    sorted_candidates = sorted(candidates, key=lambda j: j["mtime"], reverse=True)

    # Display the table for all modes
    table = Table(box=box.SIMPLE, title="Remote jobs")
    table.add_column("Job Key", style="bold")
    table.add_column("Status")
    table.add_column("Size")
    table.add_column("Last Modified")
    for job in sorted_candidates:
        status = job["status"]
        color = status_colors.get(status, "white")
        age = datetime.fromtimestamp(job["mtime"]).strftime("%Y-%m-%d %H:%M") if job["mtime"] else ""
        table.add_row(job["job_key"], f"[{color}]{status}[/{color}]", job["size"], age)
    console.print(table)

    if dry_run:
        console.print("[dim]Dry run — nothing deleted.[/dim]")
        return

    if yes:
        # --yes: delete all candidates without prompting
        selected_keys = [j["job_key"] for j in sorted_candidates]
    else:
        choices = [
            questionary.Choice(title=job["job_key"], value=job["job_key"])
            for job in sorted_candidates
        ]

        selected_keys = questionary.checkbox(
            "Select jobs to delete:",
            choices=choices,
        ).ask()

        if selected_keys is None or len(selected_keys) == 0:
            console.print("[dim]Nothing selected.[/dim]")
            return

    keep = [k for k in [j["job_key"] for j in jobs] if k not in selected_keys]
    deleted = mgr.clean(keep_keys=keep, dry_run=False)
    console.print(f"[green]\u2713[/green] Deleted {len(deleted)} job director{'y' if len(deleted) == 1 else 'ies'}.")


# ── pick ─────────────────────────────────────────────────────────────


@remote_app.command(name="pick")
def remote_pick(
    project_dir: Path = typer.Argument(
        ...,
        help="Path to uSort-M project directory.",
        exists=True,
    ),
    tier: Optional[str] = typer.Option("A", "--tier", help="Quality tier: A/B/C or '' to disable."),
    workers: int = typer.Option(4, "--workers", "-w", help="Parallel workers for pileup generation."),
    include_cons_errors: bool = typer.Option(False, "--include-cons-errors", help="Include wells with consensus errors."),
    include_flank_errors: bool = typer.Option(False, "--include-flank-errors", help="Include wells with flank mismatches."),
    pileups: bool = typer.Option(True, "--pileups/--no-pileups", help="Generate pileup HTMLs."),
    unique_only: bool = typer.Option(True, "--unique-only/--all-hits", help="One well per variant."),
    compact: bool = typer.Option(False, "--compact/--no-compact", help="Omit empty placeholder wells."),
    target_format: int = typer.Option(384, "--target-format", help="96 or 384-well plate."),
    fill_order: str = typer.Option("row", "--fill-order", help="row or column."),
    volume: float = typer.Option(5.0, "--volume", "-v", help="Transfer volume (µL)."),
    targets: Optional[Path] = typer.Option(None, "--targets", "-t", help="CSV with variant targets."),
    round_num: int = typer.Option(1, "--round", "-r", help="Sequencing round.", min=1),
):
    """Submit a pick job to the remote server.

    Uses the demux output already on the remote (from a previous
    ``usortm remote demux``).  If demux was run locally, the necessary
    files are uploaded automatically.
    """
    from usortm.remote import RemoteDemux
    from usortm.remote.demux import _make_job_key
    import json as _json

    # Try to reuse existing remote demux job key; fall back to creating
    # a new connection + key for the local-demux → remote-pick case.
    try:
        mgr, job_key = RemoteDemux.from_project(project_dir)
    except ValueError:
        try:
            mgr = RemoteDemux()
        except Exception as e:
            console.print(f"[red]Connection failed:[/red] {e}")
            raise typer.Exit(1)
        # Read or generate a job key
        state_file = project_dir / "usortm_project.json"
        if state_file.exists():
            with open(state_file) as _f:
                _proj = _json.load(_f)
            job_key = (
                _proj.get("workflow_steps", {})
                .get("pick", {}).get("remote", {}).get("job_key")
            ) or _make_job_key()
        else:
            job_key = _make_job_key()
    except Exception as e:
        console.print(f"[red]Connection failed:[/red] {e}")
        raise typer.Exit(1)

    console.print(Panel.fit(
        "[brand]uSort-M[/brand] Remote Pick",
        border_style=BORDER_STYLE,
    ))
    console.print(f"[green]\u2713[/green] Connected to [bold]{mgr.conn.host}[/bold]")
    console.print(f"[green]\u2713[/green] Job key: [bold]{job_key}[/bold]")

    _upload_progress: dict = {}

    def _on_upload(fname: str, size_bytes: int):
        if "ctx" not in _upload_progress:
            ctx = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
            )
            ctx.start()
            _upload_progress["ctx"] = ctx
        else:
            # Complete previous task
            ctx = _upload_progress["ctx"]
        task = ctx.add_task(f"  Uploading {fname}", total=size_bytes or None)
        _upload_progress["task"] = task

    def _upload_cb(transferred: int, total: int):
        if "ctx" in _upload_progress and "task" in _upload_progress:
            _upload_progress["ctx"].update(
                _upload_progress["task"], completed=transferred, total=total,
            )

    try:
        mgr.submit_pick(
            project_dir=project_dir,
            job_key=job_key,
            tier=tier,
            workers=workers,
            include_cons_errors=include_cons_errors,
            include_flank_errors=include_flank_errors,
            pileups=pileups,
            unique_only=unique_only,
            compact=compact,
            target_format=target_format,
            fill_order=fill_order,
            volume=volume,
            targets=targets,
            round_num=round_num,
            on_upload=_on_upload,
            upload_callback=_upload_cb,
        )
    finally:
        if "ctx" in _upload_progress:
            _upload_progress["ctx"].stop()
        mgr.conn.close()

    console.print(f"[green]\u2713[/green] Pick job submitted: [bold]{job_key}[/bold]")
    tier_label = f" (Tier {tier})" if tier else ""
    console.print(f"  [dim]Pileups: {'yes' if pileups else 'no'} · Workers: {workers}{tier_label}[/dim]")
    console.print()
    console.print(f"[bold]Next:[/bold] [cyan]usortm remote pick-status {project_dir}/[/cyan]")


# ── pick-status ──────────────────────────────────────────────────────


def _render_pick_status(info: dict, project_dir):
    """Build pick status display as a Rich renderable Group."""
    from rich.console import Group
    from rich.text import Text

    parts = []
    status = info["status"]
    status_color = {"RUNNING": "yellow", "COMPLETED": "green", "FAILED": "red"}.get(status, "white")
    job_key = info.get("job_key", "?")

    parts.append(Text(""))
    parts.append(Panel.fit(
        f"[brand]uSort-M[/brand] Remote Pick  ·  [bold]{job_key}[/bold]  ·  [{status_color}]{status}[/{status_color}]",
        border_style=BORDER_STYLE,
    ))
    parts.append(Text(""))

    stages = info.get("stages", [])
    current_idx = None
    if status == "RUNNING":
        for i in range(len(stages) - 1, -1, -1):
            if stages[i]["done"]:
                current_idx = i
                break
        if current_idx is None:
            current_idx = 0

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

        close = "[/bold]" if style == "[bold]" else "[/dim]" if style == "[dim]" else ""
        parts.append(Text.from_markup(f"  {icon}  {style}{label}{close}"))

    last_line = info.get("last_log_line", "").strip()
    if last_line and status == "RUNNING":
        parts.append(Text(""))
        parts.append(Text.from_markup(f"  [dim]{last_line}[/dim]"))

    parts.append(Text(""))
    if status == "COMPLETED":
        parts.append(Text.from_markup(f"[bold]Next:[/bold] [cyan]usortm remote pick-fetch {project_dir}/[/cyan]"))
    elif status == "FAILED":
        parts.append(Text.from_markup(f"[bold]Check log:[/bold] [cyan]usortm remote pick-log {project_dir}/[/cyan]"))
    parts.append(Text(""))

    return Group(*parts)


def _get_pick_remote(project_dir: Path):
    """Get RemoteDemux manager and job_key for pick commands.

    Checks pick remote state first, falls back to demux remote state.
    """
    import json as _json
    from usortm.remote.demux import RemoteDemux

    state_file = Path(project_dir) / "usortm_project.json"
    if not state_file.exists():
        raise ValueError("No usortm_project.json found")

    with open(state_file) as f:
        project = _json.load(f)

    # Try pick remote state first
    pick_remote = project.get("workflow_steps", {}).get("pick", {}).get("remote")
    if pick_remote and pick_remote.get("job_key"):
        host = pick_remote.get("host")
        mgr = RemoteDemux(host=host)
        return mgr, pick_remote["job_key"]

    # Fall back to demux remote state
    return RemoteDemux.from_project(project_dir)


@remote_app.command(name="pick-status")
def remote_pick_status(
    project_dir: Path = typer.Argument(
        ...,
        help="Path to uSort-M project directory.",
        exists=True,
    ),
    watch: bool = typer.Option(False, "--watch", "-w", help="Auto-refresh until job completes."),
    interval: int = typer.Option(15, "--interval", "-i", help="Refresh interval in seconds (with --watch)."),
):
    """Check the status of a remote pick job."""
    import time
    from rich.live import Live

    try:
        mgr, job_key = _get_pick_remote(project_dir)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Connection failed:[/red] {e}")
        raise typer.Exit(1)

    if not watch:
        info = mgr.get_detailed_pick_status(job_key)
        console.print(_render_pick_status(info, project_dir))
        return

    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                info = mgr.get_detailed_pick_status(job_key)
                live.update(_render_pick_status(info, project_dir))
                if info["status"] in ("COMPLETED", "FAILED"):
                    break
                time.sleep(interval)
    except KeyboardInterrupt:
        pass


# ── pick-fetch ───────────────────────────────────────────────────────


@remote_app.command(name="pick-fetch")
def remote_pick_fetch(
    project_dir: Path = typer.Argument(
        ...,
        help="Path to uSort-M project directory.",
        exists=True,
    ),
):
    """Download pick results from the remote server."""
    try:
        mgr, job_key = _get_pick_remote(project_dir)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Connection failed:[/red] {e}")
        raise typer.Exit(1)

    # Check pick completed
    info = mgr.pick_status(job_key)
    if info["status"] != "COMPLETED":
        console.print(
            f"[yellow]Pick job is {info['status']}.[/yellow] Wait for completion first."
        )
        raise typer.Exit(1)

    console.print(f"Fetching pick results for [bold]{job_key}[/bold]...")

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    _active_task = None

    def _on_file(fname: str, size_bytes: int):
        nonlocal _active_task
        _active_task = progress.add_task(f"  {fname}", total=size_bytes or None)

    def _transfer_cb(transferred: int, total: int):
        if _active_task is not None:
            progress.update(_active_task, completed=transferred, total=total)

    progress.start()
    try:
        local_pick = mgr.fetch_pick(
            job_key, project_dir,
            on_file=_on_file,
            transfer_callback=_transfer_cb,
        )
    finally:
        progress.stop()

    console.print(f"[green]\u2713[/green] Pick results saved to {local_pick}")
    console.print()
    console.print(f"[bold]Next:[/bold] [cyan]usortm report {project_dir}/[/cyan]")


# ── pick-log ─────────────────────────────────────────────────────────


@remote_app.command(name="pick-log")
def remote_pick_log(
    project_dir: Path = typer.Argument(
        ...,
        help="Path to uSort-M project directory.",
        exists=True,
    ),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show."),
):
    """Show the remote pick job log."""
    try:
        mgr, job_key = _get_pick_remote(project_dir)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Connection failed:[/red] {e}")
        raise typer.Exit(1)

    log = mgr.get_pick_log(job_key, lines=lines)
    console.print(log)
