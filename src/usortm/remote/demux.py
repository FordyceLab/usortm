"""Remote demux job manager following beak's SSH/nohup lifecycle."""

from __future__ import annotations

import json
import random
import time
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

# Memorable job key generation — no external deps
_ADJECTIVES = [
    "amber", "bold", "calm", "deft", "eager", "firm", "glad", "hale",
    "idle", "just", "keen", "lithe", "mellow", "nimble", "opal", "prime",
    "quiet", "rapid", "sharp", "tidy", "umber", "vivid", "warm", "xenial",
    "young", "zesty",
]
_NOUNS = [
    "alcove", "basin", "crane", "delta", "ember", "fjord", "grove", "haven",
    "inlet", "jetty", "kelp", "locus", "marsh", "notch", "orbit", "plume",
    "quarry", "ridge", "shore", "tide", "uplift", "vale", "wedge", "xenon",
    "yawl", "zenith",
]


def _make_job_key() -> str:
    """Generate a memorable, human-readable job key like 'vivid-shore'."""
    return f"{random.choice(_ADJECTIVES)}-{random.choice(_NOUNS)}"

from .connection import get_connection, expand_remote_tilde, load_config


class RemoteDemux:
    """Manage demux jobs on a remote server via SSH.

    Lifecycle: submit → status/wait → fetch_metadata → (optionally) fetch_read_data.
    """

    REQUIRED_TOOLS = {
        "minimap2": "https://github.com/lh3/minimap2",
        "samtools": "https://www.htslib.org/download/",
        "dorado": "https://github.com/nanoporetech/dorado/releases",
        "usortm": "pip install usortm[demux]",
    }

    def __init__(
        self,
        host: Optional[str] = None,
        user: Optional[str] = None,
        key_path: Optional[str] = None,
        remote_job_dir: Optional[str] = None,
        connection=None,
    ):
        if connection is not None:
            self.conn = connection
        else:
            self.conn = get_connection(host=host, user=user, key_path=key_path)

        cfg = load_config().get("connection", {})
        remote_job_dir = remote_job_dir or cfg.get("remote_job_dir", "~/usortm_jobs")
        self.remote_job_dir = expand_remote_tilde(self.conn, remote_job_dir)

        # Ensure remote directory exists
        self.conn.run(f"mkdir -p {self.remote_job_dir}", hide=True)

    # ── Submit ───────────────────────────────────────────────────────

    def submit(
        self,
        project_dir: Path,
        *,
        fastq: Optional[Path] = None,
        remote_fastq: Optional[str] = None,
        fastq_url: Optional[str] = None,
        reference: Optional[Path] = None,
        library_csv: Optional[Path] = None,
        vector_fasta: Optional[Path] = None,
        mask_config: Optional[Path] = None,
        threads: int = 8,
        workers: int = 4,
        subsample: Optional[int] = None,
        extra_args: Optional[list[str]] = None,
        upload_callback=None,
    ) -> tuple[str, bool]:
        """Upload inputs, generate run script, and launch demux on remote.

        The project gets a stable *job key* (e.g. ``vivid-shore``) stored in
        ``usortm_project.json`` on first submit.  Re-submitting the same
        project reuses the same remote directory; inputs that are already
        present are not re-uploaded.

        *fastq_url* causes the remote server to download the file directly
        via wget (useful when the remote has faster network access than the
        local machine).  Gzipped archives (``.fastq.gz``) are passed through
        to the pipeline as-is.

        Returns ``(job_key, fastq_uploaded)`` where *fastq_uploaded* is
        ``True`` when a new FASTQ transfer was performed (local→remote upload
        or remote wget).
        """
        if not fastq and not remote_fastq and not fastq_url:
            raise ValueError("Provide --fastq, --remote-fastq, or --fastq-url")
        if not reference and not library_csv:
            raise ValueError("Provide either reference or library_csv")

        project_dir = Path(project_dir)
        state_file = project_dir / "usortm_project.json"

        # Load (or create) project state
        if state_file.exists():
            with open(state_file) as f:
                project = json.load(f)
        else:
            project = {"workflow_steps": {}}

        # Stable job key — generated once, reused forever
        existing_remote = (
            project.get("workflow_steps", {}).get("demux", {}).get("remote", {})
        )
        job_key = existing_remote.get("job_key") or _make_job_key()

        job_dir = f"{self.remote_job_dir}/{job_key}"
        inputs_dir = f"{job_dir}/inputs"
        self.conn.run(f"mkdir -p {inputs_dir}", hide=True)

        def _remote_exists(path: str) -> bool:
            r = self.conn.run(f'[ -e "{path}" ] && echo 1 || echo 0', hide=True, warn=True)
            return r.stdout.strip() == "1"

        # Upload small inputs unconditionally (they may have changed)
        if reference:
            self.conn.put(str(reference), f"{inputs_dir}/reference.fasta")
        if library_csv:
            self.conn.put(str(library_csv), f"{inputs_dir}/library.csv")
        if vector_fasta:
            self.conn.put(str(vector_fasta), f"{inputs_dir}/vector.fasta")
        if mask_config:
            self.conn.put(str(mask_config), f"{inputs_dir}/mask_config.toml")

        # FASTQ — only transfer if not already present on remote
        fastq_uploaded = False
        if remote_fastq:
            if not _remote_exists(remote_fastq):
                raise FileNotFoundError(f"Remote FASTQ not found: {remote_fastq}")
            fastq_path = remote_fastq
        elif fastq_url:
            # The run script downloads and normalises to a canonical path.
            # Use a shell variable reference so the demux command resolves
            # the path at runtime (needed because ZIP archives contain files
            # with unknown names, and the file may or may not be gzipped).
            fastq_path = "$FASTQ_PATH"
            canonical = f"{inputs_dir}/reads.fastq"
            already = (
                _remote_exists(canonical)
                or _remote_exists(f"{canonical}.gz")
            )
            fastq_uploaded = not already
        else:
            remote_fastq_name = Path(str(fastq)).name
            remote_fastq_path = f"{inputs_dir}/{remote_fastq_name}"
            if _remote_exists(remote_fastq_path):
                fastq_path = remote_fastq_path  # already there from previous submit
            else:
                sftp = self.conn.sftp()
                sftp.put(str(fastq), remote_fastq_path, callback=upload_callback)
                fastq_path = remote_fastq_path
                fastq_uploaded = True

        # Resolve usortm path
        cfg = load_config().get("connection", {})
        usortm_path = cfg.get("usortm_path") or self._find_remote_usortm()

        # (Re-)write run script and (re-)launch
        script = self._generate_run_script(
            job_dir=job_dir,
            fastq_path=fastq_path,
            fastq_url=fastq_url,
            inputs_dir=inputs_dir,
            reference=reference is not None,
            library_csv=library_csv is not None,
            vector_fasta=vector_fasta is not None,
            mask_config=mask_config is not None,
            threads=threads,
            workers=workers,
            subsample=subsample,
            extra_args=extra_args or [],
            usortm_path=usortm_path,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as tmp:
            tmp.write(script)
            tmp_path = tmp.name
        self.conn.put(tmp_path, f"{job_dir}/run.sh")
        self.conn.run(f"chmod +x {job_dir}/run.sh", hide=True)

        # Reset status for the new run
        self.conn.run(
            f'echo "Submitted: $(date)" > "{job_dir}/status.txt"',
            hide=True,
        )

        # Launch
        result = self.conn.run(
            f"cd {job_dir} && nohup ./run.sh > nohup.out 2>&1 & echo $!",
            hide=True,
        )
        pid = result.stdout.strip()

        # Persist remote info (preserve metadata_downloaded etc. if re-submitting)
        remote_info = {
            **existing_remote,
            "job_key": job_key,
            "host": self.conn.host,
            "remote_path": job_dir,
            "pid": pid,
            "submitted_at": datetime.now().isoformat(),
            "usortm_path": usortm_path,
        }
        remote_info.setdefault("metadata_downloaded", False)
        remote_info.setdefault("read_data_downloaded", False)

        project.setdefault("workflow_steps", {}).setdefault("demux", {})
        project["workflow_steps"]["demux"]["remote"] = remote_info
        with open(state_file, "w") as f:
            json.dump(project, f, indent=2)

        return job_key, fastq_uploaded

    def _generate_run_script(
        self,
        job_dir: str,
        fastq_path: str,
        inputs_dir: str,
        reference: bool,
        library_csv: bool,
        vector_fasta: bool,
        mask_config: bool,
        threads: int,
        workers: int,
        subsample: Optional[int],
        extra_args: list[str],
        usortm_path: str = "usortm",
        fastq_url: Optional[str] = None,
    ) -> str:
        """Build the bash script that runs usortm demux on the remote."""
        # Build the usortm demux command
        cmd_parts = [
            f'"{usortm_path}" demux',
            f'"{job_dir}/project"',
            f'--fastq "{fastq_path}"',
            f"--threads {threads}",
            f"--workers {workers}",
        ]
        if reference:
            cmd_parts.append(f'--reference "{inputs_dir}/reference.fasta"')
        if library_csv:
            cmd_parts.append(f'--library-csv "{inputs_dir}/library.csv"')
        if vector_fasta:
            cmd_parts.append(f'--vector-fasta "{inputs_dir}/vector.fasta"')
        if mask_config:
            cmd_parts.append(f'--mask-config "{inputs_dir}/mask_config.toml"')
        if subsample:
            cmd_parts.append(f"--subsample {subsample}")
        for arg in extra_args:
            cmd_parts.append(arg)

        demux_cmd = " \\\n    ".join(cmd_parts)

        wget_block = ""
        if fastq_url:
            canonical = f"{inputs_dir}/reads.fastq"
            wget_block = f"""
# Download and normalise FASTQ (skipped if already present)
FASTQ_PATH="{canonical}"
if [ -f "{canonical}.gz" ]; then
    FASTQ_PATH="{canonical}.gz"
elif [ ! -f "{canonical}" ]; then
    TMP="{inputs_dir}/download.tmp"
    echo "Downloading FASTQ from: {fastq_url}" | tee -a "$JOB_DIR/usortm.log"
    wget -q -L -O "$TMP" "{fastq_url}"
    echo "Download complete: $(date)" | tee -a "$JOB_DIR/usortm.log"

    # Detect format by magic bytes
    FILE_TYPE=$(python3 -c "
import sys
d = open('$TMP','rb').read(4)
if d[:2] == b'PK': print('zip')
elif d[:2] == b'\\x1f\\x8b': print('gz')
else: print('plain')
")
    echo "Detected format: $FILE_TYPE" | tee -a "$JOB_DIR/usortm.log"

    if [ "$FILE_TYPE" = "zip" ]; then
        UNZIP_DIR="{inputs_dir}/unzipped"
        mkdir -p "$UNZIP_DIR"
        unzip -o "$TMP" -d "$UNZIP_DIR/" >> "$JOB_DIR/usortm.log" 2>&1
        FOUND=$(find "$UNZIP_DIR" \\( -name "*.fastq.gz" -o -name "*.fastq" \\) | head -1)
        if [[ "$FOUND" == *.gz ]]; then
            mv "$FOUND" "{canonical}.gz"
            FASTQ_PATH="{canonical}.gz"
        else
            mv "$FOUND" "{canonical}"
            FASTQ_PATH="{canonical}"
        fi
        rm -rf "$TMP" "$UNZIP_DIR"
    elif [ "$FILE_TYPE" = "gz" ]; then
        mv "$TMP" "{canonical}.gz"
        FASTQ_PATH="{canonical}.gz"
    else
        mv "$TMP" "{canonical}"
        FASTQ_PATH="{canonical}"
    fi
    echo "FASTQ ready at: $FASTQ_PATH" | tee -a "$JOB_DIR/usortm.log"
fi
"""

        return f"""#!/bin/bash -l
set -euo pipefail

JOB_DIR="{job_dir}"

echo "Job started: $(date)" > "$JOB_DIR/status.txt"
echo "RUNNING" >> "$JOB_DIR/status.txt"

# Create minimal project directory for usortm
mkdir -p "$JOB_DIR/project"
cat > "$JOB_DIR/project/usortm_project.json" <<'PROJEOF'
{{"workflow_steps": {{}}}}
PROJEOF
{wget_block}
# Run demux
{demux_cmd} \\
    2>&1 | tee "$JOB_DIR/usortm.log"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Job completed: $(date)" >> "$JOB_DIR/status.txt"
    echo "COMPLETED" >> "$JOB_DIR/status.txt"

    # Copy results to top level for easy fetching
    PROJECT_DEMUX="$JOB_DIR/project/demux_output"
    if [ -d "$PROJECT_DEMUX" ]; then
        ln -sf "$PROJECT_DEMUX" "$JOB_DIR/demux_output"
    fi
else
    echo "Job failed (exit $EXIT_CODE): $(date)" >> "$JOB_DIR/status.txt"
    echo "FAILED" >> "$JOB_DIR/status.txt"
fi

exit $EXIT_CODE
"""

    # ── Status ───────────────────────────────────────────────────────

    def status(self, job_key: str) -> dict:
        """Check job status on the remote server."""
        job_dir = f"{self.remote_job_dir}/{job_key}"

        # Check if process is running
        pid_result = self.conn.run(
            f"cat {job_dir}/nohup.out 2>/dev/null | head -0; "
            f"cat {job_dir}/run.sh 2>/dev/null | head -0; "
            f"ps -p $(cat {job_dir}/pid.txt 2>/dev/null || echo 0) -o pid= 2>/dev/null || true",
            hide=True,
            warn=True,
        )

        # Read status.txt
        status_result = self.conn.run(
            f'cat {job_dir}/status.txt 2>/dev/null || echo "NO_STATUS"',
            hide=True,
            warn=True,
        )
        status_lines = status_result.stdout.strip().split("\n")

        if "COMPLETED" in status_lines:
            status = "COMPLETED"
        elif "FAILED" in status_lines:
            status = "FAILED"
        elif "RUNNING" in status_lines:
            status = "RUNNING"
        else:
            # Fallback: check PID directly
            pid_check = self.conn.run(
                f"ps -p $(cat {job_dir}/pid.txt 2>/dev/null || echo 0) -o pid= 2>/dev/null",
                hide=True,
                warn=True,
            )
            status = "RUNNING" if pid_check.ok and pid_check.stdout.strip() else "UNKNOWN"

        return {
            "job_key": job_key,
            "status": status,
            "status_lines": status_lines,
        }

    def wait(self, job_key: str, interval: int = 30, quiet: bool = False) -> dict:
        """Block until the job completes or fails."""
        while True:
            info = self.status(job_key)
            if info["status"] in ("COMPLETED", "FAILED"):
                return info
            if not quiet:
                print(f"  [{info['status']}] Waiting... (checking every {interval}s)")
            time.sleep(interval)

    # ── Fetch ────────────────────────────────────────────────────────

    def fetch_metadata(self, job_key: str, project_dir: Path) -> Path:
        """Download metadata files (small) to local demux_output/.

        Downloads: well_df.csv, well_assignments.csv, demux_summary.json,
        plate_map.html (if present).

        Returns the local demux_output directory.
        """
        project_dir = Path(project_dir)
        local_demux = project_dir / "demux_output"
        local_demux.mkdir(parents=True, exist_ok=True)

        job_dir = f"{self.remote_job_dir}/{job_key}"
        remote_demux = f"{job_dir}/demux_output"

        metadata_files = [
            "well_df.csv",
            "well_assignments.csv",
            "demux_summary.json",
        ]

        for fname in metadata_files:
            remote_path = f"{remote_demux}/{fname}"
            exists = self.conn.run(
                f'[ -f "{remote_path}" ] && echo OK || echo MISSING',
                hide=True,
            )
            if "OK" in exists.stdout:
                self.conn.get(remote_path, str(local_demux / fname))

        # Optional files
        for fname in ("plate_map.html", "demux_plate_map.html"):
            remote_path = f"{remote_demux}/{fname}"
            exists = self.conn.run(
                f'[ -f "{remote_path}" ] && echo OK || echo MISSING',
                hide=True,
            )
            if "OK" in exists.stdout:
                self.conn.get(remote_path, str(local_demux / fname))

        # Update project state
        self._update_project_state(project_dir, job_key, metadata_downloaded=True)

        return local_demux

    def fetch_read_data(self, job_key: str, project_dir: Path) -> Path:
        """Download read_df.csv and per-variant reference FASTAs.

        These are needed for pileup generation during pick.
        """
        project_dir = Path(project_dir)
        local_demux = project_dir / "demux_output"
        local_demux.mkdir(parents=True, exist_ok=True)

        job_dir = f"{self.remote_job_dir}/{job_key}"
        remote_demux = f"{job_dir}/demux_output"

        # Download read_df.csv (can be large — check if gzipped version exists)
        for candidate in ("read_df.csv.gz", "read_df.csv"):
            remote_path = f"{remote_demux}/{candidate}"
            exists = self.conn.run(
                f'[ -f "{remote_path}" ] && echo OK || echo MISSING',
                hide=True,
            )
            if "OK" in exists.stdout:
                self.conn.get(remote_path, str(local_demux / candidate))
                break

        # Download reference_fasta/single_ref_fastas/ directory
        ref_dir = f"{remote_demux}/reference_fasta/single_ref_fastas"
        local_ref_dir = local_demux / "reference_fasta" / "single_ref_fastas"
        local_ref_dir.mkdir(parents=True, exist_ok=True)

        ls_result = self.conn.run(
            f'ls {ref_dir}/*.fasta 2>/dev/null || true',
            hide=True,
        )
        for line in ls_result.stdout.strip().split("\n"):
            line = line.strip()
            if line and line.endswith(".fasta"):
                fname = Path(line).name
                self.conn.get(line, str(local_ref_dir / fname))

        # Update project state
        self._update_project_state(project_dir, job_key, read_data_downloaded=True)

        return local_demux

    # ── Detailed status ──────────────────────────────────────────────

    # Ordered pipeline stages with (label, artifact_path_relative_to_demux_output)
    # artifact_path = None means: inferred from status.txt only
    PIPELINE_STAGES = [
        ("Check dependencies",          None),
        ("Generate barcode configs",    "dorado_config"),
        ("Orient/align reads",          "alignment/oriented_reads.fastq"),
        ("Forward barcode demux",       "fbc"),
        ("Reverse barcode demux",       "rbc"),
        ("Map barcodes → wells",        "wells/fastqs"),
        ("Generate consensus",          "wells/consensus"),
        ("Call variants",               "well_df.csv"),
        ("Screen for streak-out",       "streakout"),
        ("Finalize results",            "demux_summary.json"),
    ]

    def get_detailed_status(self, job_key: str) -> dict:
        """Return status + per-stage progress based on remote filesystem artifacts."""
        basic = self.status(job_key)
        job_dir = f"{self.remote_job_dir}/{job_key}"
        demux_dir = f"{job_dir}/project/demux_output"
        inputs_dir = f"{job_dir}/inputs"

        # Check if this job used --fastq-url (canonical reads.fastq[.gz] present or job running)
        canonical_fastq = f"{inputs_dir}/reads.fastq"
        has_url_stage = self.conn.run(
            f'[ -f "{canonical_fastq}" ] || [ -f "{canonical_fastq}.gz" ] || '
            f'[ -f "{inputs_dir}/download.tmp" ] && echo 1 || echo 0',
            hide=True, warn=True,
        ).stdout.strip() == "1"

        # Build dynamic stage list — prepend download stage if applicable
        stages_def = []
        if has_url_stage:
            stages_def.append((
                "Download FASTQ",
                f"__abs__{canonical_fastq}||{canonical_fastq}.gz",
            ))
        stages_def.extend(self.PIPELINE_STAGES)

        # Build a shell one-liner that tests each artifact and prints 1/0
        tests = []
        for _label, artifact in stages_def:
            if artifact is None:
                tests.append("echo STATUS")
            elif artifact.startswith("__abs__"):
                abs_spec = artifact[len("__abs__"):]
                # Support "path1||path2" for OR checks
                paths = abs_spec.split("||")
                checks = " || ".join(f'[ -e "{p}" ]' for p in paths)
                tests.append(f'{{ {checks}; }} && echo 1 || echo 0')
            else:
                tests.append(f'[ -e "{demux_dir}/{artifact}" ] && echo 1 || echo 0')

        probe = " ; ".join(tests)
        result = self.conn.run(probe, hide=True, warn=True)
        lines = result.stdout.strip().split("\n")

        stages = []
        is_running = basic["status"] == "RUNNING"
        is_done = basic["status"] in ("COMPLETED", "FAILED")

        for i, (label, _artifact) in enumerate(stages_def):
            raw = lines[i].strip() if i < len(lines) else "0"
            if raw == "STATUS":
                done = is_running or is_done
            else:
                done = raw == "1"
            stages.append({"label": label, "done": done})

        # If download stage is active, show tmp file size as progress hint
        download_active = (
            has_url_stage
            and stages
            and not stages[0]["done"]
            and basic["status"] == "RUNNING"
        )
        if download_active:
            size_result = self.conn.run(
                f'du -sh "{inputs_dir}/download.tmp" 2>/dev/null | cut -f1 || echo ""',
                hide=True, warn=True,
            )
            size = size_result.stdout.strip()
            last_line = f"Downloading... {size} so far" if size else "Downloading..."
        else:
            log_tail = self.conn.run(
                f"tail -n 5 {job_dir}/usortm.log 2>/dev/null | "
                r"sed 's/\x1b\[[0-9;]*m//g' | grep -v '^\s*$' | tail -n 1",
                hide=True,
                warn=True,
            )
            last_line = log_tail.stdout.strip()

        return {**basic, "stages": stages, "last_log_line": last_line}

    # ── Cancel ───────────────────────────────────────────────────────

    def cancel(self, job_key: str) -> bool:
        """Kill the remote process for this job."""
        job_dir = f"{self.remote_job_dir}/{job_key}"
        pid_result = self.conn.run(
            f"cat {job_dir}/pid.txt 2>/dev/null || echo ''",
            hide=True,
            warn=True,
        )
        pid = pid_result.stdout.strip()
        if not pid:
            return False

        self.conn.run(f"kill {pid} 2>/dev/null || true", hide=True, warn=True)

        # Update status.txt
        self.conn.run(
            f'echo "CANCELLED: $(date)" >> {job_dir}/status.txt',
            hide=True,
            warn=True,
        )
        return True

    # ── Log ──────────────────────────────────────────────────────────

    def get_log(self, job_key: str, lines: int = 50) -> str:
        """Return the last *lines* lines of the remote usortm.log."""
        job_dir = f"{self.remote_job_dir}/{job_key}"
        result = self.conn.run(
            f"tail -n {lines} {job_dir}/usortm.log 2>/dev/null || "
            f"tail -n {lines} {job_dir}/nohup.out 2>/dev/null || "
            f'echo "(no log found)"',
            hide=True,
            warn=True,
        )
        return result.stdout

    # ── Verify ───────────────────────────────────────────────────────

    def verify_remote(self) -> dict:
        """Check that required tools are installed on the remote server."""
        results = {"tools": {}, "ok": True}

        for tool, install_hint in self.REQUIRED_TOOLS.items():
            # Use a login shell so /etc/profile.d/*.sh entries (e.g. dorado, conda
            # environments) are sourced before the check.
            check = self.conn.run(
                f"bash -l -c 'command -v {tool} 2>/dev/null && {tool} --version 2>&1 | head -1 "
                f"|| echo \"__NOT_FOUND__\"'",
                hide=True,
                warn=True,
            )
            found = "__NOT_FOUND__" not in check.stdout
            version = check.stdout.strip().split("\n")[-1] if found else None

            results["tools"][tool] = {
                "found": found,
                "version": version,
                "install": install_hint,
            }
            if not found:
                results["ok"] = False

        # Disk space
        disk = self.conn.run(
            f"df -h {self.remote_job_dir} 2>/dev/null | tail -1",
            hide=True,
            warn=True,
        )
        if disk.ok and disk.stdout.strip():
            parts = disk.stdout.strip().split()
            if len(parts) >= 4:
                results["disk"] = {
                    "total": parts[1],
                    "used": parts[2],
                    "available": parts[3],
                }

        return results

    # ── Path detection ────────────────────────────────────────────────

    def _find_remote_usortm(self) -> str:
        """Try multiple strategies to find the usortm executable on the remote."""
        strategies = [
            # Login shell (sources /etc/profile.d/*.sh and .bash_profile)
            "bash -l -c 'which usortm 2>/dev/null'",
            # Interactive login shell (sources .bashrc with conda init)
            "bash -l -i -c 'which usortm 2>/dev/null'",
            # Search common conda install locations (Linux and macOS)
            "find $HOME/miniconda3 $HOME/anaconda3 $HOME/.conda $HOME/opt/anaconda3 "
            "/opt/conda /opt/anaconda3 /opt/miniconda3 "
            "-name usortm -type f 2>/dev/null | head -1",
        ]
        for cmd in strategies:
            result = self.conn.run(cmd, hide=True, warn=True)
            path = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
            if path and "usortm" in path and "not found" not in path:
                return path
        return "usortm"

    # ── Helpers ───────────────────────────────────────────────────────

    def _update_project_state(
        self,
        project_dir: Path,
        job_key: str,
        metadata_downloaded: Optional[bool] = None,
        read_data_downloaded: Optional[bool] = None,
    ):
        """Update the remote tracking info in usortm_project.json."""
        state_file = project_dir / "usortm_project.json"
        if not state_file.exists():
            return

        with open(state_file) as f:
            project = json.load(f)

        remote = project.get("workflow_steps", {}).get("demux", {}).get("remote", {})
        if remote.get("job_key") != job_key:
            return

        if metadata_downloaded is not None:
            remote["metadata_downloaded"] = metadata_downloaded
            project["workflow_steps"]["demux"]["completed"] = True
        if read_data_downloaded is not None:
            remote["read_data_downloaded"] = read_data_downloaded

        with open(state_file, "w") as f:
            json.dump(project, f, indent=2)

    # ── Clean ─────────────────────────────────────────────────────────

    def list_jobs(self) -> list[dict]:
        """List all remote job directories with their status and size."""
        result = self.conn.run(
            f"ls -1 {self.remote_job_dir}/ 2>/dev/null || true",
            hide=True,
            warn=True,
        )
        job_keys = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]

        jobs = []
        for key in job_keys:
            job_dir = f"{self.remote_job_dir}/{key}"
            # Get status and disk usage in one call
            info_result = self.conn.run(
                f'tail -1 "{job_dir}/status.txt" 2>/dev/null || echo "UNKNOWN"; '
                f'du -sh "{job_dir}" 2>/dev/null | cut -f1 || echo "?"; '
                f'stat -c "%Y" "{job_dir}" 2>/dev/null || stat -f "%m" "{job_dir}" 2>/dev/null || echo "0"',
                hide=True,
                warn=True,
            )
            lines = info_result.stdout.strip().split("\n")
            status = lines[0].strip() if lines else "UNKNOWN"
            size = lines[1].strip() if len(lines) > 1 else "?"
            mtime = int(lines[2].strip()) if len(lines) > 2 and lines[2].strip().isdigit() else 0

            jobs.append({
                "job_key": key,
                "status": status,
                "size": size,
                "mtime": mtime,
            })

        return jobs

    def clean(self, keep_keys: list[str], dry_run: bool = False) -> list[str]:
        """Delete remote job directories not in *keep_keys*.

        Returns the list of job keys that were (or would be) deleted.
        """
        all_jobs = self.list_jobs()
        to_delete = [j["job_key"] for j in all_jobs if j["job_key"] not in keep_keys]

        for key in to_delete:
            job_dir = f"{self.remote_job_dir}/{key}"
            if not dry_run:
                self.conn.run(f"rm -rf {job_dir}", hide=True, warn=True)

        return to_delete

    @classmethod
    def from_project(cls, project_dir: Path, **kwargs) -> tuple["RemoteDemux", str]:
        """Create a RemoteDemux from an existing project's remote state.

        Returns ``(manager, job_key)`` so callers can immediately check
        status or fetch results.
        """
        state_file = Path(project_dir) / "usortm_project.json"
        with open(state_file) as f:
            project = json.load(f)

        remote = project.get("workflow_steps", {}).get("demux", {}).get("remote")
        if not remote:
            raise ValueError("No remote demux job found in project state")

        host = remote.get("host")
        job_key = remote.get("job_key") or remote.get("job_id")  # backward compat

        manager = cls(host=host, **kwargs)
        return manager, job_key
