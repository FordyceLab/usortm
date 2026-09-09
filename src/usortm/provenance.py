"""The commands a project was built by, kept so they can be reported.

A project's state already records what each step was given -- the tier a pick
used, the reads a demux saw -- but not the command that was run.  Those are
not the same thing: a parameter table has to be read back into an invocation
by whoever writes the methods section, and doing that from memory months later
is where a methods section stops matching the run.

So each step records its own command line as it completes, and
:func:`write_commands` collects them in the order they ran.  A step that
finished before this was recorded has no command to give; it is listed with
what is known and said to be missing one, rather than reconstructed from its
parameters into something that was never typed.
"""
from __future__ import annotations

import os
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

#: Where the collected commands are written, at the top of the project.
COMMANDS_FILE = "commands.txt"

#: The order steps are reported in when timestamps cannot separate them --
#: the order the workflow runs in.
STEP_ORDER = ("plan", "demux", "pick", "reorder", "merge", "report",
              "pileups")


def current_command(argv: Optional[list] = None) -> str:
    """The invocation being run, as it would be typed again.

    The interpreter's own path is dropped when the CLI was reached through
    it, so ``python -m usortm demux`` and the installed entry point both come
    back as ``usortm demux``: what belongs in a methods section is the command,
    not the environment it happened to be run from.
    """
    argv = list(sys.argv if argv is None else argv)
    if not argv:
        return ""
    first = os.path.basename(argv[0])
    if first.startswith("python") or first in ("-c", "-m"):
        argv = ["usortm"] + [a for a in argv[1:] if a not in ("-m", "usortm")]
    else:
        argv = ["usortm"] + argv[1:]
    return shlex.join(argv)


def record(state: dict, argv: Optional[list] = None) -> dict:
    """Note the command and the time on a step's state, and return it.

    Called where a step records that it completed, so the command is captured
    from the run that actually did the work rather than inferred afterwards.
    """
    state["command"] = current_command(argv)
    state.setdefault("timestamp", datetime.now().isoformat())
    return state


def _steps(project: dict):
    """Every recorded step, as ``(when, round, name, state)``.

    Rounds keep their own copies, so a re-order round's demux is reported
    beside the first round's rather than replacing it.
    """
    out = []
    for name, state in (project.get("workflow_steps") or {}).items():
        if isinstance(state, dict) and state.get("completed"):
            out.append((state.get("timestamp") or "", 1, name, state))
    for rnd, block in sorted((project.get("rounds") or {}).items()):
        for name, state in (block.get("workflow_steps") or {}).items():
            if isinstance(state, dict) and state.get("completed"):
                out.append((state.get("timestamp") or "", int(rnd), name,
                            state))
    return sorted(out, key=lambda s: (s[0] or "", s[1],
                                      STEP_ORDER.index(s[2])
                                      if s[2] in STEP_ORDER else 99))


def render_commands(project: dict, project_dir) -> str:
    """The project's commands, in the order they ran, as a text file.

    Written for a supplement: one command per step, each under the date it ran
    and the round it belongs to, with nothing between them that would have to
    be stripped before it could be re-run.
    """
    name = os.path.basename(os.path.normpath(str(project_dir)))
    lines = [
        f"# uSort-M commands for {name}",
        f"# Collected {datetime.now().strftime('%Y-%m-%d')} by "
        f"`usortm methods`.",
        "#",
        "# Each command is the invocation that produced the step below it, as",
        "# recorded when it ran. Paths are as they were given.",
        "",
    ]
    steps = _steps(project)
    if not steps:
        lines.append("# No completed steps are recorded for this project.")
        return "\n".join(lines) + "\n"

    missing = 0
    for when, rnd, step, state in steps:
        day = (when or "")[:10] or "date not recorded"
        where = f"round {rnd}" if rnd != 1 else "round 1"
        lines.append(f"# {day} · {where} · {step}")
        command = state.get("command")
        if command:
            lines.append(command)
        else:
            missing += 1
            lines.append(f"# command not recorded; this step predates it.")
            for key in ("tier", "format", "library_size", "n_plates",
                        "input_reads", "total_hits", "dropouts"):
                if state.get(key) is not None:
                    lines.append(f"#   {key}: {state[key]}")
        lines.append("")

    if missing:
        lines += [
            f"# {missing} step(s) ran before their command was recorded and",
            "# are shown by their parameters instead. Re-running one records",
            "# the command it was given.",
            "",
        ]
    return "\n".join(lines)


def write_commands(project: dict, project_dir) -> Path:
    """Write the project's commands to :data:`COMMANDS_FILE` and return it."""
    path = Path(project_dir) / COMMANDS_FILE
    path.write_text(render_commands(project, project_dir))
    return path
