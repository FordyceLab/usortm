"""Where everything in a project lives.

Output is split by how long it is worth keeping rather than by the stage that
produced it:

``results/``
    Small and permanent — the summary, the plate map, the per-well tables, the
    pileups. What you would keep, share, or come back to.

``demux/``
    Large and rebuildable — alignments, barcode calls, per-well reads,
    references. Deleting it loses nothing that ``results/`` does not hold, and
    ``usortm clean`` does exactly that.

Round 1 writes to the top of the project and later rounds nest under
``rounds/<n>/``, because most projects only ever have one round and a
``rounds/1/`` in every path would be noise. That asymmetry is the reason this
module exists: computing it once here keeps it out of the fourteen places that
previously worked it out for themselves, each of which was somewhere a path
could be got wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_STATE_FILE = "usortm_project.json"
ROUND_STATE_FILE = "usortm_round.json"
INDEX_FILE = "index.html"


@dataclass(frozen=True)
class ProjectPaths:
    """Every path one round of a project uses.

    Build with :func:`paths_for`; nothing here touches the filesystem until
    :meth:`ensure` is called.
    """

    root: Path
    round_num: int

    # --- the round's own root -------------------------------------------
    @property
    def round_root(self) -> Path:
        """Where this round writes. The project root for round 1."""
        return self.root if self.round_num == 1 else (
            self.root / "rounds" / str(self.round_num)
        )

    # --- inputs and configuration ---------------------------------------
    @property
    def inputs(self) -> Path:
        return self.root / "inputs"

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def variants(self) -> Path:
        """The library for this round; later rounds order their own subset."""
        return (self.inputs / "variants.csv" if self.round_num == 1
                else self.round_root / "variants.csv")

    @property
    def plate_map(self) -> Path:
        return self.config / "plate_map.toml"

    @property
    def mask_config(self) -> Path:
        return self.config / "mask_config.toml"

    @property
    def barcodes(self) -> Path:
        return self.config / "barcodes"

    # --- results: small, permanent ---------------------------------------
    @property
    def results(self) -> Path:
        return self.round_root / "results"

    @property
    def summary(self) -> Path:
        return self.results / "summary.html"

    @property
    def plate_map_html(self) -> Path:
        return self.results / "plate_map.html"

    @property
    def wells_csv(self) -> Path:
        """One row per well: variant, depth, consensus."""
        return self.results / "wells.csv"

    @property
    def well_details_csv(self) -> Path:
        """Per-well detail, including reference lengths and sequences."""
        return self.results / "well_details.csv"

    @property
    def run_stats(self) -> Path:
        return self.results / "run_stats.json"

    @property
    def pileups(self) -> Path:
        return self.results / "pileups"

    @property
    def picks(self) -> Path:
        return self.results / "picks"

    # --- demux: large, rebuildable ---------------------------------------
    @property
    def demux(self) -> Path:
        return self.round_root / "demux"

    @property
    def live(self) -> Path:
        return self.demux / "live.html"

    @property
    def reads_csv(self) -> Path:
        """One row per read: identity and assignment, no sequences."""
        return self.demux / "reads.csv"

    @property
    def references(self) -> Path:
        return self.demux / "references"

    @property
    def segments(self) -> Path:
        return self.demux / "segments"

    def segment(self, name: str) -> Path:
        return self.segments / name

    # --- state files ------------------------------------------------------
    @property
    def state(self) -> Path:
        """The project state file, which is shared across rounds."""
        return self.root / PROJECT_STATE_FILE

    @property
    def round_state(self) -> Path:
        """A later round's own state; round 1 keeps everything in the project."""
        return self.round_root / ROUND_STATE_FILE

    @property
    def index(self) -> Path:
        return self.root / INDEX_FILE

    # --- helpers ----------------------------------------------------------
    def ensure(self) -> "ProjectPaths":
        """Create the directories a run writes into."""
        for path in (self.inputs, self.config, self.results, self.demux):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def rebuildable(self) -> tuple:
        """Directories ``usortm clean`` may remove for this round."""
        return (self.demux,)


def paths_for(project_dir, round_num: int = 1) -> ProjectPaths:
    """Return the paths for *round_num* of the project at *project_dir*."""
    if round_num < 1:
        raise ValueError(f"round must be 1 or greater, got {round_num}")
    return ProjectPaths(root=Path(project_dir), round_num=round_num)


def _resolve(root, subdir: str, name: str) -> Path:
    """Find *name* under *subdir*, or loose at the top of the project.

    New projects put their inputs and configuration in subdirectories; older
    ones left everything at the top level.  Looking in both means a project
    made before the split still runs, without a migration step that would have
    to be got right on data nobody wants to re-derive.

    Returns the subdirectory path when neither exists, so a caller reporting a
    missing file names the place it should now be put.
    """
    root = Path(root)
    organised = root / subdir / name
    if organised.exists():
        return organised
    loose = root / name
    if loose.exists():
        return loose
    return organised


def input_file(project_dir, name: str) -> Path:
    """Find one of a project's inputs: the library, the FASTQs, the vector."""
    return _resolve(project_dir, "inputs", name)


def config_file(project_dir, name: str) -> Path:
    """Find one of a project's configuration files: plate map, masks, barcodes."""
    return _resolve(project_dir, "config", name)
