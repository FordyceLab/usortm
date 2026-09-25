# Project layout

**Status: agreed target, partly implemented.** Only `7_pick_plate/` is
built so far. Everything else is still written in the layout described in
`src/usortm/paths.py`: round 1 at the top of the project, later rounds under
`rounds/<n>/`, and demultiplexing output in `demux_output/`. This note
records the layout a project will move to and how to get there.

## Target layout

A project directory lists its steps in the order they happen. Each step
writes one numbered directory.

```
usortm_project/
  usortm_project.json      project state
  commands.txt             every command run on the project, in order
  summary.html             the report; the file to open
  0_inputs/                library, read template, vector, final plate layout
  1_plan/                  barcodes, plate map, QC mask
  2_sort/                  round 1: the sorted plates, sequenced
    fastqs/
    results/
    demux/
  3_pick/                  round 1 pick list, pick plate map, pileups
  4_reorder/               dropouts re-ordered and re-sequenced
    order/                 the synthesis order sheets
    round_2/               plate_map.toml, replicate_map.toml, fastqs/,
                           results/, demux/, and the round's pick
  5_merge/                 the consolidated pick across rounds
  6_hitpick/               liquid-handler worklists, one file per source plate
  7_pick_plate/            sequencing of the built pick plate, one run per
                           sequencing ID
    GQKLWM/
    G3Y8KW/
      plate_map.toml
      expected_layout.csv  the placements the run is judged against
      qc_mask.toml         optional; the run's own sequencing artefacts
      results/
      demux/
  report/                  the report's tables, figures and zip
```

## Rules

- **Numbers belong to steps, not to projects.** A step keeps its number in
  every project, so code and documentation can name any path without first
  checking which steps a project ran. A project that never re-orders has no
  `4_reorder/`, and the gap is expected.
- **The top level holds only state, the command log, the summary, and step
  directories.** Nothing else is written there.
- **Every sequenced step splits `results/` from `demux/`.** `results/` holds
  what is kept and shared: per-well tables, plate maps, pileups, verdicts.
  `demux/` holds what can be rebuilt from the FASTQs: alignments, barcode
  calls, per-well reads, references. Cleaning a project deletes the `demux/`
  directories and nothing else.
- **Round 1 has no special case.** It lives in `2_sort/` like any other step.
  The current code writes it at the top of the project, which is the
  asymmetry `paths.py` exists to hide.
- **Re-order rounds nest under `4_reorder/`,** as `round_2/`, `round_3/` and
  so on. Their picks stay with them and are combined in `5_merge/`.
- **A pick-plate run is not a round.** It sequences the plate the merge built
  and checks each well against what was placed there. It samples nothing new,
  never feeds a merge, and never counts toward library recovery. Runs are
  named by their sequencing ID. The summary shows the newest run and lists
  earlier ones.
- **A pick-plate run is judged against the worklists the robot ran.** The
  expected layout is a snapshot taken the first time a run is
  demultiplexed, from the worklists in `6_hitpick/`. The pick in `5_merge/`
  is read beside them to name each well's source, and every well where the
  two disagree is reported; the worklists win. When a project has no
  worklists the layout comes from the pick. A layout already written is
  kept, so a re-run is judged against what was recorded before its reads
  were seen.
- **A pick-plate run may carry its own mask.** The plate is its own
  preparation and can carry sequencing artefacts the sort does not. A
  `qc_mask.toml` in the run's directory replaces the project's mask for that
  run only, the way a round's own mask does, so it lists every position the
  run forgives, with its evidence.
- **Worklists become a record once the plate is built.** A merge re-run
  after that point rewrites `6_hitpick/` and removes files it did not write.
  The merge should refuse to overwrite worklists once a pick-plate run
  exists; it does not yet.

## Commands

Pick-plate runs get their own option instead of a round number. `--pick-plate`
and `--round` are mutually exclusive.

```
usortm demux <project> --pick-plate G3Y8KW --plate-map <toml> [--min-read-length 1900] ...
usortm verify <project> [--pick-plate G3Y8KW]
```

`verify` with no run name takes the newest run. Both commands are
implemented. Demux writes the pipeline's output to `demux/` and copies the
per-well table and run summary into `results/`; verify writes its verdicts
and pileups to `results/`. Until the full migration, the worklists are read
from `integra_assist_input/` and the library reference from
`demux_output/`.

`usortm plan --pick-plate` is still present, as is reading a pick plate
recorded as a round, so a project made before `7_pick_plate/` still works.
Both go when the project is migrated.

## Current layout to target

The mapping for the AFMtag project (`~/AFMtag_demux/usortm_project`), which
has every step:

| Current | Target |
|---|---|
| `inputs/` | `0_inputs/` |
| `config/`, `barcodes/` | `1_plan/` |
| `fastqs/` | `2_sort/fastqs/` |
| `demux_output/` | `2_sort/demux/`, with per-well tables, plate map and pileups in `2_sort/results/` |
| `pick/` | `3_pick/` |
| `reorder/` | `4_reorder/order/` |
| `rounds/2/` | `4_reorder/round_2/` |
| `merged/` | `5_merge/` |
| `integra_assist_input/` | `6_hitpick/` |
| `rounds/3/` | `7_pick_plate/GQKLWM/` |
| `report/`, `usortm_report_round1.zip` | `report/` |
| `run_archive/` | removed, after review |

In the current code `rounds/3/` is a round whose state carries
`kind = "pick_plate"`. That was a mistake of category. The fix is to move it
to `7_pick_plate/GQKLWM/` and to record it under a `pick_plate` key in the
project state rather than under `rounds`.

## Migration plan

1. **`paths.py` implements the target layout** and remains the single source
   of every path.
2. **Every hard-coded path goes through `paths.py`.** Several modules still
   name `demux_output` and `rounds/` directly, and all of them move in the
   same change.
3. **Pick-plate runs leave the round machinery.** This removes the
   `pick_plate` round kind, `plan --pick-plate`, and the special case in
   `report/summary.py` that stops a pick-plate round from dating the merged
   pick. That special case exists only because the runs lived under
   `rounds/`.
4. **`usortm migrate <project>` moves an existing project.** It prints its
   moves by default and changes nothing until run with `--apply`.
5. **`commands.txt` records the equivalent new commands** for steps run under
   the old layout, each with a note of how it was originally run, so the file
   stays reproducible.
6. **The user documentation moves with the code.** `docs/getting-started.html`,
   `docs/cli.html`, `docs/demultiplexing.html`, `docs/hitpicking.html` and
   `README.md` describe the current paths and are updated when the migration
   lands, not before.

## Costs

- **Adding a step later means renumbering.** The code change is one line in
  `paths.py`, but existing projects need migrating again.
- **Single digits sort correctly up to nine.** The layout uses eight, `0` to
  `7`, which leaves room for two more steps before the prefixes need a
  leading zero.

## Open decisions

- **Step names.** The names above follow the bench: sort, hitpick, pick
  plate. The alternative mirrors the commands, as in `2_demux/` and
  `7_verify/`, so the command that writes a directory shares its name.
- **`index.html`.** A small landing page that overlaps with `summary.html`.
  The proposal is to drop it.
- **When to move `rounds/3/` to `7_pick_plate/GQKLWM/`.** It can move ahead
  of the rest of the migration, since the run code already reads that
  location.
