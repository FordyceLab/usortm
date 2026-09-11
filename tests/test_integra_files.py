"""The pick is written one file per source plate, as the robot is loaded.

The Integra ASSIST PLUS runs one source plate at a time.  The pick used to
write one file per *target* plate with every source plate mixed together, so
a run of fourteen plates arrived as a single file the robot could not be
loaded from.  The form the lab picks from is one file per source plate,
integra_assist_plate{N}.csv, with the volume written as the robot reads it.
"""
import csv

from usortm.cli.reorder import _find_hitlists, _load_recovered
from usortm.integra import (HEADER, format_volume, integra_filename,
                            write_integra_files)


def _hit(variant, plate, well, target_well, **kw):
    d = {"variant": variant, "source_plate": str(plate), "source_well": well,
         "target_plate": "0", "target_well": target_well}
    d.update(kw)
    return d


def _read(path):
    rows = list(csv.reader(open(path), delimiter=";"))
    return rows[0], rows[1:]


def test_one_file_per_source_plate_named_for_it(tmp_path):
    picks = [_hit("G3A", 13, "O12", "A1"), _hit("G15A", 3, "N23", "C1"),
             _hit("G15F", 2, "G22", "D1"), _hit("G15M", 3, "B2", "E1")]
    files = write_integra_files(picks, tmp_path, 5.0)
    assert [f.name for f in files] == ["integra_assist_plate2.csv",
                                       "integra_assist_plate3.csv",
                                       "integra_assist_plate13.csv"]
    header, rows = _read(tmp_path / "integra_assist_plate3.csv")
    assert header == HEADER
    assert [r[0] for r in rows] == ["G15A", "G15M"]
    assert {r[1] for r in rows} == {"3"}, "a plate's file holds only its own transfers"


def test_the_volume_is_written_as_the_robot_reads_it(tmp_path):
    write_integra_files([_hit("G3A", 1, "A1", "A1")], tmp_path, 5.0)
    _, rows = _read(tmp_path / "integra_assist_plate1.csv")
    assert rows[0][5] == "5"
    assert format_volume(2.5) == "2.5" and format_volume(10) == "10"


def test_the_file_matches_the_reference_byte_for_byte(tmp_path):
    """The exact bytes of a file the lab has run from: semicolons, "\\n"
    line endings, integer volume, no quoting."""
    write_integra_files([_hit("D_opt_13", 2, "B1", "A4")], tmp_path, 5.0)
    raw = (tmp_path / "integra_assist_plate2.csv").read_bytes()
    assert raw == (b"SampleID;SourcePlateID;SourceWell;TargetPlateID;TargetWell;TransferVolume\n"
                   b"D_opt_13;2;B1;0;A4;5\n")


def test_a_plate_with_nothing_to_pick_still_gets_its_file(tmp_path):
    files = write_integra_files([_hit("G3A", 1, "A1", "A1")], tmp_path, 5.0,
                                source_plates={"1", "2", "7"})
    assert [f.name for f in files] == ["integra_assist_plate1.csv",
                                       "integra_assist_plate2.csv",
                                       "integra_assist_plate7.csv"]
    header, rows = _read(tmp_path / "integra_assist_plate7.csv")
    assert header == HEADER and rows == []


def test_placeholders_and_streakouts_are_not_transfers(tmp_path):
    picks = [_hit("G3A", 1, "A1", "A1"),
             _hit("N4A", "", "", "B1", empty=True),
             _hit("V47M", 2, "D1", "C1", tier_override="Streakout")]
    files = write_integra_files(picks, tmp_path, 5.0)
    assert [f.name for f in files] == ["integra_assist_plate1.csv"]
    _, rows = _read(files[0])
    assert [r[0] for r in rows] == ["G3A"]


def test_round_prefixed_plates_keep_the_round_in_the_name():
    assert integra_filename("R1_3") == "integra_assist_R1_plate3.csv"
    assert integra_filename("R2_1") == "integra_assist_R2_plate1.csv"
    assert integra_filename("14") == "integra_assist_plate14.csv"


def test_a_stop_codon_is_written_without_the_asterisk_and_read_back(tmp_path):
    """The ASSIST PLUS rejects '*' in SampleID.  It is written as tag (the amber codon) and
    turned back into '*' by the code that reads the files (reorder)."""
    from usortm.integra import library_name, robot_sample_id

    assert robot_sample_id("K16*") == "K16tag"
    assert robot_sample_id("ATF4;25;171") == "ATF4.25.171"
    assert library_name("K16tag") == "K16*"
    assert library_name("AFMtag_G3A") == "AFMtag_G3A"   # letters alone are not a stop

    pick_dir = tmp_path / "pick" / "integra_assist_input"
    write_integra_files([_hit("K16*", 1, "A1", "A1"), _hit("G3A", 1, "B1", "B2")],
                        pick_dir, 5.0)
    _, rows = _read(pick_dir / "integra_assist_plate1.csv")
    assert [r[0] for r in rows] == ["K16tag", "G3A"]
    assert "*" not in (pick_dir / "integra_assist_plate1.csv").read_text()
    recovered = set()
    for f in _find_hitlists(tmp_path):
        recovered |= _load_recovered(f)
    assert recovered == {"K16*", "G3A"}


def test_the_source_plate_cell_is_the_number_alone(tmp_path):
    """The ASSIST PLUS reads SourcePlateID as a number.  It rejected R1_1 with
    "please make sure that all columns contain the same number of entries";
    the round lives in the filename, which is what each file is one of."""
    write_integra_files([_hit("Q27F", "R1_1", "I21", "F1"),
                         _hit("G3F", "R2_2", "A1", "B1")], tmp_path, 5.0)
    _, rows = _read(tmp_path / "integra_assist_R1_plate1.csv")
    assert rows == [["Q27F", "1", "I21", "0", "F1", "5"]]
    _, rows = _read(tmp_path / "integra_assist_R2_plate2.csv")
    assert rows == [["G3F", "2", "A1", "0", "B1", "5"]]


def test_plates_sort_numerically_and_by_round(tmp_path):
    picks = [_hit("a", "R2_1", "A1", "A1"), _hit("b", "R1_10", "A1", "B1"),
             _hit("c", "R1_2", "A1", "C1")]
    files = write_integra_files(picks, tmp_path, 5.0)
    assert [f.name for f in files] == ["integra_assist_R1_plate2.csv",
                                       "integra_assist_R1_plate10.csv",
                                       "integra_assist_R2_plate1.csv"]


def test_the_old_per_target_files_are_removed(tmp_path):
    (tmp_path / "hitlist_plate_0.csv").write_text("stale\n")
    (tmp_path / "integra_assist_plate9.csv").write_text("stale\n")
    write_integra_files([_hit("G3A", 1, "A1", "A1")], tmp_path, 5.0)
    assert not (tmp_path / "hitlist_plate_0.csv").exists()
    assert not (tmp_path / "integra_assist_plate9.csv").exists()


def test_reorder_reads_recovered_variants_across_the_plate_files(tmp_path):
    """A variant is recovered if any plate's file transfers it."""
    pick_dir = tmp_path / "pick" / "integra_assist_input"
    picks = [_hit("G3A", 1, "A1", "A1"), _hit("G15A", 3, "N23", "C1")]
    write_integra_files(picks, pick_dir, 5.0, source_plates={"1", "2", "3"})
    files = _find_hitlists(tmp_path)
    assert [f.name for f in files] == ["integra_assist_plate1.csv",
                                       "integra_assist_plate2.csv",
                                       "integra_assist_plate3.csv"]
    recovered = set()
    for f in files:
        recovered |= _load_recovered(f)
    assert recovered == {"G3A", "G15A"}
