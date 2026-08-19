"""Tests for read templates.

A read template describes a whole read with its three variable spans masked:
forward barcode, variable region, reverse barcode.  The barcode masks and the
vector flanks are both read off it, so they cannot disagree with each other
the way separately-supplied files can.
"""

import pytest

from usortm.demux.read_template import (
    MAX_BARCODE_SPAN,
    ReadTemplateError,
    parse_read_template,
    write_mask_config,
    write_vector_fasta,
)

FRONT_PAD = "GGATCCTTAAGCACTCAATG"     # 20 bp before the forward barcode
FLANK_5P = "ACGT" * 40                  # 160 bp
FLANK_3P = "TTGA" * 50                  # 200 bp
REAR_PAD = "CCTAGGATTCGTGAGTTACC"      # 20 bp after the reverse barcode
_COMP = str.maketrans("ACGT", "TGCA")


def _rc(s):
    return s.translate(_COMP)[::-1]


def _template(tmp_path, fbc=24, var=300, rbc=24, front=FRONT_PAD, rear=REAR_PAD,
              name="template.fa"):
    seq = (front + "N" * fbc + FLANK_5P + "N" * var + FLANK_3P
           + "N" * rbc + rear)
    path = tmp_path / name
    path.write_text(f">Reference_read\n{seq}\n")
    return path


class TestParsing:

    def test_spans_are_located(self, tmp_path):
        t = parse_read_template(_template(tmp_path))
        (f0, f1), (v0, v1), (r0, r1) = t.spans

        assert f1 - f0 == 24
        assert v1 - v0 == 300
        assert r1 - r0 == 24

    def test_flanks_are_the_constant_regions(self, tmp_path):
        """The flanks sit between the barcodes and the variable region — they
        must not include the barcodes, which differ per well."""
        t = parse_read_template(_template(tmp_path))

        assert t.flank_5p == FLANK_5P
        assert t.flank_3p == FLANK_3P
        assert "N" not in t.flank_5p + t.flank_3p

    def test_masks_come_from_around_the_barcode_spans(self, tmp_path):
        t = parse_read_template(_template(tmp_path), mask_length=20)

        assert t.masks["mask1_front"] == FRONT_PAD
        assert t.masks["mask1_rear"] == FLANK_5P[:20]
        assert t.masks["mask2_front"] == FLANK_3P[-20:]
        assert t.masks["mask2_rear"] == REAR_PAD

    def test_mask_length_is_respected(self, tmp_path):
        t = parse_read_template(_template(tmp_path), mask_length=10)
        assert t.masks["mask1_front"] == FRONT_PAD[-10:]
        assert t.masks["mask1_rear"] == FLANK_5P[:10]

    def test_lowercase_and_x_masking_accepted(self, tmp_path):
        seq = (FRONT_PAD + "x" * 24 + FLANK_5P.lower() + "X" * 300
               + FLANK_3P + "n" * 24 + REAR_PAD)
        path = tmp_path / "mixed.fa"
        path.write_text(f">r\n{seq}\n")

        t = parse_read_template(path)
        assert t.variable_length == 300
        assert t.flank_5p == FLANK_5P          # uppercased

    def test_vector_sequence_round_trips_through_the_vector_parser(self, tmp_path):
        """The derived vector must be readable by the existing --vector-fasta
        path, since that is what consumes it."""
        from usortm.demux.utils import parse_vector_fasta

        t = parse_read_template(_template(tmp_path))
        out = write_vector_fasta(t, tmp_path / "vector.fasta")
        flank_5p, flank_3p = parse_vector_fasta(str(out))

        assert flank_5p == t.flank_5p
        assert flank_3p == t.flank_3p

    def test_mask_config_round_trips_through_the_loader(self, tmp_path):
        from usortm.cli.demux_cmd import _load_mask_config

        t = parse_read_template(_template(tmp_path))
        out = write_mask_config(t, tmp_path / "masks.toml", source="template.fa")
        loaded = _load_mask_config(out)

        assert loaded["fbc"]["mask1_front"] == t.masks["mask1_front"]
        assert loaded["fbc"]["mask1_rear"] == t.masks["mask1_rear"]


class TestValidation:

    def test_one_masked_span_is_not_a_read_template(self, tmp_path):
        """That is a --vector-fasta; say so rather than guessing."""
        path = tmp_path / "vector.fa"
        path.write_text(f">v\n{FLANK_5P}{'N' * 300}{FLANK_3P}\n")

        with pytest.raises(ReadTemplateError, match="vector-fasta"):
            parse_read_template(path)

    def test_two_masked_spans_rejected(self, tmp_path):
        path = tmp_path / "two.fa"
        path.write_text(f">t\n{FRONT_PAD}{'N' * 24}{FLANK_5P}{'N' * 300}{FLANK_3P}\n")

        with pytest.raises(ReadTemplateError, match="found 2"):
            parse_read_template(path)

    def test_no_masked_spans_reports_none(self, tmp_path):
        path = tmp_path / "plain.fa"
        path.write_text(f">t\n{FLANK_5P}\n")

        with pytest.raises(ReadTemplateError, match="none"):
            parse_read_template(path)

    def test_implausible_barcode_span_rejected(self, tmp_path):
        """Guards against the spans being given in the wrong order."""
        with pytest.raises(ReadTemplateError, match="forward barcode span"):
            parse_read_template(_template(tmp_path, fbc=MAX_BARCODE_SPAN + 50))

    def test_barcode_at_the_very_start_has_no_mask(self, tmp_path):
        with pytest.raises(ReadTemplateError, match="no sequence in front"):
            parse_read_template(_template(tmp_path, front=""))

    def test_barcode_at_the_very_end_has_no_mask(self, tmp_path):
        with pytest.raises(ReadTemplateError, match="no sequence after"):
            parse_read_template(_template(tmp_path, rear=""))

    def test_multi_record_fasta_rejected(self, tmp_path):
        path = tmp_path / "multi.fa"
        path.write_text(f">a\n{FRONT_PAD}{'N'*24}{FLANK_5P}{'N'*300}{FLANK_3P}"
                        f"{'N'*24}{REAR_PAD}\n>b\nACGT\n")

        with pytest.raises(ReadTemplateError, match="expected one record"):
            parse_read_template(path)

    def test_empty_file_rejected(self, tmp_path):
        path = tmp_path / "empty.fa"
        path.write_text("")

        with pytest.raises(ReadTemplateError, match="no sequence"):
            parse_read_template(path)
