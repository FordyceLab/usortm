"""Tests for the cost model in costs/cost_functions.py.

Covers the two silent-$0 regressions: synthesis cost for sequences longer than
one oligo pool (>350 bp), and plate-count rounding for barcoding/sorting (a
partial plate must still bill as a full plate).
"""
import math

import pytest

import usortm.costs.cost_functions as cf
from usortm.costs.method_loader import compute_cost, load_all_methods

BARCODE_PER_PLATE = 97.73


# --- synthesis ---------------------------------------------------------------

def test_synthesis_pooled_le_350_is_positive():
    assert cf.usortm_synthesis_cost(500, 300) > 0


def test_synthesis_gt_350_uses_gene_pools():
    # >350 bp used to look up a nonexistent method and silently return $0.
    cost = cf.usortm_synthesis_cost(500, 500)
    assert cost > 0
    gene_pools = load_all_methods()["twist_gene_pools"]
    assert cost == compute_cost(gene_pools, 500, 500)


def test_synthesis_short_insert_model_is_positive():
    # Substitution/tiled libraries are priced via the ~30 bp insert length.
    assert cf.usortm_synthesis_cost(500, 30) > 0


def test_synthesis_very_long_still_prices_via_gene_pools():
    # The gene-pools lookup clamps to the nearest tabulated length, so a very
    # long sequence still returns a (top-bucket) price rather than a silent $0.
    cost = cf.usortm_synthesis_cost(500, 5000)
    assert cost > 0
    gene_pools = load_all_methods()["twist_gene_pools"]
    assert cost == compute_cost(gene_pools, 500, 5000)


# --- barcoding: ceil plate count ---------------------------------------------

def test_barcoding_subplate_bills_one_plate():
    # 320 wells is < 384, so floor division used to bill $0. It is one plate.
    assert cf.usortm_barcoding_cost(320) == pytest.approx(BARCODE_PER_PLATE)


def test_barcoding_zero_wells_is_zero():
    assert cf.usortm_barcoding_cost(0) == 0


def test_barcoding_exact_plate_boundary():
    assert cf.usortm_barcoding_cost(384) == pytest.approx(BARCODE_PER_PLATE)
    assert cf.usortm_barcoding_cost(385) == pytest.approx(2 * BARCODE_PER_PLATE)


def test_barcoding_rounds_up_partial_plate():
    # 4000 wells -> ceil(4000/384) == 11 plates (was floor -> 10).
    assert cf.usortm_barcoding_cost(4000) == pytest.approx(11 * BARCODE_PER_PLATE)


# --- sorting: ceil plate count -----------------------------------------------

def test_sorting_subplate_counts_one_plate():
    # 40 variants x 8 = 320 wells -> 1 plate (6 min) + 60 min setup.
    cost = cf.usortm_sorting_cost(40, fold_sampling=8)
    expected = (66 / 60) * (70 + 65)
    assert cost == pytest.approx(expected)
    # Strictly greater than the setup-only floor (which floor division gave).
    assert cost > (60 / 60) * (70 + 65)


def test_sorting_rounds_up_partial_plate():
    # 500 x 8 = 4000 wells -> ceil == 11 plates (66 min sort + 60 setup).
    cost = cf.usortm_sorting_cost(500, fold_sampling=8)
    expected = ((11 * 6 + 60) / 60) * (70 + 65)
    assert cost == pytest.approx(expected)
