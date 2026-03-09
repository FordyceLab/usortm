"""Tests for synthesis method TOML loading and cost computation."""

import math
import pytest

from usortm.costs.method_loader import (
    load_all_methods,
    load_method,
    compute_cost,
    find_methods,
    METHODS_DIR,
)


@pytest.fixture(autouse=True)
def _clear_cost_cache():
    """Clear the methods cache before each test to avoid stale data."""
    import usortm.costs.cost_functions as cf
    cf._methods_cache = None
    yield
    cf._methods_cache = None


@pytest.fixture
def methods():
    return load_all_methods()


class TestLoadMethods:
    def test_all_toml_files_load(self, methods):
        expected_slugs = {
            "twist_oligo_pools",
            "twist_multiplexed_gene_fragments",
            "twist_gene_pools",
            "twist_cloned_oligo_pools",
            "idt_eblocks",
            "idt_gblocks",
            "twist_gene_fragments",
        }
        assert set(methods.keys()) == expected_slugs

    def test_twist_oligo_pools_meta(self, methods):
        m = methods["twist_oligo_pools"]
        assert m.name == "Twist Oligo Pools"
        assert m.vendor == "Twist Bioscience"
        assert m.type == "pooled"

    def test_twist_oligo_pools_extended_tiers(self, methods):
        m = methods["twist_oligo_pools"]
        assert m.library_size_max == 54000
        # Verify tier 13 data exists
        cost = compute_cost(m, 50000, 300)
        assert cost is not None

    def test_twist_oligo_pools_350nt_column(self, methods):
        m = methods["twist_oligo_pools"]
        # Tier 1, 350 nt -> 1288 * 0.667
        cost = compute_cost(m, 50, 350)
        assert cost == pytest.approx(1288.0 * 0.667, rel=1e-3)

    def test_idt_eblocks_capabilities(self, methods):
        m = methods["idt_eblocks"]
        assert m.seq_length_min == 300
        assert m.seq_length_max == 1500
        assert m.library_size_min is None  # arrayed, no library size

    def test_simulation_fields(self, methods):
        m = methods["twist_oligo_pools"]
        assert len(m.error_rate) == 2
        assert isinstance(m.skew_q90_q10, float)
        assert m.skew_q90_q10 > 1.0
        assert m.error_rate[0] < m.error_rate[1]
        # arrayed methods have no skew_q90_q10
        assert methods["idt_eblocks"].skew_q90_q10 is None


class TestComputeCostLookup:
    """Test Twist Oligo Pool lookup pricing matches original hardcoded values."""

    def test_small_library_short_seq(self, methods):
        m = methods["twist_oligo_pools"]
        # 50 seqs, 120 bp -> tier (2,100), length 120, cost 400 * 0.667
        cost = compute_cost(m, 50, 120)
        assert cost == pytest.approx(400.0 * 0.667, rel=1e-3)

    def test_medium_library(self, methods):
        m = methods["twist_oligo_pools"]
        # 500 seqs, 200 bp -> tier (101,500), length 200, cost 1040 * 0.667
        cost = compute_cost(m, 500, 200)
        assert cost == pytest.approx(1040.0 * 0.667, rel=1e-3)

    def test_nearest_length_rounding(self, methods):
        m = methods["twist_oligo_pools"]
        # 50 seqs, 130 bp -> nearest to 120, cost 400 * 0.667
        cost = compute_cost(m, 50, 130)
        assert cost == pytest.approx(400.0 * 0.667, rel=1e-3)

    def test_outside_tiers_returns_none(self, methods):
        m = methods["twist_oligo_pools"]
        cost = compute_cost(m, 60000, 200)  # above max tier (54,000)
        assert cost is None


class TestComputeCostPerBase:
    """Test per-base pricing matches original hardcoded values."""

    def test_idt_eblocks_short(self, methods):
        m = methods["idt_eblocks"]
        # 100 seqs, 330 bp -> $0.07/bp
        cost = compute_cost(m, 100, 330)
        assert cost == pytest.approx(330 * 100 * 0.07)

    def test_idt_eblocks_long(self, methods):
        m = methods["idt_eblocks"]
        # 100 seqs, 1200 bp -> $0.07/bp
        cost = compute_cost(m, 100, 1200)
        assert cost == pytest.approx(1200 * 100 * 0.07)

    def test_idt_gblocks(self, methods):
        m = methods["idt_gblocks"]
        # 100 seqs, 500 bp -> $0.09/bp
        cost = compute_cost(m, 100, 500)
        assert cost == pytest.approx(500 * 100 * 0.09)

    def test_twist_gene_fragments_flat(self, methods):
        m = methods["twist_gene_fragments"]
        # 50 seqs, 400 bp -> $35/fragment
        cost = compute_cost(m, 50, 400)
        assert cost == pytest.approx(50 * 35)

    def test_twist_gene_fragments_per_base(self, methods):
        m = methods["twist_gene_fragments"]
        # 50 seqs, 600 bp -> $0.07/bp
        cost = compute_cost(m, 50, 600)
        assert cost == pytest.approx(600 * 50 * 0.07)


class TestComputeCostTiered:
    """Test Gene Pools tiered pricing (lookup model, large library)."""

    def test_small_library(self, methods):
        m = methods["twist_gene_pools"]
        cost = compute_cost(m, 500, 650)
        assert cost == pytest.approx(17335.00)

    def test_large_library(self, methods):
        m = methods["twist_gene_pools"]
        cost = compute_cost(m, 5000, 1400)
        assert cost == pytest.approx(51089.00)


class TestFindMethods:
    def test_short_seq_pooled(self):
        results = find_methods(200, library_size=500, method_type="pooled")
        slugs = {m.slug for m in results}
        assert "twist_oligo_pools" in slugs

    def test_overlap_zone(self):
        """300 bp is valid for both Twist Oligo Pools and Twist Gene Fragments."""
        results = find_methods(300, library_size=500)
        slugs = {m.slug for m in results}
        assert "twist_oligo_pools" in slugs
        assert "twist_gene_fragments" in slugs

    def test_long_seq_finds_gene_pools(self):
        results = find_methods(500, library_size=1000, method_type="pooled")
        slugs = {m.slug for m in results}
        assert "twist_gene_pools" in slugs

    def test_arrayed_filter(self):
        results = find_methods(400, method_type="arrayed")
        for m in results:
            assert m.type == "arrayed"


class TestNewTwistMethods:
    """Test Twist pooled services from June 2025 price sheet."""

    def test_mgf_loads(self, methods):
        m = methods["twist_multiplexed_gene_fragments"]
        assert m.seq_length_min == 301
        assert m.seq_length_max == 500
        assert m.library_size_max == 696000

    def test_mgf_tier1_350bp(self, methods):
        m = methods["twist_multiplexed_gene_fragments"]
        cost = compute_cost(m, 500, 350)
        assert cost == pytest.approx(4944.00)

    def test_mgf_large_pool(self, methods):
        m = methods["twist_multiplexed_gene_fragments"]
        cost = compute_cost(m, 100000, 400)
        assert cost == pytest.approx(63096.00)

    def test_gene_pools_loads(self, methods):
        m = methods["twist_gene_pools"]
        assert m.seq_length_min == 300
        assert m.seq_length_max == 1800
        assert m.library_size_max == 696000

    def test_gene_pools_small_library(self, methods):
        m = methods["twist_gene_pools"]
        # 500 seqs, 650 bp (nearest to 650 key) -> $17,335
        cost = compute_cost(m, 500, 650)
        assert cost == pytest.approx(17335.00)

    def test_gene_pools_large_library(self, methods):
        m = methods["twist_gene_pools"]
        # 5000 seqs, 1400 bp -> $51,089
        cost = compute_cost(m, 5000, 1400)
        assert cost == pytest.approx(51089.00)

    def test_cloned_oligo_pools_loads(self, methods):
        m = methods["twist_cloned_oligo_pools"]
        assert m.seq_length_min == 1
        assert m.seq_length_max == 250

    def test_cloned_oligo_pools_tier1(self, methods):
        m = methods["twist_cloned_oligo_pools"]
        cost = compute_cost(m, 50, 120)
        assert cost == pytest.approx(5000.00)

    def test_find_methods_mgf_overlap(self):
        """350 bp falls in both Oligo Pools (≤350) and MGF (301-500) range."""
        results = find_methods(350, library_size=500, method_type="pooled")
        slugs = {m.slug for m in results}
        assert "twist_oligo_pools" in slugs
        assert "twist_multiplexed_gene_fragments" in slugs

    def test_find_methods_gene_pools(self):
        results = find_methods(1000, library_size=5000, method_type="pooled")
        slugs = {m.slug for m in results}
        assert "twist_gene_pools" in slugs


class TestCostFunctionsRegression:
    """Ensure refactored cost_functions.py matches original behavior."""

    def test_usortm_synthesis_short(self):
        from usortm.costs.cost_functions import usortm_synthesis_cost
        # 500 seqs, 200 bp -> Twist Oligo Pools lookup
        cost = usortm_synthesis_cost(500, 200)
        assert cost == pytest.approx(1040.0 * (2 / 3), rel=1e-3)

    def test_usortm_synthesis_long(self):
        from usortm.costs.cost_functions import usortm_synthesis_cost
        # 1000 seqs, 500 bp -> no usortm_substitution method exists, returns 0
        cost = usortm_synthesis_cost(1000, 500)
        assert cost == 0

    def test_parsed_genefragments_idt_eblocks(self):
        from usortm.costs.cost_functions import parsed_genefragments_synthesis_cost
        cost = parsed_genefragments_synthesis_cost(330, 100, 'idt_eblocks')
        assert cost == pytest.approx(330 * 100 * 0.07)

    def test_parsed_genefragments_twist_below_min(self):
        import numpy as np
        from usortm.costs.cost_functions import parsed_genefragments_synthesis_cost
        cost = parsed_genefragments_synthesis_cost(200, 100, 'twist_genefragments')
        assert np.isnan(cost)

    def test_parsed_genefragments_twist_flat_rate(self):
        from usortm.costs.cost_functions import parsed_genefragments_synthesis_cost
        cost = parsed_genefragments_synthesis_cost(400, 100, 'twist_genefragments')
        assert cost == pytest.approx(100 * 35)
