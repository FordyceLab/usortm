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
        """Every .toml in the methods directory loads without error."""
        assert len(methods) > 0

    def test_required_fields_present(self, methods):
        """Every method has the required schema fields with correct types."""
        for slug, m in methods.items():
            assert isinstance(m.name, str) and m.name, slug
            assert isinstance(m.vendor, str) and m.vendor, slug
            assert m.type in ("pooled", "arrayed"), slug
            assert isinstance(m.seq_length_min, int) and m.seq_length_min >= 0, slug
            assert isinstance(m.seq_length_max, int) and m.seq_length_max >= m.seq_length_min, slug
            assert len(m.error_rate) == 2 and m.error_rate[0] <= m.error_rate[1], slug

    def test_pooled_methods_have_skew(self, methods):
        """Pooled methods must have a scalar skew_q90_q10 > 1; arrayed must not."""
        for slug, m in methods.items():
            if m.type == "pooled":
                assert isinstance(m.skew_q90_q10, float) and m.skew_q90_q10 > 1.0, slug
            else:
                assert m.skew_q90_q10 is None, slug

    def test_pricing_model_is_known(self, methods):
        """Every method uses a recognised pricing model."""
        known_models = {"lookup", "per_base", "per_fragment", "tiered"}
        for slug, m in methods.items():
            assert m.pricing["model"] in known_models, slug


class TestComputeCost:
    """Pricing models return plausible values; behavior is more important than exact prices."""

    def test_returns_positive_or_none_for_all_methods(self, methods):
        """compute_cost never raises and never returns a negative number."""
        for slug, m in methods.items():
            mid_len = (m.seq_length_min + m.seq_length_max) // 2 or 200
            lib = m.library_size_min or 100
            cost = compute_cost(m, lib, mid_len)
            assert cost is None or cost > 0, slug

    def test_outside_library_size_returns_none(self, methods):
        """Requests beyond a method's max library size return None, not an error."""
        for slug, m in methods.items():
            if m.library_size_max is None:
                continue
            cost = compute_cost(m, m.library_size_max + 99999, m.seq_length_min or 200)
            assert cost is None, slug

    def test_larger_library_not_cheaper(self, methods):
        """Larger libraries should cost the same or more than smaller ones."""
        for slug, m in methods.items():
            if m.library_size_min is None:
                continue
            mid_len = (m.seq_length_min + m.seq_length_max) // 2 or 200
            small = m.library_size_min
            large = min(m.library_size_min * 10, m.library_size_max or small * 10)
            cost_small = compute_cost(m, small, mid_len)
            cost_large = compute_cost(m, large, mid_len)
            if cost_small is not None and cost_large is not None:
                assert cost_large >= cost_small, slug


class TestFindMethods:
    def test_results_cover_requested_seq_length(self):
        """Every returned method's seq_length range includes the query."""
        for seq_len in (100, 300, 500, 1000):
            for m in find_methods(seq_len):
                assert m.seq_length_min <= seq_len <= m.seq_length_max

    def test_type_filter_respected(self):
        for method_type in ("pooled", "arrayed"):
            for m in find_methods(300, method_type=method_type):
                assert m.type == method_type

    def test_library_size_filter_respected(self):
        """Methods with a library_size_max below the query are excluded."""
        for m in find_methods(200, library_size=500):
            if m.library_size_max is not None:
                assert m.library_size_max >= 500

    def test_very_long_seq_returns_no_short_methods(self):
        """No method designed for short oligos should appear for 2000 bp queries."""
        for m in find_methods(2000):
            assert m.seq_length_max >= 2000
