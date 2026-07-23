"""Synthesis-method access for uSort-M cost models.

Method definitions — capabilities, sequence-length and library-size limits,
error/skew estimates, and pricing tables — come from the shared
``synthesis_methods`` registry, so uSort-M and library_designer read the same
numbers instead of each carrying a copy of the TOMLs.

This module re-exports that data layer and adds the uSort-M-specific
``compute_cost``, which the shared package intentionally does not provide (it
holds the data; the caller does the cost math off ``method.pricing``).
"""
from __future__ import annotations

from synthesis_methods import (
    METHODS_DIR,
    SynthesisMethod,
    find_methods,
    get_method,
    load_all_methods,
    load_method,
    platform_type,
)

__all__ = [
    "SynthesisMethod",
    "load_method",
    "load_all_methods",
    "get_method",
    "find_methods",
    "platform_type",
    "METHODS_DIR",
    "compute_cost",
]


def compute_cost(method, n_seqs, seq_length):
    """Compute synthesis cost for a method given sequence count and length.

    Returns cost in USD, or None if the method cannot handle the inputs.
    """
    pricing = method.pricing
    model = pricing["model"]

    if model == "lookup":
        return _compute_lookup(method, n_seqs, seq_length)
    elif model == "per_base":
        return _compute_per_base(method, n_seqs, seq_length)
    elif model == "per_fragment":
        return _compute_per_fragment(method, n_seqs, seq_length)
    elif model == "tiered":
        return _compute_tiered(method, n_seqs, seq_length)
    else:
        raise ValueError(f"Unknown pricing model: {model}")


def _compute_lookup(method, n_seqs, seq_length):
    """Lookup-table pricing (e.g. Twist Oligo Pools)."""
    pricing = method.pricing
    discount = pricing.get("commercial_discount", 1.0)

    for tier in pricing["tiers"]:
        low, high = tier["library_size"]
        if low <= n_seqs <= high:
            # costs keys are strings in TOML — convert to int
            length_dict = {int(k): v for k, v in tier["costs"].items()}
            valid_lengths = sorted(length_dict.keys())
            nearest_len = min(valid_lengths, key=lambda x: abs(x - seq_length))
            return length_dict[nearest_len] * discount

    return None  # outside defined tiers


def _compute_per_base(method, n_seqs, seq_length):
    """Per-base or per-fragment pricing (e.g. IDT eBlocks, gBlocks, Twist Gene Fragments)."""
    pricing = method.pricing

    for rule in pricing["rules"]:
        low, high = rule["seq_length"]
        if low <= seq_length <= high:
            if "per_fragment" in rule:
                return n_seqs * rule["per_fragment"]
            elif "per_base" in rule:
                return seq_length * n_seqs * rule["per_base"]

    return None  # no matching rule


def _compute_per_fragment(method, n_seqs, seq_length):
    """Flat per-fragment pricing."""
    pricing = method.pricing
    return n_seqs * pricing.get("per_fragment", 0)


def _compute_tiered(method, n_seqs, seq_length):
    """Tiered pricing by library size (e.g. substitution library)."""
    pricing = method.pricing
    insert_length = pricing.get("insert_length", seq_length)
    total_bp = insert_length * n_seqs

    for tier in pricing["tiers"]:
        low, high = tier["library_size"]
        if low <= n_seqs <= high:
            if "per_base" in tier:
                return tier["per_base"] * total_bp
            elif "per_fragment" in tier:
                return tier["per_fragment"] * n_seqs

    return None  # outside defined tiers
