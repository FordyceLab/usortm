"""Load synthesis method pricing from TOML config files."""

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

METHODS_DIR = Path(__file__).parent / "methods"


class SynthesisMethod:
    """A synthesis method loaded from a TOML config file."""

    __slots__ = (
        "name", "vendor", "type", "date_collected", "notes",
        "seq_length_min", "seq_length_max",
        "library_size_min", "library_size_max",
        "error_rate", "abundance_skew",
        "pricing", "slug",
    )

    def __init__(self, slug, data):
        meta = data["meta"]
        caps = data["capabilities"]
        sim = data["simulation"]

        self.slug = slug
        self.name = meta["name"]
        self.vendor = meta["vendor"]
        self.type = meta["type"]
        self.date_collected = meta["date_collected"]
        self.notes = meta.get("notes", "")

        self.seq_length_min = caps["seq_length_min"]
        self.seq_length_max = caps["seq_length_max"]
        self.library_size_min = caps.get("library_size_min")
        self.library_size_max = caps.get("library_size_max")

        er = sim.get("error_rate", [1e-4, 5e-4])
        if len(er) == 1:
            er = [er[0], er[0]]
        self.error_rate = tuple(er)

        # abundance_skew only applies to pooled synthesis
        if meta["type"] == "pooled":
            self.abundance_skew = tuple(sim["abundance_skew"])
        else:
            self.abundance_skew = None

        self.pricing = data["pricing"]

    def __repr__(self):
        return f"SynthesisMethod({self.slug!r}, {self.name!r})"


def load_method(path):
    """Load a single SynthesisMethod from a TOML file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return SynthesisMethod(path.stem, data)


def load_all_methods(methods_dir=None):
    """Load all methods from a directory, returning a dict keyed by slug."""
    d = Path(methods_dir) if methods_dir else METHODS_DIR
    methods = {}
    for p in sorted(d.glob("*.toml")):
        m = load_method(p)
        methods[m.slug] = m
    return methods


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


def find_methods(seq_length, library_size=None, method_type=None, methods_dir=None):
    """Return all methods whose capabilities overlap the given parameters.

    Args:
        seq_length: Length of sequence in bp.
        library_size: Number of sequences (used for pooled method filtering).
        method_type: Filter by "pooled" or "arrayed". None returns both.
        methods_dir: Optional custom methods directory.

    Returns:
        List of matching SynthesisMethod objects.
    """
    all_methods = load_all_methods(methods_dir)
    results = []

    for method in all_methods.values():
        if method_type and method.type != method_type:
            continue
        if not (method.seq_length_min <= seq_length <= method.seq_length_max):
            continue
        if library_size is not None and method.library_size_min is not None:
            if not (method.library_size_min <= library_size <= method.library_size_max):
                continue
        results.append(method)

    return results
