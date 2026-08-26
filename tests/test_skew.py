"""Tests for library skew measurement and sorting-depth recommendation.

The anchor is a closed loop: draw a pool of known skew, Poisson-sample it at
realistic depth, and check the estimator recovers the skew that was put in
while the raw Q90/Q10 ratio does not.  Everything else guards a specific
failure mode that would silently produce a wrong sorting depth.

Tests needing minimap2 are skipped when it is not installed.
"""

import csv
import json
import shutil

import numpy as np
import pytest
from typer.testing import CliRunner

from usortm.cli import app
from usortm.demux.deps import DependencyError
from usortm.qc import (
    LibraryProfile,
    check_resolvability,
    count_variant_reads,
    measure_skew,
    profile_library,
    read_variant_sequences,
    recommend_sampling,
)
from usortm.qc.counting import count_fastq_reads
from usortm.qc.resolve import _find_duplicate_groups
from usortm.qc.synthetic import make_synthetic_library
from usortm.qc.skew import (
    VariantCounts,
    ci_to_json,
    log10_histogram,
    predicted_count_distribution,
    sigma_to_skew,
    skew_to_sigma,
)
from usortm.simulate.sample import generate_pool
from usortm.simulate.sortm import find_fold_sampling, sortm

runner = CliRunner()


def _tool_available(name: str) -> bool:
    """Check if an external tool is available using the project's finders."""
    from usortm.demux import deps
    try:
        finder = getattr(deps, f"find_{name}", None)
        if finder:
            finder()
            return True
    except DependencyError:
        return False
    return shutil.which(name) is not None


requires_minimap2 = pytest.mark.skipif(
    not _tool_available("minimap2"),
    reason="minimap2 not installed",
)

BASES = "ACGT"

# Coverage is estimated by simulation, so assertions against a target allow
# a little slack; tests here use small n_sims to stay fast.
_MC_SLACK = 0.03


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def synthetic_counts(lib_size, skew, depth, dropout=0.0, seed=0):
    """Poisson-sample a pool of known skew at a given reads-per-variant depth."""
    rng = np.random.default_rng(seed)
    p = generate_pool(lib_size, skew, seed)
    if dropout > 0:
        p = p * ~(rng.random(lib_size) < dropout)
        p = p / p.sum()
    counts = rng.poisson(depth * lib_size * p)
    return VariantCounts(
        counts={f"var{i:04d}": int(c) for i, c in enumerate(counts)},
        total_reads=int(counts.sum()),
    )


def realized_skew(p):
    """Q90/Q10 of an actual drawn pool, which differs from the population value."""
    return float(np.percentile(p, 90) / np.percentile(p, 10))


def random_seq(rng, n):
    return "".join(rng.choice(list(BASES), n))


def write_variants_csv(path, sequences):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Name", "Sequence"])
        for name, seq in sequences.items():
            writer.writerow([name, seq])
    return path


def write_fastq(path, reads):
    """Write (name, sequence) pairs as a FASTQ with uniform quality."""
    with open(path, "w") as fh:
        for name, seq in reads:
            fh.write(f"@{name}\n{seq}\n+\n{'I' * len(seq)}\n")
    return path


# ---------------------------------------------------------------------------
# Parameterization
# ---------------------------------------------------------------------------

def test_sigma_skew_roundtrip():
    """skew <-> sigma conversions invert each other."""
    for skew in (1.0, 2.0, 4.0, 8.0, 20.0):
        assert sigma_to_skew(skew_to_sigma(skew)) == pytest.approx(skew, rel=1e-9)


def test_sigma_matches_generate_pool_parameterization():
    """The estimator inverts the same sigma the pool generator uses."""
    lib_size, skew = 20000, 4.0
    p = generate_pool(lib_size, skew, seed=1)
    assert realized_skew(p) == pytest.approx(skew, rel=0.05)


# ---------------------------------------------------------------------------
# Skew estimation
# ---------------------------------------------------------------------------

def test_corrected_skew_recovers_truth_while_raw_is_inflated():
    """The whole point: deconvolution beats the raw ratio at real depth.

    2000 variants over ~15k reads is 7.5 reads each, where Poisson noise
    badly inflates the observed spread.
    """
    counts = synthetic_counts(2000, skew=4.0, depth=7.5, seed=3)
    stats = measure_skew(counts)

    assert stats.q90_q10_corrected == pytest.approx(4.0, rel=0.15)
    # The raw statistic is materially worse, and always in the same direction.
    assert stats.q90_q10_observed > stats.q90_q10_corrected
    assert stats.q90_q10_observed > 5.5


def test_uniform_library_is_not_reported_as_skewed():
    """A perfectly even library must not measure as skewed at low depth."""
    counts = synthetic_counts(500, skew=1.0, depth=30, seed=5)
    stats = measure_skew(counts)

    assert stats.q90_q10_observed > 1.3      # counting noise alone
    assert stats.q90_q10_corrected < 1.25    # deconvolved away


@pytest.mark.parametrize("skew", [2.0, 4.0, 8.0])
def test_corrected_skew_tracks_across_skews(skew):
    counts = synthetic_counts(1000, skew=skew, depth=20, seed=11)
    stats = measure_skew(counts)
    assert stats.q90_q10_corrected == pytest.approx(skew, rel=0.2)


def test_dropouts_recovered_without_inflating_skew():
    """Absent variants must be reported as dropout, not as extra skew."""
    counts = synthetic_counts(500, skew=4.0, depth=30, dropout=0.10, seed=7)
    stats = measure_skew(counts)

    assert stats.dropout_fraction == pytest.approx(0.10, abs=0.05)
    assert stats.q90_q10_corrected == pytest.approx(4.0, rel=0.25)
    assert stats.coverage_ceiling == pytest.approx(1.0 - stats.dropout_fraction)


def test_zero_dropout_when_every_variant_is_seen():
    counts = synthetic_counts(300, skew=2.0, depth=60, seed=13)
    stats = measure_skew(counts)
    assert stats.n_undetected == 0
    assert stats.dropout_fraction == pytest.approx(0.0, abs=1e-6)
    assert stats.coverage_ceiling == pytest.approx(1.0)


def test_confidence_interval_brackets_the_estimate_and_the_truth():
    counts = synthetic_counts(1000, skew=4.0, depth=20, seed=17)
    stats = measure_skew(counts)
    low, high = stats.q90_q10_ci

    assert low < stats.q90_q10_corrected < high
    assert low < 4.0 < high


def test_confidence_interval_widens_for_smaller_libraries():
    """Small libraries carry genuinely less information about skew."""
    def width(lib_size):
        stats = measure_skew(synthetic_counts(lib_size, 4.0, depth=30, seed=23))
        low, high = stats.q90_q10_ci
        return high - low

    assert width(80) > width(1500)


def test_undetected_variants_are_listed():
    counts = synthetic_counts(400, skew=6.0, depth=8, dropout=0.05, seed=29)
    stats = measure_skew(counts)

    assert stats.n_undetected == len(stats.undetected_names)
    assert stats.n_detected + stats.n_undetected == counts.library_size
    for name in stats.undetected_names:
        assert counts.counts[name] == 0


def test_shrunk_abundance_is_a_normalized_distribution():
    counts = synthetic_counts(200, skew=4.0, depth=25, seed=31)
    stats = measure_skew(counts)

    assert stats.shrunk_abundance.shape == (200,)
    assert stats.shrunk_abundance.sum() == pytest.approx(1.0)
    assert (stats.shrunk_abundance >= 0).all()


def test_shrinkage_pulls_extremes_toward_the_fit():
    """Empirical Bayes must denoise, or the sim inherits the counting noise."""
    counts = synthetic_counts(1000, skew=4.0, depth=8, seed=37)
    stats = measure_skew(counts)

    observed = counts.as_array() / counts.assigned_reads
    detected = counts.as_array() > 0
    observed_spread = observed[detected].max() / observed[detected].min()
    shrunk_spread = (
        stats.shrunk_abundance[detected].max() / stats.shrunk_abundance[detected].min()
    )
    assert shrunk_spread < observed_spread


def test_effective_library_size_at_most_library_size():
    counts = synthetic_counts(500, skew=5.0, depth=30, seed=41)
    stats = measure_skew(counts)
    assert 0 < stats.effective_library_size <= 500


def test_gini_larger_for_more_skewed_library():
    even = measure_skew(synthetic_counts(500, 1.0, depth=40, seed=43))
    uneven = measure_skew(synthetic_counts(500, 8.0, depth=40, seed=43))
    assert uneven.gini > even.gini


def test_depth_flag_tracks_reads_per_variant():
    assert measure_skew(synthetic_counts(300, 4.0, depth=40, seed=2)).depth_sufficient
    assert not measure_skew(synthetic_counts(300, 4.0, depth=4, seed=2)).depth_sufficient


def test_moments_fit_is_available_and_close():
    counts = synthetic_counts(1000, skew=4.0, depth=25, seed=47)
    mle = measure_skew(counts, fit="mle")
    moments = measure_skew(counts, fit="moments")

    assert moments.fit_method == "moments"
    assert moments.q90_q10_corrected == pytest.approx(mle.q90_q10_corrected, rel=0.25)


def test_observed_skew_undefined_when_bottom_decile_is_empty():
    counts = VariantCounts(counts={f"v{i}": (10 if i > 50 else 0) for i in range(100)})
    stats = measure_skew(counts)
    assert stats.q90_q10_observed is None


def test_measure_skew_rejects_bad_input():
    with pytest.raises(ValueError, match="no variants"):
        measure_skew(VariantCounts(counts={}))
    with pytest.raises(ValueError, match="no reads"):
        measure_skew(VariantCounts(counts={"a": 0, "b": 0}))
    with pytest.raises(ValueError, match="fit must be"):
        measure_skew(synthetic_counts(50, 2.0, 20, seed=1), fit="bogus")


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def test_more_skew_demands_deeper_sorting():
    even = recommend_sampling(
        measure_skew(synthetic_counts(200, 1.5, depth=40, seed=3)),
        n_sims=20, target_coverage=0.90,
    )
    uneven = recommend_sampling(
        measure_skew(synthetic_counts(200, 8.0, depth=40, seed=3)),
        n_sims=20, target_coverage=0.90,
    )
    assert uneven.fold_sampling > even.fold_sampling


def test_recommendation_reports_wells_and_plates():
    stats = measure_skew(synthetic_counts(200, 4.0, depth=40, seed=5))
    rec = recommend_sampling(stats, n_sims=20)

    assert rec.n_wells == int(np.ceil(200 * rec.fold_sampling))
    assert rec.n_plates == int(np.ceil(rec.n_wells / 384))
    assert rec.basis == "empirical"


def test_target_above_dropout_ceiling_is_flagged():
    """Sorting cannot recover variants that were never synthesized."""
    stats = measure_skew(synthetic_counts(300, 3.0, depth=40, dropout=0.20, seed=9))
    rec = recommend_sampling(stats, target_coverage=0.95, n_sims=20)

    assert rec.coverage_ceiling < 0.95
    assert rec.target_reachable is False
    assert rec.expected_coverage <= rec.coverage_ceiling + 0.02


def test_reachable_target_is_not_flagged():
    stats = measure_skew(synthetic_counts(200, 3.0, depth=40, seed=15))
    rec = recommend_sampling(stats, target_coverage=0.85, n_sims=20)
    assert rec.target_reachable is True


def test_lognormal_basis_is_available():
    stats = measure_skew(synthetic_counts(200, 4.0, depth=40, seed=19))
    rec = recommend_sampling(stats, basis="lognormal", n_sims=20)
    assert rec.basis == "lognormal"
    assert rec.fold_sampling > 0


def test_recommend_sampling_rejects_unknown_basis():
    stats = measure_skew(synthetic_counts(100, 2.0, depth=30, seed=21))
    with pytest.raises(ValueError, match="basis must be"):
        recommend_sampling(stats, basis="bogus", n_sims=10)


# ---------------------------------------------------------------------------
# Simulation passthrough
# ---------------------------------------------------------------------------

def test_sortm_pool_matches_equivalent_generated_pool():
    """pool= must reproduce what generate_pool would have produced."""
    pool = generate_pool(200, 4.0, seed=1)
    with_pool = sortm(n_sims=40, fold_sampling=8, p_grow=0.67, seed=42, pool=pool)
    baseline = sortm(
        n_sims=40, lib_size=200, skew=4.0, fold_sampling=8, p_grow=0.67, seed=42
    )
    assert np.mean(with_pool) / 200 == pytest.approx(np.mean(baseline) / 200, abs=0.05)


def test_sortm_pool_accepts_unnormalized_counts():
    counts = np.array([10.0, 20.0, 30.0, 40.0])
    result = sortm(n_sims=5, fold_sampling=4, p_grow=1.0, seed=1, pool=counts)
    assert len(result) == 5
    assert (result <= 4).all()


def test_sortm_pool_overrides_lib_size():
    pool = generate_pool(50, 2.0, seed=1)
    result = sortm(n_sims=5, lib_size=9999, fold_sampling=4, seed=1, pool=pool)
    assert (result <= 50).all()


@pytest.mark.parametrize("bad,message", [
    (np.zeros(10), "sums to zero"),
    (np.array([1.0]), "at least 2"),
    (np.array([1.0, -1.0, 2.0]), "negative"),
    (np.ones((3, 3)), "1-D"),
])
def test_sortm_pool_validation(bad, message):
    with pytest.raises(ValueError, match=message):
        sortm(n_sims=2, fold_sampling=2, pool=bad)


def test_find_fold_sampling_does_not_return_the_search_bound():
    """Regression: the tolerance break used to return the untightened bound.

    Breaking out of the binary search on a midpoint that fell just *below*
    target left best_fold at the upper bound, roughly doubling the
    recommended well count.
    """
    fold, coverage = find_fold_sampling(
        target_coverage=0.90, lib_size=300, skew=3.0,
        n_sims=40, seed=42, p_grow=0.67,
    )
    assert fold < 20.0
    # Monte Carlo slack: the returned depth is re-evaluated after rounding up,
    # so a small shortfall at low n_sims is noise rather than a search failure.
    assert coverage >= 0.90 - _MC_SLACK


def test_find_fold_sampling_accepts_a_pool():
    pool = generate_pool(200, 4.0, seed=1)
    fold, coverage = find_fold_sampling(
        target_coverage=0.85, pool=pool, n_sims=30, seed=42, p_grow=0.67
    )
    assert fold > 0
    assert coverage >= 0.85 - _MC_SLACK


# ---------------------------------------------------------------------------
# Variant CSV parsing
# ---------------------------------------------------------------------------

def test_read_variant_sequences_strips_lowercase_flanks(tmp_path):
    path = write_variants_csv(
        tmp_path / "v.csv", {"a": "acgtATGCATGCgtca", "b": "TTTTGGGG"}
    )
    sequences = read_variant_sequences(path)
    assert sequences == {"a": "ATGCATGC", "b": "TTTTGGGG"}


def test_read_variant_sequences_tolerates_header_whitespace(tmp_path):
    path = tmp_path / "v.csv"
    path.write_text("Name , Sequence \nx,ACGT\n")
    assert read_variant_sequences(path) == {"x": "ACGT"}


def test_read_variant_sequences_requires_expected_columns(tmp_path):
    path = tmp_path / "v.csv"
    path.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="must have 'Name' and 'Sequence'"):
        read_variant_sequences(path)


def test_duplicate_sequences_are_grouped():
    groups = _find_duplicate_groups({"a": "ACGT", "b": "ACGT", "c": "TTTT"})
    assert groups == [["a", "b"]]


def test_count_fastq_reads(tmp_path):
    path = write_fastq(tmp_path / "r.fastq", [(f"r{i}", "ACGT") for i in range(7)])
    assert count_fastq_reads(path) == 7


# ---------------------------------------------------------------------------
# Alignment-backed counting
# ---------------------------------------------------------------------------

@pytest.fixture
def distinct_library(tmp_path):
    """Six unrelated 300 bp variants, plus reads with vector flanks."""
    rng = np.random.default_rng(2024)
    sequences = {f"var{i}": random_seq(rng, 300) for i in range(6)}
    csv_path = write_variants_csv(tmp_path / "variants.csv", sequences)
    flank5, flank3 = random_seq(rng, 50), random_seq(rng, 50)
    return csv_path, sequences, flank5, flank3


@requires_minimap2
def test_counting_recovers_exact_abundances(tmp_path, distinct_library):
    csv_path, sequences, flank5, flank3 = distinct_library
    truth = {"var0": 40, "var1": 25, "var2": 15, "var3": 10, "var4": 5, "var5": 0}

    reads = []
    for name, n in truth.items():
        for j in range(n):
            reads.append((f"{name}_{j}", flank5 + sequences[name] + flank3))
    fastq = write_fastq(tmp_path / "lib.fastq", reads)

    counts = count_variant_reads(fastq, csv_path, tmp_path / "work")

    assert counts.counts == truth
    assert counts.total_reads == sum(truth.values())
    assert counts.ambiguous == 0
    assert counts.unmapped == 0


@requires_minimap2
def test_counting_includes_zero_count_variants(tmp_path, distinct_library):
    """Undetected variants must survive into the statistics."""
    csv_path, sequences, flank5, flank3 = distinct_library
    fastq = write_fastq(
        tmp_path / "lib.fastq",
        [(f"r{j}", flank5 + sequences["var0"] + flank3) for j in range(10)],
    )
    counts = count_variant_reads(fastq, csv_path, tmp_path / "work")

    assert set(counts.counts) == set(sequences)
    assert counts.counts["var0"] == 10
    assert counts.counts["var3"] == 0


@requires_minimap2
def test_unrelated_reads_are_unmapped(tmp_path, distinct_library):
    csv_path, _, _, _ = distinct_library
    rng = np.random.default_rng(99)
    fastq = write_fastq(
        tmp_path / "lib.fastq", [(f"r{j}", random_seq(rng, 300)) for j in range(20)]
    )
    counts = count_variant_reads(fastq, csv_path, tmp_path / "work")

    assert counts.assigned_reads == 0
    assert counts.unmapped == 20


@requires_minimap2
def test_partial_reads_fail_the_coverage_filter(tmp_path, distinct_library):
    """A read covering a third of its variant should not count as that variant."""
    csv_path, sequences, _, _ = distinct_library
    fragment = sequences["var0"][:100]
    fastq = write_fastq(tmp_path / "lib.fastq", [(f"r{j}", fragment) for j in range(10)])

    counts = count_variant_reads(fastq, csv_path, tmp_path / "work", min_ref_cov=0.8)
    assert counts.counts["var0"] == 0
    assert counts.low_cov == 10

    relaxed = count_variant_reads(
        fastq, csv_path, tmp_path / "work2", min_ref_cov=0.2
    )
    assert relaxed.counts["var0"] == 10


@requires_minimap2
def test_near_identical_variants_produce_ambiguous_reads(tmp_path):
    """Reads that cannot pick a winner must not be forced onto one."""
    rng = np.random.default_rng(5)
    base = random_seq(rng, 300)
    twin = base[:150] + ("A" if base[150] != "A" else "C") + base[151:]
    csv_path = write_variants_csv(tmp_path / "v.csv", {"a": base, "b": twin})

    fastq = write_fastq(tmp_path / "lib.fastq", [(f"r{j}", base) for j in range(20)])
    counts = count_variant_reads(fastq, csv_path, tmp_path / "work", margin=0.05)

    assert counts.ambiguous == 20
    assert counts.assigned_reads == 0


# ---------------------------------------------------------------------------
# Resolvability
# ---------------------------------------------------------------------------

@requires_minimap2
def test_distinct_library_is_clean(tmp_path, distinct_library):
    csv_path, _, _, _ = distinct_library
    summary = check_resolvability(csv_path)

    assert summary.verdict == "clean"
    assert summary.is_usable
    assert summary.n_below_threshold == 0


@requires_minimap2
def test_single_substitution_library_is_smeared(tmp_path):
    """A DMS-style library cannot be counted read-by-read."""
    rng = np.random.default_rng(8)
    base = list(random_seq(rng, 300))
    sequences = {}
    for i in range(30):
        variant = list(base)
        variant[i * 7] = BASES[(BASES.index(variant[i * 7]) + 1) % 4]
        sequences[f"sub{i}"] = "".join(variant)
    csv_path = write_variants_csv(tmp_path / "v.csv", sequences)

    summary = check_resolvability(csv_path)
    assert summary.verdict == "smeared"
    assert not summary.is_usable
    assert summary.median_nn_distance < summary.warn_below


@requires_minimap2
def test_identical_sequences_are_reported(tmp_path):
    rng = np.random.default_rng(12)
    shared = random_seq(rng, 300)
    sequences = {"twin_a": shared, "twin_b": shared, "other": random_seq(rng, 300)}
    csv_path = write_variants_csv(tmp_path / "v.csv", sequences)

    summary = check_resolvability(csv_path)
    assert ["twin_a", "twin_b"] in summary.duplicate_groups
    assert summary.n_unique_sequences == 2
    assert summary.min_distance == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@pytest.fixture
def skew_project(tmp_path, distinct_library):
    """A planned project plus a library FASTQ, ready for `usortm skew`."""
    csv_path, sequences, flank5, flank3 = distinct_library
    project = tmp_path / "project"
    project.mkdir()
    shutil.copy(csv_path, project / "variants.csv")
    (project / "usortm_project.json").write_text(json.dumps({
        "status": "planned",
        "library_size": len(sequences),
        "skew": 4.0,
        "fold_sampling": 8.0,
        "total_wells": 48,
        "workflow_steps": {"plan": {"completed": True}},
    }))

    reads = []
    for i, (name, seq) in enumerate(sequences.items()):
        for j in range(40 - i * 5):
            reads.append((f"{name}_{j}", flank5 + seq + flank3))
    fastq = write_fastq(tmp_path / "library.fastq", reads)
    return project, fastq


@requires_minimap2
def test_cli_writes_outputs_and_updates_project(skew_project):
    project, fastq = skew_project
    result = runner.invoke(app, [
        "skew", str(fastq), "--project", str(project),
        "--n-sims", "20", "--no-html",
    ])

    assert result.exit_code == 0, result.output
    assert "Abundance Distribution" in result.output
    assert "Measured Library Skew" in result.output
    assert "Recommended Sorting Depth" in result.output

    counts_csv = project / "skew" / "variant_counts.csv"
    report_json = project / "skew" / "skew_report.json"
    assert counts_csv.exists()
    assert report_json.exists()

    with open(counts_csv) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6
    assert [int(r["rank"]) for r in rows] == sorted(int(r["rank"]) for r in rows)

    report = json.loads(report_json.read_text())
    assert report["skew"]["q90_q10_corrected"] > 0
    assert report["recommendation"]["fold_sampling"] > 0

    state = json.loads((project / "usortm_project.json").read_text())
    assert "measured_skew" in state
    assert state["measured_skew"]["recommended_fold_sampling"] > 0
    # The planning assumption is preserved rather than overwritten.
    assert state["skew"] == 4.0
    assert state["fold_sampling"] == 8.0


@requires_minimap2
def test_cli_no_update_plan_leaves_state_alone(skew_project):
    project, fastq = skew_project
    result = runner.invoke(app, [
        "skew", str(fastq), "--project", str(project),
        "--n-sims", "20", "--no-html", "--no-update-plan",
    ])

    assert result.exit_code == 0, result.output
    state = json.loads((project / "usortm_project.json").read_text())
    assert "measured_skew" not in state


@requires_minimap2
def test_cli_accepts_bare_variants_csv(tmp_path, skew_project):
    project, fastq = skew_project
    out = tmp_path / "out"
    result = runner.invoke(app, [
        "skew", str(fastq), "--variants", str(project / "variants.csv"),
        "--output", str(out), "--n-sims", "20", "--no-html",
    ])

    assert result.exit_code == 0, result.output
    assert (out / "variant_counts.csv").exists()


def test_cli_requires_variants_or_project(tmp_path):
    fastq = write_fastq(tmp_path / "r.fastq", [("r0", "ACGT")])
    result = runner.invoke(app, ["skew", str(fastq)])
    assert result.exit_code == 1
    assert "--project or --variants" in result.output


def test_cli_rejects_unknown_basis(tmp_path):
    fastq = write_fastq(tmp_path / "r.fastq", [("r0", "ACGT")])
    result = runner.invoke(app, ["skew", str(fastq), "--basis", "bogus"])
    assert result.exit_code == 1
    assert "basis" in result.output


@requires_minimap2
def test_cli_refuses_unresolvable_library_without_force(tmp_path):
    rng = np.random.default_rng(77)
    base = list(random_seq(rng, 300))
    sequences = {}
    for i in range(20):
        variant = list(base)
        variant[i * 11] = BASES[(BASES.index(variant[i * 11]) + 1) % 4]
        sequences[f"sub{i}"] = "".join(variant)
    csv_path = write_variants_csv(tmp_path / "v.csv", sequences)
    fastq = write_fastq(
        tmp_path / "r.fastq",
        [(f"r{j}", sequences["sub0"]) for j in range(10)],
    )

    result = runner.invoke(app, [
        "skew", str(fastq), "--variants", str(csv_path),
        "--output", str(tmp_path / "out"), "--no-html",
    ])
    assert result.exit_code == 1
    assert "not separable" in result.output


# ---------------------------------------------------------------------------
# Synthetic datasets with ground truth
# ---------------------------------------------------------------------------

def test_synthetic_library_writes_expected_files(tmp_path):
    lib = make_synthetic_library(tmp_path / "lib", library_size=40,
                                 n_reads=400, seed=1)
    assert lib.variants_csv.exists()
    assert lib.fastq.exists()
    assert lib.truth_json.exists()
    assert lib.library_size == 40
    assert count_fastq_reads(lib.fastq) == lib.n_reads
    assert sum(lib.true_counts.values()) + lib.n_junk == lib.n_reads

    truth = json.loads(lib.truth_json.read_text())
    assert truth["params"]["mode"] == "diverse"
    assert sum(truth["true_abundance"].values()) == pytest.approx(1.0)


def test_synthetic_csv_uses_lowercase_flanks(tmp_path):
    """Matches a real Twist order, and exercises the flank-stripping path."""
    lib = make_synthetic_library(tmp_path / "lib", library_size=10, seq_length=90,
                                 n_reads=100, flank_length=12, seed=2)
    with open(lib.variants_csv) as fh:
        row = next(csv.DictReader(fh))
    full = row["Sequence"]
    assert len(full) == 90 + 24
    assert full[:12].islower() and full[-12:].islower()
    assert read_variant_sequences(lib.variants_csv)[row["Name"]] == full[12:-12]


def test_synthetic_is_reproducible(tmp_path):
    a = make_synthetic_library(tmp_path / "a", library_size=30, n_reads=300, seed=5)
    b = make_synthetic_library(tmp_path / "b", library_size=30, n_reads=300, seed=5)
    assert a.true_counts == b.true_counts
    assert a.fastq.read_text() == b.fastq.read_text()

    c = make_synthetic_library(tmp_path / "c", library_size=30, n_reads=300, seed=6)
    assert c.true_counts != a.true_counts


def test_synthetic_dropouts_get_zero_abundance(tmp_path):
    lib = make_synthetic_library(tmp_path / "lib", library_size=200, dropout=0.15,
                                 n_reads=2000, seed=11)
    assert lib.n_absent > 0
    for name in lib.absent:
        assert lib.true_abundance[name] == 0.0
        assert lib.true_counts[name] == 0


def test_synthetic_reads_carry_errors(tmp_path):
    """Reads must not be verbatim copies, or the aligner is never tested."""
    lib = make_synthetic_library(tmp_path / "lib", library_size=5, seq_length=300,
                                 n_reads=200, error_rate=0.05, junk_fraction=0.0,
                                 truncation_rate=0.0, seed=13)
    sequences = set(read_variant_sequences(lib.variants_csv).values())
    reads = [l.strip() for i, l in enumerate(open(lib.fastq)) if i % 4 == 1]
    exact = sum(1 for r in reads if any(s in r for s in sequences))
    assert exact < len(reads) * 0.5


def test_codon_scan_mode_builds_near_identical_variants(tmp_path):
    lib = make_synthetic_library(tmp_path / "lib", library_size=100, seq_length=300,
                                 mode="codon_scan", n_reads=500, seed=17)
    seqs = read_variant_sequences(lib.variants_csv)
    assert "WT" in seqs
    wt = seqs["WT"]
    others = [s for n, s in seqs.items() if n != "WT"]
    # Each variant differs from WT within a single codon.
    for seq in others[:20]:
        diffs = [i for i in range(len(wt)) if wt[i] != seq[i]]
        assert diffs
        assert len({d // 3 for d in diffs}) == 1


def test_codon_scan_rejects_infeasible_requests(tmp_path):
    with pytest.raises(ValueError, match="divisible by 3"):
        make_synthetic_library(tmp_path / "a", seq_length=100,
                               mode="codon_scan", n_reads=10)
    with pytest.raises(ValueError, match="only has"):
        make_synthetic_library(tmp_path / "b", library_size=4000, seq_length=60,
                               mode="codon_scan", n_reads=10)


def test_synthetic_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="mode must be"):
        make_synthetic_library(tmp_path / "lib", mode="bogus", n_reads=10)


@requires_minimap2
def test_end_to_end_recovers_known_skew(tmp_path):
    """The headline check: FASTQ in, true skew out.

    Asserted against the pool's *realized* skew, not the requested value —
    a finite draw differs from the distribution it came from.
    """
    lib = make_synthetic_library(tmp_path / "lib", library_size=400, skew=4.0,
                                 n_reads=12000, seed=23)
    profile = profile_library(lib.fastq, lib.variants_csv, tmp_path / "work",
                              n_sims=25, threads=2)

    assert profile.resolvability.verdict == "clean"
    assert profile.stats.q90_q10_corrected == pytest.approx(
        lib.realized_skew, rel=0.25
    )
    # The raw statistic is worse, and biased the same way every time.
    assert profile.stats.q90_q10_observed > profile.stats.q90_q10_corrected
    assert profile.recommendation.fold_sampling > 0


@requires_minimap2
def test_end_to_end_recovers_known_dropout(tmp_path):
    lib = make_synthetic_library(tmp_path / "lib", library_size=400, skew=3.0,
                                 dropout=0.10, n_reads=12000, seed=29)
    profile = profile_library(lib.fastq, lib.variants_csv, tmp_path / "work",
                              n_sims=25, threads=2)

    true_dropout = lib.n_absent / lib.library_size
    assert profile.stats.dropout_fraction == pytest.approx(true_dropout, abs=0.05)
    # Every truly absent variant must come back with no reads.
    for name in lib.absent:
        assert profile.counts.counts[name] == 0


@requires_minimap2
def test_end_to_end_counting_is_accurate(tmp_path):
    """Per-variant counts should match the reads actually written."""
    lib = make_synthetic_library(tmp_path / "lib", library_size=200, skew=4.0,
                                 n_reads=6000, error_rate=0.03, seed=31)
    counts = count_variant_reads(lib.fastq, lib.variants_csv, tmp_path / "work",
                                 threads=2)
    truth = np.array([lib.true_counts[n] for n in counts.names], dtype=float)
    got = counts.as_array()
    accuracy = 1 - np.abs(truth - got).sum() / truth.sum()
    assert accuracy > 0.90
    assert np.corrcoef(truth, got)[0, 1] > 0.98


@requires_minimap2
def test_end_to_end_accounts_for_junk_and_truncated_reads(tmp_path):
    lib = make_synthetic_library(tmp_path / "lib", library_size=100, n_reads=3000,
                                 junk_fraction=0.05, truncation_rate=0.10, seed=37)
    counts = count_variant_reads(lib.fastq, lib.variants_csv, tmp_path / "work",
                                 threads=2)
    assert counts.unmapped > 0, "unrelated reads should not map"
    assert counts.low_cov > 0, "truncated reads should fail the coverage filter"
    assert counts.total_reads == lib.n_reads


@requires_minimap2
def test_end_to_end_codon_scan_is_refused(tmp_path):
    """The amber-scan failure mode, reproducible without private data."""
    lib = make_synthetic_library(tmp_path / "lib", library_size=120, seq_length=300,
                                 mode="codon_scan", n_reads=3000, seed=41)
    summary = check_resolvability(lib.variants_csv)
    assert summary.verdict == "smeared"
    assert not summary.is_usable

    result = runner.invoke(app, [
        "skew", str(lib.fastq), "--variants", str(lib.variants_csv),
        "--output", str(tmp_path / "out"), "--no-html",
    ])
    assert result.exit_code == 1
    assert "not separable" in result.output


def test_high_skew_is_flagged_as_a_lower_bound():
    """Above ~10x the fit reads low, so it must be labelled."""
    modest = measure_skew(synthetic_counts(500, 4.0, depth=30, seed=43))
    assert not modest.beyond_validated_range
    assert modest.to_dict()["beyond_validated_range"] is False

    extreme = measure_skew(synthetic_counts(500, 25.0, depth=30, seed=43))
    assert extreme.q90_q10_corrected > 10
    assert extreme.beyond_validated_range


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_predicted_distribution_conserves_probability():
    """Expected variant counts must sum to the library size."""
    counts = synthetic_counts(500, 4.0, depth=30, dropout=0.06, seed=71)
    stats = measure_skew(counts)
    _, expected = predicted_count_distribution(stats, counts.library_size, 400)
    assert expected.sum() == pytest.approx(counts.library_size, rel=1e-6)


def test_predicted_distribution_is_broader_than_underlying():
    """The observable spread exceeds the true spread by the counting noise.

    That gap is the entire premise of the correction, so it must widen as
    depth falls.
    """
    def widths(depth):
        counts = synthetic_counts(1000, 4.0, depth=depth, seed=73)
        stats = measure_skew(counts)
        edges, _, predicted, underlying = log10_histogram(counts, stats)
        centers = 0.5 * (edges[:-1] + edges[1:])

        def sd(weights):
            w = weights + 1e-12
            mean = np.average(centers, weights=w)
            return np.sqrt(np.average((centers - mean) ** 2, weights=w))

        return sd(predicted), sd(underlying)

    shallow_pred, shallow_true = widths(8)
    deep_pred, deep_true = widths(60)

    assert shallow_pred > shallow_true
    assert deep_pred > deep_true
    # Counting noise contributes more of the observed width at low depth.
    assert (shallow_pred - shallow_true) > (deep_pred - deep_true)


def test_predicted_histogram_tracks_observed():
    """Goodness of fit: the noise-broadened curve should follow the bars."""
    counts = synthetic_counts(1000, 4.0, depth=25, seed=79)
    stats = measure_skew(counts)
    _, observed, predicted, _ = log10_histogram(counts, stats)

    assert observed.sum() == stats.n_detected
    mask = predicted > 5
    assert mask.sum() >= 3
    chi2_per_bin = (((observed[mask] - predicted[mask]) ** 2) / predicted[mask]).sum()
    chi2_per_bin /= mask.sum()
    assert chi2_per_bin < 4.0


def test_uniform_library_has_narrow_log_width():
    """The histogram width is the uniformity readout the user reads first."""
    even = measure_skew(synthetic_counts(500, 1.0, depth=30, seed=83))
    uneven = measure_skew(synthetic_counts(500, 8.0, depth=30, seed=83))

    even_sd = even.sigma_log / np.log(10)
    uneven_sd = uneven.sigma_log / np.log(10)
    assert even_sd < 0.1
    assert uneven_sd > 0.25
    assert even.to_dict()["sigma_log10"] == pytest.approx(even_sd, abs=1e-3)


def test_mu_log_is_consistent_with_the_fit():
    counts = synthetic_counts(400, 3.0, depth=40, seed=89)
    stats = measure_skew(counts)
    # exp(mu + sigma^2/2) is the mean reads per present variant
    present = counts.library_size * (1 - stats.dropout_fraction)
    implied = np.exp(stats.mu_log + 0.5 * stats.sigma_log**2)
    assert implied == pytest.approx(counts.assigned_reads / present, rel=1e-6)


def test_log10_histogram_needs_two_detected_variants():
    counts = VariantCounts(counts={"a": 5, "b": 0, "c": 0})
    stats = measure_skew(counts)
    with pytest.raises(ValueError, match="at least 2 detected"):
        log10_histogram(counts, stats)


@pytest.mark.parametrize("lo,hi,expected", [
    (0.0, 0.30, "1"),            # log10(2) = 0.301, so only k = 1 fits
    (0.0, 0.31, "1–2"),          # ...and just barely admits k = 2
    (0.95, 1.31, "9–20"),        # 10^0.95 = 8.91, so 9 is the first integer
    (0.0, 0.0, "—"),             # covers no integer at all
])
def test_integer_bin_labels(lo, hi, expected):
    from usortm.cli.skew_cmd import _integer_bin_label
    assert _integer_bin_label(lo, hi) == expected


def test_bar_renders_proportionally():
    from usortm.cli.skew_cmd import _bar
    assert _bar(0, 10, width=8) == ""
    assert _bar(10, 10, width=8) == "█" * 8
    assert len(_bar(5, 10, width=8)) <= 8
    assert len(_bar(10, 10, width=8)) == 8


def test_abundance_histogram_figure_builds():
    from usortm.qc.viz import bokeh_available, make_abundance_histogram_figure
    if not bokeh_available():
        pytest.skip("bokeh not installed")
    counts = synthetic_counts(300, 4.0, depth=30, dropout=0.05, seed=97)
    fig = make_abundance_histogram_figure(counts, measure_skew(counts))
    assert fig is not None
    assert fig.yaxis[0].axis_label == "Variants"


def test_confidence_interval_survives_json_as_null_not_nan():
    """JSON has no NaN literal; a failed fit must serialize as null."""
    assert ci_to_json((3.0, 5.0)) == [3.0, 5.0]
    assert ci_to_json((float("nan"), float("nan"))) is None
    assert json.loads(json.dumps({"ci": ci_to_json((np.nan, 2.0))})) == {"ci": None}


def test_profile_to_dict_is_json_serializable():
    counts = synthetic_counts(150, 4.0, depth=30, seed=61)
    stats = measure_skew(counts)
    rec = recommend_sampling(stats, n_sims=20)
    profile = LibraryProfile(counts=counts, stats=stats, recommendation=rec)

    payload = json.dumps(profile.to_dict())
    restored = json.loads(payload)
    assert restored["library_size"] == 150
    assert restored["skew"]["q90_q10_corrected"] > 0
    assert restored["resolvability"] is None


def test_skew_from_well_counts():
    """Skew estimated from wells per variant, at the depth a sort gives.

    The report reads skew from how many wells carried each designed variant
    rather than from read counts, and a sort gives far fewer wells per variant
    than a sequenced pool gives reads. This pins the behaviour at that depth,
    and the numbers it asserts are the ones quoted in
    :func:`usortm.report.summary.estimate_skew`.
    """
    from usortm.report.summary import estimate_skew

    lib_size, depth, n_seeds = 376, 2.86, 12
    z2 = 2 * 1.2815515655446004
    designed = {f"v{i}" for i in range(lib_size)}

    for true_skew, expected_median in ((2.0, 1.9), (4.0, 3.9)):
        sigma = np.log(true_skew) / z2
        estimates, covered = [], 0
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 + seed)
            abundance = rng.lognormal(0.0, sigma, size=lib_size)
            abundance = abundance / abundance.mean() * depth
            wells = rng.poisson(abundance)
            # One record per well, as the demux writes them.
            well_data = [
                {"variant": f"v{i}", "reads": 100}
                for i, n in enumerate(wells) for _ in range(int(n))
            ]
            est = estimate_skew(well_data, designed)
            assert est is not None
            estimates.append(est["skew"])
            if est["ci"] and est["ci"][0] <= true_skew <= est["ci"][1]:
                covered += 1

        median = float(np.median(estimates))
        assert median == pytest.approx(expected_median, abs=0.15), (
            f"true {true_skew}: median {median:.2f}")
        assert covered >= n_seeds - 1, f"true {true_skew}: covered {covered}"


def test_skew_from_well_counts_needs_wells():
    """No designed variant seen means no estimate rather than a fabricated one."""
    from usortm.report.summary import estimate_skew

    assert estimate_skew([], {"v0", "v1"}) is None
    assert estimate_skew([{"variant": "v0", "reads": 1}], {"v0"}) is None
    assert estimate_skew([{"variant": "v0", "reads": 100}], set()) is None


def test_skew_is_q90_q10_everywhere():
    """Skew is Q90/Q10, in the estimator and in the pool it is handed to.

    The report shows the figure beside a 95% confidence interval, which reads
    as a 95/5 ratio if the ratio's own quantiles are not stated. They differ:
    at the sigma this run fitted, Q90/Q10 is 1.68 and Q95/Q5 is 1.95. A
    mismatch between the two sides would also mis-parameterise the recovery
    curve, which takes the estimate directly.
    """
    from scipy.stats import norm

    from usortm.qc.skew import _Z_SPREAD, sigma_to_skew, skew_to_sigma

    q90_q10 = norm.ppf(0.9) - norm.ppf(0.1)
    q95_q5 = norm.ppf(0.95) - norm.ppf(0.05)
    assert _Z_SPREAD == pytest.approx(q90_q10)
    assert _Z_SPREAD != pytest.approx(q95_q5)

    # The two conversions are inverses, and land on the decile ratio.
    sigma = 0.203
    assert sigma_to_skew(sigma) == pytest.approx(np.exp(sigma * q90_q10))
    assert sigma_to_skew(sigma) == pytest.approx(1.683, abs=0.002)
    assert skew_to_sigma(sigma_to_skew(sigma)) == pytest.approx(sigma)

    # generate_pool takes the same ratio, so an estimate can be handed to it
    # without conversion.  Drawn large, its own deciles reproduce the skew.
    pool = generate_pool(lib_size=200_000, skew=4.0, seed=7)
    lo, hi = np.percentile(pool, [10, 90])
    assert hi / lo == pytest.approx(4.0, rel=0.05)
