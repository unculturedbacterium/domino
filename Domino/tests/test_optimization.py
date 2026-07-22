"""Tests for bounded-memory and multivariate execution paths."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

import domino
from domino.assoc import scan_chromosome_gls_eigen
from domino.cli import build_parser
from domino.grm import iter_loco_grms
from domino.resources import ResourceConfig, plan_execution
from domino.testutils import simulate, write_plink
from domino.vc import estimate_h2_many, score_variance_components


@pytest.fixture(scope="module")
def small_sim(tmp_path_factory):
    prefix = str(tmp_path_factory.mktemp("optimized") / "sim")
    metadata = simulate(
        prefix,
        n_fam=20,
        sibs=3,
        n_chrom=2,
        snps_per_chrom=100,
        n_causal_add=5,
        n_causal_dom=5,
        seed=19,
    )
    return prefix, metadata


def test_memory_planner_downscales_and_rejects_dense():
    config = ResourceConfig(memory_budget_mb=512, safety_fraction=0.80)
    plan = plan_execution(
        n_samples=2000,
        n_traits=200,
        n_covariates=5,
        rank=100,
        requested_block_size=8192,
        decomposition="randomized",
        resource_config=config,
    )
    assert 1 <= plan.variant_block_size < 8192
    assert 1 <= plan.trait_tile_size <= 200
    assert plan.estimated_peak_mb <= 512
    with pytest.raises(MemoryError, match="LOCO decomposition"):
        plan_execution(
            n_samples=5000,
            n_traits=1,
            n_covariates=1,
            rank=5000,
            requested_block_size=128,
            decomposition="exact",
            resource_config=config,
        )


def test_cli_defaults_preserve_exact_profile_reml_contract():
    arguments = build_parser().parse_args(
        [
            "--bfile", "study", "--pheno", "phenotypes.tsv",
            "--traits", "trait", "--out", "results/scan",
        ]
    )
    assert arguments.trait_mode == "independent"
    assert arguments.variance_estimator == "auto"
    assert arguments.decomposition == "exact"
    assert arguments.compute_dtype == "float64"
    assert arguments.operator_error_tolerance == 0.02


def test_lazy_grm_cache_is_reused(small_sim, tmp_path):
    prefix, _ = small_sim
    reader = domino.PlinkReader(prefix)
    first = dict(
        iter_loco_grms(
            reader,
            chroms=["1"],
            decomposition="exact",
            grm_cache=tmp_path,
            verbose=False,
        )
    )["1"]
    second = dict(
        iter_loco_grms(
            reader,
            chroms=["1"],
            decomposition="exact",
            grm_cache=tmp_path,
            verbose=False,
        )
    )["1"]
    assert not first["cache_hit"]
    assert second["cache_hit"]
    assert isinstance(second["U"], np.memmap)
    np.testing.assert_array_equal(first["s"], second["s"])


def test_randomized_operator_reports_error(small_sim):
    prefix, _ = small_sim
    reader = domino.PlinkReader(prefix)
    result = dict(
        iter_loco_grms(
            reader,
            chroms=["1"],
            decomposition="randomized",
            n_components=40,
            randomized_power_iterations=1,
            operator_error_tolerance=None,
            verbose=False,
        )
    )["1"]
    diagnostic = result["diagnostics"]
    assert len(result["s"]) == 40
    assert result["residual_eigenvalue"] >= 0
    assert 0 <= diagnostic["operator_relative_error"] < 1
    assert 0 < diagnostic["retained_trace_fraction"] <= 1.01
    assert diagnostic["attempted_ranks"] == [40]


def test_randomized_rank_doubles_until_probe_tolerance(small_sim):
    prefix, _ = small_sim
    reader = domino.PlinkReader(prefix)
    coarse = dict(
        iter_loco_grms(
            reader,
            chroms=["1"],
            decomposition="randomized",
            n_components=5,
            operator_error_tolerance=None,
            verbose=False,
        )
    )["1"]
    threshold = coarse["diagnostics"]["operator_relative_error"] * 0.999
    adapted = dict(
        iter_loco_grms(
            reader,
            chroms=["1"],
            decomposition="randomized",
            n_components=5,
            randomized_max_components=reader.n_samples,
            operator_error_tolerance=threshold,
            verbose=False,
        )
    )["1"]
    diagnostic = adapted["diagnostics"]
    assert diagnostic["attempted_ranks"][0] == 5
    assert len(diagnostic["attempted_ranks"]) > 1
    assert diagnostic["operator_relative_error"] <= threshold


def test_score_covariance_is_psd_and_covariate_adjusted(small_sim):
    prefix, metadata = small_sim
    reader = domino.PlinkReader(prefix)
    eigensystem = dict(
        iter_loco_grms(reader, chroms=["1"], decomposition="exact", verbose=False)
    )["1"]
    Y = metadata["pheno"][["y_add", "y_dom"]].to_numpy()
    C = np.column_stack([np.ones(len(Y)), np.linspace(-1.0, 1.0, len(Y))])
    score = score_variance_components(
        Y,
        eigensystem["U"],
        eigensystem["s"],
        covar=C,
        full_covariance=True,
    )
    assert score["fixed_effect_rank"] == 2
    assert np.linalg.eigvalsh(score["genetic_covariance"]).min() >= -1e-10
    assert np.linalg.eigvalsh(score["residual_covariance"]).min() > 0
    assert all(0 <= fit["h2"] < 1 for fit in score["fits"])


def test_trait_tiles_are_numerically_identical(small_sim):
    prefix, metadata = small_sim
    reader = domino.PlinkReader(prefix)
    eigensystem = dict(
        iter_loco_grms(reader, chroms=["1"], decomposition="exact", verbose=False)
    )["1"]
    traits = ["y_add", "y_dom"]
    Y = metadata["pheno"][traits].to_numpy()
    C = np.ones((len(Y), 1))
    fits = estimate_h2_many(Y, eigensystem["U"], eigensystem["s"], covar=C)
    arguments = (
        reader,
        "1",
        Y,
        traits,
        np.arange(len(Y)),
        eigensystem["U"],
        eigensystem["s"],
        fits,
        C,
    )
    tiled = scan_chromosome_gls_eigen(*arguments, block_size=31, trait_tile_size=1)
    combined = scan_chromosome_gls_eigen(*arguments, block_size=31, trait_tile_size=2)
    keys = ["snp", "trait"]
    tiled = tiled.sort_values(keys).reset_index(drop=True)
    combined = combined.sort_values(keys).reset_index(drop=True)
    for column in [
        "beta_additive_raw",
        "neglog_p_additive",
        "beta_dom_joint_raw",
        "neglog_p_dom_joint",
    ]:
        np.testing.assert_allclose(
            tiled[column], combined[column], rtol=0, atol=1e-12, equal_nan=True
        )


def test_streamed_parquet_has_multiple_atomic_row_groups(small_sim, tmp_path):
    prefix, metadata = small_sim
    output = tmp_path / "streamed" / "scan"
    result = domino.run_gwas(
        prefix,
        metadata["pheno"][["y_add", "y_dom"]],
        chroms=["1"],
        block_size=17,
        out=str(output),
        return_results=False,
        verbose=False,
    )
    parquet_path = Path(f"{output}.chr1.parquet")
    parquet = pq.ParquetFile(parquet_path)
    assert result.empty
    assert parquet.metadata.num_rows == 200
    assert parquet.metadata.num_row_groups > 1
    assert Path(f"{output}.execution.json").exists()
    execution = json.loads(Path(f"{output}.execution.json").read_text(encoding="utf-8"))
    chromosome = execution["groups"][0]["chromosomes"]["1"]
    assert set(chromosome["timing_s"]) == {
        "decomposition_or_cache_load", "variance_components", "association_and_output"
    }
    assert execution["observed_memory"]["peak_rss_mb"] > 0
    assert not list(parquet_path.parent.glob("*.partial"))
    assert not list(parquet_path.parent.glob(".*.partial"))


def test_streamed_and_in_memory_row_order_match(small_sim, tmp_path):
    prefix, metadata = small_sim
    phenotype = metadata["pheno"][["y_add", "y_dom"]]
    retained = domino.run_gwas(
        prefix,
        phenotype,
        chroms=["1"],
        block_size=17,
        verbose=False,
    )
    output = tmp_path / "ordered" / "scan"
    domino.run_gwas(
        prefix,
        phenotype,
        chroms=["1"],
        block_size=17,
        out=str(output),
        return_results=False,
        verbose=False,
    )
    streamed = pd.read_parquet(f"{output}.chr1.parquet")
    keys = ["variant_index", "snp", "trait"]
    pd.testing.assert_frame_equal(
        retained[keys].reset_index(drop=True),
        streamed[keys].reset_index(drop=True),
        check_dtype=False,
    )
    np.testing.assert_allclose(
        retained["neglog_p_dom_joint"],
        streamed["neglog_p_dom_joint"],
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def test_multivariate_score_outputs_original_and_joint_rows(small_sim):
    prefix, metadata = small_sim
    result = domino.run_gwas(
        prefix,
        metadata["pheno"][["y_add", "y_dom"]],
        chroms=["1"],
        trait_mode="multivariate",
        variance_estimator="score",
        decomposition="exact",
        block_size=29,
        verbose=False,
    )
    assert set(result["trait"]) == {"y_add", "y_dom", "__joint__"}
    assert len(result) == 300
    joint = result[result["trait"] == "__joint__"]
    assert joint["neglog_p_add_dom_multivariate"].notna().all()
    assert (joint["df_additive_multivariate"] == 2).all()
    assert (joint["df_add_dom_multivariate"] == 4).all()
    original = result[result["trait"] != "__joint__"]
    assert original["beta_add_joint_raw"].notna().any()


def test_one_trait_multivariate_reduces_to_independent_score(small_sim):
    prefix, metadata = small_sim
    phenotype = metadata["pheno"][["y_add"]]
    independent = domino.run_gwas(
        prefix,
        phenotype,
        chroms=["1"],
        variance_estimator="score",
        decomposition="exact",
        verbose=False,
    )
    multivariate = domino.run_gwas(
        prefix,
        phenotype,
        chroms=["1"],
        trait_mode="multivariate",
        variance_estimator="auto",
        decomposition="exact",
        verbose=False,
    )
    recovered = multivariate[multivariate["trait"] == "y_add"].reset_index(drop=True)
    for column in ["beta_additive_raw", "beta_add_joint_raw", "beta_dom_joint_raw"]:
        np.testing.assert_allclose(
            independent[column], recovered[column], rtol=1e-9, atol=1e-10, equal_nan=True
        )


def test_nested_f_and_conditional_dominance_are_identical(small_sim):
    prefix, metadata = small_sim
    result = domino.run_gwas(
        prefix,
        metadata["pheno"][["y_add"]],
        chroms=["1"],
        verbose=False,
    )
    np.testing.assert_allclose(
        result["neglog_p_avsad"],
        result["neglog_p_dom_joint"],
        rtol=0,
        atol=1e-12,
        equal_nan=True,
    )


def test_multivariate_mode_accepts_duplicate_snp_ids(tmp_path):
    rng = np.random.default_rng(912)
    n, m = 36, 24
    genotype = rng.binomial(2, rng.uniform(0.2, 0.45, m), size=(n, m)).astype(float)
    prefix = str(tmp_path / "duplicates")
    ids = [f"sample_{index}" for index in range(n)]
    write_plink(
        prefix,
        genotype,
        ids,
        np.repeat(["1", "2"], m // 2),
        np.tile(np.arange(m // 2), 2),
    )
    bim = pd.read_csv(f"{prefix}.bim", sep=r"\s+", header=None)
    bim.iloc[0, 1] = "duplicate_id"
    bim.iloc[1, 1] = "duplicate_id"
    bim.to_csv(f"{prefix}.bim", sep="\t", header=False, index=False)
    phenotype = pd.DataFrame(
        {"trait_1": rng.normal(size=n), "trait_2": rng.normal(size=n)},
        index=ids,
    )
    result = domino.run_gwas(
        prefix,
        phenotype,
        chroms=["1"],
        trait_mode="multivariate",
        variance_estimator="score",
        decomposition="exact",
        verbose=False,
    )
    assert result["variant_index"].nunique() == m // 2
    duplicated = result[result["snp"] == "duplicate_id"]
    assert duplicated["variant_index"].nunique() == 2
    assert len(duplicated) == 2 * 3
