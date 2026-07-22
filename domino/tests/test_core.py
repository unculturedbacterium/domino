"""Automated tests for the domino core."""
import numpy as np
import pandas as pd
import pytest

import domino
from domino import (
    PlinkReader, compute_loco_grms, estimate_h2, covariance_matrix,
    classify_inheritance,
)
from domino.testutils import simulate, write_plink

MED = 0.45493642


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    prefix = str(tmp_path_factory.mktemp("d") / "sim")
    meta = simulate(prefix, n_fam=200, sibs=4, n_chrom=3, snps_per_chrom=1000, seed=1)
    res = domino.run_gwas(prefix, meta["pheno"], model="add-dom", verbose=False)
    return prefix, meta, res


def test_io_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    g = rng.integers(0, 3, size=(50, 20)).astype(float)
    g[0, 0] = np.nan
    prefix = str(tmp_path / "rt")
    write_plink(prefix, g, [f"i{i}" for i in range(50)],
                ["1"] * 20, np.arange(20))
    back = PlinkReader(prefix).read_block(0, 20)
    assert np.allclose(back, g, equal_nan=True)


def test_selected_sample_decode_matches_full(sim):
    prefix, _, _ = sim
    reader = PlinkReader(prefix)
    samples = np.array([0, 3, 17, 201, 799])
    full = reader.read_block(11, 29)[samples]
    selected = reader.read_block_samples(11, 29, samples)
    np.testing.assert_equal(selected, full)


def test_single_snp_matches_exact_gls(sim):
    prefix, meta, res = sim
    reader = PlinkReader(prefix)
    grms = compute_loco_grms(reader, chroms=["1"], verbose=False)
    U, s = grms["1"]["U"], grms["1"]["s"]
    y = meta["pheno"]["y_mix"].values
    C = np.ones((len(y), 1))
    hh = estimate_h2(y, U, s, covar=C)
    L = covariance_matrix(U, s, hh["h2"], hh["yvar"], power=-0.5)
    ystar = L @ y
    idx1 = reader.chrom_variant_index("1")
    G1 = reader.read_block(int(idx1.min()), len(idx1))
    sub = res[(res["chrom"] == "1") & (res["trait"] == "y_mix")]
    from scipy.stats import t as tdist
    for j in [0, 17, 123, 500, 999]:
        g = G1[:, j].astype(float)
        X = L @ np.column_stack([np.ones_like(g), g, (g == 1).astype(float)])
        b, *_ = np.linalg.lstsq(X, ystar, rcond=None)
        r = ystar - X @ b
        dof = len(ystar) - 3
        se = np.sqrt(np.diag((r @ r) / dof * np.linalg.inv(X.T @ X)))
        p = 2 * tdist.sf(abs(b[2] / se[2]), df=dof)
        row = sub[sub["snp"] == f"rs{idx1[j]}"].iloc[0]
        assert abs(b[2] - row["beta_dom_joint_raw"]) < 1e-9
        assert abs(-np.log10(p) - row["neglog_p_dom_joint"]) < 1e-8


def test_per_trait_missingness_and_covariates(sim):
    prefix, meta, _ = sim
    pheno = meta["pheno"][["y_mix", "y_dom"]].copy()
    pheno.iloc[:37, 0] = np.nan
    covar = pd.DataFrame(
        {"sex": np.where(np.arange(len(pheno)) % 2, "M", "F"),
         "batch": np.arange(len(pheno)) % 3},
        index=pheno.index,
    )
    result = domino.run_gwas(prefix, pheno, covar=covar, chroms=["1"], verbose=False)
    sizes = result.groupby("trait")["n_samples"].first()
    assert sizes["y_mix"] == len(pheno) - 37
    assert sizes["y_dom"] == len(pheno)


def test_genotype_cell_filter_and_signed_classification(sim):
    prefix, meta, _ = sim
    result = domino.run_gwas(
        prefix, meta["pheno"][["y_mix"]], chroms=["1"],
        min_genotype_count=200, verbose=False,
    )
    failed = result[~result["genotype_filter_pass"]]
    assert len(failed) > 0
    assert failed["neglog_p_dom_joint"].isna().all()
    signed, magnitude, coarse, mode = classify_inheritance(
        np.array([1.0, 1.0, -1.0, 0.01]),
        np.array([1.0, -1.0, 1.0, 1.0]),
        np.array([0.1, 0.1, 0.1, 0.1]),
    )
    assert list(mode) == ["complete_dominant_A1", "complete_dominant_A0",
                          "complete_dominant_A0", "additive_near_zero"]
    assert coarse[-1] == "UNSTABLE"


def test_binary_trait_is_rejected(sim):
    prefix, meta, _ = sim
    binary = pd.DataFrame({"case": (meta["pheno"]["y_mix"] > 0).astype(int)}, index=meta["pheno"].index)
    with pytest.raises(ValueError, match="binary phenotypes are not supported"):
        domino.run_gwas(prefix, binary, chroms=["1"], verbose=False)


def test_null_calibration(sim):
    _, meta, res = sim
    from scipy.stats import chi2
    null = np.setdiff1d(np.arange(meta["m"]),
                        np.r_[meta["causal_add"], meta["causal_dom"], meta["big_dom_snp"]])
    names = {f"rs{i}" for i in null}
    d = res[(res["snp"].isin(names)) & (res["trait"] == "y_null")]
    for col in ["neglog_p_additive", "neglog_p_dom_joint"]:
        lam = np.nanmedian(chi2.isf(10 ** (-d[col].values), 1)) / MED
        assert 0.85 < lam < 1.20, f"{col} lambda={lam}"


def test_overdominant_snp_detected_and_classified(sim):
    _, meta, res = sim
    d = res[res["trait"] == "y_dom"].sort_values("neglog_p_dom_joint", ascending=False)
    top = d.iloc[0]
    assert top["snp"] == meta["big_dom_name"]
    assert top["dominance_class"] == "OD"


def test_additive_model_runs(sim):
    prefix, meta, _ = sim
    res = domino.run_gwas(prefix, meta["pheno"], model="additive", verbose=False)
    assert "neglog_p_additive" in res.columns
    assert len(res) == meta["m"] * 4
