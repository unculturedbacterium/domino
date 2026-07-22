"""One-call, resource-bounded dominance-aware GWAS driver."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import chi2, t as _t
from threadpoolctl import threadpool_limits

from ._utils import neglog10p
from .assoc import iter_chromosome_gls_eigen
from .backend import resolve_backend_name
from .classify import classify_inheritance
from .grm import iter_loco_grms
from .io import PlinkReader
from .output import AtomicParquetWriter
from .resources import PeakRSSMonitor, ResourceConfig, plan_execution
from .vc import (
    estimate_h2_many,
    multivariate_score_transform,
    score_variance_components,
)


def _load_pheno(pheno):
    if isinstance(pheno, pd.DataFrame):
        return pheno.copy()
    frame = pd.read_csv(pheno, sep=r"\s+")
    columns = {column.lower(): column for column in frame.columns}
    id_column = columns.get("iid") or columns.get("id") or frame.columns[1]
    return frame.set_index(id_column)


def _prepare_covariates(covar, ids):
    if covar is None:
        return pd.DataFrame(index=ids)
    if isinstance(covar, pd.DataFrame):
        covar = covar.copy()
        covar.index = covar.index.astype(str)
        frame = covar.loc[ids].copy()
        categorical = frame.select_dtypes(
            include=["object", "category", "string", "bool"]
        ).columns
        numeric = frame.columns.difference(categorical, sort=False)
        parts = []
        if len(numeric):
            parts.append(frame[numeric].apply(pd.to_numeric, errors="coerce"))
        if len(categorical):
            parts.append(pd.get_dummies(frame[categorical], drop_first=True, dtype=float))
        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=ids)
    values = np.asarray(covar, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) != len(ids):
        raise ValueError("covariate array must have one row per aligned phenotype sample")
    return pd.DataFrame(values, index=ids)


def _full_rank_design(covariates):
    columns = [np.ones(len(covariates), dtype=float)]
    rank = 1
    for column in np.asarray(covariates, dtype=float).T:
        candidate = np.column_stack(columns + [column])
        new_rank = np.linalg.matrix_rank(candidate)
        if new_rank > rank:
            columns.append(column)
            rank = new_rank
    return np.column_stack(columns)


def _resolved_rank(n_samples, n_components, decomposition):
    method = decomposition.lower()
    if method == "auto":
        method = "exact" if n_samples <= 4000 else "randomized"
    if n_components is None:
        return n_samples if method == "exact" else min(2000, max(n_samples - 1, 1))
    return min(int(n_components), n_samples)


def _matrix_by_trait(frame, value, traits, variants):
    return (
        frame.pivot(index="variant_index", columns="trait", values=value)
        .reindex(index=variants, columns=traits)
        .to_numpy(dtype=float)
    )


def _recover_multivariate_block(
    frame,
    component_names,
    original_traits,
    inverse_transform,
    n_samples,
    fixed_rank,
    model,
    da_thresholds,
    stability_z,
):
    """Recover original-trait effects and cross-trait Wald tests."""
    variants = frame["variant_index"].drop_duplicates().tolist()
    base = (
        frame.drop_duplicates("variant_index")
        .set_index("variant_index")
        .reindex(variants)
    )
    t = len(component_names)
    inverse = np.asarray(inverse_transform, dtype=np.float64)
    inverse_squared = inverse * inverse

    def matrix(column):
        return _matrix_by_trait(frame, column, component_names, variants)

    beta_add_marginal_star = matrix("beta_additive_raw")
    variance_add_marginal_star = matrix("se_additive_raw") ** 2
    beta_add_marginal = beta_add_marginal_star @ inverse
    variance_add_marginal = variance_add_marginal_star @ inverse_squared
    se_add_marginal = np.sqrt(np.maximum(variance_add_marginal, 0.0))
    stat_add_marginal = beta_add_marginal / se_add_marginal
    df_additive = n_samples - fixed_rank - 1

    m = len(variants)
    repeated = {
        "variant_index": np.repeat(np.asarray(variants, dtype=np.int64), t),
        "snp": np.repeat(base["snp"].to_numpy(), t),
        "chrom": np.repeat(base["chrom"].to_numpy(), t),
        "pos": np.repeat(base["pos"].to_numpy(), t),
        "trait": np.tile(np.asarray(original_traits, dtype=object), m),
    }
    result = pd.DataFrame(repeated)
    result["beta_additive_raw"] = beta_add_marginal.ravel(order="C")
    result["se_additive_raw"] = se_add_marginal.ravel(order="C")
    result["stat_additive"] = stat_add_marginal.ravel(order="C")
    result["neglog_p_additive"] = neglog10p(
        2.0 * _t.sf(np.abs(stat_add_marginal), df=df_additive)
    ).ravel(order="C")

    add_scale = np.divide(
        matrix("beta_additive"),
        beta_add_marginal_star,
        out=np.full_like(beta_add_marginal_star, np.nan),
        where=np.abs(beta_add_marginal_star) > np.finfo(float).eps,
    )
    marker_add_scale = np.nanmedian(add_scale, axis=1)
    result["beta_additive"] = (
        beta_add_marginal * marker_add_scale[:, None]
    ).ravel(order="C")

    invariant_columns = [
        "n_obs",
        "count_AA",
        "count_AB",
        "count_BB",
        "maf",
        "genotype_filter_pass",
    ]
    for column in invariant_columns:
        result[column] = np.repeat(base[column].to_numpy(), t)
    result["dominance_class"] = np.where(
        result["genotype_filter_pass"].to_numpy(), "NA", "FILTERED"
    )
    result["inheritance_mode"] = result["dominance_class"]

    additive_multivariate = np.nansum(
        beta_add_marginal_star ** 2 / variance_add_marginal_star, axis=1
    )
    multi_statistics = {
        "stat_additive_multivariate": additive_multivariate,
        "neglog_p_additive_multivariate": neglog10p(
            chi2.sf(additive_multivariate, t)
        ),
        "df_additive_multivariate": np.full(m, t, dtype=float),
    }

    if model == "add-dom":
        beta_dom_marginal_star = matrix("beta_dominance_marginal_raw")
        variance_dom_marginal_star = matrix("se_dominance_marginal_raw") ** 2
        beta_dom_marginal = beta_dom_marginal_star @ inverse
        variance_dom_marginal = variance_dom_marginal_star @ inverse_squared
        se_dom_marginal = np.sqrt(np.maximum(variance_dom_marginal, 0.0))
        stat_dom_marginal = beta_dom_marginal / se_dom_marginal
        result["beta_dominance_marginal_raw"] = beta_dom_marginal.ravel(order="C")
        result["se_dominance_marginal_raw"] = se_dom_marginal.ravel(order="C")
        result["stat_dominance_marginal"] = stat_dom_marginal.ravel(order="C")
        result["neglog_p_dominance_marginal"] = neglog10p(
            2.0 * _t.sf(np.abs(stat_dom_marginal), df=df_additive)
        ).ravel(order="C")

        add_star = matrix("beta_add_joint_raw")
        dom_star = matrix("beta_dom_joint_raw")
        var_add_star = matrix("se_add_joint_raw") ** 2
        var_dom_star = matrix("se_dom_joint_raw") ** 2
        covariance_star = matrix("cov_add_dom_joint_raw")
        add = add_star @ inverse
        dom = dom_star @ inverse
        var_add = var_add_star @ inverse_squared
        var_dom = var_dom_star @ inverse_squared
        covariance = covariance_star @ inverse_squared
        se_add = np.sqrt(np.maximum(var_add, 0.0))
        se_dom = np.sqrt(np.maximum(var_dom, 0.0))
        stat_add = add / se_add
        stat_dom = dom / se_dom
        df_joint = n_samples - fixed_rank - 2
        fstat = stat_dom * stat_dom
        signed, magnitude, coarse, mode = classify_inheritance(
            add,
            dom,
            se_add,
            thresholds=da_thresholds,
            stability_z=stability_z,
        )
        valid = result["genotype_filter_pass"].to_numpy().reshape(m, t)
        coarse[~valid] = "FILTERED"
        mode[~valid] = "FILTERED"
        values = {
            "beta_add_joint_raw": add,
            "se_add_joint_raw": se_add,
            "beta_dom_joint_raw": dom,
            "se_dom_joint_raw": se_dom,
            "cov_add_dom_joint_raw": covariance,
            "stat_add_joint": stat_add,
            "stat_dom_joint": stat_dom,
            "neglog_p_add_joint": neglog10p(2.0 * _t.sf(np.abs(stat_add), df=df_joint)),
            "neglog_p_dom_joint": neglog10p(2.0 * _t.sf(np.abs(stat_dom), df=df_joint)),
            "f_avsad": fstat,
            "neglog_p_avsad": neglog10p(chi2.sf(fstat, 1)),
            "degree_of_dominance": signed,
            "degree_of_dominance_abs": magnitude,
            "dominance_class": coarse,
            "inheritance_mode": mode,
        }
        for column, value in values.items():
            result[column] = np.asarray(value).ravel(order="C")

        additive_joint_multivariate = np.nansum(add_star ** 2 / var_add_star, axis=1)
        dominance_joint_multivariate = np.nansum(dom_star ** 2 / var_dom_star, axis=1)
        determinant = var_add_star * var_dom_star - covariance_star ** 2
        component_joint = np.divide(
            var_dom_star * add_star ** 2
            - 2.0 * covariance_star * add_star * dom_star
            + var_add_star * dom_star ** 2,
            determinant,
            out=np.full_like(determinant, np.nan),
            where=determinant > np.finfo(float).eps,
        )
        add_dom_multivariate = np.nansum(component_joint, axis=1)
        multi_statistics.update(
            {
                "stat_add_joint_multivariate": additive_joint_multivariate,
                "neglog_p_add_joint_multivariate": neglog10p(
                    chi2.sf(additive_joint_multivariate, t)
                ),
                "df_add_joint_multivariate": np.full(m, t, dtype=float),
                "stat_dom_joint_multivariate": dominance_joint_multivariate,
                "neglog_p_dom_joint_multivariate": neglog10p(
                    chi2.sf(dominance_joint_multivariate, t)
                ),
                "df_dom_joint_multivariate": np.full(m, t, dtype=float),
                "stat_add_dom_multivariate": add_dom_multivariate,
                "neglog_p_add_dom_multivariate": neglog10p(
                    chi2.sf(add_dom_multivariate, 2 * t)
                ),
                "df_add_dom_multivariate": np.full(m, 2 * t, dtype=float),
            }
        )

    for column in multi_statistics:
        result[column] = np.nan
    result["multivariate_df"] = np.nan
    multivariate = result.iloc[::t].copy().reset_index(drop=True)
    multivariate["trait"] = "__joint__"
    for column in result.columns:
        if column not in {"variant_index", "snp", "chrom", "pos", "trait", *invariant_columns}:
            multivariate[column] = np.nan
    for column, values in multi_statistics.items():
        multivariate[column] = values
    multivariate["multivariate_df"] = t
    multivariate["dominance_class"] = "NA"
    multivariate["inheritance_mode"] = "NA"
    return pd.concat([result, multivariate], ignore_index=True, sort=False)


def _write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _write_npz_atomic(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def run_gwas(
    bfile,
    pheno,
    traits=None,
    covar=None,
    model="add-dom",
    chroms=None,
    block_size=8192,
    n_components=None,
    y_correction="ystar",
    out=None,
    da_thresholds=(0.25, 0.75, 1.25),
    min_genotype_count=0,
    stability_z=1.0,
    per_trait_missing=True,
    grm_bfile=None,
    return_results=True,
    verbose=True,
    trait_mode="independent",
    variance_estimator="auto",
    decomposition="exact",
    operator_error_tolerance=0.02,
    randomized_power_iterations=1,
    adaptive_randomized_rank=True,
    randomized_max_components=None,
    compute_dtype="float64",
    memory_budget_mb=None,
    grm_cache=None,
    n_jobs=1,
    resource_config=None,
    backend="cpu",
    gpu_devices=(),
    gpu_memory_budget_mb=None,
    scratch_dir=None,
    target_runtime_hours=None,
):
    """Run a quantitative-trait dominance-aware GWAS over PLINK1 files.

    Defaults preserve independent-trait profile REML and exact float64
    arithmetic. Explicit ``decomposition='exact'`` always requests the
    pairwise-missingness-aware dense LOCO GRM and is rejected during planning
    when it cannot fit the memory budget.
    """
    run_started = time.perf_counter()
    reader = PlinkReader(bfile)
    grm_reader = reader if grm_bfile is None else PlinkReader(grm_bfile)
    phenotype = _load_pheno(pheno)
    if traits is None:
        traits = [column for column in phenotype.columns if column != "FID"]
    traits = list(traits)
    if model not in {"add-dom", "additive"}:
        raise ValueError("model must be 'add-dom' or 'additive'")
    if y_correction not in {"ystar", None}:
        raise ValueError("only exact GLS ('ystar') and uncorrected OLS (None) are supported")
    if trait_mode not in {"independent", "multivariate"}:
        raise ValueError("trait_mode must be 'independent' or 'multivariate'")
    estimator_aliases = {
        "auto": "score" if trait_mode == "multivariate" else "profile_reml",
        "reml": "profile_reml",
        "profile_reml": "profile_reml",
        "ml": "profile_ml",
        "profile_ml": "profile_ml",
        "score": "score",
    }
    if variance_estimator not in estimator_aliases:
        raise ValueError(
            "variance_estimator must be auto, reml/profile_reml, ml/profile_ml, or score"
        )
    resolved_estimator = estimator_aliases[variance_estimator]
    if trait_mode == "multivariate" and resolved_estimator != "score":
        raise ValueError("multivariate trait_mode requires variance_estimator='score'")
    if trait_mode == "multivariate" and y_correction is None:
        raise ValueError("multivariate trait_mode requires y_correction='ystar'")
    if not return_results and out is None:
        raise ValueError("out is required when return_results=False")

    config = resource_config or ResourceConfig(
        memory_budget_mb=memory_budget_mb,
        n_jobs=n_jobs,
        backend=backend,
        gpu_devices=gpu_devices,
        gpu_memory_budget_mb=gpu_memory_budget_mb,
        scratch_dir=scratch_dir,
        target_runtime_hours=target_runtime_hours,
    )
    config = config.resolved()
    config = replace(config, backend=resolve_backend_name(config.backend))
    memory_monitor = PeakRSSMonitor()

    phenotype.index = phenotype.index.astype(str)
    fam_iid = reader.fam["iid"].astype(str).to_numpy()
    fam_position = pd.Index(fam_iid)
    phenotype_ids = set(phenotype.index)
    common = [iid for iid in fam_iid if iid in phenotype_ids]
    sub = phenotype.loc[common, traits].apply(pd.to_numeric, errors="coerce")
    binary = [trait for trait in traits if sub[trait].dropna().nunique() == 2]
    if binary:
        raise ValueError(
            f"binary phenotypes are not supported by the quantitative-trait model: {binary}"
        )
    covariate_frame = _prepare_covariates(covar, sub.index)
    covariate_complete = np.isfinite(covariate_frame.to_numpy(dtype=float)).all(axis=1)
    trait_masks = {
        trait: sub[trait].notna().to_numpy() & covariate_complete for trait in traits
    }
    if not per_trait_missing or trait_mode == "multivariate":
        joint = np.logical_and.reduce(list(trait_masks.values()))
        if trait_mode == "multivariate" and any(
            not np.array_equal(mask, joint) for mask in trait_masks.values()
        ):
            warnings.warn("multivariate mode uses complete cases shared across all traits")
        trait_masks = {trait: joint for trait in traits}

    groups = {}
    for trait, mask in trait_masks.items():
        groups.setdefault(mask.tobytes(), {"mask": mask, "traits": []})["traits"].append(trait)

    requested_chroms = (
        [str(chrom) for chrom in chroms] if chroms is not None else reader.chroms
    )
    writers = {}
    retained_frames = {}
    execution_records = []
    output_prefix = None if out is None else str(out)

    def emit(chrom, frame, h2_map, n_analysis):
        frame["test_scope"] = np.where(
            frame["trait"].astype(str).to_numpy() == "__joint__",
            "multivariate_joint",
            "trait",
        )
        frame["test_scope"] = pd.Categorical(
            frame["test_scope"], categories=["trait", "multivariate_joint"]
        )
        frame["dominance_class"] = pd.Categorical(
            frame["dominance_class"],
            categories=["FILTERED", "NA", "A", "PD", "D", "OD", "UNSTABLE"],
        )
        frame["inheritance_mode"] = pd.Categorical(
            frame["inheritance_mode"],
            categories=[
                "FILTERED", "NA", "additive", "partial_dominant_A1",
                "partial_dominant_A0", "complete_dominant_A1",
                "complete_dominant_A0", "overdominant_high",
                "underdominant_low", "additive_near_zero",
            ],
        )
        frame["h2"] = frame["trait"].map(h2_map)
        frame["n_samples"] = n_analysis
        if output_prefix is not None:
            if chrom not in writers:
                writers[chrom] = AtomicParquetWriter(
                    f"{output_prefix}.chr{chrom}.parquet", compression="zstd"
                )
            writers[chrom].write(frame)
        if return_results:
            retained_frames.setdefault(str(chrom), []).append(frame)
        memory_monitor.check_budget(
            config.memory_budget_mb, context=f"chromosome {chrom} output"
        )

    memory_monitor.start()
    try:
        with threadpool_limits(limits=config.n_jobs):
            for group_index, group in enumerate(groups.values()):
                mask = group["mask"]
                group_traits = group["traits"]
                analysis_ids = sub.index[mask].tolist()
                if len(analysis_ids) < 5:
                    warnings.warn(f"skipping {group_traits}: fewer than five complete samples")
                    continue
                sample_index = fam_position.get_indexer(analysis_ids)
                grm_fam_position = pd.Index(grm_reader.fam["iid"].astype(str).to_numpy())
                grm_sample_index = grm_fam_position.get_indexer(analysis_ids)
                if np.any(grm_sample_index < 0):
                    missing = [
                        analysis_ids[index]
                        for index in np.where(grm_sample_index < 0)[0][:5]
                    ]
                    raise ValueError(f"analysis samples missing from grm_bfile: {missing}")
                Y = sub.loc[analysis_ids, group_traits].to_numpy(dtype=np.float64)
                C = _full_rank_design(
                    covariate_frame.loc[analysis_ids].to_numpy(dtype=float)
                )
                rank = 0 if y_correction is None else _resolved_rank(
                    len(Y), n_components, decomposition
                )
                planning_decomposition = decomposition if y_correction is not None else "randomized"
                planning_rank = max(rank, 1)
                resolved_decomposition = planning_decomposition
                if resolved_decomposition == "auto":
                    resolved_decomposition = "exact" if len(Y) <= 4000 else "randomized"
                if resolved_decomposition == "randomized" and adaptive_randomized_rank:
                    planning_rank = min(
                        len(Y),
                        randomized_max_components
                        if randomized_max_components is not None
                        else max(planning_rank, 4 * planning_rank),
                    )
                plan = plan_execution(
                    len(Y),
                    len(group_traits),
                    C.shape[1],
                    planning_rank,
                    block_size,
                    model=model,
                    compute_dtype=compute_dtype,
                    decomposition=planning_decomposition,
                    n_variants=reader.n_variants,
                    resource_config=config,
                )
                record = {
                    "group_index": group_index,
                    "traits": group_traits,
                    "n_samples": len(Y),
                    "fixed_effect_rank": C.shape[1],
                    "plan": plan.as_dict(),
                    "chromosomes": {},
                }
                execution_records.append(record)
                if verbose:
                    print(
                        f"analysis set: {len(Y)} samples x {len(group_traits)} traits; "
                        f"{reader.n_variants} variants; model={model}; fixed-effect "
                        f"rank={C.shape[1]}"
                    )
                    print(
                        f"  resource plan: block={plan.variant_block_size}, "
                        f"trait tile={plan.trait_tile_size}, estimated peak="
                        f"{plan.estimated_peak_mb:.0f} MiB / {plan.memory_budget_mb} MiB"
                    )

                if y_correction is None:
                    eigensystems = (
                        (
                            chrom,
                            {
                                "U": np.empty((len(Y), 0), dtype=np.float64),
                                "s": np.empty(0, dtype=np.float64),
                                "residual_eigenvalue": 0.0,
                                "decomposition": "none",
                                "diagnostics": {},
                            },
                        )
                        for chrom in requested_chroms
                    )
                else:
                    eigensystems = iter_loco_grms(
                        grm_reader,
                        sample_index=grm_sample_index,
                        block_size=plan.variant_block_size,
                        n_components=n_components,
                        chroms=requested_chroms,
                        decomposition=decomposition,
                        compute_dtype=compute_dtype,
                        grm_cache=grm_cache,
                        randomized_power_iterations=randomized_power_iterations,
                        operator_error_tolerance=operator_error_tolerance,
                        adaptive_randomized_rank=adaptive_randomized_rank,
                        randomized_max_components=randomized_max_components,
                        verbose=verbose,
                    )

                eigensystem_iterator = iter(eigensystems)
                while True:
                    decomposition_started = time.perf_counter()
                    try:
                        chrom, eigensystem = next(eigensystem_iterator)
                    except StopIteration:
                        break
                    decomposition_s = time.perf_counter() - decomposition_started
                    U = eigensystem["U"]
                    s = eigensystem["s"]
                    residual_eigenvalue = eigensystem.get("residual_eigenvalue", 0.0)
                    variance_started = time.perf_counter()
                    transformed = None
                    if y_correction is None:
                        scan_Y = Y
                        scan_traits = group_traits
                        fits = [
                            {"h2": 0.0, "yvar": 1.0, "estimator": "ols"}
                            for _ in group_traits
                        ]
                        original_fits = fits
                    elif trait_mode == "multivariate":
                        transformed = multivariate_score_transform(
                            Y,
                            U,
                            s,
                            covar=C,
                            residual_eigenvalue=residual_eigenvalue,
                        )
                        scan_Y = transformed["Y"]
                        scan_traits = [
                            f"__component_{index + 1}" for index in range(len(group_traits))
                        ]
                        fits = transformed["fits"]
                        original_fits = transformed["original_fits"]
                    elif resolved_estimator == "score":
                        score = score_variance_components(
                            Y,
                            U,
                            s,
                            covar=C,
                            residual_eigenvalue=residual_eigenvalue,
                        )
                        fits = score["fits"]
                        original_fits = fits
                        scan_Y = Y
                        scan_traits = group_traits
                    else:
                        fits = estimate_h2_many(
                            Y,
                            U,
                            s,
                            covar=C,
                            method="REML" if resolved_estimator == "profile_reml" else "ML",
                            residual_eigenvalue=residual_eigenvalue,
                        )
                        original_fits = fits
                        scan_Y = Y
                        scan_traits = group_traits

                    variance_metadata = {
                        "estimator": (
                            "score_multivariate"
                            if transformed is not None
                            else fits[0].get("estimator", resolved_estimator)
                        ),
                        "fits": original_fits,
                        "boundary_h2_count": int(
                            sum(
                                fit["h2"] <= 1e-8 or fit["h2"] >= 1.0 - 1e-8
                                for fit in original_fits
                            )
                        ),
                    }
                    if transformed is not None:
                        genetic_covariance = transformed["genetic_covariance"]
                        residual_covariance = transformed["residual_covariance"]
                        genetic_eigenvalues = np.linalg.eigvalsh(genetic_covariance)
                        residual_eigenvalues = np.linalg.eigvalsh(residual_covariance)
                        covariance_summary = {
                            "shape": list(genetic_covariance.shape),
                            "genetic_min_eigenvalue": float(genetic_eigenvalues.min()),
                            "genetic_max_eigenvalue": float(genetic_eigenvalues.max()),
                            "residual_min_eigenvalue": float(residual_eigenvalues.min()),
                            "residual_max_eigenvalue": float(residual_eigenvalues.max()),
                        }
                        if len(group_traits) <= 50:
                            covariance_summary["genetic"] = genetic_covariance.tolist()
                            covariance_summary["residual"] = residual_covariance.tolist()
                        elif output_prefix is not None:
                            safe_chrom = "".join(
                                character if character.isalnum() or character in "._-" else "_"
                                for character in str(chrom)
                            )
                            covariance_path = (
                                f"{output_prefix}.group{group_index}.chr{safe_chrom}."
                                "trait_covariance.npz"
                            )
                            _write_npz_atomic(
                                covariance_path,
                                genetic=genetic_covariance,
                                residual=residual_covariance,
                                transform=transformed["transform"],
                                inverse_transform=transformed["inverse_transform"],
                            )
                            covariance_summary["path"] = covariance_path
                        variance_metadata["trait_covariance"] = covariance_summary
                    variance_component_s = time.perf_counter() - variance_started
                    h2_map = {
                        trait: fit["h2"] for trait, fit in zip(group_traits, original_fits)
                    }
                    h2_map["__joint__"] = np.nan
                    if verbose and y_correction is not None:
                        print(
                            f"chr {chrom}: h2 = "
                            f"{np.round([fit['h2'] for fit in original_fits], 3).tolist()}"
                        )
                    association_started = time.perf_counter()
                    block_iterator = iter_chromosome_gls_eigen(
                        reader,
                        chrom,
                        scan_Y,
                        scan_traits,
                        sample_index,
                        U,
                        s,
                        fits,
                        C,
                        block_size=plan.variant_block_size,
                        trait_tile_size=plan.trait_tile_size,
                        model=model,
                        da_thresholds=da_thresholds,
                        min_genotype_count=min_genotype_count,
                        stability_z=stability_z,
                        residual_eigenvalue=residual_eigenvalue,
                        compute_dtype=compute_dtype,
                        combine_trait_tiles=True,
                        backend=config.backend,
                        gpu_devices=config.gpu_devices,
                        gpu_memory_budget_mb=config.gpu_memory_budget_mb,
                    )
                    rows = 0
                    for frame in block_iterator:
                        if transformed is not None:
                            frame = _recover_multivariate_block(
                                frame,
                                scan_traits,
                                group_traits,
                                transformed["inverse_transform"],
                                len(Y),
                                C.shape[1],
                                model,
                                da_thresholds,
                                stability_z,
                            )
                        rows += len(frame)
                        emit(chrom, frame, h2_map, len(Y))
                    association_output_s = time.perf_counter() - association_started
                    record["chromosomes"][str(chrom)] = {
                        "rows": rows,
                        "decomposition": eigensystem.get("decomposition"),
                        "cache_hit": bool(eigensystem.get("cache_hit", False)),
                        "cache_key": eigensystem.get("cache_key"),
                        "diagnostics": eigensystem.get("diagnostics", {}),
                        "h2": h2_map,
                        "variance_components": variance_metadata,
                        "timing_s": {
                            "decomposition_or_cache_load": decomposition_s,
                            "variance_components": variance_component_s,
                            "association_and_output": association_output_s,
                        },
                        "memory_after_chromosome": memory_monitor.snapshot(),
                    }
        for writer in writers.values():
            writer.close()
    except BaseException:
        memory_monitor.stop()
        for writer in writers.values():
            writer.abort()
        raise

    observed_memory = memory_monitor.stop()
    total_runtime_s = time.perf_counter() - run_started

    metadata = {
        "resource_config": {
            "memory_budget_mb": config.memory_budget_mb,
            "n_jobs": config.n_jobs,
            "backend": config.backend,
            "gpu_devices": list(config.gpu_devices),
            "gpu_memory_budget_mb": config.gpu_memory_budget_mb,
            "scratch_dir": config.scratch_dir,
            "target_runtime_hours": config.target_runtime_hours,
        },
        "trait_mode": trait_mode,
        "variance_estimator_requested": variance_estimator,
        "variance_estimator": resolved_estimator,
        "decomposition": decomposition,
        "compute_dtype": np.dtype(compute_dtype).name,
        "runtime_s": total_runtime_s,
        "target_runtime_hours": config.target_runtime_hours,
        "target_runtime_exceeded": (
            None
            if config.target_runtime_hours is None
            else total_runtime_s > config.target_runtime_hours * 3600
        ),
        "observed_memory": observed_memory,
        "groups": execution_records,
        "output": {
            str(chrom): {
                "rows": writer.rows_written,
                "row_groups": writer.row_groups_written,
                "path": str(writer.path),
            }
            for chrom, writer in writers.items()
        },
    }
    if output_prefix is not None:
        _write_json_atomic(f"{output_prefix}.execution.json", metadata)
    if not return_results:
        empty = pd.DataFrame()
        empty.attrs["execution"] = metadata
        return empty
    if not retained_frames:
        raise ValueError("no traits had enough complete samples")
    result = pd.concat(
        [
            pd.concat(frames, ignore_index=True, sort=False)
            for frames in retained_frames.values()
        ],
        ignore_index=True,
        sort=False,
    )
    result.attrs["execution"] = metadata
    return result
