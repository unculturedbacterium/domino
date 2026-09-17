"""Best linear unbiased predictions from Domino covariance models.

The primary ``run_blup`` entry point fits a full-genome additive GRM model and
writes one prediction row per animal and quantitative trait. A lower-level
component function also supports additive and dominance BLUPs when both GRMs
and their estimated variance components are supplied.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, eigh
from threadpoolctl import threadpool_limits

from .grm import compute_grm
from .io import PlinkReader
from .pipeline import _full_rank_design, _load_pheno, _prepare_covariates
from .vc import estimate_h2_many


def _phenotype_matrix(phenotypes, sample_ids=None, trait_names=None):
    if isinstance(phenotypes, pd.DataFrame):
        frame = phenotypes.copy()
        if sample_ids is None:
            sample_ids = frame.index.astype(str).tolist()
        if trait_names is None:
            trait_names = frame.columns.astype(str).tolist()
        values = frame.to_numpy(dtype=np.float64)
    else:
        values = np.asarray(phenotypes, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if sample_ids is None:
            sample_ids = [str(index) for index in range(values.shape[0])]
        if trait_names is None:
            trait_names = [f"trait_{index + 1}" for index in range(values.shape[1])]
    if values.ndim != 2:
        raise ValueError("phenotypes must be a vector or two-dimensional matrix")
    if len(sample_ids) != values.shape[0]:
        raise ValueError("sample_ids does not match the phenotype rows")
    if len(trait_names) != values.shape[1]:
        raise ValueError("trait_names does not match the phenotype columns")
    if not np.isfinite(values).all():
        raise ValueError("phenotypes must be complete and finite")
    return values, [str(value) for value in sample_ids], [str(value) for value in trait_names]


def _fixed_design(covar, n_samples):
    if covar is None:
        return np.ones((n_samples, 1), dtype=np.float64)
    values = np.asarray(covar, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] != n_samples:
        raise ValueError("covar does not match the phenotype rows")
    if not np.isfinite(values).all():
        raise ValueError("covar must be complete and finite")
    return values


def _variance_fits_by_trait(variance_fits, trait_names):
    if isinstance(variance_fits, dict) and all(
        trait in variance_fits for trait in trait_names
    ):
        fits = [variance_fits[trait] for trait in trait_names]
    elif isinstance(variance_fits, dict) and "h2" in variance_fits:
        if len(trait_names) != 1:
            raise ValueError("one variance-component fit is required per trait")
        fits = [variance_fits]
    else:
        fits = list(variance_fits)
    if len(fits) != len(trait_names):
        raise ValueError("one variance-component fit is required per trait")
    for fit in fits:
        if "h2" not in fit or "yvar" not in fit:
            raise ValueError("each variance-component fit must contain h2 and yvar")
    return fits


def _apply_retained_operator(x, U, retained_values, bulk_value):
    projection = U.T @ x
    return bulk_value * x + U @ ((retained_values - bulk_value) * projection)


def _assemble_output(
    Y,
    sample_ids,
    trait_names,
    fixed,
    additive,
    dominance,
    fits,
    model,
):
    n_samples, n_traits = Y.shape
    total_genetic = additive + np.nan_to_num(dominance, nan=0.0)
    fitted = fixed + total_genetic
    residual = Y - fitted
    h2_values = []
    sigma_a2_values = []
    sigma_e2_values = []
    for fit in fits:
        if "h2" in fit:
            h2_value = float(fit["h2"])
        else:
            total = (
                float(fit["sigma_a2"])
                + float(fit.get("sigma_d2", 0.0))
                + float(fit["sigma_e2"])
            )
            h2_value = float(fit["sigma_a2"]) / max(total, 1e-30)
        h2_values.append(h2_value)
        sigma_a2_values.append(
            float(fit["sigma_a2"])
            if "sigma_a2" in fit
            else h2_value * float(fit["yvar"])
        )
        sigma_e2_values.append(
            float(fit["sigma_e2"])
            if "sigma_e2" in fit
            else (1.0 - h2_value) * float(fit["yvar"])
        )
    h2 = np.asarray(h2_values, dtype=float)
    sigma_a2 = np.asarray(sigma_a2_values, dtype=float)
    sigma_d2 = np.asarray([fit.get("sigma_d2", np.nan) for fit in fits], dtype=float)
    sigma_e2 = np.asarray(sigma_e2_values, dtype=float)
    return pd.DataFrame(
        {
            "iid": np.repeat(np.asarray(sample_ids, dtype=object), n_traits),
            "trait": np.tile(np.asarray(trait_names, dtype=object), n_samples),
            "phenotype": Y.ravel(order="C"),
            "fixed_prediction": fixed.ravel(order="C"),
            "additive_blup": additive.ravel(order="C"),
            "dominance_blup": dominance.ravel(order="C"),
            "total_genetic_value": total_genetic.ravel(order="C"),
            "fitted_value": fitted.ravel(order="C"),
            "residual": residual.ravel(order="C"),
            "h2_additive": np.tile(h2, n_samples),
            "sigma_additive2": np.tile(sigma_a2, n_samples),
            "sigma_dominance2": np.tile(sigma_d2, n_samples),
            "sigma_residual2": np.tile(sigma_e2, n_samples),
            "blup_model": model,
        }
    )


def calculate_blups(
    phenotypes,
    U,
    s,
    variance_fits,
    covar=None,
    sample_ids=None,
    trait_names=None,
    residual_eigenvalue=0.0,
    tol=1e-10,
):
    """Calculate additive BLUPs from a fitted additive GRM eigensystem.

    Parameters
    ----------
    phenotypes
        Complete phenotype vector or ``n_samples x n_traits`` matrix.
    U, s
        Retained eigenvectors and eigenvalues of the additive GRM.
    variance_fits
        One Domino variance-component fit per trait. Each fit must contain
        ``h2`` and ``yvar``.
    covar
        Complete fixed-effect design including an intercept.
    residual_eigenvalue
        Bulk GRM eigenvalue used by a randomized decomposition.

    Returns
    -------
    pandas.DataFrame
        One row per animal and trait. ``dominance_blup`` is missing because the
        current Domino variance model contains no dominance GRM.
    """
    Y, ids, names = _phenotype_matrix(phenotypes, sample_ids, trait_names)
    C = _fixed_design(covar, len(Y))
    U = np.asarray(U, dtype=np.float64)
    s = np.maximum(np.asarray(s, dtype=np.float64), 0.0)
    if U.shape != (len(Y), len(s)):
        raise ValueError("U, s, and phenotype dimensions do not agree")
    fits = _variance_fits_by_trait(variance_fits, names)

    fixed = np.empty_like(Y)
    additive = np.empty_like(Y)
    coefficients = []
    for index, fit in enumerate(fits):
        h2 = float(fit["h2"])
        if not 0.0 <= h2 <= 1.0:
            raise ValueError("h2 must be between zero and one")
        retained_covariance = np.maximum(h2 * s + 1.0 - h2, tol)
        bulk_covariance = max(
            h2 * float(residual_eigenvalue) + 1.0 - h2,
            tol,
        )
        inverse_retained = 1.0 / retained_covariance
        inverse_bulk = 1.0 / bulk_covariance
        inverse_y = _apply_retained_operator(
            Y[:, index], U, inverse_retained, inverse_bulk
        )
        inverse_c = np.column_stack(
            [
                _apply_retained_operator(C[:, column], U, inverse_retained, inverse_bulk)
                for column in range(C.shape[1])
            ]
        )
        information = C.T @ inverse_c
        beta = np.linalg.pinv(information) @ (C.T @ inverse_y)
        fixed[:, index] = C @ beta
        conditional = Y[:, index] - fixed[:, index]

        retained_shrinkage = h2 * s / retained_covariance
        bulk_shrinkage = h2 * float(residual_eigenvalue) / bulk_covariance
        additive[:, index] = _apply_retained_operator(
            conditional,
            U,
            retained_shrinkage,
            bulk_shrinkage,
        )
        coefficients.append(beta.tolist())

    dominance = np.full_like(additive, np.nan)
    result = _assemble_output(
        Y,
        ids,
        names,
        fixed,
        additive,
        dominance,
        fits,
        model="additive_grm",
    )
    result.attrs["fixed_effect_coefficients"] = dict(zip(names, coefficients))
    return result


def calculate_component_blups(
    phenotypes,
    additive_grm,
    variance_components,
    dominance_grm=None,
    covar=None,
    sample_ids=None,
    trait_names=None,
    tol=1e-10,
):
    """Calculate exact additive and optional dominance BLUPs from dense GRMs.

    The caller must supply externally estimated ``sigma_a2``, ``sigma_e2``,
    and, when ``dominance_grm`` is present, ``sigma_d2`` for every trait.
    This function calculates predictions only; it does not estimate the
    multi-GRM variance components.
    """
    Y, ids, names = _phenotype_matrix(phenotypes, sample_ids, trait_names)
    C = _fixed_design(covar, len(Y))
    K_add = np.asarray(additive_grm, dtype=np.float64)
    if K_add.shape != (len(Y), len(Y)):
        raise ValueError("additive_grm must be square with one row per sample")
    K_dom = None if dominance_grm is None else np.asarray(dominance_grm, dtype=np.float64)
    if K_dom is not None and K_dom.shape != K_add.shape:
        raise ValueError("dominance_grm must have the same shape as additive_grm")

    if isinstance(variance_components, dict) and all(
        trait in variance_components for trait in names
    ):
        fits = [variance_components[trait] for trait in names]
    elif isinstance(variance_components, dict) and "sigma_a2" in variance_components:
        if len(names) != 1:
            raise ValueError("one variance-component mapping is required per trait")
        fits = [variance_components]
    else:
        fits = list(variance_components)
    if len(fits) != len(names):
        raise ValueError("one variance-component mapping is required per trait")

    fixed = np.empty_like(Y)
    additive = np.empty_like(Y)
    dominance = np.full_like(Y, np.nan)
    coefficients = []
    identity = np.eye(len(Y), dtype=np.float64)
    for index, fit in enumerate(fits):
        sigma_a2 = float(fit["sigma_a2"])
        sigma_e2 = float(fit["sigma_e2"])
        sigma_d2 = float(fit.get("sigma_d2", 0.0))
        if min(sigma_a2, sigma_d2, sigma_e2) < 0.0 or sigma_e2 <= tol:
            raise ValueError("variance components must be nonnegative with sigma_e2 > 0")
        if K_dom is None and sigma_d2 != 0.0:
            raise ValueError("dominance_grm is required when sigma_d2 is nonzero")

        covariance = sigma_a2 * K_add + sigma_e2 * identity
        if K_dom is not None:
            covariance = covariance + sigma_d2 * K_dom
        factor = cho_factor(covariance, lower=True, check_finite=False)
        inverse_y = cho_solve(factor, Y[:, index], check_finite=False)
        inverse_c = cho_solve(factor, C, check_finite=False)
        beta = np.linalg.pinv(C.T @ inverse_c) @ (C.T @ inverse_y)
        fixed[:, index] = C @ beta
        conditional = Y[:, index] - fixed[:, index]
        alpha = cho_solve(factor, conditional, check_finite=False)
        additive[:, index] = sigma_a2 * (K_add @ alpha)
        if K_dom is not None:
            dominance[:, index] = sigma_d2 * (K_dom @ alpha)
        coefficients.append(beta.tolist())

    normalized_fits = [
        {
            **fit,
            "h2": float(fit["sigma_a2"])
            / max(
                float(fit["sigma_a2"])
                + float(fit.get("sigma_d2", 0.0))
                + float(fit["sigma_e2"]),
                tol,
            ),
        }
        for fit in fits
    ]
    result = _assemble_output(
        Y,
        ids,
        names,
        fixed,
        additive,
        dominance,
        normalized_fits,
        model="additive_dominance_grm" if K_dom is not None else "additive_grm_dense",
    )
    result.attrs["fixed_effect_coefficients"] = dict(zip(names, coefficients))
    return result


def write_blups(blups, output):
    """Write a BLUP table as Parquet, CSV, or tab-separated text."""
    path = Path(output)
    if not path.suffix:
        path = Path(f"{path}.blup.parquet")
    suffix = path.suffix.lower()
    if suffix not in {".parquet", ".csv", ".tsv", ".txt"}:
        raise ValueError("BLUP output must end in .parquet, .csv, .tsv, or .txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial{suffix}")
    if suffix == ".parquet":
        blups.to_parquet(temporary, index=False)
    else:
        blups.to_csv(temporary, index=False, sep="\t" if suffix in {".tsv", ".txt"} else ",")
    os.replace(temporary, path)
    return path


def run_blup(
    bfile,
    pheno,
    traits=None,
    covar=None,
    grm_bfile=None,
    method="REML",
    block_size=8192,
    out=None,
    n_jobs=1,
    verbose=True,
):
    """Fit full-genome additive GRM models and calculate per-animal BLUPs.

    Traits sharing a complete-case mask reuse one GRM eigendecomposition.
    The output is additive because Domino's fitted background covariance model
    currently contains an additive GRM only.
    """
    reader = PlinkReader(bfile)
    grm_reader = reader if grm_bfile is None else PlinkReader(grm_bfile)
    try:
        phenotype = _load_pheno(pheno)
        phenotype.index = phenotype.index.astype(str)
        if phenotype.index.has_duplicates:
            raise ValueError("phenotype IIDs must be unique")
        if traits is None:
            traits = [column for column in phenotype.columns if column != "FID"]
        traits = list(traits)
        method = method.upper()
        if method not in {"REML", "ML"}:
            raise ValueError("method must be REML or ML")

        fam_iid = reader.fam["iid"].astype(str).to_numpy()
        common = [iid for iid in fam_iid if iid in set(phenotype.index)]
        sub = phenotype.loc[common, traits].apply(pd.to_numeric, errors="coerce")
        binary = [trait for trait in traits if sub[trait].dropna().nunique() == 2]
        if binary:
            raise ValueError(
                f"binary phenotypes are not supported by the quantitative-trait model: {binary}"
            )
        covariate_frame = _prepare_covariates(covar, sub.index)
        covariate_complete = np.isfinite(
            covariate_frame.to_numpy(dtype=float)
        ).all(axis=1)
        groups = {}
        for trait in traits:
            mask = sub[trait].notna().to_numpy() & covariate_complete
            groups.setdefault(mask.tobytes(), {"mask": mask, "traits": []})[
                "traits"
            ].append(trait)

        fam_position = pd.Index(fam_iid)
        grm_position = pd.Index(grm_reader.fam["iid"].astype(str))
        frames = []
        metadata_groups = []
        with threadpool_limits(limits=None if n_jobs == -1 else n_jobs):
            for group_index, group in enumerate(groups.values()):
                mask = group["mask"]
                group_traits = group["traits"]
                analysis_ids = sub.index[mask]
                if len(analysis_ids) < 5:
                    raise ValueError(
                        f"fewer than five complete samples for traits {group_traits}"
                    )
                sample_index = fam_position.get_indexer(analysis_ids)
                grm_sample_index = grm_position.get_indexer(analysis_ids)
                if np.any(sample_index < 0) or np.any(grm_sample_index < 0):
                    raise ValueError("not all phenotype samples are present in the genotype files")
                Y = sub.loc[analysis_ids, group_traits].to_numpy(dtype=np.float64)
                covariate_values = covariate_frame.loc[analysis_ids].to_numpy(dtype=float)
                C = _full_rank_design(covariate_values)

                if verbose:
                    print(
                        f"BLUP group {group_index + 1}: {len(Y)} samples x "
                        f"{len(group_traits)} traits"
                    )
                grm = compute_grm(
                    grm_reader,
                    sample_index=grm_sample_index,
                    block_size=block_size,
                    compute_dtype="float64",
                )
                s, U = eigh(grm, check_finite=False, overwrite_a=True)
                order = np.argsort(s)[::-1]
                s = np.maximum(s[order], 0.0)
                U = U[:, order]
                fits = estimate_h2_many(Y, U, s, covar=C, method=method)
                frame = calculate_blups(
                    Y,
                    U,
                    s,
                    fits,
                    covar=C,
                    sample_ids=analysis_ids,
                    trait_names=group_traits,
                )
                frames.append(frame)
                metadata_groups.append(
                    {
                        "group": group_index,
                        "traits": group_traits,
                        "n_samples": len(Y),
                        "fixed_effect_rank": int(C.shape[1]),
                        "variance_components": dict(zip(group_traits, fits)),
                        "fixed_effect_coefficients": frame.attrs[
                            "fixed_effect_coefficients"
                        ],
                    }
                )
        result = pd.concat(frames, ignore_index=True)
        metadata = {
            "model": "full_genome_additive_grm_blup",
            "bfile": str(Path(bfile).resolve()),
            "grm_bfile": (
                str(Path(grm_bfile).resolve()) if grm_bfile is not None else None
            ),
            "traits": traits,
            "method": method,
            "groups": metadata_groups,
            "dominance_blup_available": False,
            "dominance_note": (
                "A dominance BLUP requires a dominance GRM and an estimated "
                "dominance variance component."
            ),
        }
        result.attrs["blup_metadata"] = metadata
        if out is not None:
            output_path = write_blups(result, out)
            metadata_path = output_path.with_suffix(".metadata.json")
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            result.attrs["output_path"] = str(output_path)
            result.attrs["metadata_path"] = str(metadata_path)
        return result
    finally:
        if grm_reader is not reader:
            grm_reader.close()
        reader.close()


def _names(value):
    return [] if not value else [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(
        prog="domino-blup",
        description="Calculate full-genome additive GRM BLUPs for quantitative traits",
    )
    parser.add_argument("--bfile", required=True, help="PLINK 1 prefix")
    parser.add_argument("--grm-bfile", help="Optional separate relationship-panel prefix")
    parser.add_argument("--pheno", required=True, type=Path, help="Tab-separated phenotype table")
    parser.add_argument("--id-column", default="iid")
    parser.add_argument("--traits", required=True, help="Comma-separated quantitative traits")
    parser.add_argument("--covariates", help="Comma-separated numeric covariates")
    parser.add_argument("--categorical-covariates", help="Comma-separated categorical covariates")
    parser.add_argument("--method", choices=["REML", "ML"], default="REML")
    parser.add_argument("--block-size", type=int, default=8192)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--out", required=True, help="Output prefix or table path")
    args = parser.parse_args()

    frame = pd.read_csv(args.pheno, sep="\t", dtype={args.id_column: "string"})
    frame[args.id_column] = frame[args.id_column].str.strip()
    frame = frame.set_index(args.id_column)
    traits = _names(args.traits)
    numeric = _names(args.covariates)
    categorical = _names(args.categorical_covariates)
    requested = traits + numeric + categorical
    missing = sorted(set(requested) - set(frame.columns))
    if missing:
        raise ValueError(f"columns missing from phenotype table: {missing}")
    if numeric:
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    covar = frame[numeric + categorical] if numeric or categorical else None
    result = run_blup(
        args.bfile,
        frame[traits],
        traits=traits,
        covar=covar,
        grm_bfile=args.grm_bfile,
        method=args.method,
        block_size=args.block_size,
        out=args.out,
        n_jobs=args.n_jobs,
    )
    print(f"Wrote {len(result)} BLUP rows to {result.attrs['output_path']}")


if __name__ == "__main__":
    main()
