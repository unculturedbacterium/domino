"""Relationship-matrix and heritability summaries for Domino."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .vc import estimate_h2_many, score_variance_components


def summarize_grm(grm, ids: Optional[Sequence[str]] = None) -> dict:
    """Return basic diagnostics for a genomic relationship matrix."""
    matrix = np.asarray(grm, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("grm must be a square matrix")
    n = matrix.shape[0]
    if ids is not None and len(ids) != n:
        raise ValueError("ids length does not match grm dimensions")
    diag = np.diag(matrix)
    off = matrix[np.triu_indices(n, k=1)] if n > 1 else np.array([], dtype=float)
    finite = np.isfinite(matrix)
    return {
        "n": int(n),
        "finite_fraction": float(finite.mean()),
        "diag_mean": float(np.nanmean(diag)),
        "diag_sd": float(np.nanstd(diag)),
        "diag_min": float(np.nanmin(diag)),
        "diag_max": float(np.nanmax(diag)),
        "offdiag_mean": float(np.nanmean(off)) if off.size else np.nan,
        "offdiag_sd": float(np.nanstd(off)) if off.size else np.nan,
        "offdiag_min": float(np.nanmin(off)) if off.size else np.nan,
        "offdiag_max": float(np.nanmax(off)) if off.size else np.nan,
        "trace": float(np.trace(matrix)),
        "symmetry_error_max": float(np.nanmax(np.abs(matrix - matrix.T))),
    }


def summarize_loco_grms(grms: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Summarize a dictionary of chromosome-specific LOCO GRMs."""
    rows = []
    for chrom, matrix in grms.items():
        row = summarize_grm(matrix)
        row["chrom"] = str(chrom)
        rows.append(row)
    return pd.DataFrame(rows).set_index("chrom") if rows else pd.DataFrame()


def eigensystem_summary(U, s, residual_eigenvalue: float = 0.0) -> dict:
    """Summarize an eigensystem used by Domino's covariance operator."""
    U = np.asarray(U, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if U.ndim != 2 or U.shape[1] != len(s):
        raise ValueError("U must have one column per eigenvalue")
    return {
        "n_samples": int(U.shape[0]),
        "rank": int(U.shape[1]),
        "residual_eigenvalue": float(residual_eigenvalue),
        "eigen_min": float(np.min(s)) if len(s) else np.nan,
        "eigen_median": float(np.median(s)) if len(s) else np.nan,
        "eigen_max": float(np.max(s)) if len(s) else np.nan,
        "eigen_sum": float(np.sum(s)),
        "orthogonality_error_max": float(np.max(np.abs(U.T @ U - np.eye(U.shape[1])))),
    }


def estimate_heritability_table(
    phenotypes,
    U,
    s,
    covar=None,
    traits: Optional[Sequence[str]] = None,
    method: str = "REML",
    residual_eigenvalue: float = 0.0,
) -> pd.DataFrame:
    """Estimate one profile-REML heritability row per trait."""
    if isinstance(phenotypes, pd.DataFrame):
        Y = phenotypes.to_numpy(dtype=np.float64)
        names = list(phenotypes.columns if traits is None else traits)
        if traits is not None:
            Y = phenotypes.loc[:, names].to_numpy(dtype=np.float64)
    else:
        Y = np.asarray(phenotypes, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        names = [f"trait_{index + 1}" for index in range(Y.shape[1])] if traits is None else list(traits)
    fits = estimate_h2_many(
        Y,
        U,
        s,
        covar=covar,
        method=method,
        residual_eigenvalue=residual_eigenvalue,
    )
    rows = []
    for name, fit in zip(names, fits):
        rows.append({"trait": name, **fit})
    return pd.DataFrame(rows)


def score_heritability_table(
    phenotypes,
    U,
    s,
    covar=None,
    traits: Optional[Sequence[str]] = None,
    residual_eigenvalue: float = 0.0,
    full_covariance: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Estimate SCORE heritability rows and return the full SCORE result."""
    if isinstance(phenotypes, pd.DataFrame):
        names = list(phenotypes.columns if traits is None else traits)
        Y = phenotypes.loc[:, names].to_numpy(dtype=np.float64)
    else:
        Y = np.asarray(phenotypes, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        names = [f"trait_{index + 1}" for index in range(Y.shape[1])] if traits is None else list(traits)
    result = score_variance_components(
        Y,
        U,
        s,
        covar=covar,
        residual_eigenvalue=residual_eigenvalue,
        full_covariance=full_covariance,
    )
    rows = [{"trait": name, **fit} for name, fit in zip(names, result["fits"])]
    return pd.DataFrame(rows), result
