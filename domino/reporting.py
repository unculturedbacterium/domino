"""GWAS result summaries and agreement metrics."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr


DEFAULT_P_COLUMNS = {
    "additive": "neglog_p_additive",
    "dominance_marginal": "neglog_p_dominance_marginal",
    "add_joint": "neglog_p_add_joint",
    "dom_joint": "neglog_p_dom_joint",
    "add_vs_add_dom": "neglog_p_avsad",
    "add_dom_multivariate": "neglog_p_add_dom_multivariate",
}


def neglog10_to_p(values):
    """Convert -log10(P) values to P values."""
    values = np.asarray(values, dtype=np.float64)
    return np.power(10.0, -values)


def p_to_neglog10(values):
    """Convert P values to -log10(P), preserving NaN values."""
    values = np.asarray(values, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log10(values)


def genomic_inflation(values, already_neglog10: bool = True) -> float:
    """Return lambda GC from P values or -log10(P) values."""
    p = neglog10_to_p(values) if already_neglog10 else np.asarray(values, dtype=np.float64)
    p = p[np.isfinite(p) & (p > 0.0) & (p <= 1.0)]
    if p.size == 0:
        return np.nan
    observed = chi2.isf(p, df=1)
    return float(np.nanmedian(observed) / chi2.ppf(0.5, df=1))


def bonferroni_threshold(n_tests: int, alpha: float = 0.05, neglog10: bool = True) -> float:
    """Return the Bonferroni threshold for ``n_tests``."""
    if n_tests < 1:
        raise ValueError("n_tests must be positive")
    threshold = alpha / float(n_tests)
    return float(-np.log10(threshold)) if neglog10 else threshold


def _available_tests(frame: pd.DataFrame, tests: Optional[Sequence[str]] = None) -> dict:
    if tests is None:
        return {name: column for name, column in DEFAULT_P_COLUMNS.items() if column in frame.columns}
    result = {}
    for name in tests:
        column = DEFAULT_P_COLUMNS.get(name, name)
        if column not in frame.columns:
            raise KeyError(f"test column not found: {column}")
        result[name] = column
    return result


def top_hits(
    results: pd.DataFrame,
    test: str = "additive",
    n: int = 10,
    trait: Optional[str] = None,
    chromosome: Optional[str] = None,
) -> pd.DataFrame:
    """Return the top ``n`` rows for a Domino test column."""
    column = DEFAULT_P_COLUMNS.get(test, test)
    if column not in results.columns:
        raise KeyError(f"test column not found: {column}")
    frame = results.copy()
    if trait is not None:
        frame = frame[frame["trait"].astype(str) == str(trait)]
    if chromosome is not None:
        frame = frame[frame["chrom"].astype(str) == str(chromosome)]
    frame = frame[np.isfinite(pd.to_numeric(frame[column], errors="coerce"))]
    return frame.sort_values(column, ascending=False).head(n).reset_index(drop=True)


def summarize_gwas_results(
    results: pd.DataFrame,
    tests: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Summarize lambda, maximum signal, and Bonferroni hits by trait and test."""
    test_columns = _available_tests(results, tests)
    rows = []
    group_columns = ["trait"] if "trait" in results.columns else [None]
    groups = results.groupby("trait", dropna=False) if group_columns[0] else [(None, results)]
    for trait, frame in groups:
        n_variants = int(frame["snp"].nunique()) if "snp" in frame.columns else int(len(frame))
        threshold = bonferroni_threshold(max(n_variants, 1), alpha=alpha)
        for test_name, column in test_columns.items():
            values = pd.to_numeric(frame[column], errors="coerce")
            finite = values[np.isfinite(values)]
            if finite.empty:
                rows.append(
                    {
                        "trait": trait,
                        "test": test_name,
                        "n_tests": n_variants,
                        "lambda_gc": np.nan,
                        "max_neglog10p": np.nan,
                        "n_bonferroni_hits": 0,
                    }
                )
                continue
            top_index = finite.idxmax()
            rows.append(
                {
                    "trait": trait,
                    "test": test_name,
                    "n_tests": n_variants,
                    "lambda_gc": genomic_inflation(finite, already_neglog10=True),
                    "max_neglog10p": float(finite.max()),
                    "bonferroni_neglog10p": threshold,
                    "n_bonferroni_hits": int((finite >= threshold).sum()),
                    "top_snp": results.loc[top_index, "snp"] if "snp" in results.columns else None,
                    "top_chrom": results.loc[top_index, "chrom"] if "chrom" in results.columns else None,
                    "top_pos": results.loc[top_index, "pos"] if "pos" in results.columns else None,
                }
            )
    return pd.DataFrame(rows)


def compare_rankings(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_column: str,
    right_column: str,
    keys: Sequence[str] = ("snp", "trait"),
    top_n: int = 10,
) -> dict:
    """Compare two GWAS result tables on shared marker or marker-trait keys."""
    left_frame = left[list(keys) + [left_column]].rename(columns={left_column: "_left_value"})
    right_frame = right[list(keys) + [right_column]].rename(columns={right_column: "_right_value"})
    merged = left_frame.merge(
        right_frame,
        on=list(keys),
        how="inner",
    )
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["_left_value", "_right_value"]
    )
    if merged.empty:
        return {
            "n_matched": 0,
            "spearman_rho": np.nan,
            "top_n_overlap": 0,
            "same_top_key": False,
        }
    rho = spearmanr(merged["_left_value"], merged["_right_value"], nan_policy="omit").statistic
    left_top = merged.sort_values("_left_value", ascending=False).head(top_n)
    right_top = merged.sort_values("_right_value", ascending=False).head(top_n)
    left_keys = {tuple(row) for row in left_top[list(keys)].to_numpy()}
    right_keys = {tuple(row) for row in right_top[list(keys)].to_numpy()}
    left_best = tuple(left_top.iloc[0][list(keys)].to_numpy())
    right_best = tuple(right_top.iloc[0][list(keys)].to_numpy())
    return {
        "n_matched": int(len(merged)),
        "spearman_rho": float(rho),
        "top_n": int(top_n),
        "top_n_overlap": int(len(left_keys & right_keys)),
        "same_top_key": bool(left_best == right_best),
    }


def dominance_class_counts(results: pd.DataFrame) -> pd.DataFrame:
    """Count Domino dominance classes by trait."""
    if "dominance_class" not in results.columns:
        raise KeyError("results must contain dominance_class")
    if "trait" in results.columns:
        return (
            results.groupby(["trait", "dominance_class"], observed=False)
            .size()
            .rename("n")
            .reset_index()
        )
    return results["dominance_class"].value_counts(dropna=False).rename_axis("dominance_class").reset_index(name="n")
