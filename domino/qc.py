"""Input-format and QC helpers for Domino tutorials and pipelines."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .io import PlinkReader, read_bim, read_fam


def plink_paths(prefix) -> dict:
    """Return the expected PLINK1 file paths for a prefix without extension."""
    prefix = Path(prefix)
    return {ext: prefix.with_suffix(f".{ext}") for ext in ("bed", "bim", "fam")}


def validate_plink_files(prefix) -> dict:
    """Validate a PLINK1 fileset and return a compact status dictionary."""
    paths = plink_paths(prefix)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing PLINK files: {missing}")
    fam = read_fam(prefix)
    bim = read_bim(prefix)
    with PlinkReader(prefix) as reader:
        n_samples = reader.n_samples
        n_variants = reader.n_variants
    if n_samples != len(fam) or n_variants != len(bim):
        raise ValueError("PLINK metadata counts do not match BED dimensions")
    return {
        "prefix": str(prefix),
        "n_samples": int(n_samples),
        "n_variants": int(n_variants),
        "n_chromosomes": int(bim["chrom"].nunique()),
        "duplicate_iids": int(fam["iid"].duplicated().sum()),
        "duplicate_snps": int(bim["snp"].duplicated().sum()),
        "chromosomes": list(dict.fromkeys(bim["chrom"].astype(str))),
    }


def summarize_plink(prefix, block_size: int = 8192, max_variants: Optional[int] = None) -> dict:
    """Summarize variant counts, missingness, and MAF from a PLINK1 fileset.

    ``max_variants`` can be used for a fast preview. When omitted, all variants
    are streamed without materializing the full genotype matrix.
    """
    info = validate_plink_files(prefix)
    reader = PlinkReader(prefix)
    try:
        n_seen = 0
        missing_total = 0
        call_total = 0
        maf_values = []
        per_chrom = reader.bim.groupby("chrom", sort=False).size().astype(int).to_dict()
        for _, block in reader.iter_blocks(block_size=block_size):
            if max_variants is not None and n_seen + block.shape[1] > max_variants:
                block = block[:, : max(0, max_variants - n_seen)]
            if block.shape[1] == 0:
                break
            mask = np.isfinite(block)
            missing_total += int((~mask).sum())
            call_total += int(mask.sum())
            allele_sum = np.nansum(block, axis=0)
            observed = np.sum(mask, axis=0)
            freq = allele_sum / np.maximum(2.0 * observed, 1.0)
            maf_values.append(np.minimum(freq, 1.0 - freq))
            n_seen += block.shape[1]
            if max_variants is not None and n_seen >= max_variants:
                break
        maf = np.concatenate(maf_values) if maf_values else np.array([], dtype=float)
    finally:
        reader.close()
    info.update(
        {
            "variants_summarized": int(n_seen),
            "variants_per_chromosome": {str(k): int(v) for k, v in per_chrom.items()},
            "missing_call_rate": (
                float(missing_total / (missing_total + call_total))
                if missing_total + call_total
                else np.nan
            ),
            "mean_maf": float(np.nanmean(maf)) if maf.size else np.nan,
            "min_maf": float(np.nanmin(maf)) if maf.size else np.nan,
            "max_maf": float(np.nanmax(maf)) if maf.size else np.nan,
        }
    )
    return info


def _as_indexed_frame(data, id_column="iid"):
    if isinstance(data, pd.Series):
        return data.to_frame()
    frame = data.copy()
    if id_column is not None and id_column in frame.columns:
        frame = frame.set_index(id_column, drop=True)
    frame.index = frame.index.astype(str)
    return frame


def encode_covariates(covariates, categorical: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Return numeric covariates with selected columns one-hot encoded."""
    frame = _as_indexed_frame(covariates)
    categorical = [] if categorical is None else list(categorical)
    missing = [column for column in categorical if column not in frame.columns]
    if missing:
        raise KeyError(f"categorical covariates not found: {missing}")
    numeric_columns = [column for column in frame.columns if column not in categorical]
    parts = []
    if numeric_columns:
        parts.append(frame[numeric_columns].apply(pd.to_numeric, errors="coerce"))
    if categorical:
        parts.append(pd.get_dummies(frame[categorical].astype("category"), drop_first=True, dtype=float))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=frame.index)


def align_phenotype_to_fam(
    phenotype,
    fam_or_prefix,
    traits: Optional[Sequence[str]] = None,
    covariates=None,
    categorical_covariates: Optional[Sequence[str]] = None,
    id_column: str = "iid",
    require_complete: bool = False,
) -> dict:
    """Align phenotype and optional covariates to PLINK FAM sample order."""
    fam = read_fam(fam_or_prefix) if not isinstance(fam_or_prefix, pd.DataFrame) else fam_or_prefix.copy()
    fam["iid"] = fam["iid"].astype(str)
    pheno = _as_indexed_frame(phenotype, id_column=id_column)
    traits = list(pheno.columns if traits is None else traits)
    missing_traits = [trait for trait in traits if trait not in pheno.columns]
    if missing_traits:
        raise KeyError(f"traits not found: {missing_traits}")
    common = fam.loc[fam["iid"].isin(pheno.index), "iid"].astype(str).tolist()
    aligned_pheno = pheno.loc[common, traits].apply(pd.to_numeric, errors="coerce")
    aligned_cov = None
    if covariates is not None:
        encoded = encode_covariates(covariates, categorical=categorical_covariates)
        aligned_cov = encoded.reindex(common)
    complete = aligned_pheno.notna().all(axis=1)
    if aligned_cov is not None:
        complete &= aligned_cov.notna().all(axis=1)
    if require_complete:
        aligned_pheno = aligned_pheno.loc[complete]
        if aligned_cov is not None:
            aligned_cov = aligned_cov.loc[complete]
        common = list(aligned_pheno.index)
    return {
        "ids": common,
        "phenotype": aligned_pheno,
        "covariates": aligned_cov,
        "n_fam": int(len(fam)),
        "n_matched": int(len(common)),
        "n_complete": int(complete.sum()),
        "trait_missing": aligned_pheno.isna().sum().astype(int).to_dict(),
    }


def read_gcta_grm(prefix) -> tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
    """Read a GCTA binary GRM triplet into ``(ids, grm, marker_counts)``."""
    prefix = Path(prefix)
    id_path = prefix.with_suffix(".grm.id")
    bin_path = prefix.with_suffix(".grm.bin")
    n_path = prefix.with_suffix(".grm.N.bin")
    if not id_path.exists() or not bin_path.exists():
        raise FileNotFoundError(f"expected {id_path} and {bin_path}")
    ids = pd.read_csv(id_path, sep=r"\s+", header=None, names=["fid", "iid"], dtype=str)
    n = len(ids)
    expected = n * (n + 1) // 2
    values = np.fromfile(bin_path, dtype=np.float32)
    if values.size != expected:
        raise ValueError(f"{bin_path} has {values.size} values, expected {expected}")
    matrix = np.zeros((n, n), dtype=np.float64)
    rows, cols = np.tril_indices(n)
    matrix[rows, cols] = values
    matrix[cols, rows] = values
    counts = None
    if n_path.exists():
        raw_counts = np.fromfile(n_path, dtype=np.float32)
        if raw_counts.size != expected:
            raise ValueError(f"{n_path} has {raw_counts.size} values, expected {expected}")
        counts = np.zeros((n, n), dtype=np.float64)
        counts[rows, cols] = raw_counts
        counts[cols, rows] = raw_counts
    return ids, matrix, counts


def complete_case_traits(frame: pd.DataFrame, prefix: str, min_samples: int = 5) -> list[str]:
    """Return trait columns with enough genotype-matched nonmissing samples."""
    fam_ids = set(read_fam(prefix)["iid"].astype(str))
    phenotype = _as_indexed_frame(frame)
    matched = phenotype.loc[phenotype.index.astype(str).isin(fam_ids)]
    return [
        column
        for column in matched.columns
        if pd.to_numeric(matched[column], errors="coerce").notna().sum() >= min_samples
    ]


def select_traits_by_prefix(columns: Iterable[str], prefix: str) -> list[str]:
    """Return columns whose names start with ``prefix`` in the original order."""
    return [column for column in columns if str(column).startswith(prefix)]
