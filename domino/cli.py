"""Command-line interface for quantitative-trait Domino scans."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from . import __version__, run_gwas


def _names(value: str | None) -> list[str]:
    return [] if not value else [item.strip() for item in value.split(",") if item.strip()]


def _integers(value: str | None) -> list[int]:
    return [int(item) for item in _names(value)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="domino",
        description="Run a quantitative-trait, dominance-aware GWAS with Domino",
    )
    parser.add_argument("--version", action="version", version=f"Domino {__version__}")
    parser.add_argument("--bfile", required=True, help="PLINK 1 prefix without .bed/.bim/.fam")
    parser.add_argument("--grm-bfile", help="Optional separate PLINK 1 relationship-panel prefix")
    parser.add_argument("--pheno", required=True, type=Path, help="Tab-separated phenotype/covariate table")
    parser.add_argument("--id-column", default="iid", help="Sample ID column in --pheno")
    parser.add_argument("--traits", required=True, help="Comma-separated quantitative phenotype columns")
    parser.add_argument("--covariates", help="Comma-separated numeric covariate columns")
    parser.add_argument("--categorical-covariates", help="Comma-separated categorical covariate columns")
    parser.add_argument("--chromosomes", help="Comma-separated chromosome labels")
    parser.add_argument("--model", choices=["add-dom", "additive"], default="add-dom")
    parser.add_argument("--trait-mode", choices=["independent", "multivariate"], default="independent")
    parser.add_argument(
        "--variance-estimator",
        choices=["auto", "reml", "ml", "profile_reml", "profile_ml", "score"],
        default="auto",
        help="auto selects profile REML for independent mode and SCORE for multivariate mode",
    )
    parser.add_argument("--decomposition", choices=["auto", "exact", "randomized"], default="exact")
    parser.add_argument("--n-components", type=int)
    parser.add_argument("--randomized-power-iterations", type=int, default=1)
    parser.add_argument("--randomized-max-components", type=int)
    parser.add_argument("--no-adaptive-rank", action="store_true")
    parser.add_argument("--operator-error-tolerance", type=float, default=0.02)
    parser.add_argument("--no-operator-error-check", action="store_true")
    parser.add_argument("--compute-dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--block-size", type=int, default=8192, help="Maximum marker block size")
    parser.add_argument("--memory-budget-mb", type=int)
    parser.add_argument("--n-jobs", type=int, default=1, help="BLAS CPU threads; -1 uses all logical CPUs")
    parser.add_argument("--grm-cache", help="Directory for reusable memory-mapped LOCO eigensystems")
    parser.add_argument("--backend", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--gpu-devices", help="Comma-separated zero-based CUDA device IDs")
    parser.add_argument("--gpu-memory-budget-mb", type=int, help="Maximum CuPy pool size per GPU")
    parser.add_argument("--scratch-dir", help="Scratch directory recorded in the execution plan")
    parser.add_argument("--target-runtime-hours", type=float)
    parser.add_argument("--min-genotype-count", type=int, default=5)
    parser.add_argument("--complete-case-across-traits", action="store_true")
    parser.add_argument("--stream", action="store_true", help="Write row groups without retaining results")
    parser.add_argument("--out", required=True, type=Path, help="Output prefix")
    return parser


def main() -> None:
    args = build_parser().parse_args()
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

    covar = None
    if numeric or categorical:
        if numeric:
            frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
        covar = frame[numeric + categorical]

    variance_estimator = args.variance_estimator
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = run_gwas(
        args.bfile,
        frame[traits],
        traits=traits,
        covar=covar,
        model=args.model,
        chroms=_names(args.chromosomes) or None,
        block_size=args.block_size,
        min_genotype_count=args.min_genotype_count,
        per_trait_missing=not args.complete_case_across_traits,
        grm_bfile=args.grm_bfile,
        out=str(args.out),
        return_results=not args.stream,
        trait_mode=args.trait_mode,
        variance_estimator=variance_estimator,
        decomposition=args.decomposition,
        n_components=args.n_components,
        randomized_power_iterations=args.randomized_power_iterations,
        adaptive_randomized_rank=not args.no_adaptive_rank,
        randomized_max_components=args.randomized_max_components,
        operator_error_tolerance=(
            None if args.no_operator_error_check else args.operator_error_tolerance
        ),
        compute_dtype=args.compute_dtype,
        memory_budget_mb=args.memory_budget_mb,
        n_jobs=args.n_jobs,
        grm_cache=args.grm_cache,
        backend=args.backend,
        gpu_devices=_integers(args.gpu_devices),
        gpu_memory_budget_mb=args.gpu_memory_budget_mb,
        scratch_dir=args.scratch_dir,
        target_runtime_hours=args.target_runtime_hours,
        verbose=True,
    )
    elapsed = time.perf_counter() - started
    if not args.stream:
        result.to_parquet(Path(f"{args.out}.all.parquet"), index=False)
    metadata = {
        "software": "Domino",
        "domino_version": __version__,
        "python_namespace": "domino",
        "bfile": str(Path(args.bfile).resolve()),
        "grm_bfile": str(Path(args.grm_bfile).resolve()) if args.grm_bfile else None,
        "phenotype": str(args.pheno.resolve()),
        "traits": traits,
        "numeric_covariates": numeric,
        "categorical_covariates": categorical,
        "model": args.model,
        "trait_mode": args.trait_mode,
        "variance_estimator": variance_estimator,
        "decomposition": args.decomposition,
        "n_components": args.n_components,
        "adaptive_randomized_rank": not args.no_adaptive_rank,
        "randomized_max_components": args.randomized_max_components,
        "compute_dtype": args.compute_dtype,
        "block_size_ceiling": args.block_size,
        "memory_budget_mb": args.memory_budget_mb,
        "n_jobs": args.n_jobs,
        "grm_cache": args.grm_cache,
        "backend": args.backend,
        "gpu_devices": _integers(args.gpu_devices),
        "gpu_memory_budget_mb": args.gpu_memory_budget_mb,
        "min_genotype_count": args.min_genotype_count,
        "per_trait_missingness": not args.complete_case_across_traits,
        "streamed_output": args.stream,
        "trait_type": "quantitative_only",
        "runtime_s": elapsed,
        "n_output_rows_in_memory": len(result),
        "execution": result.attrs.get("execution"),
    }
    Path(f"{args.out}.metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
