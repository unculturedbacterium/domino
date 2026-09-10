"""Domino: a pure-Python, pip-installable, dominance-aware GWAS pipeline.

Public API
----------
    run_gwas         one-call dominance-aware GWAS over a PLINK1 fileset
    PlinkReader      streaming PLINK1 reader
    compute_loco_grms, compute_grm
    estimate_h2, whiten_y, blup_resid
    calculate_blups, calculate_component_blups, run_blup, write_blups
    input QC, relationship summaries, GWAS reporting, and plotting helpers
    scan_chromosome
    degree_of_dominance, classify_da
"""
from .io import PlinkReader, read_bim, read_fam
from .grm import compute_loco_grms, compute_grm, iter_loco_grms
from .vc import (
    estimate_h2, estimate_h2_many, score_variance_components,
    multivariate_score_transform, whiten_y, whiten_matrix, covariance_matrix,
    blup_resid,
)
from .assoc import scan_chromosome, scan_chromosome_gls_eigen, iter_chromosome_gls_eigen
from .classify import degree_of_dominance, classify_da, classify_inheritance, DEFAULT_THRESHOLDS
from .pipeline import run_gwas
from .blup import calculate_blups, calculate_component_blups, run_blup, write_blups
from .resources import ResourceConfig, ExecutionPlan, plan_execution
from .qc import (
    align_phenotype_to_fam,
    complete_case_traits,
    encode_covariates,
    plink_paths,
    read_gcta_grm,
    select_traits_by_prefix,
    summarize_plink,
    validate_plink_files,
)
from .relationships import (
    eigensystem_summary,
    estimate_heritability_table,
    score_heritability_table,
    summarize_grm,
    summarize_loco_grms,
)
from .reporting import (
    bonferroni_threshold,
    compare_rankings,
    dominance_class_counts,
    genomic_inflation,
    summarize_gwas_results,
    top_hits,
)
from .plotting import additive_dominance_scatter, manhattan_plot, qq_plot, runtime_memory_plot
from .reproducibility import collect_environment, public_api_table, write_manifest

__version__ = "1.0.0"
__all__ = [
    "run_gwas", "PlinkReader", "read_bim", "read_fam",
    "compute_loco_grms", "iter_loco_grms", "compute_grm",
    "estimate_h2", "estimate_h2_many", "score_variance_components",
    "multivariate_score_transform", "whiten_y", "whiten_matrix",
    "covariance_matrix", "blup_resid", "scan_chromosome",
    "calculate_blups", "calculate_component_blups", "run_blup", "write_blups",
    "scan_chromosome_gls_eigen", "iter_chromosome_gls_eigen",
    "ResourceConfig", "ExecutionPlan", "plan_execution",
    "align_phenotype_to_fam", "complete_case_traits", "encode_covariates",
    "plink_paths", "read_gcta_grm", "select_traits_by_prefix",
    "summarize_plink", "validate_plink_files", "eigensystem_summary",
    "estimate_heritability_table", "score_heritability_table",
    "summarize_grm", "summarize_loco_grms", "bonferroni_threshold",
    "compare_rankings", "dominance_class_counts", "genomic_inflation",
    "summarize_gwas_results", "top_hits", "collect_environment",
    "public_api_table", "write_manifest", "additive_dominance_scatter",
    "manhattan_plot", "qq_plot", "runtime_memory_plot",
    "degree_of_dominance", "classify_da", "classify_inheritance",
    "DEFAULT_THRESHOLDS", "__version__",
]
