"""Domino: a pure-Python, pip-installable, dominance-aware GWAS pipeline.

Public API
----------
    run_gwas         one-call dominance-aware GWAS over a PLINK1 fileset
    PlinkReader      streaming PLINK1 reader
    compute_loco_grms, compute_grm
    estimate_h2, whiten_y, blup_resid
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
from .resources import ResourceConfig, ExecutionPlan, plan_execution

__version__ = "1.0.0"
__all__ = [
    "run_gwas", "PlinkReader", "read_bim", "read_fam",
    "compute_loco_grms", "iter_loco_grms", "compute_grm",
    "estimate_h2", "estimate_h2_many", "score_variance_components",
    "multivariate_score_transform", "whiten_y", "whiten_matrix",
    "covariance_matrix", "blup_resid", "scan_chromosome",
    "scan_chromosome_gls_eigen", "iter_chromosome_gls_eigen",
    "ResourceConfig", "ExecutionPlan", "plan_execution",
    "degree_of_dominance", "classify_da", "classify_inheritance",
    "DEFAULT_THRESHOLDS", "__version__",
]
