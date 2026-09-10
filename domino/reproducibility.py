"""Environment and API reference helpers for reproducible Domino runs."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd


def _package_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment(extra: Optional[Mapping[str, object]] = None) -> dict:
    """Collect versions and host details useful for methods reporting."""
    domino_version = _package_version("Domino") or _package_version("domino") or "unknown"
    packages = {
        "domino": domino_version,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": _package_version("scipy"),
        "scikit-learn": _package_version("scikit-learn"),
        "pyarrow": _package_version("pyarrow"),
        "psutil": _package_version("psutil"),
    }
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_executable": sys.executable,
        "packages": packages,
        "cpu_count": os.cpu_count(),
    }
    try:
        import threadpoolctl

        info["threadpools"] = threadpoolctl.threadpool_info()
    except ImportError:
        info["threadpools"] = []
    if extra:
        info["extra"] = dict(extra)
    return info


def git_revision(path=".") -> Optional[str]:
    """Return the Git commit hash for ``path`` when available."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def write_manifest(path, command=None, inputs: Optional[Mapping[str, object]] = None, extra=None) -> dict:
    """Write a JSON manifest describing a Domino run."""
    manifest = collect_environment(extra=extra)
    manifest["command"] = command
    manifest["inputs"] = dict(inputs or {})
    manifest["git_revision"] = git_revision(Path(path).resolve().parent)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def public_api_table() -> pd.DataFrame:
    """Return a compact table of public Domino API entry points."""
    rows = [
        ("run_gwas", "One-call additive and dominance GWAS over PLINK1 files"),
        ("PlinkReader", "Streaming PLINK1 BED reader"),
        ("compute_grm", "Whole-genome additive relationship matrix"),
        ("compute_loco_grms", "Leave-one-chromosome-out relationship matrices"),
        ("estimate_h2", "Single-trait profile ML or REML heritability"),
        ("estimate_h2_many", "Multi-trait profile ML or REML heritability"),
        ("score_variance_components", "Fast SCORE variance-component estimator"),
        ("multivariate_score_transform", "SCORE-based multivariate trait transform"),
        ("run_blup", "Full-genome additive BLUP workflow"),
        ("calculate_blups", "BLUP calculation from a fitted eigensystem"),
        ("calculate_component_blups", "Additive and dominance component BLUPs"),
        ("plan_execution", "Memory-aware block and trait-tile planning"),
        ("summarize_gwas_results", "Lambda, top-hit, and Bonferroni summaries"),
        ("qq_plot", "QQ plot for Domino GWAS output"),
        ("manhattan_plot", "Manhattan plot for Domino GWAS output"),
    ]
    return pd.DataFrame(rows, columns=["name", "description"])
