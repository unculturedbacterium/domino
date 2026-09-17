"""Optional plotting helpers for Domino outputs."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .reporting import DEFAULT_P_COLUMNS, bonferroni_threshold, genomic_inflation, neglog10_to_p


def _matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Domino plotting helpers require matplotlib. Install with "
            "`python -m pip install Domino[plot]` or install matplotlib directly."
        ) from exc
    return plt


def qq_plot(results, test: str = "additive", trait: Optional[str] = None, ax=None, title=None):
    """Draw a QQ plot for a Domino -log10(P) column."""
    plt = _matplotlib()
    column = DEFAULT_P_COLUMNS.get(test, test)
    frame = results if trait is None else results[results["trait"].astype(str) == str(trait)]
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    p = neglog10_to_p(values)
    p = p[np.isfinite(p) & (p > 0.0) & (p <= 1.0)]
    if p.size == 0:
        raise ValueError(f"no finite p-values found in {column}")
    observed = -np.log10(np.sort(p))
    expected = -np.log10((np.arange(1, p.size + 1) - 0.5) / p.size)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(expected, observed, s=8, alpha=0.75)
    limit = float(max(expected.max(), observed.max()) * 1.03)
    ax.plot([0, limit], [0, limit], color="black", lw=1)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Expected -log10(P)")
    ax.set_ylabel("Observed -log10(P)")
    lambda_gc = genomic_inflation(values, already_neglog10=True)
    ax.set_title(title or f"{trait or 'All traits'} {test} QQ, lambda={lambda_gc:.3f}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def manhattan_plot(
    results,
    test: str = "additive",
    trait: Optional[str] = None,
    ax=None,
    alpha: float = 0.05,
    title=None,
):
    """Draw a compact Manhattan plot from Domino GWAS results."""
    plt = _matplotlib()
    column = DEFAULT_P_COLUMNS.get(test, test)
    needed = {"chrom", "pos", column}
    missing = needed.difference(results.columns)
    if missing:
        raise KeyError(f"results missing columns: {sorted(missing)}")
    frame = results.copy()
    if trait is not None and "trait" in frame.columns:
        frame = frame[frame["trait"].astype(str) == str(trait)]
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[np.isfinite(frame[column])].copy()
    if frame.empty:
        raise ValueError(f"no finite values found in {column}")
    chroms = list(dict.fromkeys(frame["chrom"].astype(str)))
    offsets = {}
    tick_positions = []
    offset = 0
    x_values = np.empty(len(frame), dtype=float)
    for chrom in chroms:
        mask = frame["chrom"].astype(str) == chrom
        positions = pd.to_numeric(frame.loc[mask, "pos"], errors="coerce").to_numpy(dtype=float)
        x_values[mask.to_numpy()] = positions + offset
        offsets[chrom] = offset
        if positions.size:
            tick_positions.append(float(offset + np.nanmedian(positions)))
            offset += int(np.nanmax(positions)) + 1
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.5))
    palette = ["#2F6C8F", "#D1883C"]
    for index, chrom in enumerate(chroms):
        mask = frame["chrom"].astype(str) == chrom
        ax.scatter(
            x_values[mask.to_numpy()],
            frame.loc[mask, column],
            s=7,
            color=palette[index % len(palette)],
            alpha=0.8,
            linewidths=0,
        )
    threshold = bonferroni_threshold(frame["snp"].nunique() if "snp" in frame.columns else len(frame), alpha=alpha)
    ax.axhline(threshold, color="#9B2D30", lw=1, ls=":")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(chroms, fontsize=8)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("-log10(P)")
    ax.set_title(title or f"{trait or 'All traits'} {test}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def additive_dominance_scatter(results, trait: Optional[str] = None, ax=None, title=None):
    """Plot additive versus joint-dominance signal strength."""
    plt = _matplotlib()
    frame = results.copy()
    if trait is not None and "trait" in frame.columns:
        frame = frame[frame["trait"].astype(str) == str(trait)]
    add = pd.to_numeric(frame["neglog_p_additive"], errors="coerce")
    dom = pd.to_numeric(frame["neglog_p_dom_joint"], errors="coerce")
    mask = np.isfinite(add) & np.isfinite(dom)
    if not mask.any():
        raise ValueError("no finite additive and dominance values found")
    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 4.3))
    ax.scatter(add[mask], dom[mask], s=8, alpha=0.65, color="#3F7F5F")
    limit = float(max(add[mask].max(), dom[mask].max()) * 1.03)
    ax.plot([0, limit], [0, limit], color="black", lw=1)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Additive -log10(P)")
    ax.set_ylabel("Dominance joint -log10(P)")
    ax.set_title(title or f"{trait or 'All traits'} additive vs dominance")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def runtime_memory_plot(summary, method_column="method", runtime_column="runtime_seconds", memory_column="peak_memory_mb", ax=None):
    """Plot runtime and peak memory side by side for benchmark summaries."""
    plt = _matplotlib()
    frame = pd.DataFrame(summary).copy()
    for column in (method_column, runtime_column, memory_column):
        if column not in frame.columns:
            raise KeyError(f"summary missing column: {column}")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(frame))
    width = 0.38
    runtime = pd.to_numeric(frame[runtime_column], errors="coerce")
    memory = pd.to_numeric(frame[memory_column], errors="coerce")
    ax.bar(x - width / 2, runtime, width=width, label="Runtime seconds", color="#2F6C8F")
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, memory, width=width, label="Peak memory MB", color="#D1883C")
    ax.set_xticks(x)
    ax.set_xticklabels(frame[method_column].astype(str), rotation=30, ha="right")
    ax.set_ylabel("Runtime seconds")
    ax2.set_ylabel("Peak memory MB")
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    return ax, ax2
