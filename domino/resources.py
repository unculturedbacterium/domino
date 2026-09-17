"""Resource planning for bounded-memory domino execution.

The planner is deliberately conservative.  A requested variant block size is
treated as a ceiling, never a target that is allowed to exceed the process
memory budget.  The same contract is used by the GRM and association stages so
large jobs fail during planning rather than after allocating a dense matrix.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Optional, Sequence
import warnings
import threading

import numpy as np


class PeakRSSMonitor:
    """Sample process RSS for run metadata and budget overrun detection."""

    def __init__(self, interval_seconds=0.05):
        try:
            import psutil  # type: ignore

            self._process = psutil.Process()
        except ImportError:
            self._process = None
        self.interval_seconds = interval_seconds
        self.baseline_bytes = self.current_bytes
        self.peak_bytes = self.baseline_bytes
        self._stop = threading.Event()
        self._thread = None

    @property
    def current_bytes(self):
        return None if self._process is None else int(self._process.memory_info().rss)

    def start(self):
        if self._process is None or self._thread is not None:
            return self

        def sample():
            while not self._stop.wait(self.interval_seconds):
                value = self.current_bytes
                if value is not None:
                    self.peak_bytes = max(self.peak_bytes or 0, value)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def snapshot(self):
        value = self.current_bytes
        if value is not None:
            self.peak_bytes = max(self.peak_bytes or 0, value)
        return {
            "rss_mb": None if value is None else value / 1024 ** 2,
            "peak_rss_mb": None if self.peak_bytes is None else self.peak_bytes / 1024 ** 2,
            "peak_increment_mb": (
                None
                if self.peak_bytes is None or self.baseline_bytes is None
                else (self.peak_bytes - self.baseline_bytes) / 1024 ** 2
            ),
        }

    def check_budget(self, memory_budget_mb, context="run"):
        value = self.current_bytes
        if value is not None and value > memory_budget_mb * 1024 ** 2:
            raise MemoryError(
                f"observed RSS exceeded memory_budget_mb during {context}: "
                f"{value / 1024 ** 2:.0f} MiB > {memory_budget_mb} MiB"
            )

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4))
            self._thread = None
        return self.snapshot()


def _physical_memory_mb() -> int:
    """Best-effort physical-memory detection without a required dependency."""
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().total // (1024 ** 2))
    except ImportError:
        pass
    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.ullTotalPhys // (1024 ** 2))
        except (AttributeError, OSError):
            pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024 ** 2))
    except (AttributeError, ValueError, OSError):
        return 8192


@dataclass(frozen=True)
class ResourceConfig:
    """User-visible execution limits.

    ``memory_budget_mb`` is a hard process budget used for planning.  When it
    is omitted, domino uses at most 85% of physical memory and caps the
    automatic budget at 56 GiB, leaving headroom on a 64 GiB workstation.
    CUDA fields are recorded here even when a CPU execution path is selected,
    which keeps command lines and metadata stable across hosts.
    """

    memory_budget_mb: Optional[int] = None
    n_jobs: int = 1
    backend: str = "cpu"
    gpu_devices: Sequence[int] = ()
    gpu_memory_budget_mb: Optional[int] = None
    scratch_dir: Optional[str] = None
    target_runtime_hours: Optional[float] = None
    safety_fraction: float = 0.90

    def resolved(self) -> "ResourceConfig":
        physical = _physical_memory_mb()
        budget = self.memory_budget_mb
        if budget is None:
            budget = max(512, min(56 * 1024, int(physical * 0.85)))
        if budget <= 0:
            raise ValueError("memory_budget_mb must be positive")
        if self.memory_budget_mb is not None and budget > physical:
            warnings.warn(
                f"memory_budget_mb={budget} exceeds detected physical memory "
                f"({physical} MiB); explicit budget retained for scheduler/container "
                "environments, but the host must actually provide this limit",
                RuntimeWarning,
            )
        jobs = (os.cpu_count() or 1) if self.n_jobs == -1 else self.n_jobs
        if jobs < 1:
            raise ValueError("n_jobs must be -1 or a positive integer")
        backend = self.backend.lower()
        if backend not in {"cpu", "cuda", "auto"}:
            raise ValueError("backend must be 'cpu', 'cuda', or 'auto'")
        if not 0.50 <= self.safety_fraction <= 0.95:
            raise ValueError("safety_fraction must be between 0.50 and 0.95")
        if self.gpu_memory_budget_mb is not None and self.gpu_memory_budget_mb <= 0:
            raise ValueError("gpu_memory_budget_mb must be positive")
        if self.target_runtime_hours is not None and self.target_runtime_hours <= 0:
            raise ValueError("target_runtime_hours must be positive")
        scratch = None if self.scratch_dir is None else str(Path(self.scratch_dir).resolve())
        return ResourceConfig(
            memory_budget_mb=int(budget),
            n_jobs=int(jobs),
            backend=backend,
            gpu_devices=tuple(int(device) for device in self.gpu_devices),
            gpu_memory_budget_mb=self.gpu_memory_budget_mb,
            scratch_dir=scratch,
            target_runtime_hours=self.target_runtime_hours,
            safety_fraction=self.safety_fraction,
        )


@dataclass(frozen=True)
class ExecutionPlan:
    """Concrete block/tile choices and their estimated peak memory."""

    memory_budget_mb: int
    variant_block_size: int
    trait_tile_size: int
    estimated_peak_mb: float
    persistent_mb: float
    working_mb: float
    decomposition_peak_mb: float
    output_scratch_gb: Optional[float]
    n_jobs: int
    backend: str
    compute_dtype: str

    def as_dict(self):
        return asdict(self)


def plan_execution(
    n_samples: int,
    n_traits: int,
    n_covariates: int,
    rank: int,
    requested_block_size: int,
    model: str = "add-dom",
    compute_dtype="float64",
    decomposition: str = "exact",
    n_variants: Optional[int] = None,
    resource_config: Optional[ResourceConfig] = None,
) -> ExecutionPlan:
    """Choose bounded variant blocks and trait tiles for one analysis group."""
    if min(n_samples, n_traits, requested_block_size) < 1:
        raise ValueError("n_samples, n_traits, and requested_block_size must be positive")
    if n_covariates < 1:
        raise ValueError("n_covariates must include at least the intercept")
    if model not in {"add-dom", "additive"}:
        raise ValueError("model must be 'add-dom' or 'additive'")
    dtype = np.dtype(compute_dtype)
    if dtype not in {np.dtype("float32"), np.dtype("float64")}:
        raise ValueError("compute_dtype must be float32 or float64")

    config = (resource_config or ResourceConfig()).resolved()
    budget = int(config.memory_budget_mb) * 1024 ** 2
    usable = int(budget * config.safety_fraction)
    n, t, p, r = n_samples, n_traits, n_covariates, min(rank, n_samples)
    item = dtype.itemsize

    # U, Y, C, variance weights, and a fixed writer/allocator reserve.
    persistent = n * r * item + n * t * 8 + n * p * 8
    persistent += r * t * 8 + (t * p * p + t * p) * 8
    reserve = max(256 * 1024 ** 2, int(0.05 * budget))

    decomp = decomposition.lower()
    if decomp not in {"exact", "randomized", "auto"}:
        raise ValueError("decomposition must be 'exact', 'randomized', or 'auto'")
    resolved_decomp = "exact" if decomp == "auto" and n <= 4000 else decomp
    if resolved_decomp == "auto":
        resolved_decomp = "randomized"
    if resolved_decomp == "exact":
        # Genome and chromosome cross-products/counts plus LAPACK workspace.
        decomposition_peak = 6 * n * n * 8 + reserve
    else:
        q = min(n, r + min(32, max(8, r // 10)))
        decomposition_peak = (3 * n * q + q * q) * item + reserve

    if decomposition_peak > usable:
        suggestion = "use decomposition='randomized' with a smaller n_components"
        raise MemoryError(
            f"{resolved_decomp} LOCO decomposition is estimated to require "
            f"{decomposition_peak / 1024 ** 2:.0f} MiB, above the usable "
            f"{usable / 1024 ** 2:.0f} MiB budget; {suggestion}"
        )

    base = persistent + reserve
    if base >= usable:
        raise MemoryError(
            "persistent eigenbasis/phenotype storage exceeds the memory budget; "
            "reduce n_components or the number of simultaneously analyzed traits"
        )

    free = usable - base
    genotype_arrays = 3 if model == "additive" else 6
    projection_arrays = 1 if model == "additive" else 2
    bytes_per_variant = n * item * genotype_arrays + r * item * projection_arrays
    metric_count = 10 if model == "additive" else 31
    bytes_per_variant += max(64, p * 16) + t * metric_count * 8
    block_allocation = int(free * 0.62)
    output_row_cap = 1_000_000
    output_limited_block = max(1, output_row_cap // t)
    block_size = max(
        1,
        min(
            requested_block_size,
            output_limited_block,
            block_allocation // max(bytes_per_variant, 1),
        ),
    )

    block_base = block_size * bytes_per_variant
    trait_free = max(1, free - block_base)
    bytes_per_tiled_trait = (
        n * 8
        + r * 8
        + block_size * (metric_count * 8 + (2 * p + 8) * 8)
        + (p * p + 4 * p) * 8
    )
    trait_tile = max(1, min(t, trait_free // max(bytes_per_tiled_trait, 1)))
    working = block_base + trait_tile * bytes_per_tiled_trait
    peak = max(base + working, decomposition_peak)
    if peak > budget:
        raise MemoryError("no association block fits within memory_budget_mb")

    scratch_gb = None
    if n_variants is not None:
        bytes_per_row = 70 if model == "add-dom" else 38
        scratch_gb = n_variants * t * bytes_per_row / 1024 ** 3
    return ExecutionPlan(
        memory_budget_mb=int(config.memory_budget_mb),
        variant_block_size=int(block_size),
        trait_tile_size=int(trait_tile),
        estimated_peak_mb=peak / 1024 ** 2,
        persistent_mb=persistent / 1024 ** 2,
        working_mb=working / 1024 ** 2,
        decomposition_peak_mb=decomposition_peak / 1024 ** 2,
        output_scratch_gb=scratch_gb,
        n_jobs=config.n_jobs,
        backend=config.backend,
        compute_dtype=dtype.name,
    )
