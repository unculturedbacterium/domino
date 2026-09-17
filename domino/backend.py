"""Optional CPU/CUDA projection backends."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np


def resolve_backend_name(requested):
    requested = requested.lower()
    if requested == "cpu":
        return "cpu"
    try:
        import cupy as cp  # type: ignore

        available = cp.cuda.runtime.getDeviceCount() > 0
    except (ImportError, RuntimeError):
        available = False
    if requested == "auto":
        return "cuda" if available else "cpu"
    if requested != "cuda":
        raise ValueError("backend must be 'cpu', 'cuda', or 'auto'")
    if not available:
        raise RuntimeError(
            "backend='cuda' requires CuPy and a visible CUDA-capable NVIDIA GPU; "
            "install domino[cuda] or use backend='cpu'"
        )
    return "cuda"


class CPUProjectionBackend:
    name = "cpu"

    def __init__(self, U, Y, C, **_):
        self.U = U
        self.Y = Y
        self.C = C

    def project(self, X):
        return self.U.T @ X, X.T @ self.Y, X.T @ self.C

    def close(self):
        return None


class CUDAProjectionBackend:
    """Replicate fixed arrays and partition marker projections over GPUs."""

    name = "cuda"

    def __init__(self, U, Y, C, devices=(), memory_budget_mb=None):
        import cupy as cp  # type: ignore

        count = cp.cuda.runtime.getDeviceCount()
        self.devices = tuple(devices) if devices else tuple(range(count))
        if not self.devices:
            raise RuntimeError("no CUDA devices are available")
        if any(device < 0 or device >= count for device in self.devices):
            raise ValueError(
                f"gpu_devices must be between 0 and {count - 1}; got {self.devices}"
            )
        self.cp = cp
        self.memory_budget_mb = memory_budget_mb
        required = np.asarray(U).nbytes + np.asarray(Y).nbytes + np.asarray(C).nbytes
        if memory_budget_mb is not None and required > memory_budget_mb * 1024 ** 2:
            raise MemoryError(
                f"CUDA persistent arrays require {required / 1024 ** 2:.0f} MiB per device, "
                f"above gpu_memory_budget_mb={memory_budget_mb}"
            )
        self.arrays = {}
        for device in self.devices:
            with cp.cuda.Device(device):
                if memory_budget_mb is not None:
                    cp.get_default_memory_pool().set_limit(
                        size=int(memory_budget_mb * 1024 ** 2)
                    )
                self.arrays[device] = (
                    cp.asarray(U),
                    cp.asarray(Y),
                    cp.asarray(C),
                )
        self.executor = (
            ThreadPoolExecutor(max_workers=len(self.devices))
            if len(self.devices) > 1
            else None
        )

    def _project_slice(self, device, X):
        cp = self.cp
        with cp.cuda.Device(device):
            gpu_U, gpu_Y, gpu_C = self.arrays[device]
            gpu_X = cp.asarray(X)
            return (
                cp.asnumpy(gpu_U.T @ gpu_X),
                cp.asnumpy(gpu_X.T @ gpu_Y),
                cp.asnumpy(gpu_X.T @ gpu_C),
            )

    def project(self, X):
        if len(self.devices) == 1:
            return self._project_slice(self.devices[0], X)
        boundaries = np.linspace(0, X.shape[1], len(self.devices) + 1, dtype=int)
        work = [
            (device, X[:, boundaries[index]:boundaries[index + 1]])
            for index, device in enumerate(self.devices)
            if boundaries[index + 1] > boundaries[index]
        ]
        futures = [
            self.executor.submit(self._project_slice, device, block)
            for device, block in work
        ]
        results = [future.result() for future in futures]
        return (
            np.concatenate([result[0] for result in results], axis=1),
            np.concatenate([result[1] for result in results], axis=0),
            np.concatenate([result[2] for result in results], axis=0),
        )

    def close(self):
        if getattr(self, "executor", None) is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
        arrays = getattr(self, "arrays", {})
        arrays.clear()
        if hasattr(self, "cp"):
            for device in getattr(self, "devices", ()):
                with self.cp.cuda.Device(device):
                    self.cp.get_default_memory_pool().free_all_blocks()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def create_projection_backend(
    requested, U, Y, C, gpu_devices=(), gpu_memory_budget_mb=None
):
    name = resolve_backend_name(requested)
    if name == "cpu":
        return CPUProjectionBackend(U, Y, C)
    return CUDAProjectionBackend(
        U,
        Y,
        C,
        devices=gpu_devices,
        memory_budget_mb=gpu_memory_budget_mb,
    )
