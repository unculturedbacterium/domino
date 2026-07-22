"""Additive genomic relatedness matrices with LOCO decomposition.

Exact mode uses a pairwise-missingness-aware GRM definition, while
yielding one chromosome eigensystem at a time.  Randomized mode never forms an
``n x n`` GRM: it applies the mean-imputed standardized genotype operator in
blocks, estimates a retained eigenspace, and represents omitted directions by
a trace-matched bulk eigenvalue.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
from scipy.linalg import eigh
from sklearn.utils.extmath import randomized_svd

from ._utils import standardize_block

CACHE_FORMAT_VERSION = "domino-loco-v1"


def _eig(G, n_components=None, decomposition="exact", random_state=0):
    """Symmetric eigendecomposition ordered from largest to smallest."""
    G = np.asarray(G)
    n = G.shape[0]
    rank = n if n_components is None else min(int(n_components), n)
    if decomposition == "exact":
        if rank >= n:
            s, U = np.linalg.eigh(G)
        else:
            s, U = eigh(G, subset_by_index=(n - rank, n - 1), check_finite=False)
        order = np.argsort(s)[::-1]
        return U[:, order], np.abs(s[order])
    if decomposition != "randomized":
        raise ValueError("decomposition must be 'exact' or 'randomized'")
    U, s, _ = randomized_svd(
        G, n_components=rank, random_state=random_state, flip_sign=True
    )
    return U, np.abs(s)


def _resolve_decomposition(decomposition, n, n_components):
    value = decomposition.lower()
    if value not in {"auto", "exact", "randomized"}:
        raise ValueError("decomposition must be 'auto', 'exact', or 'randomized'")
    if value == "auto":
        value = "exact" if n <= 4000 else "randomized"
    if n_components is None:
        n_components = n if value == "exact" else min(2000, max(n - 1, 1))
    rank = min(int(n_components), n)
    if rank < 1:
        raise ValueError("n_components must be positive")
    return value, rank


def _cache_directory(grm_cache, reader):
    if not grm_cache:
        return None
    if grm_cache is True:
        return Path(reader.prefix).resolve().parent / ".domino_grm_cache"
    return Path(grm_cache).resolve()


def _cache_key(
    reader,
    sample_index,
    decomposition,
    rank,
    dtype,
    random_state,
    n_iter,
    adaptive_rank,
    max_adaptive_components,
):
    digest = hashlib.sha256()
    digest.update(CACHE_FORMAT_VERSION.encode("ascii"))
    normalization = (
        "pairwise_observed_standardized" if decomposition == "exact"
        else "mean_imputed_standardized_bulk"
    )
    digest.update(normalization.encode("ascii"))
    for extension in ("bed", "bim", "fam"):
        path = Path(f"{reader.prefix}.{extension}").resolve()
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    digest.update(np.asarray(sample_index, dtype=np.int64).tobytes())
    digest.update(
        f"{decomposition}:{rank}:{np.dtype(dtype).name}:{random_state}:{n_iter}:"
        f"{adaptive_rank}:{max_adaptive_components}".encode("ascii")
    )
    return digest.hexdigest()[:24]


def _cache_paths(cache_dir, key, chrom):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(chrom))
    stem = cache_dir / f"{key}.chr{safe}"
    return Path(f"{stem}.U.npy"), Path(f"{stem}.s.npy"), Path(f"{stem}.json")


def _atomic_save_npy(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def _atomic_save_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_cached(cache_dir, key, chrom):
    if cache_dir is None:
        return None
    upath, spath, mpath = _cache_paths(cache_dir, key, chrom)
    if not (upath.exists() and spath.exists() and mpath.exists()):
        return None
    metadata = json.loads(mpath.read_text(encoding="utf-8"))
    return {
        "U": np.load(upath, mmap_mode="r", allow_pickle=False),
        "s": np.load(spath, mmap_mode="r", allow_pickle=False),
        "residual_eigenvalue": float(metadata.get("residual_eigenvalue", 0.0)),
        "diagnostics": metadata.get("diagnostics", {}),
        "cache_hit": True,
        "cache_key": metadata.get("cache_key"),
        "decomposition": metadata["decomposition"],
    }


def _save_cached(cache_dir, key, chrom, result):
    if cache_dir is None:
        return result
    cache_dir.mkdir(parents=True, exist_ok=True)
    upath, spath, mpath = _cache_paths(cache_dir, key, chrom)
    _atomic_save_npy(upath, np.asarray(result["U"]))
    _atomic_save_npy(spath, np.asarray(result["s"]))
    _atomic_save_json(
        mpath,
        {
            "chrom": str(chrom),
            "cache_format_version": CACHE_FORMAT_VERSION,
            "cache_key": result.get("cache_key", key),
            "decomposition": result["decomposition"],
            "residual_eigenvalue": float(result.get("residual_eigenvalue", 0.0)),
            "diagnostics": result.get("diagnostics", {}),
        },
    )
    loaded = _load_cached(cache_dir, key, chrom)
    if loaded is None:
        return result
    loaded["cache_hit"] = False
    return loaded


def _chromosome_seed(random_state, chrom):
    digest = hashlib.sha256(str(chrom).encode("utf-8")).digest()
    return int((int(random_state) + int.from_bytes(digest[:4], "little")) % (2 ** 32))


def _iter_excluding_chrom(reader, chrom, sample_index, block_size, dtype):
    for bim_slice, genotype in reader.iter_blocks(block_size, sample_index=sample_index):
        keep = bim_slice["chrom"].to_numpy() != str(chrom)
        if not np.any(keep):
            continue
        z, _, _ = standardize_block(genotype[:, keep], dtype=dtype)
        yield z


def _apply_streamed_grm(reader, chrom, sample_index, block_size, X, dtype):
    result = np.zeros((len(sample_index), X.shape[1]), dtype=dtype)
    count = 0
    for z in _iter_excluding_chrom(reader, chrom, sample_index, block_size, dtype):
        result += z @ (z.T @ X)
        count += z.shape[1]
    if count == 0:
        raise ValueError(f"no variants remain after leaving out chromosome {chrom}")
    result /= count
    return result, count


def _randomized_loco(
    reader,
    chrom,
    sample_index,
    block_size,
    rank,
    dtype,
    random_state,
    n_iter,
    operator_error_tolerance,
):
    """Randomized eigensystem from streamed GRM operator products."""
    n = len(sample_index)
    oversample = min(32, max(8, rank // 10), max(n - rank, 0))
    q = min(n, rank + oversample)
    rng = np.random.default_rng(random_state)
    omega = rng.standard_normal((n, q)).astype(dtype, copy=False)
    projected, count = _apply_streamed_grm(
        reader, chrom, sample_index, block_size, omega, dtype
    )
    Q, _ = np.linalg.qr(projected, mode="reduced")
    del omega, projected
    for _ in range(max(int(n_iter), 0)):
        projected, _ = _apply_streamed_grm(
            reader, chrom, sample_index, block_size, Q, dtype
        )
        Q, _ = np.linalg.qr(projected, mode="reduced")
        del projected

    B = np.zeros((q, q), dtype=np.float64)
    trace = 0.0
    observed_count = 0
    for z in _iter_excluding_chrom(reader, chrom, sample_index, block_size, dtype):
        ztq = z.T @ Q
        B += np.asarray(ztq.T @ ztq, dtype=np.float64)
        trace += float(np.sum(z * z, dtype=np.float64))
        observed_count += z.shape[1]
    B /= observed_count
    trace /= observed_count
    values, vectors = np.linalg.eigh(B)
    order = np.argsort(values)[::-1][:rank]
    s = np.maximum(values[order], 0.0)
    U = np.asarray(Q @ vectors[:, order], dtype=dtype)
    omitted = max(n - rank, 0)
    residual = max((trace - float(s.sum())) / omitted, 0.0) if omitted else 0.0

    probes = min(8, n)
    probe = rng.standard_normal((n, probes)).astype(dtype, copy=False)
    exact_probe, _ = _apply_streamed_grm(
        reader, chrom, sample_index, block_size, probe, dtype
    )
    utp = U.T @ probe
    approx_probe = U @ ((s - residual)[:, None] * utp) + residual * probe
    denominator = max(float(np.linalg.norm(exact_probe)), np.finfo(float).tiny)
    operator_error = float(np.linalg.norm(exact_probe - approx_probe) / denominator)
    diagnostics = {
        "operator_relative_error": operator_error,
        "operator_error_tolerance": operator_error_tolerance,
        "rank": int(rank),
        "oversampling": int(q - rank),
        "power_iterations": int(n_iter),
        "variants_in_loco_grm": int(count),
        "trace": float(trace),
        "retained_trace_fraction": float(s.sum() / trace) if trace > 0 else 0.0,
    }
    return {
        "U": U,
        "s": s,
        "residual_eigenvalue": residual,
        "diagnostics": diagnostics,
        "cache_hit": False,
        "decomposition": "randomized",
    }


def iter_loco_grms(
    reader,
    sample_index=None,
    block_size=8192,
    n_components=None,
    chroms=None,
    decomposition="exact",
    compute_dtype="float64",
    grm_cache=None,
    random_state=0,
    randomized_power_iterations=1,
    operator_error_tolerance=0.02,
    adaptive_randomized_rank=True,
    randomized_max_components=None,
    verbose=True,
):
    """Yield ``(chromosome, eigensystem)`` without retaining all chromosomes."""
    dtype = np.dtype(compute_dtype)
    if dtype not in {np.dtype("float32"), np.dtype("float64")}:
        raise ValueError("compute_dtype must be float32 or float64")
    sidx = (
        np.arange(reader.n_samples, dtype=np.int64)
        if sample_index is None
        else np.asarray(sample_index, dtype=np.int64)
    )
    n = len(sidx)
    method, rank = _resolve_decomposition(decomposition, n, n_components)
    chrom_arr = reader.bim["chrom"].to_numpy()
    run_chroms = (
        [str(chrom) for chrom in chroms]
        if chroms is not None
        else list(dict.fromkeys(chrom_arr))
    )
    cache_dir = _cache_directory(grm_cache, reader)
    if randomized_max_components is None:
        randomized_max_components = min(n, max(rank, 4 * rank))
    else:
        randomized_max_components = min(n, int(randomized_max_components))
    if randomized_max_components < rank:
        raise ValueError("randomized_max_components must be at least n_components")
    key = _cache_key(
        reader,
        sidx,
        method,
        rank,
        dtype,
        random_state,
        randomized_power_iterations,
        adaptive_randomized_rank,
        randomized_max_components,
    )
    cached = {chrom: _load_cached(cache_dir, key, chrom) for chrom in run_chroms}

    if method == "randomized":
        for chrom in run_chroms:
            result = cached[chrom]
            if result is None:
                attempted_ranks = []
                attempted_errors = []
                attempted_rank = rank
                while True:
                    result = _randomized_loco(
                        reader,
                        chrom,
                        sidx,
                        block_size,
                        attempted_rank,
                        dtype,
                        _chromosome_seed(random_state, chrom),
                        randomized_power_iterations,
                        operator_error_tolerance,
                    )
                    error = result["diagnostics"]["operator_relative_error"]
                    attempted_ranks.append(int(attempted_rank))
                    attempted_errors.append(float(error))
                    if operator_error_tolerance is None or error <= operator_error_tolerance:
                        break
                    next_rank = min(randomized_max_components, n, attempted_rank * 2)
                    if not adaptive_randomized_rank or next_rank <= attempted_rank:
                        raise RuntimeError(
                            f"randomized LOCO operator error for chromosome {chrom} is "
                            f"{error:.4f}, above tolerance {operator_error_tolerance:.4f}; "
                            f"attempted ranks {attempted_ranks}"
                        )
                    attempted_rank = next_rank
                result["diagnostics"]["attempted_ranks"] = attempted_ranks
                result["diagnostics"]["attempted_operator_errors"] = attempted_errors
                result["cache_key"] = key
                result = _save_cached(cache_dir, key, chrom, result)
            error = result.get("diagnostics", {}).get("operator_relative_error")
            if (
                operator_error_tolerance is not None
                and error is not None
                and error > operator_error_tolerance
            ):
                raise RuntimeError(
                    f"cached randomized LOCO operator error for chromosome {chrom} is "
                    f"{error:.4f}, above tolerance {operator_error_tolerance:.4f}; "
                    "increase n_components or clear the incompatible cache entry"
                )
            if verbose:
                diagnostic = result.get("diagnostics", {})
                error = diagnostic.get("operator_relative_error")
                suffix = "cache hit" if result.get("cache_hit") else f"operator error {error:.4f}"
                print(f"  LOCO GRM chr {chrom}: rank {len(result['s'])}; {suffix}")
            yield chrom, result
        return

    missing = [chrom for chrom in run_chroms if cached[chrom] is None]
    zzt_all = w_all = None
    if missing:
        zzt_all = np.zeros((n, n), dtype=dtype)
        w_all = np.zeros((n, n), dtype=dtype)
        for _, genotype in reader.iter_blocks(block_size, sample_index=sidx):
            z, _, mask = standardize_block(genotype, dtype=dtype)
            zzt_all += z @ z.T
            w_all += mask @ mask.T

    for chrom in run_chroms:
        result = cached[chrom]
        idx_chrom = np.where(chrom_arr == chrom)[0]
        if result is None:
            zzt_chrom = np.zeros((n, n), dtype=dtype)
            w_chrom = np.zeros((n, n), dtype=dtype)
            for _, genotype in reader.iter_blocks(
                block_size, variant_index=idx_chrom, sample_index=sidx
            ):
                z, _, mask = standardize_block(genotype, dtype=dtype)
                zzt_chrom += z @ z.T
                w_chrom += mask @ mask.T
            np.subtract(zzt_all, zzt_chrom, out=zzt_chrom)
            np.subtract(w_all, w_chrom, out=w_chrom)
            np.maximum(w_chrom, 1.0, out=w_chrom)
            np.divide(zzt_chrom, w_chrom, out=zzt_chrom)
            U, s = _eig(zzt_chrom, n_components=rank, decomposition="exact")
            result = {
                "U": U,
                "s": s,
                "residual_eigenvalue": 0.0,
                "diagnostics": {"rank": int(rank), "variants_left_out": int(len(idx_chrom))},
                "cache_hit": False,
                "decomposition": "exact",
                "cache_key": key,
            }
            result = _save_cached(cache_dir, key, chrom, result)
        if verbose:
            suffix = "; cache hit" if result.get("cache_hit") else ""
            print(
                f"  LOCO GRM chr {chrom}: {len(idx_chrom)} variants left out, "
                f"rank {len(result['s'])}{suffix}"
            )
        yield chrom, result


def compute_loco_grms(reader, **kwargs):
    """Compatibility wrapper returning all LOCO decompositions in a dict."""
    return {chrom: result for chrom, result in iter_loco_grms(reader, **kwargs)}


def compute_grm(reader, sample_index=None, block_size=8192, compute_dtype="float64"):
    """Genome-wide additive GRM (dense, pairwise missingness aware)."""
    sidx = (
        np.arange(reader.n_samples, dtype=np.int64)
        if sample_index is None
        else np.asarray(sample_index, dtype=np.int64)
    )
    dtype = np.dtype(compute_dtype)
    n = len(sidx)
    zzt = np.zeros((n, n), dtype=dtype)
    counts = np.zeros((n, n), dtype=dtype)
    for _, genotype in reader.iter_blocks(block_size, sample_index=sidx):
        z, _, mask = standardize_block(genotype, dtype=dtype)
        zzt += z @ z.T
        counts += mask @ mask.T
    return zzt / np.maximum(counts, 1.0)
