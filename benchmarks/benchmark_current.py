"""Reproducible Domino runtime and peak-RSS benchmark."""
from __future__ import annotations

import argparse
import gc
import inspect
import json
from pathlib import Path
import sys
import threading
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--traits", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--grm-cache", type=Path)
    parser.add_argument("--memory-budget-mb", type=int)
    return parser.parse_args()


args = parse_args()
sys.path.insert(0, str(args.package_root.resolve()))

import psutil  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import domino  # noqa: E402
from domino.testutils import simulate  # noqa: E402


if args.generate or not Path(f"{args.prefix}.bed").exists():
    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    simulate(
        str(args.prefix),
        n_fam=250,
        sibs=4,
        n_chrom=5,
        snps_per_chrom=2000,
        seed=1234,
    )

phenotype = simulate.__module__  # Keep the selected package import explicit in metadata.
import pandas as pd  # noqa: E402

fam = domino.read_fam(str(args.prefix))
rng_seed = 8712
import numpy as np  # noqa: E402

rng = np.random.default_rng(rng_seed)
trait_names = [name.strip() for name in args.traits.split(",") if name.strip()]
trait_frame = pd.DataFrame(
    rng.normal(size=(len(fam), len(trait_names))),
    index=fam["iid"].astype(str),
    columns=trait_names,
)

process = psutil.Process()
gc.collect()
baseline_rss = process.memory_info().rss
peak_rss = [baseline_rss]
stop = threading.Event()


def sample_memory():
    while not stop.wait(0.01):
        peak_rss[0] = max(peak_rss[0], process.memory_info().rss)


sampler = threading.Thread(target=sample_memory, daemon=True)
sampler.start()
kwargs = {
    "model": "add-dom",
    "block_size": 1024,
    "return_results": not args.stream,
    "verbose": False,
}
signature = inspect.signature(domino.run_gwas)
if "decomposition" in signature.parameters:
    kwargs.update(
        {
            "decomposition": "exact",
            "compute_dtype": "float64",
            "variance_estimator": "profile_reml",
            "n_jobs": args.threads,
        }
    )
    if args.grm_cache is not None:
        kwargs["grm_cache"] = str(args.grm_cache)
    if args.memory_budget_mb is not None:
        kwargs["memory_budget_mb"] = args.memory_budget_mb
if args.stream:
    kwargs["out"] = str(args.output.with_suffix("")) + ".scan"

started = time.perf_counter()
with threadpool_limits(limits=args.threads):
    result = domino.run_gwas(str(args.prefix), trait_frame, **kwargs)
elapsed = time.perf_counter() - started
peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
stop.set()
sampler.join()

record = {
    "software": "Domino",
    "domino_version": domino.__version__,
    "package_root": str(args.package_root),
    "python": sys.version,
    "n_samples": len(fam),
    "n_variants": domino.PlinkReader(str(args.prefix)).n_variants,
    "n_traits": len(trait_names),
    "traits": trait_names,
    "threads": args.threads,
    "runtime_s": elapsed,
    "baseline_rss_mb": baseline_rss / 1024 ** 2,
    "peak_rss_mb": peak_rss[0] / 1024 ** 2,
    "peak_increment_mb": (peak_rss[0] - baseline_rss) / 1024 ** 2,
    "output_rows_in_memory": len(result),
    "streamed": args.stream,
    "grm_cache": None if args.grm_cache is None else str(args.grm_cache),
    "phenotype_seed": rng_seed,
    "phenotype_source_module": phenotype,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))
