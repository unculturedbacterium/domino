# Scaling and resource controls

Domino plans each sample-mask group before allocating a LOCO matrix or marker
block. `memory_budget_mb` is the maximum process budget, and `block_size` is a
ceiling. The planner can reduce marker blocks and trait tiles to fit the
budget.

## Conservative defaults

The defaults are:

```text
trait_mode        independent
variance_estimator auto -> profile REML
decomposition     exact
compute_dtype     float64
backend           cpu
n_jobs            1
```

Multivariate mode changes `variance_estimator=auto` to SCORE. Randomized
decomposition, SCORE in independent mode, float32, and CUDA require an
explicit option or large-sample `auto` decomposition request.

## A 64 GiB workstation

On an otherwise idle 64 GiB host, `--memory-budget-mb 57344` reserves about
56 GiB for Domino and leaves roughly 8 GiB for the operating system, file
cache, and other processes. Reduce this budget when the machine is shared.

The planner estimates persistent eigenbasis and trait storage, exact or
randomized decomposition workspaces, decoded genotype blocks, additive and
dominance projections, fixed-effect cross-products, output conversion, and a
writer/allocator reserve. It also checks sampled process RSS during output.

An infeasible exact decomposition raises `MemoryError` before the known dense
workspace is allocated. The exception recommends randomized decomposition and
a lower retained rank.

## Exact versus randomized LOCO

Exact mode stores dense sample-by-sample matrices and uses deterministic
symmetric eigendecomposition. Its memory grows as `O(n^2)` and decomposition
time as `O(n^3)`.

Randomized mode stores `O(nq)` retained eigenvectors for rank `q`, plus marker
blocks and workspaces. It streams genotype matrix products and uses a bulk
eigenvalue for the omitted subspace. Adaptive rank selection can increase `q`
when probe error exceeds the tolerance.

Use a quality-controlled LD-pruned `grm_bfile` to reduce GRM construction cost
without reducing the tested association marker set. Cache accepted LOCO
eigensystems on fast local storage and reuse them only when input signatures
and analysis settings match.

## Trait and output scaling

Independent traits with the same sample mask share genotype projections. The
planner selects a trait tile size so arrays remain within budget. Different
missingness masks form separate groups and cannot share all LOCO work.

Full long-format output contains one row per variant and trait, plus one joint
row per variant in multivariate mode. Ten million variants by 1,000 traits is
ten billion per-trait rows and can require hundreds of gigabytes even when
working memory is bounded. Use `--stream`, select phenotypes deliberately, and
plan filesystem capacity and write throughput before computation.

## Threads

`n_jobs` controls BLAS thread limits. It does not launch one Python process per
chromosome. More threads can speed dense linear algebra, but oversubscription
can increase runtime and memory pressure. Benchmark representative data with
the same BLAS implementation used for production.

## CUDA

CUDA mode offloads association projection matrix products through CuPy.
Multi-GPU mode partitions marker columns while replicating the fixed
eigenbasis and trait arrays on each selected device. `gpu_memory_budget_mb`
sets the CuPy memory-pool ceiling per selected GPU.

Randomized GRM construction, file decoding, variance-component fitting, and
Parquet output remain CPU or I/O work. Measure end-to-end runtime rather than
assuming projection speedup translates directly to total speedup.

## Execution records

The resolved plan in `{out}.execution.json` includes estimated peak memory,
block size, trait tile size, approximation diagnostics, stage timings, cache
status, and sampled peak RSS. A request is not the same as a resolved plan;
archive the sidecar with every scientific analysis.

## Large-scale qualification

Before a 50,000-sample production run:

1. Run input QC and create the intended relationship panel.
2. Benchmark one representative chromosome and trait set.
3. Confirm randomized operator error and rank stability on every chromosome.
4. Compare a feasible subset against exact float64 output.
5. Run null calibration and causal-effect simulations matching sample size,
   LD, MAF, missingness, relatedness, and covariates.
6. Measure output size and sustained write speed.
7. Record hardware, BLAS, Python, CuPy/CUDA, and Domino versions.

The planner can keep allocations under a stated budget, but it cannot promise
a host-independent completion time or scientific accuracy for an unvalidated
approximation rank.
