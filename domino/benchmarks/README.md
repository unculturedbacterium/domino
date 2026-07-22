# Benchmarks

`benchmark_current.py` generates or reuses a synthetic PLINK fileset, creates
fixed-seed null traits, runs the selected Domino source tree, samples process
RSS every 10 ms, and writes a JSON record.

From the repository root:

```bash
python benchmarks/benchmark_current.py \
  --package-root . \
  --prefix benchmarks/data/synthetic_1k_10k \
  --traits trait_01,trait_02,trait_03,trait_04 \
  --threads 1 \
  --generate \
  --output benchmarks/results/current_4traits.json
```

Add `--stream` to benchmark incremental Parquet output. Add
`--grm-cache benchmarks/cache/loco` for cache tests. The generated genotype,
cache, scan, and result JSON files are ignored by Git; commit only reviewed
aggregate summaries intended for release documentation.

The checked-in aggregate measurement table is
`results/optimization_benchmark_summary.csv`. Benchmark methods and
limitations are documented in `../docs/OPTIMIZATION_BENCHMARK.md`.
