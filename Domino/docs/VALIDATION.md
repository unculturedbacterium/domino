# Validation scope

Domino includes deterministic unit and integration tests for PLINK decoding,
sample alignment, covariate handling, missingness groups, exact GLS algebra,
genotype-cell filtering, duplicate variant identifiers, inheritance labels,
resource planning, streamed atomic output, cache reuse, adaptive randomized
rank, SCORE estimation, and multivariate output.

The exact shared-eigenbasis path has been compared with direct GLS and with a
frozen pre-optimization implementation. Current optimization benchmark results
are summarized in [OPTIMIZATION_BENCHMARK.md](OPTIMIZATION_BENCHMARK.md).

Run release checks with:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m ruff check .
python release_check.py
python examples/quickstart.py
```

## What the included checks do not establish

The repository tests do not independently prove calibration for every sample
size, ancestry, relatedness structure, trait architecture, approximation rank,
GPU, or missingness process. A production study should additionally run null
simulations, architecture-specific power simulations, exact-versus-randomized
subset comparisons, external-tool comparisons where hypotheses match, and
replication on an independent phenotype or cohort.

The 50,000-sample values in the benchmark report are planning estimates. A
measured scale test on the target host remains necessary before making runtime
or memory claims at that scale.
