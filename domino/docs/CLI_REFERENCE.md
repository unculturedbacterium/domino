# Command-line reference

Run `domino --help` to inspect the options installed with the current source.
The CLI accepts a tab-separated phenotype/covariate table and a PLINK 1 binary
fileset.

## Required arguments

| Argument | Description |
|---|---|
| `--bfile PREFIX` | Association genotype prefix for `PREFIX.bed/.bim/.fam` |
| `--pheno FILE` | Tab-separated phenotype and covariate table |
| `--traits A,B` | Comma-separated quantitative phenotype columns |
| `--out PREFIX` | Output prefix, including the desired directory |

## Samples, variants, and fixed effects

| Argument | Default | Description |
|---|---|---|
| `--id-column NAME` | `iid` | Sample-ID column in the phenotype table |
| `--covariates A,B` | None | Comma-separated numeric covariates |
| `--categorical-covariates A,B` | None | Categorical covariates to one-hot encode |
| `--chromosomes A,B` | All | Chromosome labels to scan |
| `--grm-bfile PREFIX` | Association fileset | Separate relationship-panel PLINK prefix |
| `--min-genotype-count N` | `5` | Required observed count in each AA, AB, and BB cell |
| `--complete-case-across-traits` | Off | Use one complete-case mask for every trait |

## Statistical model

| Argument | Default | Description |
|---|---|---|
| `--model {add-dom,additive}` | `add-dom` | Joint additive-dominance or additive-only scan |
| `--trait-mode {independent,multivariate}` | `independent` | Independent per-trait or multivariate SCORE mode |
| `--variance-estimator METHOD` | `auto` | `auto`, `reml`, `ml`, `profile_reml`, `profile_ml`, or `score` |

`auto` selects profile REML in independent mode and SCORE in multivariate
mode. Multivariate mode requires SCORE and complete cases across the requested
trait panel.

## LOCO decomposition

| Argument | Default | Description |
|---|---|---|
| `--decomposition {exact,randomized,auto}` | `exact` | LOCO eigensystem method |
| `--n-components N` | Planner choice | Initial randomized rank |
| `--randomized-power-iterations N` | `1` | Randomized subspace power iterations |
| `--randomized-max-components N` | Adaptive cap | Maximum retained randomized rank |
| `--no-adaptive-rank` | Off | Disable deterministic rank doubling |
| `--operator-error-tolerance X` | `0.02` | Maximum accepted relative probe error |
| `--no-operator-error-check` | Off | Disable the probe-error acceptance check |
| `--grm-cache DIR` | None | Reusable memory-mapped LOCO eigensystem cache |

With `--decomposition auto`, analyses of at most 4,000 samples select exact
decomposition and larger analyses select randomized decomposition. Explicit
`exact` always requests the dense exact path and fails during planning when it
cannot fit the memory budget.

## Runtime and memory

| Argument | Default | Description |
|---|---|---|
| `--block-size N` | `8192` | Maximum marker block; the planner may reduce it |
| `--memory-budget-mb N` | 85% of physical memory | Hard process-memory planning budget |
| `--n-jobs N` | `1` | BLAS thread limit; `-1` uses all logical CPUs |
| `--compute-dtype {float64,float32}` | `float64` | Association and randomized-work precision |
| `--scratch-dir DIR` | None | Scratch path recorded in the execution plan |
| `--target-runtime-hours X` | None | Runtime target recorded and evaluated in metadata |

## CPU and CUDA backend

| Argument | Default | Description |
|---|---|---|
| `--backend {cpu,cuda,auto}` | `cpu` | Association projection backend |
| `--gpu-devices A,B` | Visible devices | Zero-based CUDA device IDs |
| `--gpu-memory-budget-mb N` | CuPy default | CuPy pool ceiling per selected GPU |

`cuda` fails if CuPy or a compatible device is unavailable. `auto` falls back
to CPU. CUDA accelerates association projections, not every pipeline stage.

## Output behavior

| Argument | Default | Description |
|---|---|---|
| `--stream` | Off | Write Parquet row groups without retaining all result rows |
| `--version` | | Print the installed Domino package version |

For production scans, use `--stream`. Domino always writes per-chromosome
Parquet files plus execution and CLI metadata sidecars when an output prefix
is supplied. Without `--stream`, the CLI also writes a combined
`.all.parquet` file.

## Minimal command

```bash
domino \
  --bfile data/study \
  --pheno data/phenotypes.tsv \
  --traits trait_1 \
  --out results/trait_1
```

## Recommended covariate-adjusted command

```bash
domino \
  --bfile data/study \
  --grm-bfile data/study_ld_pruned \
  --pheno data/phenotypes.tsv \
  --id-column iid \
  --traits trait_1 \
  --covariates age,PC1,PC2,PC3,PC4,PC5 \
  --categorical-covariates sex,batch \
  --model add-dom \
  --min-genotype-count 5 \
  --memory-budget-mb 57344 \
  --n-jobs 16 \
  --stream \
  --out results/trait_1
```
