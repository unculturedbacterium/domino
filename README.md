# Domino

Domino is a Python pipeline for additive and dominance genome-wide association
analysis of quantitative traits. It reads PLINK 1 binary genotype files,
constructs additive leave-one-chromosome-out (LOCO) relationship models,
estimates variance components, and tests additive and conditional dominance
effects with generalized least squares (GLS). The implementation is designed
for reproducible single-trait and multi-trait analyses with explicit memory
budgets, streamed output, reusable LOCO decompositions, and detailed execution
metadata.

The GitHub repository, installed Python package, import namespace, and
command-line program are all named **Domino**:

```python
import domino
print(domino.__version__)
```

## What Domino provides

- Block-wise reading of PLINK 1 `.bed`, `.bim`, and `.fam` files.
- Additive-only and joint additive-dominance marker models.
- Additive LOCO mixed-model correction with profile REML by default.
- Independent analysis of many traits in shared, vectorized LOCO eigenbases.
- An opt-in multivariate SCORE mode with per-trait and joint Wald tests.
- Numeric covariates and one-hot-encoded categorical covariates.
- Per-trait phenotype missingness or complete-case multi-trait analysis.
- Hard process-memory planning that adjusts marker blocks and trait tiles.
- Incremental, atomic Parquet output that does not retain a full scan in RAM.
- Exact or adaptive randomized LOCO decomposition with error diagnostics.
- Memory-mapped LOCO cache reuse across traits and repeated analyses.
- Optional CuPy projection offload to one or more NVIDIA GPUs.
- Signed `d/a` estimates and inheritance-mode classifications.
- Runtime, peak-RSS, cache, approximation, and resource-plan sidecars.

Domino does not require PLINK, GCTA, R, or a project-specific compiled
extension for association testing. PLINK remains useful for upstream genotype
quality control, ancestry estimation, and construction of an LD-pruned
relationship panel.

## Requirements

- Python 3.9 or newer.
- A PLINK 1 binary fileset with matching `.bed`, `.bim`, and `.fam` files.
- A tab-separated phenotype file for command-line analyses.
- Enough storage for Parquet output. Multi-trait output can be much larger than
  the working-memory requirement.

Core dependencies are NumPy, pandas, SciPy, scikit-learn, PyArrow,
threadpoolctl, and psutil. They are installed automatically by `pip`.

## Installation

### Install from GitHub

```bash
git clone https://github.com/unculturedbacterium/Domino.git
cd Domino
python -m pip install .
```

### Install directly with pip

Install the repository without cloning it manually:

```bash
python -m pip install "git+https://github.com/unculturedbacterium/Domino.git"
```

Upgrade a previous installation from the repository:

```bash
python -m pip install --upgrade --force-reinstall \
  "git+https://github.com/unculturedbacterium/Domino.git"
```

For an editable development installation with tests and linting tools:

```bash
python -m pip install -e ".[test,plot]"
python -m pytest -q
python -m ruff check .
```

The supplied Conda environment is another reproducible option:

```bash
conda env create -f environment.yml
conda activate domino
python -m pip install -e .
```

Optional CUDA support requires a compatible NVIDIA driver and CUDA 12 CuPy
wheel:

```bash
python -m pip install ".[cuda]"
```

Verify the command-line installation:

```bash
domino --version
domino --help
python -c "import domino; print(domino.__version__)"
```

## Input files

### PLINK genotypes

Pass the common PLINK prefix without an extension. For the following files:

```text
data/study.bed
data/study.bim
data/study.fam
```

use `--bfile data/study`. Sample IDs are taken from the IID column, which is
the second column of the `.fam` file. IIDs must identify samples unambiguously.
Variant order, chromosome, identifier, base-pair position, and alleles are
read from the `.bim` file.

### Phenotypes and covariates

The CLI expects a tab-separated text file with one header row and an IID
column. Column names supplied to `--traits`, `--covariates`, and
`--categorical-covariates` must exactly match the header.

```text
iid	trait_1	trait_2	sex	batch	PC1	PC2
sample_001	0.41	-0.22	F	B1	-0.018	0.006
sample_002	-1.07	0.35	M	B1	0.011	-0.004
```

Numeric covariates are parsed as numbers. Categorical covariates are encoded
with one reference level. An intercept is always included, and redundant
fixed-effect columns are removed. Samples are aligned by IID, not by input row
position. Missing phenotype or covariate values remove that sample from the
corresponding analysis group.

Domino supports quantitative phenotypes only. Binary covariates are
supported, but binary case-control phenotypes are not.

## Quick start

Run an additive and conditional-dominance scan with numeric and categorical
covariates:

```bash
domino \
  --bfile data/study \
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

For Windows PowerShell, place the command on one line or replace each `\` with
a PowerShell backtick.

`--stream` is recommended for genome-scale scans. It writes one compressed
Parquet file per chromosome and does not retain all association rows in a
pandas DataFrame. The example produces:

```text
results/trait_1.chr1.parquet
results/trait_1.chr2.parquet
...
results/trait_1.metadata.json
results/trait_1.execution.json
```

Without `--stream`, the CLI also writes `results/trait_1.all.parquet` and
retains the returned rows until the combined file has been serialized.

The repository includes a complete synthetic example that requires no private
data or external GWAS executable:

```bash
python examples/quickstart.py
```

The complete argument reference is in
[docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md). The four required analysis
arguments are `--bfile`, `--pheno`, `--traits`, and `--out`.

## Python API

Phenotype and covariate DataFrames should be indexed by IID:

```python
import pandas as pd
import domino

data = pd.read_csv("data/phenotypes.tsv", sep="\t", dtype={"iid": "string"})
data["iid"] = data["iid"].str.strip()
data = data.set_index("iid")

results = domino.run_gwas(
    "data/study",
    data[["trait_1"]],
    traits=["trait_1"],
    covar=data[["sex", "batch", "PC1", "PC2", "PC3"]],
    model="add-dom",
    min_genotype_count=5,
    memory_budget_mb=57344,
    n_jobs=16,
    out="results/trait_1",
)
```

For streamed library output, set `return_results=False` and provide `out`:

```python
domino.run_gwas(
    "data/study",
    data[["trait_1"]],
    covar=data[["sex", "batch", "PC1", "PC2", "PC3"]],
    return_results=False,
    out="results/trait_1",
)
```

## Multiple traits

Independent mode is the default. Traits with the same sample and covariate
mask share a LOCO eigenbasis and are evaluated in trait tiles:

```bash
domino \
  --bfile data/study \
  --pheno data/phenotypes.tsv \
  --traits trait_1,trait_2,trait_3 \
  --covariates PC1,PC2,PC3,PC4,PC5 \
  --categorical-covariates sex,batch \
  --trait-mode independent \
  --variance-estimator auto \
  --stream \
  --out results/independent_traits
```

`variance-estimator=auto` resolves to profile REML in independent mode.
Different missingness masks form separate analysis groups so that each group
retains the correct sample-specific LOCO model.

Multivariate mode uses a covariate-adjusted SCORE variance estimator and a
complete-case sample set across requested traits. It emits each recovered
trait plus one `__joint__` row per variant:

```bash
domino \
  --bfile data/study \
  --pheno data/phenotypes.tsv \
  --traits trait_1,trait_2,trait_3 \
  --covariates PC1,PC2,PC3,PC4,PC5 \
  --trait-mode multivariate \
  --variance-estimator auto \
  --complete-case-across-traits \
  --stream \
  --out results/multivariate_traits
```

Multivariate covariance storage grows quadratically with the number of traits.
Review the resource plan and validate calibration for the selected phenotype
panel before interpreting joint tests.

## Large analyses and memory control

`memory_budget_mb` is a hard process budget used during planning and checked
while output is written. The requested `block_size` is a ceiling. Domino can
reduce the marker block and trait tile to fit the budget, but it will reject an
analysis before a known oversized exact decomposition is allocated.

On an otherwise idle 64 GiB workstation, a 56 GiB process budget leaves room
for the operating system and file cache. A scalable configuration for large
sample panels is:

```bash
domino \
  --bfile data/study \
  --grm-bfile data/study_ld_pruned \
  --pheno data/phenotypes.tsv \
  --traits trait_1,trait_2 \
  --covariates PC1,PC2,PC3,PC4,PC5 \
  --model add-dom \
  --variance-estimator score \
  --decomposition randomized \
  --n-components 1000 \
  --randomized-max-components 4000 \
  --operator-error-tolerance 0.02 \
  --compute-dtype float32 \
  --memory-budget-mb 57344 \
  --block-size 8192 \
  --n-jobs 16 \
  --grm-cache cache/loco \
  --stream \
  --out results/large_scan
```

The randomized method applies the standardized genotype GRM as a streamed
operator, retains a low-rank eigenbasis, and models the omitted orthogonal
space with a trace-matched bulk eigenvalue. Independent probe vectors estimate
operator error. By default, Domino doubles a failing rank up to the configured
cap and stops if the requested tolerance is not met. The accepted rank,
retained trace, probe error, cache status, and timing are recorded for every
chromosome.

For optional GPU projection offload:

```bash
domino ... \
  --backend cuda \
  --gpu-devices 0,1 \
  --gpu-memory-budget-mb 14000
```

`--backend auto` uses CUDA when CuPy and a visible device are available,
otherwise it uses the CPU. `--backend cuda` fails explicitly if CUDA cannot be
initialized. CUDA accelerates association projection products; randomized GRM
construction and all pipeline stages are not fully GPU-resident.

The 50,000-sample planning values in the optimization report are allocation
estimates, not measured runtime guarantees. Benchmark the intended host,
relationship panel, trait count, and output filesystem before a full launch.

## Statistical model

For chromosome `c`, Domino uses the additive LOCO covariance model

```text
V_c = sigma2 [h2 K_add,-c + (1 - h2) I],
```

where `K_add,-c` is built without markers on the tested chromosome. For one
variant, the joint marker model is

```text
y = C beta + a A + d D + error,    error ~ N(0, V_c),
```

where `C` contains the intercept and covariates, `A` is allele dosage 0/1/2,
and `D` is the heterozygote indicator 0/1/0. Domino evaluates the GLS inner
products in the LOCO eigenbasis rather than constructing a dense whitening
matrix for every trait.

The additive effect `a` is the per-allele slope. The dominance effect `d` is
the heterozygote deviation from the midpoint of the two homozygotes. Domino
reports marginal additive and dominance tests, conditional additive and
dominance tests from the joint model, and an additive-versus-additive-plus-
dominance nested test.

The default inheritance categories use `|d/a|` thresholds 0.25, 0.75, and
1.25 for additive, partial dominance, complete dominance, and overdominance.
When the additive estimate is too close to zero relative to its uncertainty,
the ratio is labelled unstable rather than treated as biologically reliable.

Domino and ADDO use different covariance models. ADDO includes whole-genome
additive and dominance relationship matrices in a variance-component model.
Domino uses an additive-only LOCO relationship matrix and marker-level
additive and dominance fixed effects. Results are empirically comparable, but
the methods are not mathematically interchangeable.

See [docs/METHODS.md](docs/METHODS.md) for a fuller model description.

## Output columns

Principal fields include:

- Variant identity: `variant_index`, `chrom`, `snp`, `pos`, and `trait`.
- Genotype summaries: `count_AA`, `count_AB`, `count_BB`, `n_obs`, and `maf`.
- Quality status: `genotype_filter_pass`.
- Marginal additive and dominance effects, standard errors, test statistics,
  and `-log10(p)` scores.
- Joint-model raw effects in `beta_add_joint_raw` and
  `beta_dom_joint_raw`, with standard errors and conditional `-log10(p)`
  scores.
- Nested-model statistics in `f_avsad` and `neglog_p_avsad`.
- Interpretation fields: signed `degree_of_dominance`, absolute ratio,
  `dominance_class`, and `inheritance_mode`.
- Model metadata: chromosome-specific `h2` and `n_samples`.
- Multivariate Wald statistics and degrees of freedom on `__joint__` rows.

Markers that fail `min_genotype_count` remain in the output with
`genotype_filter_pass=False`; non-estimable association statistics are
missing. Apply multiple-testing correction to eligible, estimable tests and
state the tested family clearly.

See [docs/INPUT_OUTPUT.md](docs/INPUT_OUTPUT.md) for the complete output
contract and [docs/COVARIATES.md](docs/COVARIATES.md) for fixed-effect guidance.

## Execution metadata

`{out}.execution.json` records the resolved memory budget, marker block and
trait tile sizes, decomposition method, approximation diagnostics, h2 fits,
boundary counts, cache keys, per-stage runtime, and sampled peak RSS. Large
multivariate covariance matrices can be written as compressed NPZ sidecars.

Keep these files with association output. They document the actual execution
choices that may differ from requested ceilings after resource planning.

## Validation and reproducibility

### Full Validation Report can be found here: [docs/VALIDATION_REPORT.md](docs/VALIDATION_REPORT.md)

Run the local checks from the repository root:

```bash
python -m pytest -q
python -m ruff check .
python release_check.py
python examples/quickstart.py
```

The exact optimized scan was compared with a direct/reference implementation
over 200,000 variant-trait rows and matched primary numeric outputs within
`rtol=1e-10` and `atol=1e-10`. On the recorded 1,000-sample,
10,000-variant benchmark, shared-eigenbasis vectorization and cache reuse were
most beneficial as trait count increased. See
[docs/OPTIMIZATION_BENCHMARK.md](docs/OPTIMIZATION_BENCHMARK.md).

Private genotype and phenotype files are not included. The synthetic example
and tests are generated locally from fixed random seeds.

## Scope and limitations

- Quantitative traits only. Logistic or case-control association is not
  implemented.
- Autosomal PLINK 1 analysis. Sex-aware X, Y, and mitochondrial models are not
  implemented.
- Missing marker genotypes are mean-imputed within marker for association.
- Sparse genotype cells can make dominance effects unstable. Set and report a
  defensible `min_genotype_count`.
- Exact LOCO eigendecomposition has quadratic memory and cubic decomposition
  cost. It is not appropriate for 50,000 samples on a 64 GiB host.
- Randomized decomposition, SCORE, float32, multivariate tests, and CUDA are
  optional scaling paths whose calibration should be checked for each intended
  study design.
- Output volume can dominate runtime and storage. Ten million variants by
  1,000 traits represents ten billion per-trait rows before joint rows.
- Association and inheritance labels do not establish biological causality.

Preserve the installed software version, exact command, input checksums,
execution sidecars, and analysis decisions for every scientific report.

## Repository layout

```text
domino/                 tested Python implementation
tests/                   unit and integration tests
examples/quickstart.py   self-contained synthetic run
docs/                    methods, I/O, covariates, scaling, validation
benchmarks/              reproducible current-version benchmark driver
.github/workflows/       continuous integration
```

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). Before citing a
specific scientific release, create a tagged GitHub release, archive it with a
persistent DOI service such as Zenodo, and update the citation metadata with
the DOI.

## Contributing and support

Bug reports and reproducible feature requests belong in
[GitHub Issues](https://github.com/unculturedbacterium/Domino/issues). Please
read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.
Security-sensitive reports should follow [SECURITY.md](SECURITY.md).

