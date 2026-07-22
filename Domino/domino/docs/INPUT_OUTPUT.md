# Input and output reference

## PLINK input

`bfile` is a PLINK 1 prefix. All three files must exist:

```text
PREFIX.bed    SNP-major binary genotype data
PREFIX.bim    variant chromosome, ID, position, and allele metadata
PREFIX.fam    family and individual IDs
```

The reader expects SNP-major BED data. Samples are keyed by the `.fam` IID
column. Duplicate variant IDs are permitted because `variant_index`, the
zero-based BIM row number, is the unique output key. Duplicate or ambiguous
sample IIDs should be resolved before analysis.

`grm_bfile` optionally supplies a different PLINK fileset for relationship
estimation. Its sample IDs must align with the association fileset. A
quality-controlled, LD-pruned panel is recommended for large randomized LOCO
analyses.

## CLI phenotype table

`--pheno` must be a tab-separated table. The default sample column is `iid`;
change it with `--id-column`.

```text
iid	trait	age	sex	batch	PC1
S001	0.44	62	F	B1	-0.013
S002	-0.81	55	M	B2	0.008
```

- `--traits` accepts comma-separated quantitative columns.
- `--covariates` accepts comma-separated numeric columns.
- `--categorical-covariates` accepts comma-separated label columns.
- Leading and trailing whitespace is stripped from IID strings.
- Numeric covariates fail parsing when a nonnumeric value is present.
- Missing trait or covariate values determine the usable sample mask.
- Categorical missing values also remove the affected sample.

## Library phenotype input

`domino.run_gwas` accepts a pandas Series or DataFrame indexed by IID. A
DataFrame is strongly recommended for covariates because it can be aligned by
sample identifier. An ordered numeric array can be used only when the caller
has already guaranteed sample order.

## Output files

When `out="results/scan"` is supplied, the library writes:

```text
results/scan.chr1.parquet
results/scan.chr2.parquet
...
results/scan.execution.json
```

The CLI also writes `results/scan.metadata.json`. Without `--stream`, it writes
`results/scan.all.parquet` after retaining the complete result in memory.
Parquet chromosome files are written to unique partial paths and atomically
promoted only after all row groups complete. Interrupted partial files are not
presented as completed output.

## Association columns

| Column | Meaning |
|---|---|
| `variant_index` | Zero-based row in the BIM file; stable unique variant key |
| `snp` | BIM variant identifier |
| `chrom` | BIM chromosome label |
| `pos` | BIM base-pair position |
| `trait` | Phenotype name or `__joint__` for multivariate rows |
| `n_obs` | Number of observed genotypes for the marker |
| `count_AA`, `count_AB`, `count_BB` | Observed dosage 0, 1, and 2 counts |
| `maf` | Minor allele frequency among observed calls |
| `genotype_filter_pass` | Whether variance and genotype-cell checks passed |
| `beta_additive` | Marginal additive coefficient on the standardized code |
| `beta_additive_raw` | Marginal additive effect per dosage unit |
| `se_additive_raw` | Standard error of the raw marginal additive effect |
| `stat_additive` | Marginal additive t statistic |
| `neglog_p_additive` | `-log10(p)` for the marginal additive test |
| `beta_dominance_marginal_raw` | Marginal heterozygote deviation |
| `se_dominance_marginal_raw` | Standard error of the marginal dominance effect |
| `stat_dominance_marginal` | Marginal dominance t statistic |
| `neglog_p_dominance_marginal` | `-log10(p)` for the marginal dominance test |
| `beta_add_joint_raw` | Additive effect conditional on dominance |
| `se_add_joint_raw` | Standard error of the conditional additive effect |
| `stat_add_joint` | Conditional additive t statistic |
| `neglog_p_add_joint` | `-log10(p)` for conditional additive effect |
| `beta_dom_joint_raw` | Dominance effect conditional on dosage |
| `se_dom_joint_raw` | Standard error of conditional dominance effect |
| `stat_dom_joint` | Conditional dominance t statistic |
| `neglog_p_dom_joint` | `-log10(p)` for conditional dominance effect |
| `cov_add_dom_joint_raw` | Estimated covariance of joint raw effects |
| `f_avsad` | One-df additive versus additive-plus-dominance F statistic |
| `neglog_p_avsad` | `-log10(p)` for the nested model test |
| `degree_of_dominance` | Signed `d/a` from joint raw effects |
| `degree_of_dominance_abs` | Absolute `|d/a|` ratio |
| `dominance_class` | `A`, `PD`, `D`, `OD`, `UNSTABLE`, or `FILTERED` |
| `inheritance_mode` | Sign-aware inheritance label |
| `h2` | Chromosome-specific variance ratio used for the trait |
| `n_samples` | Samples used in the analysis group |
| `test_scope` | `trait` or `multivariate_joint` |

Additive-only analyses omit the dominance-specific statistical columns.
Filtered markers remain in the output with non-estimable fields set to null.

## Multivariate joint columns

`__joint__` rows carry the following chi-square Wald statistics:

| Statistic prefix | Degrees of freedom |
|---|---:|
| `stat_additive_multivariate` | Number of traits |
| `stat_add_joint_multivariate` | Number of traits |
| `stat_dom_joint_multivariate` | Number of traits |
| `stat_add_dom_multivariate` | Twice the number of traits |

Each statistic has a matching `neglog_p_*` and `df_*` column. Per-trait rows
retain recovered effect estimates but leave joint-only fields null.

## Execution sidecar

The execution JSON records:

- Domino configuration and resolved resource budget.
- Requested and resolved decomposition settings.
- Marker block and trait tile sizes for each sample-mask group.
- LOCO cache keys and cache-hit status.
- Randomized rank attempts, retained trace, and probe errors.
- Variance-component estimator, h2 values, and boundary counts.
- Decomposition/cache-load, variance-component, association/output, and total
  runtime measurements.
- Baseline and sampled peak resident memory.
- Output rows and Parquet row groups.

Store this sidecar with the association files. Paths may reveal local directory
names, so inspect it before sharing publicly.

## Reading output

```python
import pandas as pd

chromosome_1 = pd.read_parquet("results/scan.chr1.parquet")
eligible = chromosome_1.loc[chromosome_1["genotype_filter_pass"]]
top_dominance = eligible.nlargest(20, "neglog_p_dom_joint")
```

Raw p-values can be reconstructed when needed:

```python
p_dom = 10.0 ** (-eligible["neglog_p_dom_joint"])
```

Very large scores can underflow when converted back to raw p-values. Retain
the `-log10(p)` representation for ranking and reporting in that case.
