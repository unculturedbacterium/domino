# Statistical methods

## Analysis scope

Domino performs linear mixed-model association for quantitative traits.
It tests additive dosage and a heterozygote dominance deviation while using an
additive leave-one-chromosome-out relationship matrix to control polygenic
covariance. Case-control outcomes and sex-aware chromosome models are outside
the current scope.

## Sample alignment and fixed effects

Samples are identified by the IID field in the PLINK `.fam` file. Phenotype
and covariate DataFrame rows are aligned to those IIDs. The fixed-effect design
contains an intercept, user-specified numeric covariates, and one-hot-encoded
categorical covariates with one reference level. Linearly redundant columns
are removed before fitting.

In independent mode, traits with the same complete sample and covariate mask
share one analysis group. Distinct missingness masks retain distinct LOCO
cross-products. Multivariate mode uses complete cases across all requested
traits.

## Genotype coding

For a biallelic marker, the additive code is allele dosage

```text
A = 0, 1, 2,
```

and the dominance code is the heterozygote indicator

```text
D = 0, 1, 0.
```

Codes are centered and scaled for numerical fitting. Reported raw effects and
standard errors are transformed back to the original dosage and heterozygote
scales. Missing marker genotypes are mean-imputed after observed AA, AB, and BB
counts have been recorded. A marker is ineligible when a genotype code has no
variance or when any genotype cell is smaller than `min_genotype_count`.

## Additive LOCO covariance

For a marker on chromosome `c`, the covariance model is

```text
V_c = sigma2 [h2 K_add,-c + (1 - h2) I].
```

`K_add,-c` is the additive genomic relationship matrix constructed from
markers outside chromosome `c`. Exact mode computes a pairwise-observed,
standardized GRM and a deterministic symmetric eigendecomposition. A separate
LD-pruned PLINK fileset can be supplied with `grm_bfile`; marker association is
still performed against the primary `bfile`.

The independent-mode default estimates `h2` and `sigma2` by profile REML with
the complete fixed-effect design. Profile ML and a covariate-adjusted
two-component SCORE estimator are also available. Estimates and boundary
counts are recorded in the execution sidecar.

## Marker tests

For each eligible marker, Domino fits GLS models under the chromosome-specific
covariance. Let `C` be the intercept and covariate design. The joint model is

```text
y = C beta + a A + d D + e,
e ~ N(0, V_c).
```

The principal tests are:

| Output | Model and null hypothesis |
|---|---|
| `neglog_p_additive` | Marginal additive model, `H0: a = 0` |
| `neglog_p_dominance_marginal` | Marginal dominance model, `H0: d = 0` |
| `neglog_p_add_joint` | Additive coefficient in the joint model, `H0: a = 0` conditional on `D` |
| `neglog_p_dom_joint` | Dominance coefficient in the joint model, `H0: d = 0` conditional on `A` |
| `neglog_p_avsad` | Additive model versus additive-plus-dominance model |

Adding one dominance coefficient makes the nested one-degree-of-freedom F
test algebraically equivalent to the squared conditional dominance t
statistic. Domino computes that stable form rather than subtracting nearly
equal residual sums of squares.

Outputs store `-log10(p)` rather than raw p-values. Raw p-values can be
recovered as `10 ** (-neglog_p_...)` when they are representable at the chosen
precision.

## Shared-eigenbasis GLS

If `K_add,-c = U diag(s) U'`, the inverse covariance weights are diagonal in
the columns of `U`. Domino projects the phenotype, fixed design, and marker
blocks into this shared basis and evaluates weighted cross-products directly.
It does not create one dense `n x n` whitening matrix per trait. Marker blocks
are projected once and combined with trait-specific weights in bounded trait
tiles.

This transformation is algebraically equivalent to complete-design GLS for
the exact eigensystem. It is important that the phenotype, fixed effects,
additive code, and dominance code all use the same covariance model.

## Randomized LOCO approximation

Randomized mode applies the mean-imputed standardized genotype GRM as a
streamed linear operator. It retains leading eigenpairs and represents the
omitted orthogonal complement with a trace-matched bulk eigenvalue. The
approximation is evaluated on independent random probes. Domino can
deterministically double the rank until the requested relative operator-error
tolerance is met or the configured cap is reached.

Randomized mode changes the GRM missingness convention and approximates the
eigenspectrum. It is therefore a separate numerical path, not merely a faster
call to the exact eigensolver. Scientific analyses should report the initial
and accepted ranks, power iterations, precision, retained trace fraction, and
probe error for each chromosome.

## Multivariate mode

Multivariate mode uses a covariate-adjusted SCORE estimator for genetic and
residual trait covariance components. It diagonalizes the estimated trait
covariance representation, runs the marker algebra on transformed components,
recovers effects for the original traits, and emits cross-trait Wald tests.

For `t` traits, the `__joint__` row reports:

- Marginal additive Wald test with `t` degrees of freedom.
- Conditional additive Wald test with `t` degrees of freedom.
- Conditional dominance Wald test with `t` degrees of freedom.
- Combined additive-dominance Wald test with `2t` degrees of freedom.

Trait covariance storage and work scale at least quadratically in `t`.
Calibration and rank stability must be evaluated for the intended trait panel.

## Degree of dominance

In the joint model, Domino reports signed `d/a` and `|d/a|`. The default
coarse categories are:

| `|d/a|` range | Category |
|---|---|
| `< 0.25` | Additive (`A`) |
| `0.25` to `< 0.75` | Partial dominance (`PD`) |
| `0.75` to `< 1.25` | Complete dominance (`D`) |
| `>= 1.25` | Overdominance magnitude (`OD`) |

Sign-aware labels distinguish displacement toward dosage 0 or dosage 2 and
high-heterozygote overdominance from low-heterozygote underdominance. If the
additive effect does not exceed `stability_z` times its standard error, `d/a`
is marked unstable because division by a near-zero additive effect is not
interpretable.

## Difference from ADDO

ADDO estimates a variance-component framework containing whole-genome
additive and dominance relationship matrices. Domino uses an additive-only
LOCO relationship matrix for background covariance and tests additive and
dominance marker codes as fixed effects. LOCO also avoids including the tested
chromosome in the background GRM. These differences can change variance
estimates, test statistics, and runtime; matching marker scores should not be
assumed.

## Multiple testing and interpretation

Domino does not choose a study-wide significance rule. Analysts should define
the eligible marker set, tested phenotype family, and additive, dominance, or
joint hypotheses before correction. Results from filtered or non-estimable
markers should not enter the correction denominator as successful tests.
Association signals and inheritance labels require replication and biological
follow-up and do not establish causality.
