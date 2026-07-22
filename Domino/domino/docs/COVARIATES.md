# Covariate support

Domino applies the same chromosome-specific GLS weighting to the phenotype,
intercept, covariates, additive dosage, and dominance code. Covariates are
therefore part of the complete fixed-effect design used in both variance-
component estimation and marker testing.

## Numeric covariates

Use `--covariates` for continuous or already encoded variables such as age,
genotyping intensity metrics, or ancestry principal components:

```bash
--covariates age,PC1,PC2,PC3,PC4,PC5
```

The CLI requires every value in these columns to be numeric or missing.
Standardization is not required for correctness, but rescaling very large or
very small continuous covariates can improve numerical conditioning.

## Categorical covariates

Use `--categorical-covariates` for labels such as sex, batch, site, or cohort:

```bash
--categorical-covariates sex,batch,cohort
```

Domino one-hot encodes categorical columns with one reference category. The
fixed design always includes an intercept, and redundant columns are removed.
Small levels should be merged or excluded according to a prespecified rule;
one observation in a level cannot support a stable adjustment.

## Missing values and sample masks

A sample must have a nonmissing phenotype and all requested covariates for a
given analysis group. Independent traits can use different masks. Traits that
share a mask reuse the same LOCO eigenbasis and fixed-design projections.
Multivariate mode uses complete cases shared across all requested traits.

The execution sidecar records `n_samples` per analysis group. Confirm this
against the expected cohort after every run.

## Recommended documentation

For a reproducible analysis, report:

- The exact numeric and categorical covariate columns.
- How sex, batch, cohort, and ancestry variables were encoded.
- The method and marker set used to calculate PCs.
- Any centering, scaling, transformations, or interaction terms performed
  before Domino.
- Missing-value and level-collapsing rules.
- Final sample count for every trait or complete-case panel.
- Whether covariates were selected before looking at association results.

Domino does not infer a scientifically appropriate adjustment set. Avoid
post-outcome covariates, unplanned selection, and adjustment for variables that
would change the target estimand without a clear causal rationale.
