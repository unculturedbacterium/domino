# Domino validation report

**Software:** Domino 1.0.0  
**Validation date:** 2026-07-22  
**Target use:** dominance-aware genome-wide association analysis of quantitative traits  
**Primary model tested:** exact float64 additive leave-one-chromosome-out genomic relationship matrix, independent-trait profile REML, and blockwise additive-plus-dominance GLS  

## Executive conclusion

Domino passed the software, numerical, simulation, real-data, and resource-scaling checks needed to support a software-focused methods paper. The exact model matched direct complete-design GLS to numerical precision, controlled independent null simulations, recovered simulated additive and non-additive loci with architecture-dependent power, and produced highly concordant real-data rankings with ADDO while being easier to install and faster in the tested workflows.

The validation supports the claim that Domino is a reproducible, memory-aware, dominance-aware quantitative-trait GWAS pipeline. It does not establish new biological loci, does not validate binary traits, and does not prove that Domino's additive LOCO covariance is universally superior to ADDO's additive-plus-dominance variance-component model.

![Domino workflow](figures/Figure1_Domino_workflow.png)

## Model tested

For each chromosome, Domino fits a chromosome-specific covariance model using an additive leave-one-chromosome-out GRM. The phenotype, covariates, additive marker dosage, and dominance marker code are transformed by the same covariance operator before the marker model is evaluated. The tested marker model includes additive dosage and the heterozygote dominance term in the fixed-effect design.

This differs from ADDO. The ADDO comparator used additive and dominance genomic relationship matrices through GCTA-ad. Domino models dominance at the marker-test level after additive LOCO correction, while ADDO estimates additive and dominance genomic variance components. Agreement between the two programs should therefore be interpreted as empirical concordance, not algebraic equivalence.

## Software and numerical validation

The release passed the core automated validation suite:

| Check | Result |
| --- | ---: |
| `pytest` tests | 22 passed |
| Ruff linting | passed |
| Release audit | passed |
| Direct GLS maximum dominance beta difference | 1.55e-13 |
| Direct GLS maximum conditional dominance score difference | 7.46e-13 |

The software tests cover PLINK decoding, sample alignment, covariate handling, missingness groups, exact GLS algebra, genotype-cell filtering, duplicate variant identifiers, inheritance labels, resource planning, streamed atomic output, cache reuse, adaptive randomized rank, SCORE estimation, and multivariate output.

## Null calibration

Null simulations used 400 related samples, 2,400 markers, and 1,000 replicates per scenario. Independent nulls were well calibrated, with median genomic inflation factors from 0.987 to 0.991 and marker-level type I error close to 0.05. In the genotype-matched polygenic null, the additive test showed modest inflation, while conditional dominance remained closer to expectation. The deliberately misspecified shared-family null produced stronger inflation, showing that a GRM does not replace measured family, batch, cohort, or other fixed covariates.

| Scenario | Test | Replicates | Median lambda | Nominal 0.05 rate | Bonferroni FWER |
| --- | --- | ---: | ---: | ---: | ---: |
| Independent null | additive | 1000 | 0.987 | 0.047 | 0.036 |
| Independent null | dominance conditional | 1000 | 0.991 | 0.049 | 0.027 |
| Polygenic family null | additive | 1000 | 1.075 | 0.058 | 0.084 |
| Polygenic family null | dominance conditional | 1000 | 1.006 | 0.051 | 0.049 |
| Shared family null | additive | 1000 | 1.123 | 0.064 | 0.137 |
| Shared family null | dominance conditional | 1000 | 1.051 | 0.056 | 0.058 |

![Null calibration and REML](figures/Figure2_calibration.png)

## Power and inheritance recovery

Power depended strongly on inheritance architecture, effect size, and allele-frequency structure. At 10% locus variance, Bonferroni-corrected causal-locus power was highest for additive, recessive, and overdominant architectures in this simulation design. Partial dominance was the hardest case for the conditional dominance test at this sample size. Polygenic simulations recovered at least one causal marker in 33% of additive replicates and 50% of dominance replicates.

| Architecture | Locus variance | Test component | Replicates | Power | Median causal rank |
| --- | ---: | --- | ---: | ---: | ---: |
| Additive | 0.10 | additive | 100 | 0.79 | 1 |
| Partial dominance | 0.10 | dominance | 100 | 0.01 | 121 |
| Dominant | 0.10 | dominance | 100 | 0.15 | 18 |
| Recessive | 0.10 | dominance | 100 | 0.65 | 1 |
| Overdominant | 0.10 | dominance | 100 | 0.73 | 1 |
| Polygenic additive | mixed | additive | 100 | 0.33 | NA |
| Polygenic dominance | mixed | dominance | 100 | 0.50 | NA |

Profile-REML heritability calibration was accurate across simulated heritability values, with maximum absolute mean bias of 0.033 and complete convergence in the tested grid. Sign-aware inheritance classification across 1,000 simulations showed useful but imperfect coarse classification, with mode-specific accuracy ranging from 0.74 to 0.99 in the final classification table.

![Power and inheritance classification](figures/Figure3_power.png)

## Real-data comparison with ADDO

Four previously regressed cocaine-related quantitative traits were analyzed in 811 complete-case animals at 35,723 variants without additional covariates. Domino completed the exact four-trait scan in 13.11 seconds with 475 MiB peak resident memory. The archived ADDO GCTA-ad run required 2,789.08 seconds. This was a 212.8-fold runtime difference for this workflow on the tested Windows workstation.

Across 12 trait-test comparisons, Spearman rank concordance between Domino and ADDO ranged from 0.980 to 0.999. The same top genomic position was selected in 10 of 12 comparisons, and the median top-10 overlap was 9 positions.

| Trait | Test | Spearman rho | ADDO lambda | Domino lambda | Same top position | Top-10 overlap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| pr_max | additive | 0.983 | 1.009 | 1.066 | yes | 9 |
| pr_max | dominance marginal | 0.983 | 1.109 | 1.146 | yes | 9 |
| pr_max | dominance conditional | 0.982 | 1.080 | 1.144 | no | 9 |
| addiction_index | additive | 0.981 | 1.021 | 1.095 | yes | 8 |
| addiction_index | dominance marginal | 0.982 | 1.053 | 1.099 | yes | 5 |
| addiction_index | dominance conditional | 0.980 | 1.039 | 1.098 | yes | 8 |
| lga_total_intake | additive | 0.989 | 0.956 | 1.060 | yes | 8 |
| lga_total_intake | dominance marginal | 0.992 | 1.028 | 1.055 | yes | 9 |
| lga_total_intake | dominance conditional | 0.998 | 1.063 | 1.065 | yes | 10 |
| sha_total_intake | additive | 0.995 | 1.083 | 1.141 | no | 9 |
| sha_total_intake | dominance marginal | 0.997 | 0.975 | 0.996 | yes | 10 |
| sha_total_intake | dominance conditional | 0.999 | 1.027 | 1.024 | yes | 10 |

![ADDO comparison](figures/Figure4_ADDO_comparison.png)

The full scatter and QQ panels are included below for all ADDO comparisons.

![All ADDO scatter panels](figures/FigureS1_all_ADDO_scatter.png)

![All real-data QQ panels](figures/FigureS2_all_real_QQ.png)

## Sex-adjusted body-mass real-data test

A second real-data comparison used `dissection:body_g` in a fixed-seed random sample of 2,000 animals drawn from 9,098 complete genotype-matched records. The sample contained 975 females and 1,025 males. Sex was included as the only fixed covariate in Domino. ADDO used a sex-residualized phenotype before fitting additive and dominance GCTA GRMs.

Both pipelines evaluated 36,002 variants. Domino completed the scan in 76.62 seconds with 1,092 MiB peak resident memory. ADDO required 407.94 seconds, giving a 5.32-fold runtime difference. ADDO's accelerated screen reported full marker statistics for 3,205 markers, or 8.9% of the tested variants, so ADDO genomic inflation was not estimated for this run.

| Test | Matched markers with ADDO output | Spearman rho | Domino lambda | Same top position | Top-10 overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Additive | 3,205 | 0.977 | 1.202 | yes | 9 |
| Dominance marginal | 3,205 | 0.980 | 1.095 | yes | 8 |
| Dominance conditional | 3,205 | 0.992 | 1.099 | no | 9 |

Domino identified two additive markers below the subset-wide Bonferroni threshold, led by chromosome 7 at 36,024,619 bp with P = 7.31e-07. No conditional dominance marker passed the threshold. Because the additive lambda was 1.202, this should be treated as an internal real-data demonstration rather than independent biological validation.

![Sex-adjusted body-mass analysis](figures/FigureS3_body_g_comparison.png)

## Genome-wide real-phenotype scan

The covariate-free `pr_max` scan analyzed 814 animals and 7,659,686 autosomal source markers. Of these, 4,686,655 markers passed the minimum AA/AB/BB cell count of five. Exact analysis required 25.99 minutes with peak resident memory of 4,801 MiB.

The genomic inflation factor was 1.064 for the additive test and 1.138 for the conditional dominance test. No additive or dominance marker passed the experiment-wide Bonferroni threshold. This scan demonstrates that Domino can run a full autosomal quantitative-trait analysis on the tested workstation, but it is not a claim of biological discovery.

![Genome-wide pr_max scan](figures/Figure6_real_pr_max_genomewide.png)

## Runtime and memory scaling

Scaling tests were run in isolated processes with one CPU worker, exact float64 decomposition, streamed Parquet output, and a 9 GB memory budget. Each point was repeated three times. Runtime excluded synthetic-data generation and included result writing.

| Scaling axis | Samples | Variants | Traits | Median runtime, s | Median peak RSS, MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Samples | 100 | 10,000 | 1 | 0.46 | 171 |
| Samples | 250 | 10,000 | 1 | 0.87 | 203 |
| Samples | 500 | 10,000 | 1 | 1.78 | 262 |
| Samples | 1,000 | 10,000 | 1 | 4.63 | 378 |
| Variants | 100 | 10,000 | 1 | 0.46 | 170 |
| Variants | 100 | 50,000 | 1 | 1.64 | 227 |
| Variants | 100 | 100,000 | 1 | 3.08 | 249 |
| Variants | 100 | 1,000,000 | 1 | 30.54 | 380 |
| Traits | 400 | 2,400 | 1 | 0.54 | 175 |
| Traits | 400 | 2,400 | 100 | 2.73 | 252 |
| Traits | 400 | 2,400 | 1,000 | 20.43 | 913 |

These tests show approximately linear scaling with marker count in the tested range and controlled memory growth with increasing trait count. The 50,000-sample scale remains a planning target for future low-rank or randomized-operator validation, not a measured exact dense result.

![Runtime and memory scaling](figures/Figure5_scaling.png)

## Installation and execution comparison

| Aspect | Domino | ADDO comparator used in validation |
| --- | --- | --- |
| Primary interface | Python API and `domino` CLI | R functions plus PLINK and GCTA executables |
| Install route | `pip install git+https://github.com/unculturedbacterium/Domino.git` or editable clone install | R package plus archived dependencies and external binaries |
| Input formats | PLINK1 genotypes and tabular phenotypes | PLINK1 genotypes plus ADDO-specific phenotype and covariate setup |
| Resource control | Explicit memory budget and block planning | No equivalent memory-budget guard observed in the tested workflow |
| Reproducibility | Single Python package environment | Dedicated Conda/R environment and configured executable paths |

The installation comparison is operational. It does not imply that ADDO's statistical model is inferior.

## Reproducibility commands

From a fresh clone of Domino:

```bash
python -m pip install -e ".[test,plot]"
python -m pytest -q
python -m ruff check .
python release_check.py
python examples/quickstart.py
```

The paper validation scripts and aggregate outputs are stored under `paper/`. Internal real-data runs require access to the local phenotype and PLINK genotype files and are not bundled with the public repository.

## Claims supported by this validation

1. Domino's exact quantitative-trait model agrees with direct GLS to numerical precision on the validation problem.
2. Independent null simulations were calibrated, while misspecified shared-family structure produced inflation that should be handled with measured covariates or better study design.
3. Dominance power depends strongly on inheritance architecture and allele-frequency structure.
4. Domino and ADDO produced highly concordant real-data rankings despite different covariance models.
5. Domino completed the tested real-data and synthetic scaling workflows with lower runtime and a simpler installation path than ADDO on the tested workstation.
6. Domino can stream large marker scans and multi-trait outputs under an enforced memory budget in the measured sample-size ranges.

## Limitations

1. Real-data results come from one genotype resource and should not be presented as independent biological replication.
2. ADDO body-mass output was censored by its accelerated screening step, so full ADDO QQ plots and genomic inflation values were not calculated for that run.
3. Power simulations used finite sample sizes and selected genetic architectures; they do not cover every MAF, LD, relatedness, or missingness setting.
4. Domino currently supports quantitative traits. Binary trait validation is outside scope.
5. Exact dense LOCO decomposition is not validated as a 50,000-sample, 64 GB production workflow. Large-sample randomized or low-rank modes require separate calibration before being used for primary scientific claims.
6. The covariate-free `pr_max` genome-wide run used previously regressed phenotypes. Raw phenotypes require appropriate covariates, such as sex, batch, cohort, PCs, or family effects when relevant.

## Overall assessment

Domino is ready for a software-focused methods manuscript and public GitHub release as a validated quantitative-trait, dominance-aware GWAS pipeline. The strongest paper claim is practical and reproducibility-focused: Domino provides an installable Python implementation of additive and dominance marker testing with additive LOCO GRM correction, memory-bounded execution, direct-GLS validation, simulation calibration, real-data ADDO concordance, and documented limitations.
