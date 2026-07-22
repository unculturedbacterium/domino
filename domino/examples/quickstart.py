"""Generate a small synthetic PLINK dataset and run Domino end to end."""
from pathlib import Path

import numpy as np

import domino
from domino.testutils import simulate


def main() -> None:
    output = Path("example_output")
    output.mkdir(exist_ok=True)
    prefix = output / "synthetic"

    simulated = simulate(
        str(prefix),
        n_fam=20,
        sibs=4,
        n_chrom=2,
        snps_per_chrom=200,
        n_causal_add=10,
        n_causal_dom=10,
        seed=2026,
    )
    phenotype = simulated["pheno"][["y_mix"]]
    covariates = phenotype.copy()
    covariates["sex"] = np.where(np.arange(len(phenotype)) % 2, "F", "M")
    covariates["batch"] = np.where(np.arange(len(phenotype)) % 3, "B1", "B2")
    covariates = covariates[["sex", "batch"]]
    analysis = phenotype.join(covariates)
    analysis.index.name = "iid"
    analysis.to_csv(output / "phenotypes.tsv", sep="\t")

    results = domino.run_gwas(
        str(prefix),
        phenotype,
        traits=["y_mix"],
        covar=covariates,
        min_genotype_count=2,
        out=str(output / "scan"),
        verbose=True,
    )
    results.to_csv(output / "scan.csv", index=False)
    top = results.loc[results["neglog_p_dom_joint"].idxmax()]
    print(f"Domino {domino.__version__}: tested {len(results)} marker rows")
    print(f"top conditional-dominance marker: {top['snp']}")
    print(f"results: {(output / 'scan.csv').resolve()}")


if __name__ == "__main__":
    main()
