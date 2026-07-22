"""Run a small, cross-platform end-to-end validation of the Domino core."""
from __future__ import annotations

from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
from scipy.stats import t as tdist

import domino
from domino.testutils import simulate


def main() -> None:
    with TemporaryDirectory(prefix="domino-validation-") as directory:
        prefix = str(Path(directory) / "synthetic")
        simulated = simulate(
            prefix,
            n_fam=60,
            sibs=3,
            n_chrom=2,
            snps_per_chrom=400,
            n_causal_add=20,
            n_causal_dom=20,
            seed=1127,
        )
        phenotype = simulated["pheno"][["y_mix"]]
        results = domino.run_gwas(
            prefix,
            phenotype,
            traits=["y_mix"],
            decomposition="exact",
            variance_estimator="profile_reml",
            compute_dtype="float64",
            verbose=False,
        )

        reader = domino.PlinkReader(prefix)
        eigensystem = domino.compute_loco_grms(reader, chroms=["1"], verbose=False)["1"]
        y = phenotype["y_mix"].to_numpy()
        variance = domino.estimate_h2(y, eigensystem["U"], eigensystem["s"])
        transformed_y = domino.whiten_matrix(
            y,
            eigensystem["U"],
            eigensystem["s"],
            variance["h2"],
            variance["yvar"],
        )
        chromosome_index = reader.chrom_variant_index("1")
        genotype = reader.read_block(int(chromosome_index.min()), len(chromosome_index))
        chromosome_results = results.loc[results["chrom"] == "1"].set_index("variant_index")
        eligible = chromosome_results.index[
            chromosome_results["genotype_filter_pass"].astype(bool)
        ].to_numpy(dtype=int)
        check_global = np.random.default_rng(41).choice(eligible, 8, replace=False)
        max_beta_difference = 0.0
        max_score_difference = 0.0

        for variant_index in check_global:
            local_index = int(np.flatnonzero(chromosome_index == variant_index)[0])
            g = genotype[:, local_index].astype(float)
            design = np.column_stack([np.ones_like(g), g, (g == 1).astype(float)])
            transformed_design = domino.whiten_matrix(
                design,
                eigensystem["U"],
                eigensystem["s"],
                variance["h2"],
                variance["yvar"],
            )
            beta, *_ = np.linalg.lstsq(transformed_design, transformed_y, rcond=None)
            residual = transformed_y - transformed_design @ beta
            degrees_freedom = len(transformed_y) - design.shape[1]
            sigma2 = residual @ residual / degrees_freedom
            standard_error = np.sqrt(
                np.diag(sigma2 * np.linalg.inv(transformed_design.T @ transformed_design))
            )
            score = -np.log10(
                2.0 * tdist.sf(abs(beta[2] / standard_error[2]), df=degrees_freedom)
            )
            observed = chromosome_results.loc[variant_index]
            max_beta_difference = max(
                max_beta_difference,
                abs(beta[2] - observed["beta_dom_joint_raw"]),
            )
            max_score_difference = max(
                max_score_difference,
                abs(score - observed["neglog_p_dom_joint"]),
            )

        if len(results) != simulated["m"]:
            raise AssertionError(f"expected {simulated['m']} rows, observed {len(results)}")
        if max_beta_difference > 1e-8 or max_score_difference > 1e-8:
            raise AssertionError(
                "direct GLS comparison failed: "
                f"beta={max_beta_difference:.3e}, score={max_score_difference:.3e}"
            )

        reader.close()

        print(f"PASS: Domino {domino.__version__}")
        print(f"samples={simulated['n']} variants={simulated['m']} rows={len(results)}")
        print(f"max direct-GLS beta difference={max_beta_difference:.3e}")
        print(f"max direct-GLS score difference={max_score_difference:.3e}")


if __name__ == "__main__":
    main()
