"""Classify variants by inheritance mode from the degree of dominance (d/a).

With additive code {0,1,2} and heterozygote code {0,1,0}, the joint model
    E[y] = mu + a * add + d * het
places the heterozygote at (midpoint of homozygotes) + d, so ``d`` is exactly
the dominance deviation and ``a`` the additive effect. The degree of dominance
|d/a| then maps to the classical categories, matching ADDO's scheme:

    |d/a| < ~0        additive            (A)
    0 < |d/a| < 1     partial dominance   (PD)
    |d/a| ~ 1         complete dominance  (D)
    |d/a| > 1         over-dominance      (OD)
"""
import numpy as np

DEFAULT_THRESHOLDS = (0.25, 0.75, 1.25)


def degree_of_dominance(beta_add_raw, beta_dom_raw):
    """|d / a| on the raw (per-allele) genotype scale."""
    a = np.asarray(beta_add_raw, dtype=np.float64)
    d = np.asarray(beta_dom_raw, dtype=np.float64)
    out = np.full(a.shape, np.nan)
    ok = np.isfinite(a) & np.isfinite(d) & (np.abs(a) > 0)
    out[ok] = np.abs(d[ok] / a[ok])
    return out


def classify_da(da, thresholds=DEFAULT_THRESHOLDS):
    """Map |d/a| ratios to {'A','PD','D','OD','NA'}."""
    t_a, t_d_lo, t_d_hi = thresholds
    da = np.asarray(da, dtype=np.float64)
    out = np.full(da.shape, "NA", dtype=object)
    ok = np.isfinite(da) & (da >= 0)
    out[ok & (da < t_a)] = "A"
    out[ok & (da >= t_a) & (da < t_d_lo)] = "PD"
    out[ok & (da >= t_d_lo) & (da < t_d_hi)] = "D"
    out[ok & (da >= t_d_hi)] = "OD"
    return out


def fieller_ratio_interval(
    beta_add_raw,
    beta_dom_raw,
    variance_add_raw,
    variance_dom_raw,
    covariance_add_dom_raw=0.0,
    z=1.959963984540054,
):
    """Fieller confidence interval for the signed ratio ``d/a``.

    Unbounded or undefined intervals are represented by missing endpoints and
    ``bounded=False``. This is intentional: a finite point ratio should not be
    interpreted when the additive denominator is insufficiently separated
    from zero.
    """
    a, d, va, vd, cov = np.broadcast_arrays(
        np.asarray(beta_add_raw, dtype=np.float64),
        np.asarray(beta_dom_raw, dtype=np.float64),
        np.asarray(variance_add_raw, dtype=np.float64),
        np.asarray(variance_dom_raw, dtype=np.float64),
        np.asarray(covariance_add_dom_raw, dtype=np.float64),
    )
    low = np.full(a.shape, np.nan)
    high = np.full(a.shape, np.nan)
    bounded = np.zeros(a.shape, dtype=bool)
    qa = a * a - z * z * va
    qb = -2.0 * a * d + 2.0 * z * z * cov
    qc = d * d - z * z * vd
    discriminant = qb * qb - 4.0 * qa * qc
    valid = (
        np.isfinite(qa)
        & np.isfinite(qb)
        & np.isfinite(qc)
        & (qa > 0.0)
        & (discriminant >= 0.0)
    )
    root = np.sqrt(np.maximum(discriminant, 0.0))
    first = np.divide(-qb - root, 2.0 * qa, out=np.full(a.shape, np.nan), where=valid)
    second = np.divide(-qb + root, 2.0 * qa, out=np.full(a.shape, np.nan), where=valid)
    low[valid] = np.minimum(first[valid], second[valid])
    high[valid] = np.maximum(first[valid], second[valid])
    bounded[valid] = True
    return low, high, bounded


def classify_inheritance(beta_add_raw, beta_dom_raw, se_add_raw=None,
                         thresholds=DEFAULT_THRESHOLDS, stability_z=3.0):
    """Return coarse and sign-aware inheritance labels.

    The allele suffix describes the homozygote toward which the heterozygote
    is displaced: ``A1`` for dosage 2 and ``A0`` for dosage 0.  Additive
    effects that do not exceed ``stability_z`` standard errors are labelled
    unstable because ``d/a`` is not interpretable there. Labels are descriptive
    annotations, not hypothesis tests.
    """
    a = np.asarray(beta_add_raw, dtype=np.float64)
    d = np.asarray(beta_dom_raw, dtype=np.float64)
    signed = np.full(a.shape, np.nan)
    stable = np.isfinite(a) & np.isfinite(d) & (np.abs(a) > 0)
    if se_add_raw is not None:
        se = np.asarray(se_add_raw, dtype=np.float64)
        stable &= np.isfinite(se) & (np.abs(a) > stability_z * se)
    signed[stable] = d[stable] / a[stable]
    magnitude = np.abs(signed)
    coarse = classify_da(magnitude, thresholds)
    mode = np.full(a.shape, "NA", dtype=object)
    mode[stable & (coarse == "A")] = "additive"
    for code, label in (("PD", "partial_dominant"), ("D", "complete_dominant")):
        select = stable & (coarse == code)
        mode[select & (signed > 0)] = f"{label}_A1"
        mode[select & (signed < 0)] = f"{label}_A0"
    over = stable & (coarse == "OD")
    mode[over & (d > 0)] = "overdominant_high"
    mode[over & (d < 0)] = "underdominant_low"
    unstable = np.isfinite(a) & np.isfinite(d) & ~stable
    coarse[unstable] = "UNSTABLE"
    mode[unstable] = "additive_near_zero"
    return signed, magnitude, coarse, mode
