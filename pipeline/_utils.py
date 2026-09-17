"""Shared numeric helpers used across the package."""
import numpy as np
from scipy.special import betaln, gammaln
from scipy.stats import chi2 as _chi2, f as _f, t as _t


def standardize_block(geno, center=True, scale=True, dtype=np.float64):
    """Center/scale a genotype block column-wise, NaN-safe.

    Parameters
    ----------
    geno : ndarray (n_samples, n_variants)
        Genotype dosages in {0, 1, 2} with missing values as NaN.

    Returns
    -------
    z : ndarray (n_samples, n_variants)
        Standardized codes with missing entries set to 0.
    sd : ndarray (n_variants,)
        The per-column divisor actually used (1.0 where ``scale`` is False or
        a column is monomorphic). Needed to recover raw-scale effect sizes.
    mask : ndarray (n_samples, n_variants)
        1.0 where the genotype was observed, else 0.0.
    """
    geno = np.asarray(geno, dtype=dtype)
    mask = ~np.isnan(geno)
    n_obs = mask.sum(0)
    n_obs_safe = np.maximum(n_obs, 1)
    if center:
        mu = np.where(mask, geno, 0.0).sum(0) / n_obs_safe
        z = np.where(mask, geno - mu, 0.0)
    else:
        z = np.where(mask, geno, 0.0)
    if scale:
        ss = (z * z).sum(0)
        sd = np.sqrt(ss / np.maximum(n_obs - 1, 1))
        sd_safe = np.where(sd == 0, 1.0, sd)
        z = z / sd_safe
        z[:, sd == 0] = 0.0
        sd = np.where(sd == 0, np.nan, sd)
    else:
        sd = np.ones(geno.shape[1], dtype=dtype)
    return z.astype(dtype), sd.astype(dtype), mask.astype(dtype)


def dominance_encode(geno, dtype=np.float64):
    """Heterozygote indicator: 1 if genotype == 1, 0 for homozygotes, NaN if missing."""
    geno = np.asarray(geno, dtype=dtype)
    out = np.full(geno.shape, np.nan, dtype=dtype)
    m = ~np.isnan(geno)
    out[m] = (geno[m] == 1).astype(dtype)
    return out


def neglog10p(p):
    """Convert representable probabilities to ``-log10(p)``.

    Association tests should prefer the distribution-specific log-survival
    helpers below, which remain finite when a tail probability underflows.
    """
    p = np.asarray(p, dtype=np.float64)
    tiny = np.finfo(np.float64).tiny
    return -np.log10(np.clip(p, tiny, 1.0))


def _neglog10_from_logp(logp):
    return -np.asarray(logp, dtype=np.float64) / np.log(10.0)


def _log_beta_continued_fraction(a, b, x, max_iterations=400, tolerance=3e-14):
    """Log regularized incomplete beta without exponentiating its tiny prefactor."""
    if x <= 0.0:
        return -np.inf
    if x >= 1.0:
        return 0.0
    if x >= (a + 1.0) / (a + b + 2.0):
        complement = _log_beta_continued_fraction(b, a, 1.0 - x)
        if complement == -np.inf:
            return 0.0
        return float(np.log1p(-np.exp(min(complement, 0.0))))
    floor = np.finfo(np.float64).tiny
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    fraction = d
    for iteration in range(1, max_iterations + 1):
        doubled = 2 * iteration
        coefficient = iteration * (b - iteration) * x / (
            (qam + doubled) * (a + doubled)
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        fraction *= d * c
        coefficient = -(a + iteration) * (qab + iteration) * x / (
            (a + doubled) * (qap + doubled)
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        update = d * c
        fraction *= update
        if abs(update - 1.0) <= tolerance:
            break
    return float(
        a * np.log(x)
        + b * np.log1p(-x)
        - betaln(a, b)
        + np.log(fraction)
        - np.log(a)
    )


def _log_chi2_upper_tail(stat, df):
    stat, df = np.broadcast_arrays(
        np.asarray(stat, dtype=np.float64), np.asarray(df, dtype=np.float64)
    )
    result = np.asarray(_chi2.logsf(stat, df=df), dtype=np.float64)
    for index in np.ndindex(result.shape):
        if np.isfinite(result[index]) or not np.isfinite(stat[index]) or stat[index] <= 0:
            continue
        a = float(df[index]) / 2.0
        x = float(stat[index]) / 2.0
        floor = np.finfo(np.float64).tiny
        b = x + 1.0 - a
        c = 1.0 / floor
        d = 1.0 / max(abs(b), floor)
        if b < 0:
            d = -d
        fraction = d
        for iteration in range(1, 1000):
            coefficient = -iteration * (iteration - a)
            b += 2.0
            d = coefficient * d + b
            if abs(d) < floor:
                d = floor
            c = b + coefficient / c
            if abs(c) < floor:
                c = floor
            d = 1.0 / d
            update = d * c
            fraction *= update
            if abs(update - 1.0) <= 3e-14:
                break
        result[index] = -x + a * np.log(x) - gammaln(a) + np.log(fraction)
    return result


def neglog10_t(stat, df, two_sided=True):
    """Stable t-test ``-log10(p)`` calculated in log-survival space."""
    absolute, degrees = np.broadcast_arrays(
        np.abs(np.asarray(stat, dtype=np.float64)), np.asarray(df, dtype=np.float64)
    )
    logp = np.asarray(_t.logsf(absolute, df=degrees), dtype=np.float64)
    for index in np.ndindex(logp.shape):
        if np.isfinite(logp[index]) or not np.isfinite(absolute[index]):
            continue
        value = float(absolute[index])
        dof = float(degrees[index])
        x = dof / (dof + value * value)
        logp[index] = np.log(0.5) + _log_beta_continued_fraction(
            dof / 2.0, 0.5, x
        )
    if two_sided:
        logp = np.minimum(logp + np.log(2.0), 0.0)
    return _neglog10_from_logp(logp)


def neglog10_f(stat, dfn, dfd):
    """Stable upper-tail F-test ``-log10(p)``."""
    stat, numerator_df, denominator_df = np.broadcast_arrays(
        np.asarray(stat, dtype=np.float64),
        np.asarray(dfn, dtype=np.float64),
        np.asarray(dfd, dtype=np.float64),
    )
    logp = np.asarray(_f.logsf(stat, numerator_df, denominator_df), dtype=np.float64)
    for index in np.ndindex(logp.shape):
        if np.isfinite(logp[index]) or not np.isfinite(stat[index]) or stat[index] < 0:
            continue
        value = float(stat[index])
        first = float(numerator_df[index])
        second = float(denominator_df[index])
        x = second / (second + first * value)
        logp[index] = _log_beta_continued_fraction(second / 2.0, first / 2.0, x)
    return _neglog10_from_logp(logp)


def neglog10_chi2(stat, df):
    """Stable upper-tail chi-square ``-log10(p)``."""
    return _neglog10_from_logp(_log_chi2_upper_tail(stat, df))
