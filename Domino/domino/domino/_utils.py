"""Shared numeric helpers used across the package."""
import numpy as np


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
    p = np.asarray(p, dtype=np.float64)
    tiny = np.finfo(np.float64).tiny
    return -np.log10(np.clip(p, tiny, 1.0))
